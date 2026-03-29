// alpamayo_bridge.cc
//
// ── EC2 / x86_64 dev build (mock SNPE, no DLC needed) ───────────────────────
// VENV=/home/ubuntu/openpilot/.venv/lib/python3.12/site-packages
// OP=/home/ubuntu/openpilot
// g++ -O2 -std=c++17 -DUSE_MOCK_SNPE \
//   -I${OP} -I${OP}/msgq_repo \
//   -I${VENV}/capnproto/install/include \
//   -I${VENV}/libyuv/install/include \
//   -I${VENV}/zeromq/install/include \
//   alpamayo_bridge.cc denoiser.cc \
//   -Wl,--start-group \
//   ${OP}/cereal/libcereal.a ${OP}/cereal/libsocketmaster.a \
//   ${OP}/common/libcommon.a ${OP}/third_party/libjson11.a \
//   ${OP}/msgq_repo/libmsgq.a ${OP}/msgq_repo/libvisionipc.a \
//   ${VENV}/capnproto/install/lib/libcapnp.a \
//   ${VENV}/capnproto/install/lib/libkj.a \
//   ${VENV}/libyuv/install/lib/libyuv.a \
//   ${VENV}/zeromq/install/lib/libzmq.a \
//   -Wl,--end-group -lpthread -lrt \
//   -o alpamayo_bridge_mock
//
// ── Snapdragon / larch64 on-device build (real SNPE DSP) ────────────────────
// aarch64-linux-gnu-g++ -O2 -std=c++17 \
//   -I$SNPE/include/SNPE \
//   -I/home/ubuntu/openpilot/msgq_repo \
//   alpamayo_bridge.cc denoiser.cc \
//   -L$SNPE/lib/aarch64-linux-gcc -lSNPE \
//   -lcereal -lmessaging -lyuv -lpthread \
//   -o alpamayo_bridge

#include "snpe_runner.h"
#include "denoiser.h"
#include "cereal/messaging/messaging.h"
#include "msgq/visionipc/visionipc_client.h"
#include <libyuv.h>
#include <libyuv/scale_rgb.h>
#include <cstdio>

static const int ENC_W = 512, ENC_H = 256;   // encoder input resolution
static const int DIM_COND = 256;

int main() {
    // ── load models onto DSP ─────────────────────────────────────────
    SNPERunner encoder("/data/encoder_q.dlc",  "frame",        "conditioning");
    SNPERunner denoiser("/data/denoiser_q.dlc", "packed_input", "velocity");

    printf("models loaded on DSP\n");

    // ── cereal sockets ───────────────────────────────────────────────
    PubMaster pm({"modelV2"});
    SubMaster sm({"carState"});

    VisionIpcClient vipc("camerad", VISION_STREAM_ROAD, true);
    vipc.connect(true);

    printf("sockets up, entering loop\n");

    // pre-allocate buffers
    static uint8_t rgb_buf[ENC_W * ENC_H * 3];
    static float   float_frame[ENC_W * ENC_H * 3];
    static float   cond[DIM_COND];
    static float   wx[33], wy[33];

    while (true) {
        // ── grab frame ───────────────────────────────────────────────
        VisionIpcBufExtra extra{};
        VisionBuf* buf = vipc.recv(&extra);
        if (!buf) continue;

        sm.update(0);
        float v_ego = sm["carState"].getCarState().getVEgo();

        // ── NV12 → RGB24 → resize to (256, 512) ─────────────────────
        // comma 3 road cam: 1928 × 1208 NV12 (Y plane + interleaved UV)
        int src_w = buf->width, src_h = buf->height;
        int src_stride = buf->stride;

        // libyuv: NV12 → RGB24, then scale
        static uint8_t rgb_full[1928 * 1208 * 3];
        libyuv::NV12ToRGB24(
            buf->y,  src_stride,
            buf->uv, src_stride,
            rgb_full, src_w * 3,
            src_w, src_h);

        libyuv::RGBScale(
            rgb_full, src_w * 3, src_w, src_h,
            rgb_buf,  ENC_W * 3,  ENC_W, ENC_H,
            libyuv::kFilterBilinear);

        // normalize to float [0,1]
        for (int i = 0; i < ENC_W * ENC_H * 3; i++)
            float_frame[i] = rgb_buf[i] / 255.0f;

        // ── encoder forward ──────────────────────────────────────────
        encoder.forward(float_frame, ENC_W * ENC_H * 3, cond, DIM_COND);

        // ── flow matching denoiser loop ──────────────────────────────
        run_flow_matching(cond, v_ego, wx, wy, denoiser);

        // ── publish modelV2 ──────────────────────────────────────────
        // T_IDXS: t[j] = 10*(j/32)^2  (33 points, 0..10s, quadratic spacing)
        MessageBuilder msg;
        auto mv2 = msg.initEvent(true).initModelV2();
        mv2.setFrameId(extra.frame_id);
        mv2.setFrameIdExtra(extra.frame_id);
        mv2.setConfidence(cereal::ModelDataV2::ConfidenceClass::GREEN);

        // Helper: fill an XYZTData builder with T_IDXS time axis, zeros for x/y/z
        auto fill_empty_xyzt = [](auto builder) {
            auto pt = builder.initT(33);
            builder.initX(33); builder.initY(33); builder.initZ(33);
            for (int i = 0; i < 33; i++) {
                float fj = (float)i / 32.0f;
                pt.set(i, 10.0f * fj * fj);
            }
        };

        // Position — filled from the denoiser trajectory
        {
            auto pos = mv2.initPosition();
            auto px = pos.initX(33);
            auto py = pos.initY(33);
            auto pz = pos.initZ(33);
            auto pt = pos.initT(33);
            for (int i = 0; i < 33; i++) {
                float fj = (float)i / 32.0f;
                px.set(i, wx[i]);
                py.set(i, wy[i]);
                pz.set(i, 0.0f);
                pt.set(i, 10.0f * fj * fj);
            }
        }

        // Orientation, velocity, acceleration — zero stubs required by UI
        fill_empty_xyzt(mv2.initOrientation());
        fill_empty_xyzt(mv2.initVelocity());
        fill_empty_xyzt(mv2.initAcceleration());

        // Lane lines (4 empty lines) + probabilities + road edges (2)
        // The renderer iterates laneLineProbs[i] for each of the 4 lane lines.
        {
            auto ll = mv2.initLaneLines(4);
            for (int i = 0; i < 4; i++) fill_empty_xyzt(ll[i]);
            auto llp = mv2.initLaneLineProbs(4);
            for (int i = 0; i < 4; i++) llp.set(i, 0.0f);
            auto lls = mv2.initLaneLineStds(4);
            for (int i = 0; i < 4; i++) lls.set(i, 1.0f);
        }
        {
            auto re = mv2.initRoadEdges(2);
            for (int i = 0; i < 2; i++) fill_empty_xyzt(re[i]);
            auto res = mv2.initRoadEdgeStds(2);
            for (int i = 0; i < 2; i++) res.set(i, 1.0f);
        }

        pm.send("modelV2", msg);
    }
}