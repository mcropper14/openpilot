import base64
import time
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image
from cereal import messaging
from msgq.visionipc import VisionIpcClient, VisionStreamType

ALPAMAYO_URL = "http://127.0.0.1:8084/infer"
PUBLISH_HZ   = 1          # max inference rate (server is ~2-3s per frame)
ENC_W, ENC_H = 512, 256   # resize to encoder input size before sending

# T_IDXS: openpilot's 33-point quadratic time axis  t[j] = 10*(j/32)^2
_T_IDXS = np.array([10.0 * (j / 32.0) ** 2 for j in range(33)], dtype=np.float32)


def _nv12_to_rgb(buf) -> np.ndarray:
    """Convert VisionIPC NV12 buffer → (H, W, 3) uint8 RGB array."""
    raw = np.frombuffer(buf.data, dtype=np.uint8)

    # Y plane: height × stride
    y_src = raw[:buf.stride * buf.height].reshape(buf.height, buf.stride)
    y = y_src[:, :buf.width]

    # UV plane (interleaved): (height/2) × stride
    uv_src = raw[buf.uv_offset:buf.uv_offset + buf.stride * (buf.height // 2)]
    uv_src = uv_src.reshape(buf.height // 2, buf.stride)
    uv = uv_src[:, :buf.width]

    # Build contiguous NV12 array expected by cv2: (H*3//2, W)
    nv12 = np.empty((buf.height * 3 // 2, buf.width), dtype=np.uint8)
    nv12[:buf.height, :]  = y
    nv12[buf.height:, :]  = uv

    return cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12)


def _rgb_to_jpeg_b64(rgb: np.ndarray) -> str:
    resized = cv2.resize(rgb, (ENC_W, ENC_H), interpolation=cv2.INTER_AREA)
    buf = BytesIO()
    Image.fromarray(resized).save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _resample_to_t_idxs(wx, wy, wz):
    n = len(wx)
    if n == 0:
        z = [0.0] * 33
        return z, z, z
    t_src = np.linspace(0.0, 10.0, n, dtype=np.float32)
    return (np.interp(_T_IDXS, t_src, wx).tolist(),
            np.interp(_T_IDXS, t_src, wy).tolist(),
            np.interp(_T_IDXS, t_src, wz).tolist())


def _publish_model_v2(pm, x33, y33, z33, frame_id=0):
    msg = messaging.new_message("modelV2")
    mv2 = msg.modelV2
    mv2.frameId      = frame_id
    mv2.frameIdExtra = frame_id
    mv2.confidence   = "green"

    mv2.position.x = x33
    mv2.position.y = y33
    mv2.position.z = z33
    mv2.position.t = _T_IDXS.tolist()

    for attr in ("orientation", "velocity", "acceleration"):
        f = getattr(mv2, attr)
        f.x = [0.0] * 33
        f.y = [0.0] * 33
        f.z = [0.0] * 33
        f.t = _T_IDXS.tolist()

    mv2.init("laneLines", 4)
    for ll in mv2.laneLines:
        ll.x = []; ll.y = []; ll.z = []; ll.t = []
    mv2.laneLineProbs = [0.0] * 4
    mv2.laneLineStds  = [1.0] * 4

    mv2.init("roadEdges", 2)
    for re in mv2.roadEdges:
        re.x = []; re.y = []; re.z = []; re.t = []
    mv2.roadEdgeStds = [1.0] * 2

    pm.send("modelV2", msg)


def main():
    print("alpamayod: starting")
    sm = messaging.SubMaster(["carState"])
    pm = messaging.PubMaster(["alpamayoDebug", "modelV2"])

    # Connect to the same VisionIPC road camera stream the UI uses
    vipc = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
    print("alpamayod: connecting to VisionIPC camerad/road...")
    vipc.connect(True)
    print("alpamayod: VisionIPC connected")

    dt       = 1.0 / PUBLISH_HZ
    frame_id = 0

    while True:
        loop_t0 = time.monotonic()

        # Pull latest camera frame
        buf = vipc.recv()
        if buf is None:
            time.sleep(0.05)
            continue

        sm.update(0)
        v_ego    = float(sm["carState"].vEgo)
        steer    = float(sm["carState"].steeringAngleDeg)

        rgb      = _nv12_to_rgb(buf)
        jpeg_b64 = _rgb_to_jpeg_b64(rgb)

        payload = {
            "timestamp_ns":       int(time.time() * 1e9),
            "ego_speed":          v_ego,
            "steering_angle_deg": steer,
            "camera_names":       ["road"],
            "images_b64":         [jpeg_b64],
        }

        try:
            resp = requests.post(ALPAMAYO_URL, json=payload, timeout=(2.0, 30.0))
            resp.raise_for_status()
            out = resp.json()

            wx        = out.get("waypoints_x", [])
            wy        = out.get("waypoints_y", [])
            wz        = out.get("waypoints_z", [])
            reasoning = out.get("reasoning", "")
            latency   = out.get("latency_ms", 0)

            print(f"alpamayod: frame={buf.width}x{buf.height}  "
                  f"traj_len={len(wx)}  latency={latency:.0f}ms  "
                  f"v_ego={v_ego:.1f}m/s")
            if reasoning:
                preview = reasoning.replace("\n", " | ")[:300]
                print(f"alpamayod: reasoning: {preview}"
                      f"{'...' if len(reasoning) > 300 else ''}")

            if wx:
                x33, y33, z33 = _resample_to_t_idxs(wx, wy, wz)
                _publish_model_v2(pm, x33, y33, z33, frame_id=frame_id)
                frame_id += 1

            dbg = messaging.new_message("alpamayoDebug")
            dbg.alpamayoDebug.valid      = True
            dbg.alpamayoDebug.text       = reasoning
            dbg.alpamayoDebug.confidence = float(out.get("confidence", 0.0))
            dbg.alpamayoDebug.mode       = out.get("mode", "ok")
            pm.send("alpamayoDebug", dbg)

        except Exception as e:
            print(f"alpamayod: error: {e}")
            dbg = messaging.new_message("alpamayoDebug")
            dbg.alpamayoDebug.valid      = False
            dbg.alpamayoDebug.text       = f"error: {e}"
            dbg.alpamayoDebug.confidence = 0.0
            dbg.alpamayoDebug.mode       = "error"
            pm.send("alpamayoDebug", dbg)

        elapsed = time.monotonic() - loop_t0
        if elapsed < dt:
            time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
