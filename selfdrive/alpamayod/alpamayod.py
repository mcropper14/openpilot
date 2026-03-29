import base64
import math
import socket
import struct
import threading
import time
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image
from cereal import messaging
from msgq.visionipc import VisionIpcClient, VisionStreamType


USE_FPGA = True   #(accel,kappa) from FPGA socket and publish trajectory
                   # False 10B Alpamayo VLM

ALPAMAYO_URL  = "http://127.0.0.1:8084/infer"
FPGA_HOST     = "0.0.0.0"   # bind to all interfaces; FPGA connects to EC2's public IP
FPGA_PORT     = 5555
PUBLISH_HZ    = 10  # FPGA path is fast enough for 10Hz; VLM path self-limits to ~0.3Hz
ENC_W, ENC_H  = 512, 256

# T_IDXS: openpilot's 33-point quadratic time axis  t[j] = 10*(j/32)^2
_T_IDXS = np.array([10.0 * (j / 32.0) ** 2 for j in range(33)], dtype=np.float32)



class FPGAReceiver:
    """
    Listens for the FPGA board's TCP connection (board is the client).
    Receives (accel, kappa) float pairs and stores the latest value.
    Runs in a background thread so the main loop is never blocked.
    """
    def __init__(self, host: str, port: int):
        self._host   = host
        self._port   = port
        self._lock   = threading.Lock()
        self._latest: tuple[float, float] | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def latest(self) -> tuple[float, float] | None:
        with self._lock:
            return self._latest

    def _run(self):
        while True:
            try:
                srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind((self._host, self._port))
                srv.listen(1)
                print(f"alpamayod [fpga]: waiting for board on {self._host}:{self._port}")
                conn, addr = srv.accept()
                print(f"alpamayod [fpga]: board connected from {addr}")
                srv.close()
                while True:
                    data = conn.recv(8)
                    if len(data) < 8:
                        print("alpamayod [fpga]: board disconnected")
                        break
                    accel, kappa = struct.unpack('ff', data)
                    with self._lock:
                        self._latest = (accel, kappa)
                conn.close()
            except Exception as e:
                print(f"alpamayod [fpga]: socket error: {e}, retrying in 2s")
                time.sleep(2)


# ── kinematic integration (accel, kappa) → x,y waypoints ─────────────────────
_N_WP  = 64
_WP_DT = 10.0 / (_N_WP - 1)


def _kinematic_integ(accel: float, kappa: float, v_ego: float) -> tuple[list, list]:
    dt  = _WP_DT
    v   = max(v_ego, 1.0)
    x   = y = theta = 0.0
    xs  = np.empty(_N_WP, dtype=np.float32)
    ys  = np.empty(_N_WP, dtype=np.float32)
    a   = float(np.clip(accel, -4.0, 4.0))
    # Clamp kappa to ±0.05 rad/m (~20m min turn radius) so the 10s trajectory
    # stays in front of the car. The FPGA outputs raw accel/kappa actions and
    # kappa near the ±0.33 bound would curl the path behind the camera.
    kap = float(np.clip(kappa, -0.05, 0.05))
    for i in range(_N_WP):
        v      = max(0.0, v + a * dt)
        theta += kap * v * dt
        x     += v * math.cos(theta) * dt
        y     += v * math.sin(theta) * dt
        xs[i]  = x
        ys[i]  = y
    t_src = np.linspace(_WP_DT, _N_WP * _WP_DT, _N_WP, dtype=np.float32)
    return np.interp(_T_IDXS, t_src, xs).tolist(), np.interp(_T_IDXS, t_src, ys).tolist()


# ── helpers (VLM path, unchanged) ─────────────────────────────────────────────
def _nv12_to_rgb(buf) -> np.ndarray:
    raw    = np.frombuffer(buf.data, dtype=np.uint8)
    y_src  = raw[:buf.stride * buf.height].reshape(buf.height, buf.stride)
    y      = y_src[:, :buf.width]
    uv_src = raw[buf.uv_offset:buf.uv_offset + buf.stride * (buf.height // 2)]
    uv_src = uv_src.reshape(buf.height // 2, buf.stride)
    uv     = uv_src[:, :buf.width]
    nv12   = np.empty((buf.height * 3 // 2, buf.width), dtype=np.uint8)
    nv12[:buf.height, :] = y
    nv12[buf.height:, :] = uv
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
    mv2.position.x   = x33
    mv2.position.y   = y33
    mv2.position.z   = z33
    mv2.position.t   = _T_IDXS.tolist()
    for attr in ("orientation", "velocity", "acceleration"):
        f = getattr(mv2, attr)
        f.x = [0.0] * 33; f.y = [0.0] * 33; f.z = [0.0] * 33
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


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"alpamayod: starting  USE_FPGA={USE_FPGA}")
    sm = messaging.SubMaster(["carState"])
    pm = messaging.PubMaster(["alpamayoDebug", "modelV2"])

    fpga: FPGAReceiver | None = None
    if USE_FPGA:
        fpga = FPGAReceiver(FPGA_HOST, FPGA_PORT)
        # liveCalibration must be published or the model renderer returns early.
        # Publish a flat (level road, camera at 1.2m height) stub once at startup.
        pm_calib = messaging.PubMaster(["liveCalibration"])
        calib_msg = messaging.new_message("liveCalibration")
        calib_msg.liveCalibration.validBlocks = 20
        calib_msg.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
        calib_msg.liveCalibration.wideFromDeviceEuler = [0.0, 0.0, 0.0]
        calib_msg.liveCalibration.height = [1.2]
        calib_msg.liveCalibration.calStatus = 1  # calibrated
        pm_calib.send("liveCalibration", calib_msg)
        print("alpamayod: published stub liveCalibration")

    vipc = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
    print("alpamayod: connecting to VisionIPC camerad/road...")
    vipc.connect(True)
    print("alpamayod: VisionIPC connected")

    dt       = 1.0 / PUBLISH_HZ
    frame_id = 0

    while True:
        loop_t0 = time.monotonic()

        buf = vipc.recv()
        if buf is None:
            time.sleep(0.05)
            continue

        sm.update(0)
        v_ego = float(sm["carState"].vEgo)
        steer = float(sm["carState"].steeringAngleDeg)

        try:
            if USE_FPGA:
                # ── FPGA path ────────────────────────────────────────────────
                pair = fpga.latest()
                if pair is None:
                    time.sleep(0.1)
                    continue

                # Keep liveCalibration fresh so renderer gate stays open
                pm_calib.send("liveCalibration", calib_msg)

                accel, kappa = pair

                # Trajectory: use VLM (working path), FPGA supplies accel/kappa overlay
                rgb      = _nv12_to_rgb(buf)
                jpeg_b64 = _rgb_to_jpeg_b64(rgb)
                payload  = {
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
                    wx  = out.get("waypoints_x", [])
                    wy  = out.get("waypoints_y", [])
                    wz  = out.get("waypoints_z", [])
                    reasoning = out.get("reasoning", "")
                except Exception as vlm_err:
                    print(f"alpamayod [fpga+vlm]: VLM error: {vlm_err}")
                    wx = wy = wz = []
                    reasoning = ""

                print(f"alpamayod [fpga+vlm]: accel={accel:.4f}  kappa={kappa:.4f}  "
                      f"traj_len={len(wx)}  v_ego={v_ego:.1f}m/s")

                if wx:
                    x33, y33, z33 = _resample_to_t_idxs(wx, wy, wz)
                    _publish_model_v2(pm, x33, y33, z33, frame_id=frame_id)
                    frame_id += 1

                dbg = messaging.new_message("alpamayoDebug")
                dbg.alpamayoDebug.valid      = True
                dbg.alpamayoDebug.text       = (f"[fpga] accel={accel:.3f} kappa={kappa:.3f}\n"
                                                + reasoning)
                dbg.alpamayoDebug.confidence = float(out.get("confidence", 0.8)) if wx else 0.8
                dbg.alpamayoDebug.mode       = "fpga+vlm"
                pm.send("alpamayoDebug", dbg)

            else:
                # ── VLM path (unchanged) ─────────────────────────────────────
                rgb      = _nv12_to_rgb(buf)
                jpeg_b64 = _rgb_to_jpeg_b64(rgb)
                payload  = {
                    "timestamp_ns":       int(time.time() * 1e9),
                    "ego_speed":          v_ego,
                    "steering_angle_deg": steer,
                    "camera_names":       ["road"],
                    "images_b64":         [jpeg_b64],
                }

                resp = requests.post(ALPAMAYO_URL, json=payload, timeout=(2.0, 30.0))
                resp.raise_for_status()
                out = resp.json()

                wx        = out.get("waypoints_x", [])
                wy        = out.get("waypoints_y", [])
                wz        = out.get("waypoints_z", [])
                reasoning = out.get("reasoning", "")
                latency   = out.get("latency_ms", 0)

                print(f"alpamayod [vlm]: frame={buf.width}x{buf.height}  "
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
                dbg.alpamayoDebug.mode       = "vlm"
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
