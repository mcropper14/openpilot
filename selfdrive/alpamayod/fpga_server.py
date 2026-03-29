"""
fpga_server.py — FastAPI server wrapping the quantized MLP on the FPGA PS (ARM).

Run on the Xilinx board:
    pip install fastapi uvicorn numpy
    python fpga_server.py          # listens on 0.0.0.0:8085

The server runs the full flow-matching Euler loop:
  - 64 waypoints × N_STEPS Euler steps  (N_STEPS=10 default, 4 for speed)
  - Each step: fourier_embed(accel_i, kappa_i, t) → run_mlp → velocity
  - After denoising: kinematic integration → x,y waypoints at 33 T_IDXS points
  - Returns JSON compatible with alpamayod's existing waypoints_x/waypoints_y format

Fourier embedding matches PerWaypointActionInProjV2 (action_in_proj.py):
  FourierEncoderV2(dim=20, max_freq=100) = logspace(1,100,10) → sin+cos × sqrt(2)
  Input order: [fourier(accel) | fourier(kappa) | fourier(t)]  → 60-dim
"""

import math
import time
import threading

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# ── load weights ──────────────────────────────────────────────────────────────
d0 = np.load("/home/xilinx/action_in_proj_encoder_trunk_0.npz")
d1 = np.load("/home/xilinx/action_in_proj_encoder_trunk_3.npz")
d2 = np.load("/home/xilinx/action_in_proj_encoder_trunk_6.npz")
d3 = np.load("/home/xilinx/action_out_proj_.npz")

W0 = d0["weight_q"].astype(np.int8)
W1 = d1["weight_q"].astype(np.int8)
W2 = d2["weight_q"].astype(np.int8)
W3 = d3["weight_q"].astype(np.int8)

s0 = d0["scales"].astype(np.float32)
s1 = d1["scales"].astype(np.float32)
s2 = d2["scales"].astype(np.float32)
s3 = d3["scales"].astype(np.float32)

b0 = d0["bias"].astype(np.float32)
b1 = d1["bias"].astype(np.float32)
b2 = d2["bias"].astype(np.float32)
b3 = d3["bias"].astype(np.float32)

print(f"Weights loaded: W0{W0.shape} W1{W1.shape} W2{W2.shape} W3{W3.shape}")

# ── constants ─────────────────────────────────────────────────────────────────
N_WAYPOINTS = 64
N_STEPS     = 10         # Euler integration steps (set to 4 for ~2.5× speed)
WP_DT       = 10.0 / (N_WAYPOINTS - 1)   # teacher time step between waypoints

# T_IDXS: modelV2's 33-point quadratic time axis  t[j] = 10*(j/32)^2
_T_IDXS = np.array([10.0 * (j / 32.0) ** 2 for j in range(33)], dtype=np.float32)

# Fourier frequencies: logspace(log10(1), log10(100), 10) = 10 values
_HALF    = 10
_FREQS   = np.logspace(0, math.log10(100.0), num=_HALF).astype(np.float32)  # (10,)
_SQRT2   = math.sqrt(2.0)

# ── fourier embedding (matches FourierEncoderV2 in action_in_proj.py) ────────
def _fourier20(x: float) -> np.ndarray:
    """Scalar → 20-dim Fourier features: [sin(x*f*2π), cos(x*f*2π)] × sqrt(2)"""
    arg = x * _FREQS * (2.0 * math.pi)           # (10,)
    return np.concatenate([np.sin(arg), np.cos(arg)]) * _SQRT2  # (20,)


def fourier_embed(accel: float, kappa: float, t: float) -> np.ndarray:
    """(accel, kappa, t) → 60-dim input for run_mlp."""
    return np.concatenate([_fourier20(accel), _fourier20(kappa), _fourier20(t)])


# ── quantized MLP forward ─────────────────────────────────────────────────────
def _dense(x: np.ndarray, W, scale, bias, relu: bool = True) -> np.ndarray:
    x_q = np.clip(np.round(x), -128, 127).astype(np.int8)
    acc  = W.astype(np.int32) @ x_q.astype(np.int32)
    y    = acc.astype(np.float32) * scale + bias
    return np.maximum(y, 0.0) if relu else y


def run_mlp(hidden_vec: np.ndarray) -> tuple[float, float]:
    """60-dim fourier embedding → (v_accel, v_kappa) velocity for one Euler step."""
    x = hidden_vec.astype(np.float32)
    x = _dense(x, W0, s0, b0, relu=True)
    x = _dense(x, W1, s1, b1, relu=True)
    x = _dense(x, W2, s2, b2, relu=True)
    x = _dense(x, W3, s3, b3, relu=False)
    return float(x[0]), float(x[1])


# ── Euler flow-matching loop ──────────────────────────────────────────────────
def run_denoising(n_steps: int = N_STEPS) -> tuple[np.ndarray, np.ndarray]:
    """
    Flow-matching Euler integration for all N_WAYPOINTS.
    Returns (accel_seq, kappa_seq) each shape (N_WAYPOINTS,).
    """
    rng = np.random.default_rng()
    accel = rng.standard_normal(N_WAYPOINTS).astype(np.float32)
    kappa = rng.standard_normal(N_WAYPOINTS).astype(np.float32)

    for step in range(n_steps):
        t = (step + 0.5) / n_steps   # midpoint in [0,1]
        v_accel = np.empty(N_WAYPOINTS, dtype=np.float32)
        v_kappa = np.empty(N_WAYPOINTS, dtype=np.float32)

        for i in range(N_WAYPOINTS):
            h = fourier_embed(float(accel[i]), float(kappa[i]), t)
            va, vk = run_mlp(h)
            v_accel[i] = va
            v_kappa[i] = vk

        accel += v_accel / n_steps
        kappa += v_kappa / n_steps

    return accel, kappa


# ── kinematic integration (accel/kappa → x,y) ────────────────────────────────
def kinematic_integ(accel_seq: np.ndarray, kappa_seq: np.ndarray,
                    v_ego: float) -> tuple[list, list]:
    """
    Integrate (accel, kappa) at dt=WP_DT → (x, y) in meters.
    Returns (wx, wy) resampled to 33-point modelV2 T_IDXS grid.
    """
    dt = WP_DT
    v  = max(v_ego, 1.0)
    x = y = theta = 0.0
    xs = np.empty(N_WAYPOINTS, dtype=np.float32)
    ys = np.empty(N_WAYPOINTS, dtype=np.float32)

    for i in range(N_WAYPOINTS):
        a     = float(np.clip(accel_seq[i], -4.0, 4.0))
        kap   = float(np.clip(kappa_seq[i], -0.2, 0.2))
        v     = max(0.0, v + a * dt)
        theta += kap * v * dt
        x     += v * math.cos(theta) * dt
        y     += v * math.sin(theta) * dt
        xs[i]  = x
        ys[i]  = y

    # Resample to T_IDXS quadratic grid
    t_src = np.linspace(WP_DT, N_WAYPOINTS * WP_DT, N_WAYPOINTS, dtype=np.float32)
    wx = np.interp(_T_IDXS, t_src, xs).tolist()
    wy = np.interp(_T_IDXS, t_src, ys).tolist()
    return wx, wy


# ── FastAPI server ────────────────────────────────────────────────────────────
app   = FastAPI()
_lock = threading.Lock()   # MLP weights are shared; serialize requests


class InferRequest(BaseModel):
    v_ego:   float = 0.0
    n_steps: int   = N_STEPS


class InferResponse(BaseModel):
    waypoints_x: list[float]
    waypoints_y: list[float]
    latency_ms:  float


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    t0 = time.perf_counter()
    with _lock:
        accel_seq, kappa_seq = run_denoising(n_steps=req.n_steps)
        wx, wy = kinematic_integ(accel_seq, kappa_seq, v_ego=req.v_ego)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return InferResponse(waypoints_x=wx, waypoints_y=wy, latency_ms=latency_ms)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    print(f"FPGA MLP server starting on 0.0.0.0:8085")
    print(f"  N_WAYPOINTS={N_WAYPOINTS}  N_STEPS={N_STEPS}  WP_DT={WP_DT:.4f}s")
    uvicorn.run(app, host="0.0.0.0", port=8085)
