// denoiser.cc
#include "snpe_runner.h"
#include <cmath>
#include <cstring>
#include <random>
#include <stdio.h>

// ── constants (must match models.py / export_onnx.py) ────────────────────────
static const int N_STEPS  = 10;
static const int DIM_ACT  = 128;  // 64 waypoints × (x, y)
static const int DIM_COND = 256;  // encoder output
static const int DIM_TEMB = 64;   // fourier timestep embedding
static const int DIM_IN   = DIM_ACT + DIM_TEMB + DIM_COND; // 448

// Teacher waypoints: 64 points linearly spaced over [0, 10] seconds
// t_teacher[i] = i * (10.0 / 63.0)
static const int   N_WP       = 64;
static const float WP_DT      = 10.0f / 63.0f;  // ~0.1587s between waypoints

// ── fourier timestep embedding ────────────────────────────────────────────────
// fills out[0..DIM_TEMB-1] with sin/cos features for scalar t in [0,1]
static void fourier_embed(float t, float* out) {
    for (int i = 0; i < DIM_TEMB / 2; i++) {
        float freq = powf(10000.0f, -2.0f * i / (float)DIM_TEMB);
        out[2 * i]     = sinf(t * freq);
        out[2 * i + 1] = cosf(t * freq);
    }
}

// ── gaussian noise via Box-Muller ─────────────────────────────────────────────
static std::mt19937 rng(42);
static std::normal_distribution<float> normal_dist(0.0f, 1.0f);
static float randn() { return normal_dist(rng); }

// ── resample 64-waypoint trajectory to modelV2 T_IDXS (33 points) ────────────
// wp:      (N_WP, 2) flattened as [x0,y0, x1,y1, ...]  — teacher coordinate frame
// v_ego:   vehicle speed (unused here; kept for API consistency)
// wx, wy:  output arrays, each (33,) floats
//
// T_IDXS: t[j] = 10*(j/32)^2  (quadratic, 0..10s)
// Teacher: t[i] = i * WP_DT   (linear,    0..10s)
// Simple linear interpolation between teacher waypoints.
static void resample_to_t_idxs(const float* wp, float /*v_ego*/,
                                float* wx, float* wy)
{
    for (int j = 0; j < 33; j++) {
        float fj  = (float)j / 32.0f;
        float t   = 10.0f * fj * fj;         // T_IDXS[j] seconds

        float raw = t / WP_DT;               // fractional index into teacher wp
        if (raw <= 0.0f) {
            wx[j] = wp[0];
            wy[j] = wp[1];
        } else if (raw >= (float)(N_WP - 1)) {
            // extrapolate beyond teacher horizon at last velocity
            float dt_extra = t - (float)(N_WP - 1) * WP_DT;
            int   last     = N_WP - 1;
            float vx = wp[last * 2]     - wp[(last - 1) * 2];
            float vy = wp[last * 2 + 1] - wp[(last - 1) * 2 + 1];
            wx[j] = wp[last * 2]     + vx * (dt_extra / WP_DT);
            wy[j] = wp[last * 2 + 1] + vy * (dt_extra / WP_DT);
        } else {
            int   lo = (int)raw;
            int   hi = lo + 1;
            float fr = raw - (float)lo;
            wx[j] = wp[lo * 2]     + fr * (wp[hi * 2]     - wp[lo * 2]);
            wy[j] = wp[lo * 2 + 1] + fr * (wp[hi * 2 + 1] - wp[lo * 2 + 1]);
        }
    }
}

// ── main denoising entry point ────────────────────────────────────────────────
// cond:     conditioning vector from encoder, (DIM_COND,) floats
// v_ego:    current vehicle speed in m/s (from carState cereal msg)
// wx, wy:   output arrays, each (33,) floats — filled by this function
// denoiser: pre-loaded SNPERunner pointing at denoiser_q.dlc on DSP
void run_flow_matching(const float* cond,
                       float v_ego,
                       float* wx,
                       float* wy,
                       SNPERunner& denoiser)
{
    // x_0 ~ N(0, I)  shape: (DIM_ACT,) = (128,)
    float x_t[DIM_ACT];
    for (int i = 0; i < DIM_ACT; i++)
        x_t[i] = randn();

    float packed[DIM_IN];
    float t_emb[DIM_TEMB];
    float v_out[DIM_ACT];

    for (int step = 0; step < N_STEPS; step++) {
        // midpoint timestep in [0,1] (flow matching convention)
        float t = (step + 0.5f) / (float)N_STEPS;

        // pack: [x_t (128) | fourier(t) (64) | cond (256)]  → 448 floats
        memcpy(packed,                        x_t,   DIM_ACT  * sizeof(float));
        fourier_embed(t, t_emb);
        memcpy(packed + DIM_ACT,              t_emb, DIM_TEMB * sizeof(float));
        memcpy(packed + DIM_ACT + DIM_TEMB,   cond,  DIM_COND * sizeof(float));

        // denoiser forward: packed(448) → velocity v(128)
        denoiser.forward(packed, DIM_IN, v_out, DIM_ACT);

        // Euler step: x_t += (1/N) * v
        for (int i = 0; i < DIM_ACT; i++)
            x_t[i] += (1.0f / N_STEPS) * v_out[i];
    }

    // x_1 now holds 64 (x, y) waypoints in teacher coordinate frame
    // resample to modelV2's 33-point quadratic T_IDXS grid
    resample_to_t_idxs(x_t, v_ego, wx, wy);
}
