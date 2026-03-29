// denoiser.h
#pragma once
#include "snpe_runner.h"

// Run the flow-matching denoiser loop.
// cond:     conditioning vector from encoder, (256,) floats
// v_ego:    current vehicle speed in m/s (from carState)
// wx, wy:   output trajectory arrays, each (33,) floats — aligned to T_IDXS
// denoiser: pre-loaded SNPERunner for denoiser_q.dlc
void run_flow_matching(const float* cond,
                       float v_ego,
                       float* wx,
                       float* wy,
                       SNPERunner& denoiser);
