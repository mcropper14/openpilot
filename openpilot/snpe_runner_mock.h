// snpe_runner_mock.h
// CPU-only stub with the same interface as snpe_runner.h.
// Used for EC2 / x86_64 development builds where no Qualcomm DSP is present.
//
// The encoder stub returns a fixed sinusoidal conditioning vector so the
// denoiser has a deterministic, non-trivial input to work with.
// The denoiser stub returns small Gaussian noise so the flow-matching loop
// produces a plausible (but random) trajectory -- good enough to verify that
// VisionIPC ingestion, frame processing, and cereal publishing all work.
#pragma once
#include <cstring>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>

class SNPERunner {
public:
    std::string input_name;
    std::string output_name;
    std::string dlc_path;

    // mt19937 seeded per-instance so encoder and denoiser produce independent noise
    std::mt19937 rng;
    std::normal_distribution<float> normal_dist{0.0f, 0.1f};

    SNPERunner(const char* dlc_path_,
               const char* in_name,
               const char* out_name,
               int /*runtime_ignored*/ = 0)
        : input_name(in_name), output_name(out_name), dlc_path(dlc_path_),
          rng(std::hash<std::string>{}(std::string(dlc_path_)))
    {
        printf("[mock SNPE] loaded %s  in=%s  out=%s\n", dlc_path_, in_name, out_name);
    }

    void forward(const float* in, int in_len, float* out, int out_len)
    {
        if (dlc_path.find("encoder") != std::string::npos) {
            // Encoder stub: sinusoidal features derived from mean input energy.
            // Gives a stable, deterministic conditioning signal for debugging.
            float energy = 0.0f;
            for (int i = 0; i < in_len; i++) energy += in[i];
            energy /= (float)in_len;

            for (int i = 0; i < out_len; i++) {
                float freq = (float)(i + 1) * 0.1f;
                out[i] = 0.5f * sinf(energy * freq) + normal_dist(rng) * 0.05f;
            }
        } else {
            // Denoiser stub: small Gaussian noise → flow-matching produces a
            // gentle curved trajectory.
            for (int i = 0; i < out_len; i++)
                out[i] = normal_dist(rng);
        }
    }
};
