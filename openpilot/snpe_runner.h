// snpe_runner.h
// Compile with -DUSE_MOCK_SNPE to use the CPU stub (EC2 / x86_64 dev builds).
// Without that flag the real SNPE SDK is used (Snapdragon on-device builds).
#pragma once
#ifdef USE_MOCK_SNPE
#include "snpe_runner_mock.h"
#else
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/ITensor.hpp"
#include "DlSystem/TensorMap.hpp"
#include "DlSystem/TensorShape.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include <memory>
#include <cstring>
#include <stdexcept>

// Wraps one SNPE model. Call forward() per inference.
class SNPERunner {
public:
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    std::string input_name;
    std::string output_name;

    SNPERunner(const char* dlc_path,
               const char* in_name,
               const char* out_name,
               zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::DSP)
        : input_name(in_name), output_name(out_name)
    {
        // load container
        auto container = zdl::DlContainer::IDlContainer::open(dlc_path);
        if (!container)
            throw std::runtime_error(std::string("failed to open DLC: ") + dlc_path);

        // build SNPE instance targeting the DSP
        zdl::DlSystem::RuntimeList rt_list;
        rt_list.add(runtime);

        zdl::SNPE::SNPEBuilder builder(container.get());
        snpe = builder
            .setRuntimeProcessorOrder(rt_list)
            .setUseUserSuppliedBuffers(false)   // use ITensor path
            .build();

        if (!snpe)
            throw std::runtime_error("SNPE build failed");
    }

    // Run one forward pass.
    // in:      pointer to float input data
    // in_len:  number of floats
    // out:     pointer to float output buffer (pre-allocated)
    // out_len: number of floats expected
    void forward(const float* in, int in_len, float* out, int out_len)
    {
        // ── build input tensor ────────────────────────────────────────
        zdl::DlSystem::TensorShape in_shape({1, (unsigned long)in_len});
        auto in_tensor = zdl::SNPE::SNPEFactory::getTensorFactory()
                             .createTensor(in_shape);
        if (!in_tensor)
            throw std::runtime_error("failed to create input tensor");

        // copy float data into the tensor's backing store
        // ITensor exposes begin()/end() iterators over float
        std::copy(in, in + in_len, in_tensor->begin());

        // ── set up tensor maps ────────────────────────────────────────
        zdl::DlSystem::TensorMap input_map;
        input_map.add(input_name.c_str(), in_tensor.get());

        zdl::DlSystem::TensorMap output_map;
        // output tensor is allocated by SNPE on execute()
        // we'll read it back after

        // ── execute ───────────────────────────────────────────────────
        bool ok = snpe->execute(input_map, output_map);
        if (!ok)
            throw std::runtime_error("SNPE execute failed");

        // ── read output ───────────────────────────────────────────────
        auto out_tensor_ref = output_map.getTensor(output_name.c_str());
        if (!out_tensor_ref)
            throw std::runtime_error("output tensor not found in result map");

        int i = 0;
        for (auto it = out_tensor_ref->cbegin();
             it != out_tensor_ref->cend() && i < out_len;
             ++it, ++i)
        {
            out[i] = *it;
        }
    }
};
#endif // USE_MOCK_SNPE