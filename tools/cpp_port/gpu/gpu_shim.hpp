// GPU layer API (implementation: gpu_shim.cpp on CPU, fused_kernels.cu on CUDA).
#pragma once

#include <cstdint>

namespace distribai {
namespace gpu {

// True when a CUDA device is available at runtime.
bool cuda_available();

// Fused MLP forward+backward+AdamW step (CUDA path). Returns 0 on success,
// nonzero when the caller should stay on the portable CPU loop.
extern "C" int distribai_gpu_fused_step(
    const float* x, const float* y,
    float* w1, float* b1, float* w2, float* b2, float* w3, float* b3,
    float* m_all, float* v_all,
    int n, float lr, float wd, int64_t t);

}  // namespace gpu
}  // namespace distribai
