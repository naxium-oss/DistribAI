// CPU fallback for the GPU acceleration layer.
// When fused_kernels.cu is not compiled (no nvcc) or no CUDA device exists
// at runtime, these symbols keep the same API and route to the portable
// CPU implementations. The bench binary reports gpu_available=false and
// every benchmark result is honestly labeled CPU.
#include <cstdint>

namespace distribai {
namespace gpu {

bool cuda_available() { return false; }

// Same signature as the CUDA host entry; CPU path does nothing here because
// the portable training loop in core/tensor.hpp performs the equivalent
// fused operation inline (single pass, cache-resident at 1K params).
extern "C" int distribai_gpu_fused_step(
    const float* /*x*/, const float* /*y*/,
    float* /*w1*/, float* /*b1*/, float* /*w2*/, float* /*b2*/, float* /*w3*/, float* /*b3*/,
    float* /*m_all*/, float* /*v_all*/,
    int /*n*/, float /*lr*/, float /*wd*/, int64_t /*t*/) {
  return 1;  // signals "not accelerated"; caller stays on the CPU loop
}

}  // namespace gpu
}  // namespace distribai
