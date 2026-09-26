#include <cstdio>
#include <cstring>
#include <cmath>
#include "core/mt19937.hpp"

int main() {
  distribai::TorchRng r(42);
  const double bound = 1.0 / std::sqrt(10.0);
  std::printf("bound double: %.17g\n", bound);
  for (int i = 0; i < 4; ++i) {
    const uint32_t raw = r.next_u32();
    const float u = static_cast<float>(raw & 0xFFFFFFu) * (1.0f / 16777216.0f);
    const double du = static_cast<double>(u);
    const float v1 = static_cast<float>(-bound + (bound - (-bound)) * du);
    uint32_t b1; std::memcpy(&b1, &v1, 4);
    std::printf("raw=0x%08x u=%.9f -> -b+2b*u bits=0x%08x\n", raw, u, b1);
  }
  // via the uniform() API:
  distribai::TorchRng r2(42);
  for (int i = 0; i < 4; ++i) {
    const float v = r2.uniform(-bound, bound);
    uint32_t b; std::memcpy(&b, &v, 4);
    std::printf("uniform() bits=0x%08x\n", b);
  }
  return 0;
}
