#include <cstdio>
#include <cstring>
#include "../core/parity.hpp"
int main() {
  distribai::ParityMlp m(42);
  const auto w = m.flat_weights();
  for (int i = 0; i < 4; ++i) {
    uint32_t b; std::memcpy(&b, &w[i], 4);
    std::printf("W1[%d] bits=0x%08x val=%.9f\n", i, b, w[i]);
  }
  distribai::TorchRng r2(42);
  const float v = r2.uniform(-1.0/std::sqrt(10.0), 1.0/std::sqrt(10.0));
  uint32_t b2; std::memcpy(&b2, &v, 4);
  std::printf("direct uniform bits=0x%08x\n", b2);
  return 0;
}
