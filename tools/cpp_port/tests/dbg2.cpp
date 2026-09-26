#include <cstdio>
#include <cstring>
#include "../core/parity.hpp"
int main() {
  distribai::ParityMlp m(42);
  const auto w = m.flat_weights();
  for (int i = 0; i < 4; ++i) { uint32_t b; std::memcpy(&b, &w[300+i], 4); std::printf("b1[%d]=0x%08x\n", i, b); }
  for (int i = 0; i < 4; ++i) { uint32_t b; std::memcpy(&b, &w[1260+i], 4); std::printf("W3[%d]=0x%08x\n", i, b); }
  return 0;
}
