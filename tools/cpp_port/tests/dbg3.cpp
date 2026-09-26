#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "../core/parity.hpp"
#include "golden_seed42.hpp"
using namespace distribai;
static float from_bits(uint32_t b){ float f; std::memcpy(&f,&b,4); return f; }
int main() {
  distribai::ParityMlp m(42);
  auto r = m.run(200);
  double max_abs = 0; int first_diverge = -1; int ulp1 = 0, ulp2 = 0, big = 0;
  for (int i = 0; i < 200; ++i) {
    const float a = r.losses[i], b = from_bits(golden::LOSS_ALL200[i]);
    const double d = std::fabs(double(a) - double(b));
    if (d > max_abs) max_abs = d;
    if (d > 0 && first_diverge < 0) first_diverge = i;
    uint32_t ba, bb; std::memcpy(&ba, &a, 4); std::memcpy(&bb, &b, 4);
    const long long off = std::abs((long long)ba - (long long)bb);
    if (off <= 1) ++ulp1; else if (off <= 2) ++ulp2; else ++big;
  }
  std::printf("max|dloss|=%.3g first_diverge_step=%d ulp<=1:%d ulp2:%d big:%d final: mine=%.9f torch=%.9f\n",
              max_abs, first_diverge, ulp1, ulp2, big, r.losses[199], from_bits(golden::LOSS_ALL200[199]));
  return 0;
}
