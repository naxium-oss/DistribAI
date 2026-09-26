// framework.hpp - shared mini test framework for every suite category binary.
// Conventions (tests/suite/README.md):
//   * exit 0 = category green; 1 = failures
//   * prints one "CHECK <name>: ok|FAIL" line per check and a final
//     "CATEGORY <name>: <passed>/<total> checks, <fails> failures" summary
//   * a machine-readable JSON tail line: SUITE_JSON {"category":..., ...}
#pragma once

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace suite {

static int g_checks = 0;
static int g_fails = 0;
static std::vector<std::string> g_failed_names;

#define CHECK(cond, name)                                                    \
  do {                                                                       \
    ++suite::g_checks;                                                       \
    if (cond) {                                                              \
      std::printf("CHECK %s: ok\n", name);                                   \
    } else {                                                                 \
      ++suite::g_fails;                                                      \
      suite::g_failed_names.push_back(name);                                 \
      std::printf("CHECK %s: FAIL\n", name);                                 \
    }                                                                        \
  } while (0)

// Section banner for readable logs.
inline void section(const char* name) { std::printf("== %s ==\n", name); }

// Call at the end of main(). Returns the process exit code.
inline int finish(const char* category) {
  std::printf("CATEGORY %s: %d/%d checks, %d failures\n", category, g_checks - g_fails, g_checks,
              g_fails);
  std::printf("SUITE_JSON {\"category\":\"%s\",\"passed\":%d,\"total\":%d,\"fails\":%d}\n",
              category, g_checks - g_fails, g_checks, g_fails);
  return g_fails == 0 ? 0 : 1;
}

// Approximate float compare with absolute tolerance.
inline bool near(float a, float b, float tol) {
  const float d = a > b ? a - b : b - a;
  return d <= tol;
}
inline bool near(double a, double b, double tol) {
  const double d = a > b ? a - b : b - a;
  return d <= tol;
}

}  // namespace suite
