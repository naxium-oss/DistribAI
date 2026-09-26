#include "workspace.hpp"

namespace distribai {

Scratch& thread_scratch() {
  // 4 MiB per training thread: enough for 100K-width steps with slack
  // (100K model step needs ~250 KB of activations + grads).
  static thread_local Scratch scratch(4u << 20);
  return scratch;
}

}  // namespace distribai
