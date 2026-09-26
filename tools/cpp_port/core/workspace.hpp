// Reusable per-step workspace (improvement O1).
//
// Parity constraint: math and float op order must stay identical to the
// golden-verified path. A bump allocator satisfies this because the parity
// code only ever allocates, writes, and reads within a step - the allocator
// is invisible to numerics. Reuse eliminates per-step heap traffic and the
// page-fault cost that dominates on a shared box.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <vector>

namespace distribai {

class Scratch {
 public:
  explicit Scratch(size_t bytes)
      : buf_(new (std::nothrow) uint8_t[bytes]), cap_(bytes), used_(0) {}
  ~Scratch() { delete[] buf_; }

  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;

  void reset() { used_ = 0; }

  void* alloc(size_t bytes, size_t align = 64) {
    const size_t aligned = (used_ + align - 1) & ~(align - 1);
    if (!buf_ || aligned + bytes > cap_) return nullptr;
    void* p = buf_ + aligned;
    used_ = aligned + bytes;
    return p;
  }

  size_t used() const { return used_; }
  size_t capacity() const { return cap_; }

 private:
  uint8_t* buf_;
  size_t cap_, used_;
};

// Per-thread workspace: reset at the top of every training step.
Scratch& thread_scratch();

}  // namespace distribai
