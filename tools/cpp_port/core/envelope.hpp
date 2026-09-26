// Versioned envelope schema v1 (see runtime/baselines/design_envelope_checkpoint.md).
//
// Self-describing TLV after a fixed header, CRC32-checked, forward-compatible:
// readers skip unknown tags and reject versions newer than they understand.
// Serves four transports: POD pipe bytes, checkpoint files, CLI stdout (via
// to_json), and the future C ABI (via dai_envelope_* accessors).
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace distribai {

namespace env {

constexpr uint32_t kMagic = 0x4449454Eu;  // "DIEN" little-endian
constexpr uint16_t kVersion = 1;

enum class Kind : uint16_t {
  TrainResult = 1,
  GradReport = 2,
  JobResult = 3,
  Error = 4,
  CheckpointRef = 5,
};

enum class ValType : uint8_t {
  I64 = 0,
  F64 = 1,
  Blob = 2,
  Str = 3,
  U64 = 4,
  F64Array = 5,
  Bool = 6,
};

// Standard tags, v1.
enum Tag : uint8_t {
  TAG_SEED = 1,
  TAG_STEPS = 2,
  TAG_N_PARAMS = 3,
  TAG_WALL_S = 4,
  TAG_STEPS_PER_S = 5,
  TAG_MS_PER_STEP = 6,
  TAG_FINAL_LOSS = 7,
  TAG_GRAD_LEN = 8,
  TAG_GRAD_SUM = 9,
  TAG_GRAD_VALUES = 10,
  TAG_LOSS_FIRST5 = 11,
  TAG_MODEL_NAME = 12,
  TAG_MODEL_INDEX = 13,
  TAG_CHECKPOINT_ID = 14,
  TAG_ERROR = 15,
  TAG_RESUME_FROM_STEP = 16,
  TAG_SANDBOX = 17,
  TAG_OWN_SYSTEM = 18,
  TAG_GRAD_FIRST3 = 19,
  TAG_RL_MEM_MB = 20,
  TAG_RL_CPU_SEC = 21,
  TAG_OPTIM_STEPS = 22,  // AdamW bias-correction counter (checkpoint metadata)
  TAG_CKPT_WEIGHTS_LEN = 30,
  TAG_CKPT_OPTIM_LEN = 31,
  TAG_CKPT_RNG_LEN = 32,
  TAG_CKPT_PAYLOAD_LEN = 33,
};

struct Field {
  uint8_t tag;
  ValType type;
  std::vector<uint8_t> raw;  // encoded value bytes

  static Field make(uint8_t tag, ValType t, const void* data, size_t n) {
    Field f;
    f.tag = tag;
    f.type = t;
    const auto* p = static_cast<const uint8_t*>(data);
    f.raw.assign(p, p + n);
    return f;
  }
};

inline Field make_u64(uint8_t tag, uint64_t v) {
  return Field::make(tag, ValType::U64, &v, sizeof(v));
}
inline Field make_i64(uint8_t tag, int64_t v) {
  return Field::make(tag, ValType::I64, &v, sizeof(v));
}
inline Field make_f64(uint8_t tag, double v) {
  return Field::make(tag, ValType::F64, &v, sizeof(v));
}
inline Field make_bool(uint8_t tag, bool v) {
  return Field::make(tag, ValType::Bool, &v, 1);
}
inline Field make_str(uint8_t tag, const std::string& s) {
  return Field::make(tag, ValType::Str, s.data(), s.size());
}
inline Field make_f64_array(uint8_t tag, const std::vector<double>& xs) {
  return Field::make(tag, ValType::F64Array, xs.data(), xs.size() * sizeof(double));
}

// IEEE CRC-32 (reflected, poly 0xEDB88320).
uint32_t crc32_ieee(const uint8_t* data, size_t n, uint32_t crc = 0xFFFFFFFFu);

class Envelope {
 public:
  Envelope() = default;
  Envelope(Kind kind, uint32_t seq) : kind_(kind), seq_(seq) {}

  void add(Field f) { fields_.push_back(std::move(f)); }
  void add_u64(uint8_t tag, uint64_t v) { add(make_u64(tag, v)); }
  void add_f64(uint8_t tag, double v) { add(make_f64(tag, v)); }
  void add_bool(uint8_t tag, bool v) { add(make_bool(tag, v)); }
  void add_str(uint8_t tag, const std::string& s) { add(make_str(tag, s)); }
  void add_f64_array(uint8_t tag, const std::vector<double>& xs) { add(make_f64_array(tag, xs)); }

  Kind kind() const { return kind_; }
  uint16_t version() const { return version_; }
  uint32_t seq() const { return seq_; }
  const std::vector<Field>& fields() const { return fields_; }

  // ---- serialization ----
  std::vector<uint8_t> encode() const;

  // Decodes and verifies magic/version/CRC. Unknown tags are kept (skipped
  // by typed accessors). Returns false with err set on any structural fault.
  static bool decode(const uint8_t* data, size_t n, Envelope& out, std::string& err);

  // ---- typed getters (first match wins) ----
  bool get_u64(uint8_t tag, uint64_t& out) const;
  bool get_f64(uint8_t tag, double& out) const;
  bool get_str(uint8_t tag, std::string& out) const;
  bool get_bool(uint8_t tag, bool& out) const;
  bool get_f64_array(uint8_t tag, std::vector<double>& out) const;
  // (free byte-order helpers live in envelope.cpp as ld_u16/ld_u32/ld_u64)

  // ---- JSON emission (baseline-contract keys + extras) ----
  std::string to_json() const;

 private:
  uint16_t version_ = kVersion;
  Kind kind_ = Kind::TrainResult;
  uint32_t seq_ = 0;
  std::vector<Field> fields_;
};

}  // namespace env
}  // namespace distribai
