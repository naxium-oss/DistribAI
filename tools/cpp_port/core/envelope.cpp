#include "envelope.hpp"

#include <cmath>
#include <cstdio>
#include <mutex>

namespace distribai {
namespace env {

uint32_t crc32_ieee(const uint8_t* data, size_t n, uint32_t crc) {
  static uint32_t table[256];
  static std::once_flag once;
  std::call_once(once, [] {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      table[i] = c;
    }
  });
  crc ^= 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i) crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFFu;
}

namespace {

void put_u16(std::vector<uint8_t>& b, uint16_t v) {
  b.push_back(uint8_t(v & 0xFF));
  b.push_back(uint8_t(v >> 8));
}
void put_u32(std::vector<uint8_t>& b, uint32_t v) {
  for (int i = 0; i < 4; ++i) b.push_back(uint8_t((v >> (8 * i)) & 0xFF));
}
void put_u64(std::vector<uint8_t>& b, uint64_t v) {
  for (int i = 0; i < 8; ++i) b.push_back(uint8_t((v >> (8 * i)) & 0xFF));
}
uint16_t ld_u16(const uint8_t* p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }
uint32_t ld_u32(const uint8_t* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
uint64_t ld_u64(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= uint64_t(p[i]) << (8 * i);
  return v;
}

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

std::string f2s(double v) {
  if (std::isnan(v) || std::isinf(v)) return "null";
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return buf;
}

}  // namespace

std::vector<uint8_t> Envelope::encode() const {
  // TLV payload
  std::vector<uint8_t> payload;
  for (const auto& f : fields_) {
    payload.push_back(f.tag);
    payload.push_back(static_cast<uint8_t>(f.type));
    payload.push_back(0);
    payload.push_back(0);
    payload.push_back(0);
    payload.push_back(0);
    payload.push_back(0);
    payload.push_back(0);
    put_u64(payload, f.raw.size());
    payload.insert(payload.end(), f.raw.begin(), f.raw.end());
  }

  std::vector<uint8_t> out;
  out.reserve(32 + payload.size());
  put_u32(out, kMagic);
  put_u16(out, version_);
  put_u16(out, static_cast<uint16_t>(kind_));
  put_u32(out, 0);  // flags
  put_u32(out, static_cast<uint32_t>(fields_.size()));
  put_u32(out, static_cast<uint32_t>(payload.size()));
  put_u32(out, seq_);
  put_u32(out, 0);  // reserved
  put_u32(out, 0);  // crc placeholder

  const uint32_t crc = crc32_ieee(payload.data(), payload.size(),
                                  crc32_ieee(out.data(), 28));
  for (int i = 0; i < 4; ++i) out[28 + i] = uint8_t((crc >> (8 * i)) & 0xFF);
  out.insert(out.end(), payload.begin(), payload.end());
  return out;
}

bool Envelope::decode(const uint8_t* data, size_t n, Envelope& out, std::string& err) {
  if (n < 32) { err = "too short for header"; return false; }
  if (ld_u32(data) != kMagic) { err = "bad magic"; return false; }
  const uint16_t version = ld_u16(data + 4);
  if (version > kVersion) {
    err = "envelope version " + std::to_string(version) + " newer than supported " +
          std::to_string(kVersion);
    return false;
  }
  const uint32_t stored_crc = ld_u32(data + 28);
  const uint32_t payload_len = ld_u32(data + 16);
  if (32 + static_cast<size_t>(payload_len) > n) { err = "payload_len exceeds buffer"; return false; }

  // CRC over header 0..27 with crc zeroed, then payload.
  std::vector<uint8_t> head(data, data + 28);
  head.resize(32, 0);  // zeroed crc field
  const uint32_t expect = crc32_ieee(data + 32, payload_len, crc32_ieee(head.data(), 28));
  if (expect != stored_crc) { err = "crc mismatch"; return false; }

  out = Envelope();
  out.version_ = version;
  out.kind_ = static_cast<Kind>(ld_u16(data + 6));
  out.seq_ = ld_u32(data + 20);

  const uint32_t field_count = ld_u32(data + 12);
  size_t off = 32;
  for (uint32_t i = 0; i < field_count; ++i) {
    if (off + 16 > 32 + payload_len) { err = "truncated TLV header"; return false; }
    Field f;
    f.tag = data[off];
    f.type = static_cast<ValType>(data[off + 1]);
    const uint64_t len = ld_u64(data + off + 8);
    off += 16;
    if (off + len > 32 + payload_len) { err = "truncated TLV value"; return false; }
    f.raw.assign(data + off, data + off + len);
    off += len;
    out.fields_.push_back(std::move(f));
  }
  return true;
}

bool Envelope::get_u64(uint8_t tag, uint64_t& out) const {
  for (const auto& f : fields_) {
    if (f.tag == tag && f.type == ValType::U64 && f.raw.size() == 8) {
      out = ld_u64(f.raw.data());
      return true;
    }
  }
  return false;
}
bool Envelope::get_f64(uint8_t tag, double& out) const {
  for (const auto& f : fields_) {
    if (f.tag == tag && f.type == ValType::F64 && f.raw.size() == 8) {
      std::memcpy(&out, f.raw.data(), 8);
      return true;
    }
  }
  return false;
}
bool Envelope::get_str(uint8_t tag, std::string& out) const {
  for (const auto& f : fields_) {
    if (f.tag == tag && f.type == ValType::Str) {
      out.assign(f.raw.begin(), f.raw.end());
      return true;
    }
  }
  return false;
}
bool Envelope::get_bool(uint8_t tag, bool& out) const {
  for (const auto& f : fields_) {
    if (f.tag == tag && f.type == ValType::Bool && f.raw.size() == 1) {
      out = f.raw[0] != 0;
      return true;
    }
  }
  return false;
}
bool Envelope::get_f64_array(uint8_t tag, std::vector<double>& out) const {
  for (const auto& f : fields_) {
    if (f.tag == tag && f.type == ValType::F64Array && f.raw.size() % 8 == 0) {
      out.resize(f.raw.size() / 8);
      for (size_t i = 0; i < out.size(); ++i) std::memcpy(&out[i], f.raw.data() + i * 8, 8);
      return true;
    }
  }
  return false;
}

std::string Envelope::to_json() const {
  // Baseline contract keys first, then extras. Numeric f64s via f2s (round-trip),
  // matching Python envelope float rendering closely enough for diff tooling.
  std::string s = "{";
  bool first = true;
  auto push = [&](const std::string& k, const std::string& v) {
    if (!first) s += ", ";
    first = false;
    s += "\"" + k + "\": " + v;
  };

  uint64_t u;
  double d;
  std::string str;
  bool b;
  std::vector<double> arr;

  if (get_bool(TAG_SANDBOX, b)) push("sandbox", b ? "true" : "false");
  if (get_bool(TAG_OWN_SYSTEM, b)) push("own_system", b ? "true" : "false");
  if (get_u64(TAG_SEED, u)) push("seed", std::to_string(u));
  if (get_u64(TAG_N_PARAMS, u)) push("n_params", std::to_string(u));
  if (get_u64(TAG_STEPS, u)) push("steps", std::to_string(u));
  if (get_f64(TAG_WALL_S, d)) push("wall_s", f2s(d));
  if (get_f64(TAG_STEPS_PER_S, d)) push("steps_per_s", f2s(d));
  if (get_f64(TAG_MS_PER_STEP, d)) push("ms_per_step", f2s(d));
  if (get_f64(TAG_FINAL_LOSS, d)) push("final_loss", f2s(d));
  if (get_f64_array(TAG_LOSS_FIRST5, arr)) {
    std::string a = "[";
    for (size_t i = 0; i < arr.size(); ++i) {
      a += f2s(arr[i]);
      if (i + 1 < arr.size()) a += ", ";
    }
    push("first5_losses", a + "]");
  }
  if (get_u64(TAG_GRAD_LEN, u)) push("grad_len", std::to_string(u));
  if (get_f64(TAG_GRAD_SUM, d)) push("grad_sum", f2s(d));
  if (get_f64_array(TAG_GRAD_FIRST3, arr)) {
    std::string a = "[";
    for (size_t i = 0; i < arr.size(); ++i) {
      a += f2s(arr[i]);
      if (i + 1 < arr.size()) a += ", ";
    }
    push("grad_first3", a + "]");
  }
  if (get_str(TAG_MODEL_NAME, str)) push("model_name", "\"" + json_escape(str) + "\"");
  if (get_u64(TAG_MODEL_INDEX, u)) push("model_index", std::to_string(u));
  if (get_str(TAG_CHECKPOINT_ID, str)) push("checkpoint_id", "\"" + json_escape(str) + "\"");
  if (get_u64(TAG_RESUME_FROM_STEP, u)) push("resume_from_step", std::to_string(u));
  if (get_str(TAG_ERROR, str)) push("error", "\"" + json_escape(str) + "\"");
  s += " }";
  return s;
}

}  // namespace env
}  // namespace distribai
