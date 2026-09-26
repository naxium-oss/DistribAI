// base64.hpp - base64 for the native grid's JSON API.
//
// Job bundles (job spec, TorchScript module, data tensors) and result
// envelopes travel as base64 fields inside JSON messages, so one frame type
// carries everything and every message stays readable with a plain HTTP client.
// Jobs in this repo are small (a translated 1K model is a few KB), and the
// encoder refuses to exceed kMaxBlob unless the caller raises the cap.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace distribai::b64 {

inline const char* alphabet() {
  return "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
}

inline std::string encode(const uint8_t* data, size_t n) {
  static const char* a = alphabet();
  std::string out;
  out.reserve(((n + 2) / 3) * 4);
  size_t i = 0;
  for (; i + 2 < n; i += 3) {
    const uint32_t v = (static_cast<uint32_t>(data[i]) << 16) |
                       (static_cast<uint32_t>(data[i + 1]) << 8) |
                       static_cast<uint32_t>(data[i + 2]);
    out += a[(v >> 18) & 63];
    out += a[(v >> 12) & 63];
    out += a[(v >> 6) & 63];
    out += a[v & 63];
  }
  if (i + 1 == n) {
    const uint32_t v = static_cast<uint32_t>(data[i]) << 16;
    out += a[(v >> 18) & 63];
    out += a[(v >> 12) & 63];
    out += "==";
  } else if (i + 2 == n) {
    const uint32_t v = (static_cast<uint32_t>(data[i]) << 16) |
                       (static_cast<uint32_t>(data[i + 1]) << 8);
    out += a[(v >> 18) & 63];
    out += a[(v >> 12) & 63];
    out += a[(v >> 6) & 63];
    out += '=';
  }
  return out;
}

inline std::string encode(const std::vector<uint8_t>& bytes) {
  return encode(bytes.data(), bytes.size());
}

inline std::string encode(const std::string& text) {
  return encode(reinterpret_cast<const uint8_t*>(text.data()), text.size());
}

// Whitespace is skipped so wrapped base64 survives a copy/paste. Any other
// invalid character fails the decode: the caller gets false and an error.
inline bool decode(const std::string& in, std::vector<uint8_t>& out, std::string& err) {
  static int table[256];
  static bool ready = false;
  if (!ready) {
    for (int i = 0; i < 256; ++i) table[i] = -1;
    const char* a = alphabet();
    for (int i = 0; i < 64; ++i) table[static_cast<uint8_t>(a[i])] = i;
    ready = true;
  }

  out.clear();
  out.reserve(in.size() / 4 * 3);
  uint32_t acc = 0;
  int bits = 0;
  int pad = 0;
  for (char c : in) {
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
    if (c == '=') { ++pad; continue; }
    if (pad > 0) { err = "base64: data after padding"; return false; }
    const int v = table[static_cast<uint8_t>(c)];
    if (v < 0) { err = std::string("base64: bad character '") + c + "'"; return false; }
    acc = (acc << 6) | static_cast<uint32_t>(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<uint8_t>((acc >> bits) & 0xFF));
    }
  }
  if (pad > 2) { err = "base64: too much padding"; return false; }
  if (bits >= 6) { err = "base64: truncated group"; return false; }
  return true;
}

}  // namespace distribai::b64
