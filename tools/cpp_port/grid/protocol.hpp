// protocol.hpp - the native grid's wire contract.
//
// The orchestrator and the worker both speak HTTP/1.1 with JSON bodies, so a
// worker can sit behind any plain HTTP proxy or a free Cloudflare quick tunnel
// without extra tooling. Endpoint names, message types and field names live
// here so the two binaries cannot drift apart.
//
// Two directions:
//   worker -> orchestrator : register, heartbeat, claim, result, bye
//   orchestrator -> worker : welcome, task, idle, ack, error
//
// Job bundles and result envelopes ride inside these messages as base64. That
// keeps one message shape for everything and lets a worker on a notebook host
// run a job with no shared filesystem.
#pragma once

#include <ctime>
#include <string>
#include <vector>

#include "../torch/json_lite.hpp"
#include "base64.hpp"

namespace distribai::grid {

namespace json = distribai::json;

// Protocol revision. A worker that sends a different value is refused at
// register time instead of failing later with a confusing field error.
constexpr int kProto = 1;

// Worker endpoints.
constexpr const char* kRegister = "/v1/register";
constexpr const char* kHeartbeat = "/v1/heartbeat";
constexpr const char* kClaim = "/v1/claim";
constexpr const char* kResult = "/v1/result";
constexpr const char* kBye = "/v1/bye";

// Operator and dashboard endpoints.
constexpr const char* kJobs = "/v1/jobs";
constexpr const char* kNodes = "/v1/nodes";
constexpr const char* kTasks = "/v1/tasks";
constexpr const char* kSummary = "/v1/summary";
constexpr const char* kHealth = "/v1/health";

// Message types.
constexpr const char* kTWelcome = "welcome";
constexpr const char* kTTask = "task";
constexpr const char* kTIdle = "idle";
constexpr const char* kTAck = "ack";
constexpr const char* kTError = "error";

// Caps. A bundle larger than this is refused rather than silently truncated.
constexpr size_t kMaxBundleBytes = 256ull << 20;  // 256 MiB
constexpr size_t kMaxEnvelopeBytes = 64ull << 20;  // 64 MiB

inline int64_t now_s() {
  return static_cast<int64_t>(std::time(nullptr));
}

// Minimal JSON object writer. Building messages with string concatenation gets
// unreadable fast once nested objects appear, and every value must be escaped
// anyway.
class Jw {
 public:
  Jw& s(const std::string& key, const std::string& value) {
    sep(key);
    out_ += "\"" + json::escape(value) + "\"";
    return *this;
  }
  Jw& n(const std::string& key, double value) {
    sep(key);
    if (value == static_cast<double>(static_cast<int64_t>(value))) {
      out_ += std::to_string(static_cast<int64_t>(value));
    } else {
      out_ += std::to_string(value);
    }
    return *this;
  }
  Jw& b(const std::string& key, bool value) {
    sep(key);
    out_ += value ? "true" : "false";
    return *this;
  }
  // Inserts pre-built JSON (an array or object) under key.
  Jw& raw(const std::string& key, const std::string& encoded) {
    sep(key);
    out_ += encoded;
    return *this;
  }
  Jw& strs(const std::string& key, const std::vector<std::string>& values) {
    sep(key);
    out_ += "[";
    for (size_t i = 0; i < values.size(); ++i) {
      if (i) out_ += ",";
      out_ += "\"" + json::escape(values[i]) + "\"";
    }
    out_ += "]";
    return *this;
  }
  std::string str() const { return out_ + "}"; }

 private:
  void sep(const std::string& key) {
    if (!first_) out_ += ",";
    first_ = false;
    out_ += "\"" + json::escape(key) + "\":";
  }
  std::string out_ = "{";
  bool first_ = true;
};

// Parses a request body into an object, reporting a readable error. Callers
// treat a parse failure as a 400 rather than guessing at the intent.
inline bool parse_body(const std::string& body, json::Value& out, std::string& err) {
  if (body.empty()) {
    err = "empty body";
    return false;
  }
  if (!json::parse(body, out, err)) return false;
  if (!out.is_object()) {
    err = "body must be a JSON object";
    return false;
  }
  return true;
}

inline std::string str_field(const json::Value& v, const std::string& key,
                            const std::string& dflt = "") {
  return v.str(key, dflt);
}
inline double num_field(const json::Value& v, const std::string& key, double dflt = 0) {
  return v.num(key, dflt);
}
inline int64_t i64_field(const json::Value& v, const std::string& key, int64_t dflt = 0) {
  return static_cast<int64_t>(v.num(key, static_cast<double>(dflt)));
}

}  // namespace distribai::grid
