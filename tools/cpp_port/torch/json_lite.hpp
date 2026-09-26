// json_lite.hpp: a small dependency-free JSON reader for LibTorch job specs.
//
// The native port deliberately carries no third-party JSON dependency, so the
// LibTorch train path parses its job specs with this minimal reader. It
// supports exactly what a job spec needs: objects, arrays, strings, numbers,
// booleans and null. Anything else fails closed with a path-annotated error.
#pragma once

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace distribai::json {

enum class Type { Null, Bool, Number, String, Array, Object };

class Value;
using Array = std::vector<Value>;
using Object = std::map<std::string, Value>;

class Value {
 public:
  Value() : type_(Type::Null) {}
  explicit Value(bool b) : type_(Type::Bool), bool_(b) {}
  explicit Value(double n) : type_(Type::Number), num_(n) {}
  explicit Value(std::string s) : type_(Type::String), str_(std::move(s)) {}

  static Value array() { Value v; v.type_ = Type::Array; return v; }
  static Value object() { Value v; v.type_ = Type::Object; return v; }

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::Null; }
  bool is_object() const { return type_ == Type::Object; }
  bool is_array() const { return type_ == Type::Array; }
  bool is_string() const { return type_ == Type::String; }
  bool is_number() const { return type_ == Type::Number; }
  bool is_bool() const { return type_ == Type::Bool; }

  const Array& as_array() const { return arr_; }
  Array& as_array() { return arr_; }
  const Object& as_object() const { return obj_; }
  Object& as_object() { return obj_; }
  const std::string& as_string() const { return str_; }
  double as_number() const { return num_; }
  bool as_bool() const { return bool_; }

  bool has(const std::string& key) const {
    return type_ == Type::Object && obj_.find(key) != obj_.end();
  }
  const Value* get(const std::string& key) const {
    if (type_ != Type::Object) return nullptr;
    auto it = obj_.find(key);
    return it == obj_.end() ? nullptr : &it->second;
  }
  // Typed accessors with defaults; job specs are forgiving on optionals but
  // required fields are validated explicitly by the caller.
  std::string str(const std::string& key, const std::string& dflt = "") const;
  double num(const std::string& key, double dflt = 0) const;
  bool boolean(const std::string& key, bool dflt = false) const;
  std::vector<double> num_array(const std::string& key) const;
  std::vector<std::string> str_array(const std::string& key) const;

 private:
  Type type_;
  bool bool_ = false;
  double num_ = 0;
  std::string str_;
  Array arr_;
  Object obj_;
};

inline std::string Value::str(const std::string& key, const std::string& dflt) const {
  const Value* v = get(key);
  if (!v || !v->is_string()) return dflt;
  return v->as_string();
}
inline double Value::num(const std::string& key, double dflt) const {
  const Value* v = get(key);
  return (v && v->is_number()) ? v->as_number() : dflt;
}
inline bool Value::boolean(const std::string& key, bool dflt) const {
  const Value* v = get(key);
  return (v && v->is_bool()) ? v->as_bool() : dflt;
}
inline std::vector<double> Value::num_array(const std::string& key) const {
  std::vector<double> out;
  const Value* v = get(key);
  if (!v || !v->is_array()) return out;
  for (const auto& e : v->as_array()) {
    if (e.is_number()) out.push_back(e.as_number());
  }
  return out;
}
inline std::vector<std::string> Value::str_array(const std::string& key) const {
  std::vector<std::string> out;
  const Value* v = get(key);
  if (!v || !v->is_array()) return out;
  for (const auto& e : v->as_array()) {
    if (e.is_string()) out.push_back(e.as_string());
  }
  return out;
}

// Nesting limit. The parser recurses per level, and bodies arrive from the
// network, so a document folded 100k deep would otherwise exhaust the stack and
// take the process down. Nothing a job spec or an API message needs goes past a
// handful of levels.
constexpr size_t kMaxDepth = 64;

class Parser {
 public:
  explicit Parser(const std::string& text) : s_(text) {}

  bool parse(Value& out, std::string& err) {
    skip_ws();
    if (!parse_value(out, err)) return false;
    skip_ws();
    if (i_ != s_.size()) {
      err = "trailing data at offset " + std::to_string(i_);
      return false;
    }
    return true;
  }

 private:
  void skip_ws() {
    while (i_ < s_.size() &&
           (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\t' || s_[i_] == '\r')) {
      ++i_;
    }
  }

  bool parse_value(Value& out, std::string& err) {
    skip_ws();
    if (i_ >= s_.size()) { err = "unexpected end of input"; return false; }
    const char c = s_[i_];
    if (c == '{') return parse_object(out, err);
    if (c == '[') return parse_array(out, err);
    if (c == '"') {
      std::string str;
      if (!parse_string(str, err)) return false;
      out = Value(std::move(str));
      return true;
    }
    if (c == 't' || c == 'f') return parse_bool(out, err);
    if (c == 'n') return parse_null(out, err);
    if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number(out, err);
    err = std::string("unexpected character '") + c + "' at offset " + std::to_string(i_);
    return false;
  }

  // Counts depth for the lifetime of a nested parse and unwinds on every
  // return path, including the failures.
  struct DepthGuard {
    Parser& p;
    bool ok;
    explicit DepthGuard(Parser& parser) : p(parser), ok(++parser.depth_ <= kMaxDepth) {}
    ~DepthGuard() { --p.depth_; }
  };

  bool parse_object(Value& out, std::string& err) {
    DepthGuard guard(*this);
    if (!guard.ok) { err = "nesting deeper than " + std::to_string(kMaxDepth); return false; }
    out = Value::object();
    ++i_;  // '{'
    skip_ws();
    if (i_ < s_.size() && s_[i_] == '}') { ++i_; return true; }
    for (;;) {
      skip_ws();
      std::string key;
      if (!parse_string(key, err)) return false;
      skip_ws();
      if (i_ >= s_.size() || s_[i_] != ':') { err = "expected ':' after key '" + key + "'"; return false; }
      ++i_;
      Value v;
      if (!parse_value(v, err)) return false;
      out.as_object()[key] = std::move(v);
      skip_ws();
      if (i_ < s_.size() && s_[i_] == ',') { ++i_; continue; }
      if (i_ < s_.size() && s_[i_] == '}') { ++i_; return true; }
      err = "expected ',' or '}' in object";
      return false;
    }
  }

  bool parse_array(Value& out, std::string& err) {
    DepthGuard guard(*this);
    if (!guard.ok) { err = "nesting deeper than " + std::to_string(kMaxDepth); return false; }
    out = Value::array();
    ++i_;  // '['
    skip_ws();
    if (i_ < s_.size() && s_[i_] == ']') { ++i_; return true; }
    for (;;) {
      Value v;
      if (!parse_value(v, err)) return false;
      out.as_array().push_back(std::move(v));
      skip_ws();
      if (i_ < s_.size() && s_[i_] == ',') { ++i_; continue; }
      if (i_ < s_.size() && s_[i_] == ']') { ++i_; return true; }
      err = "expected ',' or ']' in array";
      return false;
    }
  }

  bool parse_string(std::string& out, std::string& err) {
    skip_ws();
    if (i_ >= s_.size() || s_[i_] != '"') { err = "expected string"; return false; }
    ++i_;
    out.clear();
    while (i_ < s_.size() && s_[i_] != '"') {
      const char c = s_[i_];
      if (c == '\\') {
        if (i_ + 1 >= s_.size()) { err = "bad escape"; return false; }
        const char e = s_[i_ + 1];
        i_ += 2;
        switch (e) {
          case '"': out += '"'; break;
          case '\\': out += '\\'; break;
          case '/': out += '/'; break;
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          case 'r': out += '\r'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'u': {
            // \uXXXX decodes to UTF-8. A high surrogate has to be followed by
            // its low surrogate, or the pair is malformed and the document is
            // rejected rather than silently mangled.
            unsigned cp = 0;
            if (!parse_hex4(cp, err)) return false;
            if (cp >= 0xD800 && cp <= 0xDBFF) {
              if (s_.compare(i_, 2, "\\u") != 0) {
                err = "high surrogate without a low surrogate";
                return false;
              }
              i_ += 2;
              unsigned low = 0;
              if (!parse_hex4(low, err)) return false;
              if (low < 0xDC00 || low > 0xDFFF) {
                err = "bad low surrogate";
                return false;
              }
              cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
            } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
              err = "stray low surrogate";
              return false;
            }
            append_utf8(out, cp);
            break;
          }
          default: err = std::string("unsupported escape \\") + e; return false;
        }
      } else {
        out += c;
        ++i_;
      }
    }
    if (i_ >= s_.size()) { err = "unterminated string"; return false; }
    ++i_;  // closing quote
    return true;
  }

  bool parse_hex4(unsigned& out, std::string& err) {
    if (i_ + 4 > s_.size()) { err = "truncated \\u escape"; return false; }
    unsigned value = 0;
    for (int k = 0; k < 4; ++k) {
      const char c = s_[i_ + static_cast<size_t>(k)];
      int digit = -1;
      if (c >= '0' && c <= '9') digit = c - '0';
      else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
      if (digit < 0) { err = "bad hex digit in \\u escape"; return false; }
      value = (value << 4) | static_cast<unsigned>(digit);
    }
    i_ += 4;
    out = value;
    return true;
  }

  static void append_utf8(std::string& out, unsigned cp) {
    if (cp < 0x80) {
      out += static_cast<char>(cp);
    } else if (cp < 0x800) {
      out += static_cast<char>(0xC0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }

  bool parse_number(Value& out, std::string& err) {
    const size_t start = i_;
    if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) ++i_;
    while (i_ < s_.size() &&
           (std::isdigit(static_cast<unsigned char>(s_[i_])) || s_[i_] == '.' ||
            s_[i_] == 'e' || s_[i_] == 'E' || s_[i_] == '-' || s_[i_] == '+')) {
      ++i_;
    }
    const std::string tok = s_.substr(start, i_ - start);
    if (tok.empty()) { err = "invalid number"; return false; }
    // The scan above is permissive (it stops at the first byte that cannot be
    // part of a number), so validate the shape and reject anything that lands
    // outside the finite range: an infinite loss or gradient would poison the
    // aggregate instead of failing loudly. Each part has to carry a digit of
    // its own, so "1e", "1." and "1e+" are malformed rather than silently 1.
    int int_digits = 0;
    int frac_digits = 0;
    int exp_digits = 0;
    size_t k = 0;
    bool seen_dot = false;
    bool seen_exp = false;
    if (k < tok.size() && (tok[k] == '-' || tok[k] == '+')) ++k;
    for (; k < tok.size(); ++k) {
      const char c = tok[k];
      if (std::isdigit(static_cast<unsigned char>(c))) {
        if (seen_exp) ++exp_digits;
        else if (seen_dot) ++frac_digits;
        else ++int_digits;
        continue;
      }
      if (c == '.' && !seen_dot && !seen_exp) { seen_dot = true; continue; }
      if ((c == 'e' || c == 'E') && !seen_exp) {
        seen_exp = true;
        if (k + 1 < tok.size() && (tok[k + 1] == '-' || tok[k + 1] == '+')) ++k;
        continue;
      }
      err = "invalid number '" + tok + "'";
      return false;
    }
    if (int_digits == 0 || (seen_dot && frac_digits == 0) || (seen_exp && exp_digits == 0)) {
      err = "invalid number '" + tok + "'";
      return false;
    }
    const double value = std::atof(tok.c_str());
    if (!std::isfinite(value)) { err = "number out of range: " + tok; return false; }
    out = Value(value);
    return true;
  }

  bool parse_bool(Value& out, std::string& err) {
    if (s_.compare(i_, 4, "true") == 0) { i_ += 4; out = Value(true); return true; }
    if (s_.compare(i_, 5, "false") == 0) { i_ += 5; out = Value(false); return true; }
    err = "invalid literal at offset " + std::to_string(i_);
    return false;
  }

  bool parse_null(Value& out, std::string&) {
    if (s_.compare(i_, 4, "null") == 0) { i_ += 4; out = Value(); return true; }
    return false;
  }

  const std::string& s_;
  size_t i_ = 0;
  size_t depth_ = 0;
};

inline bool parse(const std::string& text, Value& out, std::string& err) {
  Parser p(text);
  return p.parse(out, err);
}

inline std::string escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        // Every remaining control byte has to be escaped too. A raw 0x01 in a
        // node id used to produce a response no JSON client could read.
        if (static_cast<unsigned char>(c) < 0x20) {
          static const char* hex = "0123456789abcdef";
          const unsigned char u = static_cast<unsigned char>(c);
          out += "\\u00";
          out += hex[(u >> 4) & 0xF];
          out += hex[u & 0xF];
        } else {
          out += c;
        }
    }
  }
  return out;
}

}  // namespace distribai::json
