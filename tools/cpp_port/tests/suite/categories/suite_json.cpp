// suite_json.cpp - the port's dependency-free JSON reader and writer.
//
// The LibTorch trainer reads its job spec and the coordinator parses worker
// bodies with tools/cpp_port/torch/json_lite.hpp. A malformed document, an
// out-of-range number or a raw control byte in a node id all cross this code,
// so this category pins the behavior that keeps a reply readable and a spec
// fail-closed:
//
//   * primitives: objects, arrays, strings, numbers, booleans, null
//   * strings: every escape, unicode, surrogate pairs, astral code points,
//     truncated and stray escapes
//   * numbers: shapes accepted, shapes rejected, overflow rejected
//   * depth: the 64-level nesting guard, on both objects and arrays
//   * errors: trailing data, unterminated input, offset reporting
//   * typed accessors with defaults, and the escape/write round trip
//
// Header only, so it needs no torch and no link step.
// Run: build/cpp_port/suite_json   (or: make -C tools/cpp_port suite-json)
#include <string>
#include <vector>

#include "../framework.hpp"
#include "torch/json_lite.hpp"

using distribai::json::Type;
using distribai::json::Value;

namespace {

bool parses(const std::string& text, Value& out, std::string& err) {
  err.clear();
  return distribai::json::parse(text, out, err);
}

std::string repeat(const std::string& unit, int n) {
  std::string out;
  out.reserve(unit.size() * static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) out += unit;
  return out;
}

}  // namespace

int main() {
  Value v;
  std::string err;

  // -------------------------------------------------------------------------
  suite::section("primitives");
  CHECK(!parses("", v, err), "empty input is rejected");
  CHECK(!parses("   \n\t ", v, err), "whitespace-only input is rejected");
  CHECK(parses("{}", v, err) && v.is_object(), "an empty object parses");
  CHECK(parses("{}", v, err) && v.as_object().empty(), "an empty object has no keys");
  CHECK(parses("[]", v, err) && v.is_array(), "an empty array parses");
  CHECK(parses("[]", v, err) && v.as_array().empty(), "an empty array has no entries");
  CHECK(parses("null", v, err) && v.is_null(), "null parses as null");
  CHECK(parses("true", v, err) && v.is_bool() && v.as_bool(), "true parses as a true boolean");
  CHECK(parses("false", v, err) && v.is_bool() && !v.as_bool(), "false parses as a false boolean");
  CHECK(!parses("tru", v, err), "a truncated true is rejected");
  CHECK(!parses("TRUE", v, err), "an upper-case true is rejected");
  CHECK(!parses("nul", v, err), "a truncated null is rejected");
  CHECK(!parses("nil", v, err), "an unknown literal is rejected");
  CHECK(!parses("'x'", v, err), "single-quoted strings are rejected");
  CHECK(parses("[1, 2, 3]", v, err) && v.as_array().size() == 3, "an array of numbers parses");
  CHECK(parses("[1,2,3]", v, err) && v.as_array()[2].as_number() == 3, "array entries keep order");
  CHECK(parses("[1,[2,[3]]]", v, err) && v.as_array()[1].as_array()[0].as_number() == 2,
        "nested arrays parse");
  CHECK(!parses("[1,]", v, err), "a trailing comma in an array is rejected");
  CHECK(!parses("[1 2]", v, err), "a missing comma in an array is rejected");
  CHECK(!parses("[", v, err), "an unterminated array is rejected");
  CHECK(!parses("]", v, err), "a bare closing bracket is rejected");
  CHECK(parses("{\"a\":1}", v, err) && v.has("a"), "an object key is readable");
  CHECK(parses("{\"a\":1}", v, err) && !v.has("b"), "a missing key reports absent");
  CHECK(parses("{\"a\":{\"b\":{\"c\":7}}}", v, err), "a nested object parses");
  CHECK(parses("{\"a\":{\"b\":{\"c\":7}}}", v, err) &&
            v.get("a")->get("b")->num("c") == 7,
        "a nested value is readable by path");
  CHECK(parses("{\"a\":1, \"b\":2}", v, err) && v.as_object().size() == 2,
        "a two-key object parses");
  CHECK(parses("{\"a\":1,\"a\":2}", v, err) && v.num("a") == 2,
        "a duplicate key keeps the last value");
  CHECK(parses("{ \"a\" : 1 }", v, err) && v.num("a") == 1, "loose whitespace is allowed");
  CHECK(!parses("{\"a\" 1}", v, err), "a missing colon is rejected");
  CHECK(!parses("{\"a\":}", v, err), "a missing value is rejected");
  CHECK(!parses("{\"a\":1,}", v, err), "a trailing comma in an object is rejected");
  CHECK(!parses("{", v, err), "an unterminated object is rejected");
  CHECK(!parses("{\"a\":1", v, err), "an object without a closing brace is rejected");
  CHECK(parses("{\"a\":1}", v, err) && v.get("a")->get("b") == nullptr,
        "get on a non-object returns nothing");
  CHECK(parses("[1]", v, err) && v.get("x") == nullptr, "get on an array returns nothing");
  CHECK(parses("{\"a\":1,\"b\":2,\"c\":3}", v, err) &&
            v.as_object().begin()->first == "a",
        "object keys are ordered by name");
  CHECK(parses("{\"b\":1,\"a\":2}", v, err) && v.as_object().begin()->first == "a",
        "object keys are sorted, not insertion ordered");

  // -------------------------------------------------------------------------
  suite::section("strings and escapes");
  CHECK(parses("\"hello\"", v, err) && v.as_string() == "hello", "a plain string parses");
  CHECK(parses("\"\"", v, err) && v.as_string().empty(), "an empty string parses");
  CHECK(parses("\"a\\nb\"", v, err) && v.as_string() == "a\nb", "\\n decodes");
  CHECK(parses("\"a\\tb\"", v, err) && v.as_string() == "a\tb", "\\t decodes");
  CHECK(parses("\"a\\rb\"", v, err) && v.as_string() == "a\rb", "\\r decodes");
  CHECK(parses("\"a\\bb\"", v, err) && v.as_string() == "a\bb", "\\b decodes");
  CHECK(parses("\"a\\fb\"", v, err) && v.as_string() == "a\fb", "\\f decodes");
  CHECK(parses("\"a\\\\b\"", v, err) && v.as_string() == "a\\b", "a backslash decodes");
  CHECK(parses("\"a\\\"b\"", v, err) && v.as_string() == "a\"b", "an escaped quote decodes");
  CHECK(parses("\"a\\/b\"", v, err) && v.as_string() == "a/b", "\\/ decodes to a slash");
  CHECK(!parses("\"a\\xb\"", v, err), "an unknown escape is rejected");
  CHECK(!parses("\"a\\", v, err), "an escape at end of input is rejected");
  CHECK(!parses("\"unterminated", v, err), "an unterminated string is rejected");

  CHECK(parses("\"\\u0041\"", v, err) && v.as_string() == "A", "\\u0041 decodes to A");
  CHECK(parses("\"\\u00e9\"", v, err) && v.as_string() == "\xc3\xa9",
        "\\u00e9 decodes to two UTF-8 bytes");
  CHECK(parses("\"\\u20ac\"", v, err) && v.as_string() == "\xe2\x82\xac",
        "\\u20ac decodes to three UTF-8 bytes");
  CHECK(parses("\"\\ud83d\\ude00\"", v, err) && v.as_string() == "\xf0\x9f\x98\x80",
        "a surrogate pair decodes to four UTF-8 bytes");
  CHECK(parses("\"\\u0001\"", v, err) && v.as_string() == "\x01",
        "a low code point decodes");
  CHECK(parses("\"\\u0000\"", v, err) && v.as_string().size() == 1,
        "a zero code point decodes to one byte");
  CHECK(parses("\"\\uFFFD\"", v, err) && v.as_string() == "\xef\xbf\xbd",
        "the replacement character decodes");
  CHECK(!parses("\"\\ud83d\"", v, err), "a high surrogate alone is rejected");
  CHECK(err.find("surrogate") != std::string::npos, "the surrogate error is named");
  CHECK(!parses("\"\\ud83dx\"", v, err), "a high surrogate without a low escape is rejected");
  CHECK(!parses("\"\\ud83d\\u0041\"", v, err), "a high surrogate with a non-surrogate is rejected");
  CHECK(!parses("\"\\ude00\"", v, err), "a stray low surrogate is rejected");
  CHECK(!parses("\"\\ude00\\ud83d\"", v, err), "reversed surrogates are rejected");
  CHECK(!parses("\"\\u00g1\"", v, err), "a bad hex digit is rejected");
  CHECK(!parses("\"\\u00\"", v, err), "a truncated \\u escape is rejected");
  CHECK(err.find("escape") != std::string::npos, "the truncated escape error is named");
  CHECK(parses("\"\\u0020\\u0020\"", v, err) && v.as_string() == "  ",
        "escaped spaces decode");
  CHECK(parses("\"tab\\tand newline\\n\"", v, err) &&
            v.as_string() == "tab\tand newline\n",
        "escapes and text mix");

  // -------------------------------------------------------------------------
  suite::section("numbers");
  CHECK(parses("0", v, err) && v.is_number() && v.as_number() == 0, "zero parses");
  CHECK(parses("42", v, err) && v.as_number() == 42, "an integer parses");
  CHECK(parses("-7", v, err) && v.as_number() == -7, "a negative integer parses");
  CHECK(parses("3.5", v, err) && v.as_number() == 3.5, "a decimal parses");
  CHECK(parses("-0.25", v, err) && v.as_number() == -0.25, "a negative decimal parses");
  CHECK(parses("1e3", v, err) && v.as_number() == 1000, "an exponent parses");
  CHECK(parses("1E3", v, err) && v.as_number() == 1000, "an upper-case exponent parses");
  CHECK(parses("1e-3", v, err) && v.as_number() == 0.001, "a negative exponent parses");
  CHECK(parses("1e+3", v, err) && v.as_number() == 1000, "an explicit plus exponent parses");
  CHECK(parses("2.5e2", v, err) && v.as_number() == 250, "a decimal with an exponent parses");
  CHECK(parses("[-1,-2]", v, err) && v.as_array()[0].as_number() == -1,
        "a negative number inside an array parses");
  CHECK(!parses("+8", v, err), "a leading plus at the top level is rejected");
  CHECK(!parses("-", v, err), "a lone minus is rejected");
  CHECK(!parses(".", v, err), "a lone dot is rejected");
  CHECK(!parses("1.2.3", v, err), "two decimal points are rejected");
  CHECK(!parses("1e", v, err), "an exponent with no digits is rejected");
  CHECK(err.find("invalid number") != std::string::npos, "the bad-number error is named");
  CHECK(!parses("1e+", v, err), "an exponent sign with no digits is rejected");
  CHECK(!parses("1e-", v, err), "a negative exponent with no digits is rejected");
  CHECK(!parses("1.", v, err), "a trailing decimal point is rejected");
  CHECK(!parses("-1.", v, err), "a negative trailing decimal point is rejected");
  CHECK(!parses("[1.]", v, err), "a trailing decimal point in an array is rejected");
  CHECK(!parses("e5", v, err), "an exponent with no mantissa is rejected");
  CHECK(!parses("abcdef", v, err), "letters are rejected");
  CHECK(!parses("NaN", v, err), "NaN is rejected");
  CHECK(!parses("Infinity", v, err), "Infinity is rejected");
  CHECK(!parses("-Infinity", v, err), "-Infinity is rejected");
  CHECK(!parses("1e999", v, err), "an overflowing exponent is rejected");
  CHECK(err.find("out of range") != std::string::npos, "the overflow error is named");
  CHECK(!parses("1e400", v, err), "a huge exponent is rejected");
  CHECK(parses("1.7976931348623157e308", v, err), "the largest finite double parses");
  CHECK(parses("1e-320", v, err), "a tiny subnormal parses");
  CHECK(parses("0.000001", v, err) && v.as_number() == 1e-6, "a small decimal parses");

  // -------------------------------------------------------------------------
  suite::section("trailing data and offsets");
  CHECK(!parses("{}x", v, err), "trailing text after an object is rejected");
  CHECK(err.find("trailing data") != std::string::npos, "the trailing-data error is named");
  CHECK(err.find("offset 2") != std::string::npos, "the trailing-data error carries the offset");
  CHECK(!parses("1 2", v, err), "two top-level values are rejected");
  CHECK(!parses("[] {}", v, err), "two top-level documents are rejected");
  CHECK(!parses("\"a\"\"b\"", v, err), "two adjacent strings are rejected");
  CHECK(!parses("x", v, err), "a stray character is rejected");
  CHECK(err.find("unexpected character") != std::string::npos,
        "the stray-character error is named");
  CHECK(!parses("\x01", v, err), "a raw control byte is rejected");
  CHECK(parses("  {}  ", v, err), "surrounding whitespace is tolerated");

  // -------------------------------------------------------------------------
  suite::section("depth limit");
  const std::string at_limit = repeat("[", 64) + repeat("]", 64);
  const std::string past_limit = repeat("[", 65) + repeat("]", 65);
  CHECK(parses(at_limit, v, err), "64 levels of arrays parse");
  CHECK(!parses(past_limit, v, err), "65 levels of arrays are rejected");
  CHECK(err.find("nesting deeper") != std::string::npos, "the depth error is named");
  const std::string obj_at = repeat("{\"a\":", 64) + "1" + repeat("}", 64);
  const std::string obj_past = repeat("{\"a\":", 65) + "1" + repeat("}", 65);
  CHECK(parses(obj_at, v, err), "64 levels of objects parse");
  CHECK(!parses(obj_past, v, err), "65 levels of objects are rejected");
  CHECK(parses(repeat("[", 32) + repeat("]", 32), v, err), "32 levels parse comfortably");
  // A rejected deep document must not leave the parser wedged: the next parse
  // starts clean because each call builds its own Parser.
  CHECK(parses("{\"ok\":1}", v, err) && v.num("ok") == 1,
        "a shallow parse still works after a deep rejection");

  // -------------------------------------------------------------------------
  suite::section("typed accessors");
  CHECK(parses("{\"s\":\"x\",\"n\":3,\"b\":true}", v, err), "a mixed object parses");
  CHECK(v.str("s") == "x", "str reads a string");
  CHECK(v.str("missing", "dflt") == "dflt", "str falls back to its default");
  CHECK(v.str("n", "dflt") == "dflt", "str refuses a number");
  CHECK(v.num("n") == 3, "num reads a number");
  CHECK(v.num("missing", 9) == 9, "num falls back to its default");
  CHECK(v.num("s", -1) == -1, "num refuses a string");
  CHECK(v.boolean("b"), "boolean reads a true boolean");
  CHECK(!v.boolean("missing", false), "boolean falls back to its default");
  CHECK(v.boolean("missing", true), "boolean honours a true default");
  CHECK(!v.boolean("n", false), "boolean refuses a number");
  CHECK(parses("{\"a\":[1,2,\"x\",3]}", v, err), "a mixed array parses");
  CHECK(v.num_array("a").size() == 3, "num_array keeps only numbers");
  CHECK(v.num_array("a").size() == 3 && v.num_array("a")[2] == 3,
        "num_array keeps number order");
  CHECK(v.num_array("missing").empty(), "num_array on a missing key is empty");
  CHECK(parses("{\"a\":[\"x\",\"y\",1]}", v, err), "a mixed string array parses");
  CHECK(v.str_array("a").size() == 2, "str_array keeps only strings");
  CHECK(v.str_array("a").size() == 2 && v.str_array("a")[1] == "y",
        "str_array keeps string order");
  CHECK(v.str_array("missing").empty(), "str_array on a missing key is empty");
  CHECK(parses("{\"a\":1}", v, err) && v.num_array("a").empty(),
        "num_array on a scalar is empty");

  // -------------------------------------------------------------------------
  suite::section("writing and round trips");
  CHECK(distribai::json::escape("plain") == "plain", "escape leaves plain text alone");
  CHECK(distribai::json::escape("a\"b") == "a\\\"b", "escape quotes a quote");
  CHECK(distribai::json::escape("a\\b") == "a\\\\b", "escape quotes a backslash");
  CHECK(distribai::json::escape("a\nb") == "a\\nb", "escape turns a newline into \\n");
  CHECK(distribai::json::escape("\t\r\b\f") == "\\t\\r\\b\\f",
        "escape names the short control escapes");
  for (int i = 0; i < 32; ++i) {
    const std::string raw(1, static_cast<char>(i));
    const std::string out = distribai::json::escape(raw);
    const bool clean = out.find(static_cast<char>(i)) == std::string::npos;
    const std::string label = "escape removes raw control byte " + std::to_string(i);
    CHECK(clean, label.c_str());
  }
  CHECK(distribai::json::escape(std::string(1, static_cast<char>(0x7f))) ==
            std::string(1, static_cast<char>(0x7f)),
        "escape leaves DEL alone");
  CHECK(distribai::json::escape("\x01") == "\\u0001", "escape writes \\u0001 for 0x01");
  CHECK(distribai::json::escape("\x1f") == "\\u001f", "escape writes a lower-case hex escape");
  CHECK(distribai::json::escape("\xc3\xa9") == "\xc3\xa9", "escape passes UTF-8 through");

  const std::string tricky = "quote:\" backslash:\\ newline:\n tab:\t utf8:\xc3\xa9 \x01";
  const std::string quoted = "\"" + distribai::json::escape(tricky) + "\"";
  CHECK(parses(quoted, v, err), "an escaped string parses back");
  CHECK(v.as_string() == tricky, "the escape round trip is exact");
  CHECK(parses("{\"id\":\"" + distribai::json::escape("\x02node") + "\"}", v, err),
        "an escaped key value parses");
  CHECK(v.str("id") == "\x02node", "a control byte survives the round trip");

  // A big array exercises the loop and the accumulator.
  std::string big = "[";
  for (int i = 0; i < 1000; ++i) {
    if (i) big += ",";
    big += std::to_string(i);
  }
  big += "]";
  CHECK(parses(big, v, err), "a thousand-element array parses");
  CHECK(v.as_array().size() == 1000, "the whole array arrives");
  double sum = 0;
  for (const auto& e : v.as_array()) sum += e.as_number();
  CHECK(sum == 499500.0, "the array sum is exact");

  CHECK(parses("{\"nested\":{\"list\":[{\"k\":\"v\"}]}}", v, err), "a realistic spec parses");
  CHECK(v.get("nested")->get("list")->as_array()[0].str("k") == "v",
        "a value buried in the spec is readable");
  CHECK(v.get("nested")->type() == Type::Object, "type reports an object");
  CHECK(v.get("nested")->get("list")->type() == Type::Array, "type reports an array");

  return suite::finish("json");
}
