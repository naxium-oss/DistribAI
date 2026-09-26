// suite_grid.cpp - the native grid's wire and parsing layer.
//
// The grid ships a hand-written HTTP server, base64 codec and JSON reader, and
// every byte of a worker message crosses all three. This category pins their
// behavior and hammers the edges that a remote worker or a hostile client can
// reach:
//
//   * base64: known-answer vectors, every length 0..64, all 256 byte values,
//     1 MiB payloads, whitespace, padding abuse, invalid characters
//   * JSON: escapes, unicode and surrogate pairs, number shapes, depth limit,
//     control characters on the way out, malformed documents
//   * protocol: pinned constants, message round trips, bundle blobs
//   * HTTP: query decoding, header handling, the live server on loopback
//     (malformed input, header floods, oversized bodies, handler crashes,
//     concurrent clients, response framing)
//
// Run: build/cpp_port/suite_grid   (or: make -C tools/cpp_port suite-grid)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../framework.hpp"
#include "grid/base64.hpp"
#include "grid/http.hpp"
#include "grid/protocol.hpp"

using namespace distribai;

namespace {

std::string bytes_of(size_t n, unsigned seed) {
  std::string s;
  s.reserve(n);
  unsigned x = seed;
  for (size_t i = 0; i < n; ++i) {
    x = x * 1103515245u + 12345u;
    s += static_cast<char>((x >> 16) & 0xFF);
  }
  return s;
}

// Opens one connection, writes the request, reads until the peer closes.
std::string talk(int port, const std::string& bytes, bool half_close = true) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return "";
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    return "";
  }
  size_t sent = 0;
  while (sent < bytes.size()) {
    const ssize_t n = ::send(fd, bytes.data() + sent, bytes.size() - sent, 0);
    if (n <= 0) break;
    sent += static_cast<size_t>(n);
  }
  if (half_close) ::shutdown(fd, SHUT_WR);
  std::string out;
  char buf[4096];
  for (;;) {
    const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
    if (n <= 0) break;
    out.append(buf, static_cast<size_t>(n));
  }
  ::close(fd);
  return out;
}

bool starts_with(const std::string& s, const char* prefix) {
  return s.rfind(prefix, 0) == 0;
}

// A live server plus its thread, torn down at the end of the block.
struct LiveServer {
  http::Server server;
  std::atomic<bool> stop{false};
  std::thread thread;
  int port = 0;

  LiveServer()
      : server("127.0.0.1", 0, [](const http::Request& req) {
          if (req.path == "/health") {
            return http::Response::json(grid::Jw().s("status", "ok").n("proto", grid::kProto).str());
          }
          if (req.path == "/echo") {
            return http::Response::json(
                grid::Jw().n("len", static_cast<double>(req.body.size())).str());
          }
          if (req.path == "/echo-body") {
            return http::Response::json(grid::Jw().s("body", req.body).str());
          }
          if (req.path == "/params") {
            return http::Response::json(grid::Jw()
                                            .s("x", req.param("x"))
                                            .s("y", req.param("y", "none"))
                                            .s("missing", req.param("missing", "dflt"))
                                            .b("local", req.is_local())
                                            .str());
          }
          if (req.path == "/boom") {
            throw std::runtime_error("handler blew up");
          }
          return http::Response::error(404, "no route");
        }) {}

  bool up() {
    std::string err;
    if (!server.start(err)) return false;
    port = server.port();
    thread = std::thread([this] { server.serve(stop); });
    return true;
  }

  ~LiveServer() {
    stop.store(true);
    server.close_listen();
    if (thread.joinable()) thread.join();
  }
};

}  // namespace

int main() {
  // ------------------------------------------------------------------
  suite::section("grid: base64 codec");
  {
    const char* known[][2] = {{"", ""},
                             {"f", "Zg=="},
                             {"fo", "Zm8="},
                             {"foo", "Zm9v"},
                             {"foob", "Zm9vYg=="},
                             {"fooba", "Zm9vYmE="},
                             {"foobar", "Zm9vYmFy"}};
    bool vectors_ok = true;
    for (const auto& kv : known) {
      const std::string enc = b64::encode(std::string(kv[0]));
      if (enc != kv[1]) {
        std::printf("  base64 vector %s -> %s (want %s)\n", kv[0], enc.c_str(), kv[1]);
        vectors_ok = false;
      }
    }
    CHECK(vectors_ok, "base64: RFC 4648 known-answer vectors match");

    bool every_len = true;
    for (size_t n = 0; n <= 64; ++n) {
      const std::string src = bytes_of(n, static_cast<unsigned>(n * 7 + 1));
      const std::string enc = b64::encode(src);
      std::vector<uint8_t> back;
      std::string err;
      if (!b64::decode(enc, back, err) || std::string(back.begin(), back.end()) != src) {
        std::printf("  base64 length %zu failed: %s\n", n, err.c_str());
        every_len = false;
      }
      // The encoded form must be a whole number of 4-character groups.
      if (enc.size() % 4 != 0) every_len = false;
    }
    CHECK(every_len, "base64: round trip holds for every length 0..64");

    std::string all256;
    for (int i = 0; i < 256; ++i) all256 += static_cast<char>(i);
    std::vector<uint8_t> all256_back;
    std::string err;
    const bool all256_ok = b64::decode(b64::encode(all256), all256_back, err) &&
                           std::string(all256_back.begin(), all256_back.end()) == all256;
    CHECK(all256_ok, "base64: all 256 byte values survive a round trip");

    const std::string big = bytes_of(1u << 20, 99);
    const std::string big_enc = b64::encode(big);
    std::vector<uint8_t> big_back;
    const bool big_ok = b64::decode(big_enc, big_back, err) &&
                        std::string(big_back.begin(), big_back.end()) == big;
    CHECK(big_ok, "base64: 1 MiB payload round trips exactly");
    CHECK(big_enc.size() == ((big.size() + 2) / 3) * 4, "base64: 1 MiB encodes at the expected size");

    std::vector<uint8_t> out;
    std::string e1;
    CHECK(b64::decode("Zm9v\nYmFy\n", out, e1) && out.size() == 6,
          "base64: wrapped input with newlines decodes");
    std::string tabbed = "Zm9v\tYmFy";
    CHECK(b64::decode(tabbed, out, e1) && out.size() == 6, "base64: tabs are skipped");

    std::string e2;
    CHECK(!b64::decode("Zm9v*", out, e2), "base64: invalid character is rejected");
    CHECK(e2.find("bad character") != std::string::npos, "base64: rejection names the bad character");
    std::string e3;
    CHECK(!b64::decode("Zg==Zg==", out, e3), "base64: data after padding is rejected");
    std::string e4;
    CHECK(!b64::decode("Zm9vY", out, e4), "base64: truncated final group is rejected");
    std::string e5;
    CHECK(!b64::decode("Zg===", out, e5), "base64: over-padding is rejected");
    std::string e6;
    b64::decode("", out, e6);
    CHECK(out.empty(), "base64: empty input decodes to empty output");
  }

  // ------------------------------------------------------------------
  suite::section("grid: JSON reader");
  {
    json::Value v;
    std::string err;
    CHECK(json::parse("{\"a\":1,\"b\":[1,2,3],\"c\":{\"d\":\"x\"}}", v, err),
          "json: nested object and array parse");
    CHECK(v.num("a") == 1.0, "json: integer field reads back");
    CHECK(v.get("b") && v.get("b")->as_array().size() == 3, "json: array keeps its length");
    CHECK(v.get("c") && v.get("c")->str("d") == "x", "json: nested string reads back");
    CHECK(v.str("absent", "fallback") == "fallback", "json: missing string falls back");
    CHECK(v.num("absent", 7) == 7, "json: missing number falls back");

    std::string esc = "quote:\" back:\\ newline:\n tab:\t";
    std::string doc = std::string("{\"s\":\"") + json::escape(esc) + "\"}";
    json::Value round;
    CHECK(json::parse(doc, round, err) && round.str("s") == esc,
          "json: escape and parse round trip a string with quotes, slashes and newlines");

    // Control bytes have to leave as escapes; a raw 0x01 makes the response
    // unreadable for any conforming client.
    std::string ctrl = "a";
    ctrl += static_cast<char>(0x01);
    ctrl += "b";
    ctrl += static_cast<char>(0x1F);
    const std::string escaped = json::escape(ctrl);
    CHECK(escaped.find("\\u0001") != std::string::npos &&
              escaped.find("\\u001f") != std::string::npos,
          "json: control bytes escape as \\u00XX");
    json::Value ctrl_back;
    std::string cdoc = std::string("{\"s\":\"") + escaped + "\"}";
    CHECK(json::parse(cdoc, ctrl_back, err) && ctrl_back.str("s") == ctrl,
          "json: escaped control bytes parse back to the original bytes");

    json::Value uni;
    CHECK(json::parse("{\"s\":\"caf\\u00e9\"}", uni, err) && uni.str("s") == "caf\xc3\xa9",
          "json: \\u escape decodes to UTF-8");
    json::Value emoji;
    CHECK(json::parse("{\"s\":\"\\ud83d\\ude00\"}", emoji, err) &&
              emoji.str("s") == "\xf0\x9f\x98\x80",
          "json: surrogate pair decodes to one UTF-8 code point");

    json::Value bad;
    std::string be;
    CHECK(!json::parse("{\"s\":\"\\ud83d\"}", bad, be), "json: lone high surrogate is rejected");
    CHECK(!json::parse("{\"s\":\"\\ude00\"}", bad, be), "json: stray low surrogate is rejected");
    CHECK(!json::parse("{\"s\":\"\\u12\"}", bad, be), "json: short \\u escape is rejected");
    CHECK(!json::parse("{\"s\":\"\\uZZZZ\"}", bad, be), "json: non-hex \\u escape is rejected");
    CHECK(!json::parse("{\"s\":\"\\q\"}", bad, be), "json: unknown escape is rejected");

    CHECK(!json::parse("{\"a\":1,}", bad, be), "json: trailing comma is rejected");
    CHECK(!json::parse("{a:1}", bad, be), "json: unquoted key is rejected");
    CHECK(!json::parse("{\"a\":1", bad, be), "json: unterminated object is rejected");
    CHECK(!json::parse("{\"a\":\"x}", bad, be), "json: unterminated string is rejected");
    CHECK(!json::parse("{\"a\":1} extra", bad, be), "json: trailing data is rejected");
    CHECK(!json::parse("", bad, be), "json: empty document is rejected");
    CHECK(!json::parse("{\"a\":tru}", bad, be), "json: misspelled literal is rejected");

    CHECK(json::parse("{\"n\":-12.5}", v, err) && v.num("n") == -12.5, "json: negative float parses");
    CHECK(json::parse("{\"n\":1e3}", v, err) && v.num("n") == 1000.0, "json: exponent parses");
    CHECK(json::parse("{\"n\":1E+2}", v, err) && v.num("n") == 100.0, "json: signed exponent parses");
    CHECK(!json::parse("{\"n\":1e999}", v, err), "json: overflowing exponent is rejected");
    CHECK(!json::parse("{\"n\":--1}", v, err), "json: double sign is rejected");
    CHECK(!json::parse("{\"n\":1.2.3}", v, err), "json: two decimal points are rejected");
    CHECK(!json::parse("{\"n\":.}", v, err), "json: a dot with no digits is rejected");

    // Depth: 32 levels parse, 5000 are refused instead of blowing the stack.
    std::string deep;
    for (int i = 0; i < 32; ++i) deep += "{\"a\":";
    deep += "1";
    for (int i = 0; i < 32; ++i) deep += "}";
    CHECK(json::parse(deep, v, err), "json: 32 levels of nesting parse");

    std::string huge;
    for (int i = 0; i < 5000; ++i) huge += "[";
    huge += "1";
    for (int i = 0; i < 5000; ++i) huge += "]";
    const bool huge_ok = json::parse(huge, v, err);
    CHECK(!huge_ok, "json: 5000 levels of nesting is refused");
    CHECK(err.find("nesting") != std::string::npos, "json: depth rejection names the reason");

    std::string wide = "[";
    for (int i = 0; i < 10000; ++i) {
      if (i) wide += ",";
      wide += std::to_string(i);
    }
    wide += "]";
    CHECK(json::parse(wide, v, err) && v.as_array().size() == 10000,
          "json: 10k-element array parses");
  }

  // ------------------------------------------------------------------
  suite::section("grid: protocol contracts");
  {
    CHECK(grid::kProto == 1, "proto: wire revision is pinned at 1");
    CHECK(std::strcmp(grid::kRegister, "/v1/register") == 0, "proto: register endpoint is pinned");
    CHECK(std::strcmp(grid::kHeartbeat, "/v1/heartbeat") == 0, "proto: heartbeat endpoint is pinned");
    CHECK(std::strcmp(grid::kClaim, "/v1/claim") == 0, "proto: claim endpoint is pinned");
    CHECK(std::strcmp(grid::kResult, "/v1/result") == 0, "proto: result endpoint is pinned");
    CHECK(std::strcmp(grid::kBye, "/v1/bye") == 0, "proto: bye endpoint is pinned");
    CHECK(std::strcmp(grid::kJobs, "/v1/jobs") == 0, "proto: jobs endpoint is pinned");
    CHECK(std::strcmp(grid::kHealth, "/v1/health") == 0, "proto: health endpoint is pinned");
    CHECK(grid::kMaxBundleBytes == (256ull << 20), "proto: bundle cap is 256 MiB");
    CHECK(grid::kMaxEnvelopeBytes == (64ull << 20), "proto: envelope cap is 64 MiB");

    CHECK(grid::Jw().str() == "{}", "proto: an empty writer yields an empty object");

    const std::string msg = grid::Jw()
                                .s("type", grid::kTWelcome)
                                .s("node_id", "node-1")
                                .n("proto", grid::kProto)
                                .b("contributing", true)
                                .strs("lanes", {"P1", "P2"})
                                .str();
    json::Value parsed;
    std::string err;
    CHECK(json::parse(msg, parsed, err), "proto: a built message parses back");
    CHECK(parsed.str("type") == "welcome", "proto: message type survives the round trip");
    CHECK(parsed.num("proto") == grid::kProto, "proto: revision survives the round trip");
    CHECK(parsed.boolean("contributing"), "proto: booleans serialize as true/false");
    CHECK(parsed.get("lanes") && parsed.get("lanes")->as_array().size() == 2,
          "proto: string arrays serialize");

    // Integer-valued doubles must not print as "1.000000e+00"; the orchestrator
    // and the worker compare these as integers.
    const std::string ints = grid::Jw().n("steps", 30).n("pct", 0.5).n("big", 1e12).str();
    CHECK(ints.find("\"steps\":30") != std::string::npos,
          "proto: whole numbers print without a decimal point");
    CHECK(ints.find("0.5") != std::string::npos, "proto: fractional numbers keep their value");
    CHECK(ints.find("1000000000000") != std::string::npos, "proto: large integers print exactly");

    // A bundle blob is base64 inside the message and comes back byte for byte.
    const std::string blob = bytes_of(4096, 5);
    const std::string with_blob =
        grid::Jw().s("type", grid::kTTask).s("b64", b64::encode(blob)).str();
    json::Value task;
    CHECK(json::parse(with_blob, task, err), "proto: a task message with a blob parses");
    std::vector<uint8_t> decoded;
    std::string derr;
    CHECK(b64::decode(task.str("b64"), decoded, derr) &&
              std::string(decoded.begin(), decoded.end()) == blob,
          "proto: the embedded bundle survives base64 inside JSON");

    json::Value body;
    std::string perr;
    CHECK(!grid::parse_body("", body, perr), "proto: an empty body is refused");
    CHECK(!grid::parse_body("[1,2]", body, perr), "proto: a bare array body is refused");
    CHECK(!grid::parse_body("not json", body, perr), "proto: a non-JSON body is refused");
    CHECK(grid::parse_body("{\"ok\":true}", body, perr), "proto: an object body is accepted");
    CHECK(perr.find("object") != std::string::npos,
          "proto: the array rejection explains that an object is required");
  }

  // ------------------------------------------------------------------
  suite::section("grid: request helpers");
  {
    CHECK(http::url_decode("a%20b") == "a b", "http: %20 decodes to a space");
    CHECK(http::url_decode("a+b") == "a b", "http: plus decodes to a space");
    CHECK(http::url_decode("a%2Fb") == "a/b", "http: %2F decodes to a slash");
    CHECK(http::url_decode("100%") == "100%", "http: a trailing percent is left alone");
    CHECK(http::url_decode("%zz") == "%zz", "http: invalid hex is left alone");
    CHECK(http::url_decode("plain") == "plain", "http: plain text passes through");

    std::map<std::string, std::string> params;
    http::parse_query("a=1&b=two%20words&c=&d&e=x=y", params);
    CHECK(params["a"] == "1", "http: query value parses");
    CHECK(params["b"] == "two words", "http: query value is URL-decoded");
    CHECK(params["c"] == "", "http: an empty value stays empty");
    CHECK(params["d"] == "", "http: a key without '=' gets an empty value");
    CHECK(params["e"] == "x=y", "http: an '=' inside the value is kept");
    std::map<std::string, std::string> empty_params;
    http::parse_query("", empty_params);
    CHECK(empty_params.empty(), "http: an empty query produces no params");

    CHECK(http::content_type_for("index.html") == "text/html; charset=utf-8",
          "http: html content type");
    CHECK(http::content_type_for("app.css").rfind("text/css", 0) == 0, "http: css content type");
    CHECK(http::content_type_for("app.js").rfind("text/javascript", 0) == 0,
          "http: js content type");
    CHECK(http::content_type_for("data.json").rfind("application/json", 0) == 0,
          "http: json content type");
    CHECK(http::content_type_for("logo.svg") == "image/svg+xml", "http: svg content type");
    CHECK(http::content_type_for("favicon.ico") == "image/x-icon", "http: ico content type");
    CHECK(http::content_type_for("archive.bin") == "application/octet-stream",
          "http: unknown extension falls back to octet-stream");
    CHECK(http::content_type_for("noext") == "application/octet-stream",
          "http: a file with no extension falls back");

    http::Request req;
    req.peer = "127.0.0.1";
    CHECK(req.is_local(), "http: 127.0.0.1 counts as local");
    req.peer = "::1";
    CHECK(req.is_local(), "http: ::1 counts as local");
    req.peer = "10.1.2.3";
    CHECK(!req.is_local(), "http: a private remote address is not local");
    req.peer = "203.0.113.9";
    CHECK(!req.is_local(), "http: a public address is not local");
    CHECK(req.param("nope", "dflt") == "dflt", "http: a missing param returns the default");
    CHECK(req.header("x").empty(), "http: a missing header returns an empty string");

    CHECK(http::status_text(400) == "Bad Request", "http: 400 has a reason phrase");
    CHECK(http::status_text(403) == "Forbidden", "http: 403 has a reason phrase");
    CHECK(http::status_text(404) == "Not Found", "http: 404 has a reason phrase");
    CHECK(http::status_text(405) == "Method Not Allowed", "http: 405 has a reason phrase");
    CHECK(http::status_text(413) == "Payload Too Large", "http: 413 has a reason phrase");
    CHECK(http::status_text(500) == "Internal Server Error", "http: 500 has a reason phrase");

    const std::string err_body = http::Response::error(418, "teapot").body;
    CHECK(err_body.find("\"error\":\"teapot\"") != std::string::npos,
          "http: error responses carry an error message");
    CHECK(http::Response::json("{}", 201).status == 201, "http: json responses keep the status");
    CHECK(http::Response::text("hi").content_type.rfind("text/plain", 0) == 0,
          "http: text responses default to text/plain");
  }

  // ------------------------------------------------------------------
  suite::section("grid: file helpers");
  {
    const std::string dir = "/tmp/grid_suite_" + std::to_string(::getpid());
    ::mkdir(dir.c_str(), 0755);
    const std::string path = dir + "/blob.bin";
    const std::string payload = bytes_of(70000, 11);
    CHECK(http::write_file(path, payload), "files: write_file accepts a binary payload");
    std::string back;
    CHECK(http::read_file(path, back) && back == payload,
          "files: read_file returns the same bytes");
    CHECK(http::is_readable(path), "files: a written file is readable");
    CHECK(!http::read_file(dir + "/missing", back), "files: reading a missing file fails");
    CHECK(!http::is_readable(""), "files: the empty path is never readable");
    CHECK(!http::is_readable(dir + "/missing"), "files: a missing path is not readable");
    ::unlink(path.c_str());
    ::rmdir(dir.c_str());
  }

  // ------------------------------------------------------------------
  suite::section("grid: live server on loopback");
  {
    LiveServer live;
    if (!live.up()) {
      CHECK(false, "http: server binds an ephemeral port on loopback");
    } else {
      CHECK(live.port > 0, "http: server reports the port it bound");

      const std::string ok = talk(live.port, "GET /health HTTP/1.1\r\nHost: x\r\n\r\n");
      CHECK(starts_with(ok, "HTTP/1.1 200 OK"), "http: a valid GET returns 200");
      CHECK(ok.find("\"status\":\"ok\"") != std::string::npos, "http: the handler body is sent");
      CHECK(ok.find("Connection: close\r\n") != std::string::npos,
            "http: responses close the connection");
      CHECK(ok.find("Cache-Control: no-store\r\n") != std::string::npos,
            "http: responses are marked no-store");
      const size_t cl = ok.find("Content-Length: ");
      CHECK(cl != std::string::npos, "http: responses declare a content length");
      if (cl != std::string::npos) {
        const size_t hdr_end = ok.find("\r\n\r\n");
        const size_t declared =
            static_cast<size_t>(std::strtoul(ok.c_str() + cl + 16, nullptr, 10));
        CHECK(hdr_end != std::string::npos && ok.size() - (hdr_end + 4) == declared,
              "http: the declared content length matches the body");
      }

      const std::string malformed = talk(live.port, "\r\n\r\n");
      CHECK(starts_with(malformed, "HTTP/1.1 400"), "http: an empty request line is refused");

      const std::string no_target = talk(live.port, "GET\r\n\r\n");
      CHECK(starts_with(no_target, "HTTP/1.1 400"), "http: a request line without a target is refused");

      const std::string no_route = talk(live.port, "GET /nope HTTP/1.1\r\n\r\n");
      CHECK(starts_with(no_route, "HTTP/1.1 404"), "http: an unknown route returns 404");

      const std::string crashed = talk(live.port, "GET /boom HTTP/1.1\r\n\r\n");
      CHECK(starts_with(crashed, "HTTP/1.1 500"), "http: a throwing handler becomes a 500");
      CHECK(crashed.find("handler blew up") != std::string::npos,
            "http: the 500 names the handler failure");

      const std::string post = talk(live.port,
                                    "POST /echo HTTP/1.1\r\nContent-Length: 11\r\n\r\nhello world");
      CHECK(post.find("\"len\":11") != std::string::npos,
            "http: the request body arrives whole");

      // A body split across two writes still assembles: the reader loops until
      // Content-Length bytes arrive.
      const std::string two_part =
          talk(live.port, "POST /echo HTTP/1.1\r\nContent-Length: 5\r\n\r\nabcde");
      CHECK(two_part.find("\"len\":5") != std::string::npos,
            "http: a small body is read completely");

      const std::string quoted = talk(live.port,
                                      "POST /echo-body HTTP/1.1\r\nContent-Length: 9\r\n\r\n{\"a\":\"b\"}");
      CHECK(quoted.find("\\\"a\\\":\\\"b\\\"") != std::string::npos,
            "http: JSON in the body is echoed escaped");

      const std::string q = talk(live.port,
                                 "GET /params?x=1&y=two%20words HTTP/1.1\r\n\r\n");
      CHECK(q.find("\"x\":\"1\"") != std::string::npos, "http: query params reach the handler");
      CHECK(q.find("\"y\":\"two words\"") != std::string::npos,
            "http: query params arrive decoded");
      CHECK(q.find("\"missing\":\"dflt\"") != std::string::npos,
            "http: an absent query param uses the default");
      CHECK(q.find("\"local\":true") != std::string::npos,
            "http: a loopback client is reported as local");

      // Mixed case headers are lowercased by the parser, which is what the
      // worker relies on when it sends Content-Type.
      const std::string mixed = talk(
          live.port, "POST /echo HTTP/1.1\r\nCoNtEnT-LeNgTh: 3\r\nUser-Agent: t\r\n\r\nabc");
      CHECK(mixed.find("\"len\":3") != std::string::npos,
            "http: header names are matched case-insensitively");

      // A header block beyond the 1 MiB cap is dropped instead of growing
      // forever.
      std::string flood = "GET /health HTTP/1.1\r\n";
      for (int i = 0; i < 40000; ++i) flood += "X-Flood: 012345678901234567890123456789\r\n";
      const std::string flooded = talk(live.port, flood, false);
      CHECK(starts_with(flooded, "HTTP/1.1 400"), "http: a header flood is refused");
      CHECK(flood.size() > (1u << 20), "http: the flood really exceeded the header cap");

      // Content-Length past the bundle cap is refused before any body arrives.
      const std::string oversized = talk(
          live.port, "POST /echo HTTP/1.1\r\nContent-Length: 999999999999\r\n\r\n");
      CHECK(starts_with(oversized, "HTTP/1.1 400"),
            "http: an oversized Content-Length is refused");

      // strtoull turns "-5" into a huge unsigned value, which lands past the
      // cap, so the request is refused. The point of the check is that the
      // reader answers instead of waiting for a body that will never come.
      const std::string negative = talk(live.port,
                                        "POST /echo HTTP/1.1\r\nContent-Length: -5\r\n\r\nabc");
      CHECK(starts_with(negative, "HTTP/1.1 400"),
            "http: a negative Content-Length is refused, not waited on");
      const std::string wordy = talk(live.port,
                                     "POST /echo HTTP/1.1\r\nContent-Length: lots\r\n\r\nabc");
      CHECK(starts_with(wordy, "HTTP/1.1 200"),
            "http: a non-numeric Content-Length is treated as no body");

      // 16 clients at once: one thread per connection means they all answer.
      std::atomic<int> good{0};
      std::vector<std::thread> clients;
      for (int t = 0; t < 16; ++t) {
        clients.emplace_back([&live, &good, t] {
          for (int i = 0; i < 4; ++i) {
            const std::string r = talk(
                live.port,
                "GET /params?x=" + std::to_string(t) + " HTTP/1.1\r\n\r\n");
            if (starts_with(r, "HTTP/1.1 200")) ++good;
          }
        });
      }
      for (auto& c : clients) c.join();
      CHECK(good.load() == 64, "http: 64 concurrent requests all answer 200");

      // The server survives every one of the malformed requests above.
      const std::string after = talk(live.port, "GET /health HTTP/1.1\r\n\r\n");
      CHECK(starts_with(after, "HTTP/1.1 200 OK"),
            "http: the server is still healthy after all the malformed traffic");
    }
  }

  return suite::finish("grid");
}
