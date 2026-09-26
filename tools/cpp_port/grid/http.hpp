// http.hpp - a small HTTP/1.1 server for the native grid.
//
// The port carries no third-party HTTP dependency, so this is a direct
// implementation of the slice the grid needs: request line, headers,
// Content-Length bodies, and one response per connection with
// "Connection: close". Keep-alive and chunked uploads are deliberately absent.
// Requests are short JSON posts, and closing each connection keeps the code
// easy to audit.
//
// Threading: one detached thread per connection. The store serializes its own
// access, so no extra locking lives here.
#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "protocol.hpp"

namespace distribai::http {

using json::Value;

struct Request {
  std::string method;
  std::string target;
  std::string path;
  std::string query;
  std::string body;
  std::map<std::string, std::string> headers;
  std::map<std::string, std::string> params;
  std::string peer;

  std::string header(const std::string& key) const {
    auto it = headers.find(key);
    return it == headers.end() ? std::string() : it->second;
  }
  std::string param(const std::string& key, const std::string& dflt = "") const {
    auto it = params.find(key);
    return it == params.end() ? dflt : it->second;
  }
  bool is_local() const {
    return peer.rfind("127.", 0) == 0 || peer == "::1" || peer == "1";
  }
};

struct Response {
  int status = 200;
  std::string content_type = "application/json; charset=utf-8";
  std::string body;
  std::map<std::string, std::string> headers;

  static Response json(const std::string& payload, int code = 200) {
    Response r;
    r.status = code;
    r.content_type = "application/json; charset=utf-8";
    r.body = payload;
    return r;
  }
  static Response text(const std::string& payload, int code = 200,
                       const std::string& type = "text/plain; charset=utf-8") {
    Response r;
    r.status = code;
    r.content_type = type;
    r.body = payload;
    return r;
  }
  static Response error(int code, const std::string& message) {
    return json(grid::Jw().s("error", message).str(), code);
  }
};

using Handler = std::function<Response(const Request&)>;

inline std::string status_text(int code) {
  switch (code) {
    case 200: return "OK";
    case 201: return "Created";
    case 202: return "Accepted";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 403: return "Forbidden";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 409: return "Conflict";
    case 413: return "Payload Too Large";
    case 415: return "Unsupported Media Type";
    case 429: return "Too Many Requests";
    case 500: return "Internal Server Error";
    case 503: return "Service Unavailable";
    default: return "OK";
  }
}

inline std::string url_decode(const std::string& in) {
  std::string out;
  out.reserve(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    if (in[i] == '+') {
      out += ' ';
    } else if (in[i] == '%' && i + 2 < in.size()) {
      const int hi = std::isxdigit(static_cast<unsigned char>(in[i + 1]))
                         ? std::stoi(in.substr(i + 1, 1), nullptr, 16)
                         : -1;
      const int lo = std::isxdigit(static_cast<unsigned char>(in[i + 2]))
                         ? std::stoi(in.substr(i + 2, 1), nullptr, 16)
                         : -1;
      if (hi < 0 || lo < 0) {
        out += in[i];
      } else {
        out += static_cast<char>((hi << 4) | lo);
        i += 2;
      }
    } else {
      out += in[i];
    }
  }
  return out;
}

inline void parse_query(const std::string& query, std::map<std::string, std::string>& out) {
  size_t start = 0;
  while (start <= query.size()) {
    const size_t amp = query.find('&', start);
    const std::string pair = query.substr(start, amp == std::string::npos ? std::string::npos
                                                                         : amp - start);
    if (!pair.empty()) {
      const size_t eq = pair.find('=');
      if (eq == std::string::npos) {
        out[pair] = "";
      } else {
        out[pair.substr(0, eq)] = url_decode(pair.substr(eq + 1));
      }
    }
    if (amp == std::string::npos) break;
    start = amp + 1;
  }
}

inline std::string content_type_for(const std::string& path) {
  const size_t dot = path.find_last_of('.');
  const std::string ext = dot == std::string::npos ? "" : path.substr(dot);
  if (ext == ".html") return "text/html; charset=utf-8";
  if (ext == ".css") return "text/css; charset=utf-8";
  if (ext == ".js") return "text/javascript; charset=utf-8";
  if (ext == ".json") return "application/json; charset=utf-8";
  if (ext == ".svg") return "image/svg+xml";
  if (ext == ".ico") return "image/x-icon";
  return "application/octet-stream";
}

// Reads a file whole. Used for dashboard assets and stored job results.
inline bool read_file(const std::string& path, std::string& out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  std::ostringstream ss;
  ss << in.rdbuf();
  out = ss.str();
  return true;
}

inline bool is_readable(const std::string& path) {
  return !path.empty() && ::access(path.c_str(), R_OK) == 0;
}

inline bool write_file(const std::string& path, const std::string& data) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return false;
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
  return out.good();
}

class Server {
 public:
  Server(std::string host, int port, Handler handler)
      : host_(std::move(host)), port_(port), handler_(std::move(handler)) {}

  bool start(std::string& err) {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
      err = "socket() failed";
      return false;
    }
    int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    if (::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
      // Fall back to any-address when given a name we cannot parse as IPv4.
      addr.sin_addr.s_addr = htonl(INADDR_ANY);
    }
    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      err = "bind(" + host_ + ":" + std::to_string(port_) + ") failed: " + std::strerror(errno);
      ::close(listen_fd_);
      listen_fd_ = -1;
      return false;
    }
    if (::listen(listen_fd_, 64) != 0) {
      err = "listen() failed";
      ::close(listen_fd_);
      listen_fd_ = -1;
      return false;
    }
    socklen_t len = sizeof(addr);
    if (::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len) == 0) {
      bound_port_ = ntohs(addr.sin_port);
    }
    return true;
  }

  int port() const { return bound_port_; }

  // Accepts until `stop` is set. Each connection is handled on its own thread,
  // which is fine here: requests are short and the store serializes writes.
  void serve(const std::atomic<bool>& stop) {
    while (!stop.load()) {
      sockaddr_in peer{};
      socklen_t plen = sizeof(peer);
      const int fd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&peer), &plen);
      if (fd < 0) {
        if (errno == EINTR) continue;  // signals set the stop flag
        if (stop.load()) break;
        continue;
      }
      char buf[64] = {0};
      ::inet_ntop(AF_INET, &peer.sin_addr, buf, sizeof(buf));
      const std::string peer_str = buf;
      std::thread([this, fd, peer_str] { handle(fd, peer_str); }).detach();
    }
  }

  // Stops the accept loop. accept() cannot be woken by close() from another
  // thread, so shut the socket down first: shutdown makes the blocked accept
  // return, the loop sees the stop flag and exits. Without this, a caller that
  // joins the serving thread hangs until the next connection arrives.
  void close_listen() {
    if (listen_fd_ >= 0) {
      ::shutdown(listen_fd_, SHUT_RDWR);
      ::close(listen_fd_);
      listen_fd_ = -1;
    }
  }

 private:
  void handle(int fd, const std::string& peer) {
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    Request req;
    req.peer = peer;
    Response res;
    if (!read_request(fd, req)) {
      res = Response::error(400, "malformed request");
    } else {
      try {
        res = handler_(req);
      } catch (const std::exception& e) {
        res = Response::error(500, std::string("handler: ") + e.what());
      }
    }
    std::ostringstream out;
    out << "HTTP/1.1 " << res.status << " " << status_text(res.status) << "\r\n";
    out << "Content-Type: " << res.content_type << "\r\n";
    out << "Content-Length: " << res.body.size() << "\r\n";
    out << "Connection: close\r\n";
    out << "Cache-Control: no-store\r\n";
    for (const auto& kv : res.headers) out << kv.first << ": " << kv.second << "\r\n";
    out << "\r\n";
    out << res.body;
    const std::string payload = out.str();
    size_t sent = 0;
    while (sent < payload.size()) {
      const ssize_t n = ::send(fd, payload.data() + sent, payload.size() - sent, MSG_NOSIGNAL);
      if (n <= 0) break;
      sent += static_cast<size_t>(n);
    }
    ::close(fd);
  }

  bool read_request(int fd, Request& req) {
    std::string buf;
    buf.reserve(4096);
    char chunk[8192];
    size_t header_end = std::string::npos;
    while (header_end == std::string::npos) {
      const ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
      if (n <= 0) return false;
      buf.append(chunk, static_cast<size_t>(n));
      header_end = buf.find("\r\n\r\n");
      if (buf.size() > (1u << 20)) return false;  // header flood
    }

    const std::string head = buf.substr(0, header_end);
    std::istringstream hs(head);
    std::string line;
    if (!std::getline(hs, line)) return false;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    {
      std::istringstream rl(line);
      std::string version;
      rl >> req.method >> req.target >> version;
      if (req.method.empty() || req.target.empty()) return false;
      const size_t q = req.target.find('?');
      req.path = q == std::string::npos ? req.target : req.target.substr(0, q);
      req.query = q == std::string::npos ? "" : req.target.substr(q + 1);
      parse_query(req.query, req.params);
    }

    while (std::getline(hs, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.empty()) continue;
      const size_t colon = line.find(':');
      if (colon == std::string::npos) continue;
      std::string key = line.substr(0, colon);
      for (char& c : key) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      std::string value = line.substr(colon + 1);
      while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.erase(0, 1);
      req.headers[key] = value;
    }

    size_t want = 0;
    const std::string cl = req.header("content-length");
    if (!cl.empty()) {
      want = static_cast<size_t>(std::strtoull(cl.c_str(), nullptr, 10));
      if (want > grid::kMaxBundleBytes) return false;
    }
    const size_t already = buf.size() - (header_end + 4);
    req.body = buf.substr(header_end + 4);
    while (req.body.size() < want) {
      const ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
      if (n <= 0) break;
      req.body.append(chunk, static_cast<size_t>(n));
    }
    if (req.body.size() > want && want > 0) req.body.resize(want);
    (void)already;
    return true;
  }

  std::string host_;
  int port_;
  int bound_port_ = 0;
  int listen_fd_ = -1;
  Handler handler_;
};

}  // namespace distribai::http
