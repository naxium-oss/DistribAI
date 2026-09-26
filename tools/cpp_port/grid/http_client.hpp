// http_client.hpp - the worker's HTTP client.
//
// Plain HTTP covers a local or LAN orchestrator. TLS covers a worker that
// reaches the grid through a free Cloudflare tunnel, where the URL is https and
// the edge certificate is public, so the default trust store is enough. TLS is
// compiled in only when OpenSSL headers are present; without them an https URL
// fails with a message that says so instead of quietly sending cleartext.
//
// One connection per request, no pooling. A worker makes a few requests per
// minute, and this keeps the failure modes obvious.
#pragma once

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cstring>
#include <string>

#ifdef DISTRIBAI_HAVE_OPENSSL
#include <openssl/err.h>
#include <openssl/ssl.h>
#endif

#include "http.hpp"

namespace distribai::http {

struct Url {
  std::string scheme = "http";
  std::string host;
  std::string path = "/";
  int port = 80;
};

// Accepts "https://host[:port]/path", "http://...", or a bare "host:port"
// (bare and http default to port 80 only when no port is given).
inline bool parse_url(const std::string& in, Url& out, std::string& err) {
  std::string rest = in;
  const size_t scheme_pos = rest.find("://");
  if (scheme_pos != std::string::npos) {
    out.scheme = rest.substr(0, scheme_pos);
    rest = rest.substr(scheme_pos + 3);
  } else if (!rest.empty() && rest[0] != '/') {
    out.scheme = "http";
  }
  if (out.scheme != "http" && out.scheme != "https") {
    err = "unsupported URL scheme: " + out.scheme;
    return false;
  }
  out.port = out.scheme == "https" ? 443 : 80;

  const size_t slash = rest.find('/');
  std::string authority = slash == std::string::npos ? rest : rest.substr(0, slash);
  out.path = slash == std::string::npos ? "/" : rest.substr(slash);
  if (authority.empty()) {
    err = "URL has no host";
    return false;
  }
  // Bracketed IPv6, or host[:port].
  if (authority[0] == '[') {
    const size_t close = authority.find(']');
    if (close == std::string::npos) {
      err = "unterminated IPv6 host";
      return false;
    }
    out.host = authority.substr(1, close - 1);
    if (close + 1 < authority.size() && authority[close + 1] == ':') {
      out.port = std::atoi(authority.substr(close + 2).c_str());
    }
  } else {
    const size_t colon = authority.rfind(':');
    if (colon != std::string::npos) {
      out.host = authority.substr(0, colon);
      out.port = std::atoi(authority.substr(colon + 1).c_str());
    } else {
      out.host = authority;
    }
  }
  if (out.host.empty() || out.port <= 0 || out.port > 65535) {
    err = "bad host or port in URL: " + in;
    return false;
  }
  return true;
}

struct ClientResponse {
  int status = 0;
  std::string body;
};

inline int connect_socket(const std::string& host, int port, std::string& err, double timeout_s) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  const std::string port_str = std::to_string(port);
  const int rc = ::getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res);
  if (rc != 0 || !res) {
    err = "resolve " + host + ": " + ::gai_strerror(rc);
    return -1;
  }
  int fd = -1;
  for (addrinfo* ai = res; ai; ai = ai->ai_next) {
    fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (fd < 0) continue;
    if (::connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
    ::close(fd);
    fd = -1;
  }
  ::freeaddrinfo(res);
  if (fd < 0) {
    err = "connect " + host + ":" + port_str + " failed: " + std::strerror(errno);
    return -1;
  }
  int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  const auto secs = static_cast<time_t>(timeout_s < 1 ? 1 : timeout_s);
  timeval tv{};
  tv.tv_sec = secs;
  tv.tv_usec = 0;
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
  return fd;
}

#ifdef DISTRIBAI_HAVE_OPENSSL
inline void ensure_ssl_init() {
  static bool done = false;
  if (!done) {
    OPENSSL_init_ssl(OPENSSL_INIT_LOAD_SSL_STRINGS | OPENSSL_INIT_LOAD_CRYPTO_STRINGS, nullptr);
    done = true;
  }
}
#endif

// Sends one request and reads one response. `url` carries scheme, host, port and
// the base path; `path` is appended to it.
inline bool request(const Url& url, const std::string& method, const std::string& path,
                    const std::string& body, const std::string& content_type,
                    ClientResponse& out, std::string& err, double timeout_s = 60.0) {
  const std::string target = path.empty() ? url.path : path;
  const int fd = connect_socket(url.host, url.port, err, timeout_s);
  if (fd < 0) return false;

  std::string req;
  req += method + " " + target + " HTTP/1.1\r\n";
  req += "Host: " + url.host + (url.port == 80 || url.port == 443 ? "" : ":" + std::to_string(url.port)) + "\r\n";
  req += "User-Agent: distribai-worker/1\r\n";
  req += "Accept: application/json\r\n";
  req += "Connection: close\r\n";
  if (!body.empty()) {
    req += "Content-Type: " + content_type + "\r\n";
    req += "Content-Length: " + std::to_string(body.size()) + "\r\n";
  }
  req += "\r\n";
  req += body;

#ifdef DISTRIBAI_HAVE_OPENSSL
  if (url.scheme == "https") {
    ensure_ssl_init();
    SSL_CTX* ctx = SSL_CTX_new(TLS_client_method());
    if (!ctx) {
      ::close(fd);
      err = "SSL_CTX_new failed";
      return false;
    }
    SSL_CTX_set_default_verify_paths(ctx);
    SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr);
    SSL* ssl = SSL_new(ctx);
    if (!ssl) {
      SSL_CTX_free(ctx);
      ::close(fd);
      err = "SSL_new failed";
      return false;
    }
    SSL_set_fd(ssl, fd);
    SSL_set_tlsext_host_name(ssl, url.host.c_str());
    SSL_set1_host(ssl, url.host.c_str());
    if (SSL_connect(ssl) != 1) {
      const unsigned long e = ERR_get_error();
      char buf[256] = {0};
      ERR_error_string_n(e, buf, sizeof(buf));
      err = std::string("TLS handshake failed: ") + buf;
      SSL_free(ssl);
      SSL_CTX_free(ctx);
      ::close(fd);
      return false;
    }
    size_t sent = 0;
    while (sent < req.size()) {
      const int n = SSL_write(ssl, req.data() + sent, static_cast<int>(req.size() - sent));
      if (n <= 0) {
        err = "TLS write failed";
        SSL_free(ssl);
        SSL_CTX_free(ctx);
        ::close(fd);
        return false;
      }
      sent += static_cast<size_t>(n);
    }
    std::string raw;
    char buf[8192];
    for (;;) {
      const int n = SSL_read(ssl, buf, sizeof(buf));
      if (n <= 0) break;
      raw.append(buf, static_cast<size_t>(n));
      if (raw.size() > grid::kMaxBundleBytes) break;
    }
    SSL_shutdown(ssl);
    SSL_free(ssl);
    SSL_CTX_free(ctx);
    ::close(fd);

    const size_t head_end = raw.find("\r\n\r\n");
    if (head_end == std::string::npos) {
      err = "malformed response";
      return false;
    }
    out.status = std::atoi(raw.substr(raw.find(' ') + 1, 3).c_str());
    out.body = raw.substr(head_end + 4);
    return true;
  }
#else
  if (url.scheme == "https") {
    ::close(fd);
    err = "this build has no TLS support (rebuild with OpenSSL headers present), "
          "so an https:// orchestrator URL cannot be used";
    return false;
  }
#endif

  size_t sent = 0;
  while (sent < req.size()) {
    const ssize_t n = ::send(fd, req.data() + sent, req.size() - sent, MSG_NOSIGNAL);
    if (n <= 0) {
      err = "send failed";
      ::close(fd);
      return false;
    }
    sent += static_cast<size_t>(n);
  }
  std::string raw;
  char buf[8192];
  for (;;) {
    const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
    if (n <= 0) break;
    raw.append(buf, static_cast<size_t>(n));
    if (raw.size() > grid::kMaxBundleBytes) break;
  }
  ::close(fd);

  const size_t head_end = raw.find("\r\n\r\n");
  if (head_end == std::string::npos) {
    err = "malformed response";
    return false;
  }
  const size_t sp = raw.find(' ');
  out.status = sp == std::string::npos ? 0 : std::atoi(raw.substr(sp + 1, 3).c_str());
  out.body = raw.substr(head_end + 4);
  return true;
}

}  // namespace distribai::http
