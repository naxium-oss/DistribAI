// dai_job - Feature 3: job manifest CLI.
//
// Input: a JSON job manifest (hand-rolled parser, no dependencies):
// {
//   "job_id": "demo-001",
//   "models": [{"name": "alpha", "seed": 42, "steps": 200}, ...],
//   "sandbox_count": 2,          // concurrent sandboxed children
//   "rlimits": {"mem_mb": 64, "cpu_sec": 120},
//   "checkpoint_dir": "runtime/checkpoints/cpp",   // optional
//   "aggregate": "trimmed_mean"  // mean | median | trimmed_mean
// }
// Output: one JSON line on stdout: job_result envelope (machine-readable,
// baseline-contract keys per model plus job-level aggregate info).
//
// Usage: dai_job --manifest job.json [--dry-run]
#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

#include "../core/envelope.hpp"
#include "../core/multi_model.hpp"
#include "../core/checkpoint.hpp"

using namespace distribai;

// --- minimal JSON reader for the manifest (flat + one nested array) ---
namespace mini {

struct ModelEntry { std::string name; uint64_t seed = 42; int steps = 200; };
struct Manifest {
  std::string job_id = "job";
  std::vector<ModelEntry> models;
  int sandbox_count = 2;
  uint64_t mem_mb = 64, cpu_sec = 120;
  std::string checkpoint_dir;
  std::string aggregate = "trimmed_mean";
  bool ok = false;
  std::string error;
};

void skip_ws(const std::string& s, size_t& i) {
  while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i;
}
std::string parse_string(const std::string& s, size_t& i) {
  ++i;  // opening quote
  std::string out;
  while (i < s.size() && s[i] != '"') {
    if (s[i] == '\\' && i + 1 < s.size()) { out += s[i + 1]; i += 2; }
    else out += s[i++];
  }
  ++i;
  return out;
}
double parse_number(const std::string& s, size_t& i) {
  size_t start = i;
  while (i < s.size() && (isdigit((unsigned char)s[i]) || s[i] == '-' || s[i] == '+' ||
                          s[i] == '.' || s[i] == 'e' || s[i] == 'E')) ++i;
  return std::atof(s.substr(start, i - start).c_str());
}

Manifest parse(const std::string& text) {
  Manifest m;
  size_t i = 0;
  skip_ws(text, i);
  if (i >= text.size() || text[i] != '{') { m.error = "manifest must be a JSON object"; return m; }
  ++i;
  while (i < text.size()) {
    skip_ws(text, i);
    if (i < text.size() && text[i] == '}') { m.ok = true; return m; }
    if (i >= text.size() || text[i] != '"') { m.error = "expected key"; return m; }
    const std::string key = parse_string(text, i);
    skip_ws(text, i);
    if (i >= text.size() || text[i] != ':') { m.error = "expected ':' after " + key; return m; }
    ++i;
    skip_ws(text, i);
    if (key == "job_id") m.job_id = parse_string(text, i);
    else if (key == "sandbox_count") m.sandbox_count = static_cast<int>(parse_number(text, i));
    else if (key == "aggregate") m.aggregate = parse_string(text, i);
    else if (key == "checkpoint_dir") m.checkpoint_dir = parse_string(text, i);
    else if (key == "mem_mb" || key == "cpu_sec") {
      const double v = parse_number(text, i);
      if (key == "mem_mb") m.mem_mb = static_cast<uint64_t>(v); else m.cpu_sec = static_cast<uint64_t>(v);
    }
    else if (key == "rlimits") {
      // nested { "mem_mb": N, "cpu_sec": N }
      skip_ws(text, i);
      if (text[i] != '{') { m.error = "rlimits must be an object"; return m; }
      ++i;
      while (i < text.size() && text[i] != '}') {
        skip_ws(text, i);
        const std::string k2 = parse_string(text, i);
        skip_ws(text, i); ++i; skip_ws(text, i);
        const double v = parse_number(text, i);
        if (k2 == "mem_mb") m.mem_mb = static_cast<uint64_t>(v);
        if (k2 == "cpu_sec") m.cpu_sec = static_cast<uint64_t>(v);
        skip_ws(text, i);
        if (i < text.size() && text[i] == ',') ++i;
      }
      ++i;
    }
    else if (key == "models") {
      skip_ws(text, i);
      if (text[i] != '[') { m.error = "models must be an array"; return m; }
      ++i;
      while (i < text.size() && text[i] != ']') {
        skip_ws(text, i);
        if (text[i] == ',') { ++i; continue; }
        if (text[i] != '{') { m.error = "model entries must be objects"; return m; }
        ++i;
        ModelEntry e;
        while (i < text.size() && text[i] != '}') {
          skip_ws(text, i);
          if (text[i] == ',') { ++i; continue; }
          const std::string k2 = parse_string(text, i);
          skip_ws(text, i); ++i; skip_ws(text, i);
          if (k2 == "name") e.name = parse_string(text, i);
          else if (k2 == "seed") e.seed = static_cast<uint64_t>(parse_number(text, i));
          else if (k2 == "steps") e.steps = static_cast<int>(parse_number(text, i));
          skip_ws(text, i);
        }
        ++i;
        m.models.push_back(e);
        skip_ws(text, i);
      }
      ++i;
    }
    else { m.error = "unknown key: " + key; return m; }
    skip_ws(text, i);
    if (i < text.size() && text[i] == ',') { ++i; skip_ws(text, i); }
  }
  m.ok = true;
  return m;
}

}  // namespace mini

static std::string json_escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (c == '"' || c == '\\') { out += '\\'; out += c; }
    else if (c == '\n') out += "\\n";
    else out += c;
  }
  return out;
}

int main(int argc, char** argv) {
  const char* manifest_path = nullptr;
  bool dry = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--manifest") && i + 1 < argc) manifest_path = argv[++i];
    else if (!std::strcmp(argv[i], "--dry-run")) dry = true;
  }
  if (!manifest_path) {
    std::printf("{\"kind\": \"job_result\", \"status\": \"error\", \"error\": \"--manifest required\"}\n");
    return 2;
  }
  FILE* f = std::fopen(manifest_path, "rb");
  if (!f) {
    std::printf("{\"kind\": \"job_result\", \"status\": \"error\", \"error\": \"manifest not found: %s\"}\n",
                manifest_path);
    return 2;
  }
  std::string text;
  char buf[4096];
  size_t r;
  while ((r = std::fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, r);
  std::fclose(f);

  const mini::Manifest mf = mini::parse(text);
  if (!mf.ok) {
    std::printf("{\"kind\": \"job_result\", \"status\": \"error\", \"error\": \"manifest: %s\"}\n",
                json_escape(mf.error).c_str());
    return 2;
  }
  if (mf.models.empty()) {
    std::printf("{\"kind\": \"job_result\", \"status\": \"error\", \"error\": \"manifest has no models\"}\n");
    return 2;
  }

  SandboxLimits lim;
  lim.mem_mb = mf.mem_mb;
  lim.cpu_sec = mf.cpu_sec;
  MultiModelTrainer trainer(lim, static_cast<size_t>(std::max(1, mf.sandbox_count)));
  for (const auto& e : mf.models) trainer.register_model({e.name, e.seed, e.steps});

  if (dry) {
    std::printf("{\"kind\": \"job_result\", \"status\": \"dry_run\", \"job_id\": \"%s\", "
                "\"models\": %zu, \"sandbox_count\": %d, \"mem_mb\": %llu, \"cpu_sec\": %llu}\n",
                json_escape(mf.job_id).c_str(), mf.models.size(), mf.sandbox_count,
                static_cast<unsigned long long>(mf.mem_mb),
                static_cast<unsigned long long>(mf.cpu_sec));
    return 0;
  }

  const auto t0 = std::chrono::steady_clock::now();
  const auto outs = trainer.train_all();
  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const auto rep = MultiModelTrainer::aggregate(outs);

  // Emit job_result: one envelope per model aggregated into a single JSON.
  env::Envelope job(env::Kind::JobResult, 0);
  job.add_str(40, mf.job_id);
  std::string json = "{\"kind\": \"job_result\", \"status\": \"ok\", \"job_id\": \"" +
                     json_escape(mf.job_id) + "\", \"wall_s\": " +
                     std::to_string(wall) + ", \"contributors\": " +
                     std::to_string(rep.contributors) + ", \"models\": [";
  bool first = true;
  int ok_count = 0;
  for (const auto& o : outs) {
    if (!first) json += ", ";
    first = false;
    json += "{\"name\": \"" + json_escape(o.model_name) + "\", \"status\": \"" +
            (o.ok ? "ok" : "error") + "\"";
    if (o.ok) {
      ++ok_count;
      json += ", \"steps\": " + std::to_string(o.steps) +
              ", \"steps_per_s\": " + std::to_string(o.steps_per_s) +
              ", \"final_loss\": " + std::to_string(o.final_loss) +
              ", \"grad_len\": " + std::to_string(o.grad_values.size());
    } else {
      json += ", \"error\": \"" + json_escape(o.error) + "\"";
    }
    json += "}";
  }
  json += "]";
  if (rep.contributors > 0) {
    double tmean = 0;
    for (double v : rep.trimmed_mean) tmean += v;
    json += ", \"aggregate\": \"" + json_escape(mf.aggregate) +
            "\", \"aggregate_grad_sum\": " + std::to_string(tmean) +
            ", \"aggregate_grad_len\": " + std::to_string(rep.trimmed_mean.size());
  }
  json += ", \"models_ok\": " + std::to_string(ok_count) + "}";
  std::printf("%s\n", json.c_str());

  // Also write the envelope bytes next to nothing else (for future pipe use).
  (void)job;
  return ok_count == static_cast<int>(outs.size()) ? 0 : 1;
}
