// distribai_torch_train: the LibTorch live train path.
//
// This is the ONLY live trainer the grid runs. It consumes the job spec that
// tools/trainer_translate emits from a Python trainer:
//
//   job.json  : optimizer/loss/steps/batch shape and metadata
//   model.pt  : a TorchScript module (the translated architecture)
//   x.pt,y.pt : optional tensor data (otherwise data is synthesized
//               deterministically from the seed, identical in Python & C++)
//
// Two modes:
//   single  : `--spec job.json`            -> one model
//   manifest: spec carries a "models":[..] -> N sandboxed models + aggregation
//
// Every model runs inside the native sandbox (rlimits + namespaces) and returns
// a versioned envelope carrying its flattened gradients; job-level Byzantine
// aggregation (mean | median | trimmed_mean) reuses the port's MultiModel
// contract, so LibTorch jobs and the parity grid speak the same wire format.
//
// Contract keys on stdout mirror apps/dai_job.cpp so the orchestrator can
// consume either engine without a schema fork.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <torch/optim.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "../core/envelope.hpp"
#include "../core/multi_model.hpp"
#include "../sandbox/sandbox.hpp"
#include "json_lite.hpp"

namespace {

using distribai::AggregateReport;
using distribai::MultiModelTrainer;
using distribai::SandboxLimits;
using distribai::SandboxResult;
using distribai::TrainOutcome;
using distribai::TrainPod;

// The port's JSON reader lives in distribai::json; alias it so the job-spec
// plumbing below stays readable.
namespace json = distribai::json;

constexpr int kMaxSteps = 1000000;

struct JobCfg {
  std::string job_id = "torch-job";
  std::string name = "model";
  std::string model_path;      // TorchScript module (required)
  std::string device = "auto";  // resolved to cpu | cuda before training
  std::string optimizer = "adamw";
  std::string loss = "mse";
  double lr = 0.01;
  double weight_decay = 0.0;
  double momentum = 0.9;
  int steps = 200;
  int64_t batch_size = 64;
  uint64_t seed = 42;
  std::string input_path;   // optional tensor file
  std::string target_path;  // optional tensor file
  std::vector<int64_t> input_shape;
  std::vector<int64_t> target_shape;
  std::string target_dtype = "float";  // float | long
  std::string checkpoint_dir;
  std::string aggregate = "trimmed_mean";
  int sandbox_count = 2;
  SandboxLimits limits;
  bool sandbox = true;
};

std::string dir_of(const std::string& path) {
  const size_t p = path.find_last_of('/');
  return p == std::string::npos ? std::string(".") : path.substr(0, p);
}

std::string abspath(const std::string& base, const std::string& p) {
  if (p.empty() || p[0] == '/') return p;
  return base + "/" + p;
}

// Resolve the job's requested device against what this build actually has.
// `cuda` and `auto` (or an omitted device) both prefer CUDA; when it is not
// available the job runs on the CPU instead of failing, so a translated bundle
// still runs on a CPU-only box. An explicit `cpu` stays on the CPU even when a
// GPU is present.
std::string resolve_device_name(const std::string& requested) {
  if (requested == "cpu") return "cpu";
  return torch::cuda::is_available() ? "cuda" : "cpu";
}

// ---- data tensor transport (DAIT format) --------------------------------
// Python's torch.save writes a zip the C++ torch::load cannot read in this
// build (it routes through the jit archive expecting constants.pkl), so the
// translator and the runner speak a tiny documented format instead:
//   "DAIT" | u32 version=1 | u8 dtype(0=f32,1=i64) | u8 ndim | ndim*u64 dims | data
// little-endian, row-major, contiguous. Both sides are ~20 lines and there is
// no serialization layer to drift.
bool read_tensor_bin(const std::string& path, torch::Tensor& out, std::string& err) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    err = "cannot open " + path;
    return false;
  }
  auto fail = [&](const std::string& m) {
    err = m + " in " + path;
    std::fclose(f);
    return false;
  };
  char magic[4];
  uint32_t version = 0;
  uint8_t dtype = 0, ndim = 0;
  if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "DAIT", 4) != 0) {
    return fail("bad magic");
  }
  if (std::fread(&version, 4, 1, f) != 1 || version != 1) return fail("bad version");
  if (std::fread(&dtype, 1, 1, f) != 1 || std::fread(&ndim, 1, 1, f) != 1) {
    return fail("truncated header");
  }
  std::vector<int64_t> dims;
  for (int i = 0; i < ndim; ++i) {
    uint64_t d = 0;
    if (std::fread(&d, 8, 1, f) != 1) return fail("truncated dims");
    dims.push_back(static_cast<int64_t>(d));
  }
  torch::TensorOptions opts;
  size_t elem = 0;
  if (dtype == 0) {
    opts = opts.dtype(torch::kFloat32);
    elem = 4;
  } else if (dtype == 1) {
    opts = opts.dtype(torch::kInt64);
    elem = 8;
  } else {
    return fail("unsupported dtype");
  }
  int64_t n = 1;
  for (int64_t d : dims) n *= d;
  auto t = torch::empty(dims, opts);
  const size_t need = static_cast<size_t>(n) * elem;
  if (need > 0 && std::fread(t.data_ptr(), 1, need, f) != need) return fail("truncated data");
  std::fclose(f);
  out = t;
  return true;
}

int64_t prod(const std::vector<int64_t>& v) {
  int64_t n = 1;
  for (int64_t d : v) n *= d;
  return n;
}

std::vector<int64_t> shape_of(const json::Value& v, const std::string& key) {
  std::vector<int64_t> out;
  const json::Value* a = v.get(key);
  if (!a || !a->is_array()) return out;
  for (const auto& e : a->as_array()) {
    if (e.is_number()) out.push_back(static_cast<int64_t>(e.as_number()));
  }
  return out;
}

// ---- deterministic synthetic data (identical arithmetic in Python) ----
// x[i, j] = (((seed*131 + i*17 + j*7) % 1000) - 500) / 1000
// y[i, k] = (((seed*271 + i*13 + k*5) % 1000) - 500) / 1000   [float]
// y[i]    = (seed + i*7) % num_classes                        [long]
torch::Tensor synth_x(const JobCfg& c, int64_t bs) {
  const int64_t per = std::max<int64_t>(1, prod(c.input_shape));
  auto x = torch::empty({bs, per}, torch::kFloat32);
  float* d = x.data_ptr<float>();
  for (int64_t i = 0; i < bs; ++i) {
    for (int64_t j = 0; j < per; ++j) {
      const int64_t mod = (static_cast<int64_t>(c.seed) * 131 + i * 17 + j * 7) % 1000;
      d[i * per + j] = static_cast<float>(mod - 500) / 1000.0f;
    }
  }
  if (c.input_shape.size() > 1) {
    std::vector<int64_t> s{bs};
    s.insert(s.end(), c.input_shape.begin(), c.input_shape.end());
    return x.view(s).contiguous();
  }
  return x;
}

torch::Tensor synth_y(const JobCfg& c, int64_t bs) {
  if (c.target_dtype == "long") {
    const int64_t classes = c.target_shape.empty() ? 2 : c.target_shape.back();
    auto y = torch::empty({bs}, torch::kLong);
    int64_t* d = y.data_ptr<int64_t>();
    for (int64_t i = 0; i < bs; ++i) {
      d[i] = (static_cast<int64_t>(c.seed) + i * 7) % classes;
    }
    return y;
  }
  const int64_t per = std::max<int64_t>(1, prod(c.target_shape));
  auto y = torch::empty({bs, per}, torch::kFloat32);
  float* d = y.data_ptr<float>();
  for (int64_t i = 0; i < bs; ++i) {
    for (int64_t j = 0; j < per; ++j) {
      const int64_t mod = (static_cast<int64_t>(c.seed) * 271 + i * 13 + j * 5) % 1000;
      d[i * per + j] = static_cast<float>(mod - 500) / 1000.0f;
    }
  }
  if (c.target_shape.size() > 1) {
    std::vector<int64_t> s{bs};
    s.insert(s.end(), c.target_shape.begin(), c.target_shape.end());
    return y.view(s).contiguous();
  }
  return y;
}

struct TorchOutcome {
  bool ok = false;
  std::string error;
  int steps = 0;
  int64_t param_count = 0;
  double wall_s = 0, steps_per_s = 0, final_loss = 0, first_loss = 0;
  std::vector<double> grad_values;
  std::string checkpoint_id;
};

std::unique_ptr<torch::optim::Optimizer> make_optimizer(const JobCfg& c,
                                                      std::vector<torch::Tensor>& params) {
  if (c.optimizer == "sgd") {
    torch::optim::SGDOptions o(c.lr);
    o.momentum(c.momentum);
    o.weight_decay(c.weight_decay);
    return std::make_unique<torch::optim::SGD>(params, o);
  }
  if (c.optimizer == "adam") {
    torch::optim::AdamOptions o(c.lr);
    o.weight_decay(c.weight_decay);
    return std::make_unique<torch::optim::Adam>(params, o);
  }
  if (c.optimizer == "adamw") {
    torch::optim::AdamWOptions o(c.lr);
    o.weight_decay(c.weight_decay);
    return std::make_unique<torch::optim::AdamW>(params, o);
  }
  return nullptr;
}

torch::Tensor compute_loss(const JobCfg& c, const torch::Tensor& pred, const torch::Tensor& y) {
  if (c.loss == "cross_entropy") return torch::nn::functional::cross_entropy(pred, y);
  if (c.loss == "l1") return torch::l1_loss(pred, y);
  return torch::mse_loss(pred, y);
}

TorchOutcome run_torch(const JobCfg& c) {
  TorchOutcome out;
  out.steps = c.steps;
  try {
    torch::set_num_threads(1);
    torch::manual_seed(static_cast<int64_t>(c.seed));

    const torch::Device dev = c.device == "cuda" ? torch::kCUDA : torch::kCPU;

    torch::jit::Module module;
    try {
      module = torch::jit::load(c.model_path, dev);
    } catch (const c10::Error& e) {
      out.error = std::string("load model.pt: ") + e.what();
      return out;
    }
    module.eval();

    // TorchScript parameters are leaf tensors owned by the module; keep the
    // same TensorImpl so optimizer steps update the live weights (a .data()
    // copy would silently train a detached shadow model).
    std::vector<torch::Tensor> params;
    for (auto p : module.parameters()) {
      p.set_requires_grad(true);
      params.push_back(p);
    }
    if (params.empty()) {
      out.error = "model.pt exposes no parameters (cannot build an optimizer)";
      return out;
    }
    out.param_count = 0;
    for (auto& p : params) out.param_count += p.numel();

    auto opt = make_optimizer(c, params);
    if (!opt) {
      out.error = "unsupported optimizer: " + c.optimizer;
      return out;
    }

    torch::Tensor x, y;
    const bool have_data = !c.input_path.empty() && !c.target_path.empty();
    int64_t n = 0;
    if (have_data) {
      std::string io_err;
      if (!read_tensor_bin(c.input_path, x, io_err) ||
          !read_tensor_bin(c.target_path, y, io_err)) {
        out.error = "data: " + io_err;
        return out;
      }
      x = x.to(dev);
      y = y.to(dev);
      if (c.target_dtype == "long") y = y.to(torch::kLong);
      n = x.size(0);
      if (n <= 0) {
        out.error = "input tensor is empty";
        return out;
      }
    } else {
      x = synth_x(c, c.batch_size).to(dev);
      y = synth_y(c, c.batch_size).to(dev);
      n = c.batch_size;
    }

    const int64_t bs = std::min<int64_t>(c.batch_size, n);
    double first = 0, last = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < c.steps; ++s) {
      torch::Tensor xb, yb;
      if (have_data && bs < n) {
        const int64_t off = (static_cast<int64_t>(s) * bs) % n;
        auto idx = torch::arange(off, off + bs, torch::kLong).remainder(n);
        xb = x.index_select(0, idx.to(dev));
        yb = y.index_select(0, idx.to(dev));
      } else {
        xb = x;
        yb = y;
      }
      opt->zero_grad();
      torch::Tensor pred;
      try {
        pred = module.forward({xb}).toTensor();
      } catch (const c10::Error& e) {
        out.error = std::string("forward failed: ") + e.what();
        return out;
      }
      torch::Tensor loss;
      try {
        loss = compute_loss(c, pred, yb);
        loss.backward();
      } catch (const c10::Error& e) {
        out.error = std::string("loss/backward failed: ") + e.what();
        return out;
      }
      opt->step();
      last = loss.item<double>();
      if (s == 0) first = last;
    }
    const auto t1 = std::chrono::steady_clock::now();
    out.wall_s = std::chrono::duration<double>(t1 - t0).count();
    out.steps_per_s = c.steps > 0 && out.wall_s > 0 ? c.steps / out.wall_s : 0;
    out.final_loss = last;
    out.first_loss = first;

    for (auto& p : params) {
      if (p.grad().defined()) {
        auto g = p.grad().contiguous().to(torch::kCPU).to(torch::kFloat64).view({-1});
        const double* gd = g.data_ptr<double>();
        out.grad_values.insert(out.grad_values.end(), gd, gd + g.numel());
      }
    }
    if (!c.checkpoint_dir.empty()) {
      const std::string ck = c.checkpoint_dir + "/" + c.name + "_" + std::to_string(c.seed) + ".pt";
      try {
        module.save(ck);
        out.checkpoint_id = ck;
      } catch (const c10::Error&) {
        // checkpoint failure is non-fatal for the training result
      }
    }
    out.ok = true;
    return out;
  } catch (const std::exception& e) {
    out.error = std::string("libtorch: ") + e.what();
    return out;
  }
}

// ---- one sandboxed model: child trains, parent consumes the envelope ----
void train_sandboxed(const JobCfg& base, const std::string& name, uint64_t seed, int steps,
                     TorchOutcome& out) {
  JobCfg c = base;
  c.name = name;
  c.seed = seed;
  c.steps = steps;

  TrainPod pod{};
  std::vector<uint8_t> env_bytes;
  SandboxLimits lim = c.limits;
  const SandboxResult res = distribai::run_sandboxed(
      lim, lim.cpu_sec + 60,
      [&](int wfd) {
        TorchOutcome o = run_torch(c);
        TrainPod p{};
        p.status = o.ok ? 0 : 1;
        p.steps = static_cast<uint64_t>(o.steps);
        p.param_count = static_cast<uint64_t>(o.param_count);
        p.wall_s = o.wall_s;
        p.steps_per_s = o.steps_per_s;
        p.final_loss = o.final_loss;
        p.loss_first = o.first_loss;
        p.grad_len = o.grad_values.size();
        double gsum = 0;
        for (double g : o.grad_values) gsum += g;
        p.grad_sum = gsum;
        if (!o.ok) std::strncpy(p.error, o.error.c_str(), sizeof(p.error) - 1);

        distribai::env::Envelope e(distribai::env::Kind::GradReport, 0);
        e.add_str(distribai::env::TAG_MODEL_NAME, name);
        e.add_u64(distribai::env::TAG_SEED, seed);
        e.add_u64(distribai::env::TAG_STEPS, static_cast<uint64_t>(steps));
        e.add_u64(distribai::env::TAG_N_PARAMS, static_cast<uint64_t>(o.param_count));
        e.add_f64(distribai::env::TAG_WALL_S, o.wall_s);
        e.add_f64(distribai::env::TAG_STEPS_PER_S, o.steps_per_s);
        e.add_f64(distribai::env::TAG_FINAL_LOSS, o.final_loss);
        if (!o.checkpoint_id.empty()) {
          e.add_str(distribai::env::TAG_CHECKPOINT_ID, o.checkpoint_id);
        }
        e.add_bool(distribai::env::TAG_SANDBOX, true);
        e.add_bool(distribai::env::TAG_OWN_SYSTEM, true);
        if (o.ok) {
          e.add_u64(distribai::env::TAG_GRAD_LEN, o.grad_values.size());
          e.add_f64(distribai::env::TAG_GRAD_SUM, gsum);
          e.add_f64_array(distribai::env::TAG_GRAD_VALUES, o.grad_values);
        } else {
          e.add_str(distribai::env::TAG_ERROR, o.error);
        }
        const auto bytes = e.encode();
        p.grad_blob_len = bytes.size();
        (void)write(wfd, &p, sizeof(p));
        (void)write(wfd, bytes.data(), bytes.size());
      },
      &pod, sizeof(pod), &env_bytes);

  if (!res.ok) {
    out.error = res.error.empty() ? "sandbox failed" : res.error;
    return;
  }
  distribai::env::Envelope e;
  std::string err;
  if (!distribai::env::Envelope::decode(env_bytes.data(), env_bytes.size(), e, err)) {
    out.error = "envelope decode failed: " + err;
    return;
  }
  out.steps = steps;
  uint64_t u;
  double d;
  out.param_count = e.get_u64(distribai::env::TAG_N_PARAMS, u) ? static_cast<int64_t>(u) : 0;
  out.wall_s = e.get_f64(distribai::env::TAG_WALL_S, d) ? d : 0;
  out.steps_per_s = e.get_f64(distribai::env::TAG_STEPS_PER_S, d) ? d : 0;
  out.final_loss = e.get_f64(distribai::env::TAG_FINAL_LOSS, d) ? d : 0;
  std::string ck;
  if (e.get_str(distribai::env::TAG_CHECKPOINT_ID, ck)) out.checkpoint_id = ck;
  bool ok = false;
  e.get_bool(distribai::env::TAG_SANDBOX, ok);
  out.grad_values.clear();
  e.get_f64_array(distribai::env::TAG_GRAD_VALUES, out.grad_values);
  out.ok = true;
}

// Writes a GradReport envelope for a finished model. The grid worker hands the
// path in with --envelope-out and forwards these bytes to the orchestrator, so
// the LibTorch path reports through the same wire format as the port's own
// trainer.
bool write_outcome_envelope(const std::string& path, const JobCfg& c, const TorchOutcome& o,
                            const std::string& errors) {
  distribai::env::Envelope e(distribai::env::Kind::GradReport, 0);
  e.add_str(distribai::env::TAG_MODEL_NAME, c.name);
  e.add_u64(distribai::env::TAG_SEED, c.seed);
  e.add_u64(distribai::env::TAG_STEPS, static_cast<uint64_t>(o.steps));
  e.add_u64(distribai::env::TAG_N_PARAMS, static_cast<uint64_t>(o.param_count));
  e.add_f64(distribai::env::TAG_WALL_S, o.wall_s);
  e.add_f64(distribai::env::TAG_STEPS_PER_S, o.steps_per_s);
  e.add_f64(distribai::env::TAG_FINAL_LOSS, o.final_loss);
  e.add_bool(distribai::env::TAG_SANDBOX, c.sandbox);
  if (!o.checkpoint_id.empty()) {
    e.add_str(distribai::env::TAG_CHECKPOINT_ID, o.checkpoint_id);
  }
  if (o.ok) {
    double gsum = 0;
    for (double g : o.grad_values) gsum += g;
    e.add_u64(distribai::env::TAG_GRAD_LEN, o.grad_values.size());
    e.add_f64(distribai::env::TAG_GRAD_SUM, gsum);
    e.add_f64_array(distribai::env::TAG_GRAD_VALUES, o.grad_values);
  } else {
    e.add_str(distribai::env::TAG_ERROR, errors.empty() ? o.error : errors);
  }
  const auto bytes = e.encode();
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  const size_t written = std::fwrite(bytes.data(), 1, bytes.size(), f);
  std::fclose(f);
  return written == bytes.size();
}

}  // namespace

int main(int argc, char** argv) {
  std::string spec_path;
  std::string envelope_out;
  bool json = false;
  bool force_no_sandbox = false;
  int steps_override = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--spec" || a == "--manifest") spec_path = next();
    else if (a == "--json") json = true;
    else if (a == "--no-sandbox") force_no_sandbox = true;
    else if (a == "--sandbox") force_no_sandbox = false;
    else if (a == "--envelope-out") envelope_out = next();
    else if (a == "--steps") steps_override = std::atoi(next().c_str());
  }
  if (spec_path.empty()) {
    std::printf("{\"kind\":\"torch_result\",\"engine\":\"libtorch\",\"status\":\"error\","
                "\"error\":\"--spec job.json is required\"}\n");
    return 2;
  }

  FILE* f = std::fopen(spec_path.c_str(), "rb");
  if (!f) {
    std::printf("{\"kind\":\"torch_result\",\"engine\":\"libtorch\",\"status\":\"error\","
                "\"error\":\"spec not found: %s\"}\n",
                json::escape(spec_path).c_str());
    return 2;
  }
  std::string text;
  char buf[8192];
  size_t r;
  while ((r = std::fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, r);
  std::fclose(f);

  json::Value spec;
  std::string err;
  if (!json::parse(text, spec, err)) {
    std::printf("{\"kind\":\"torch_result\",\"engine\":\"libtorch\",\"status\":\"error\","
                "\"error\":\"spec parse: %s\"}\n",
                json::escape(err).c_str());
    return 2;
  }
  if (!spec.is_object()) {
    // Valid JSON, wrong shape. Saying so beats an empty parse error.
    std::printf("{\"kind\":\"torch_result\",\"engine\":\"libtorch\",\"status\":\"error\","
                "\"error\":\"spec must be a JSON object, got a %s\"}\n",
                spec.is_array() ? "array" : "scalar");
    return 2;
  }

  const std::string base = dir_of(spec_path);
  JobCfg c;
  c.job_id = spec.str("job_id", "torch-job");
  c.model_path = abspath(base, spec.str("model"));
  c.device = resolve_device_name(spec.str("device", "auto"));
  c.optimizer = spec.str("optimizer", "adamw");
  c.loss = spec.str("loss", "mse");
  c.lr = spec.num("lr", 0.01);
  c.weight_decay = spec.num("weight_decay", 0.0);
  c.momentum = spec.num("momentum", 0.9);
  c.steps = static_cast<int>(spec.num("steps", 200));
  c.batch_size = static_cast<int64_t>(spec.num("batch_size", 64));
  c.seed = static_cast<uint64_t>(spec.num("seed", 42));
  c.input_path = abspath(base, spec.str("input"));
  c.target_path = abspath(base, spec.str("target"));
  c.input_shape = shape_of(spec, "input_shape");
  c.target_shape = shape_of(spec, "target_shape");
  c.target_dtype = spec.str("target_dtype", "float");
  c.checkpoint_dir = abspath(base, spec.str("checkpoint_dir"));
  c.aggregate = spec.str("aggregate", "trimmed_mean");
  c.sandbox_count = static_cast<int>(spec.num("sandbox_count", 2));
  c.name = spec.str("name", "model");
  if (const json::Value* rl = spec.get("rlimits")) {
    if (rl->is_object()) {
      c.limits.mem_mb = static_cast<uint64_t>(rl->num("mem_mb", 8192));
      c.limits.cpu_sec = static_cast<uint64_t>(rl->num("cpu_sec", 300));
    }
  } else {
    c.limits.mem_mb = 8192;  // torch reserves address space; keep headroom
    c.limits.cpu_sec = 300;
  }
  if (steps_override > 0) c.steps = steps_override;
  if (c.model_path.empty()) {
    std::printf("{\"kind\":\"torch_result\",\"engine\":\"libtorch\",\"status\":\"error\","
                "\"error\":\"spec has no \\\"model\\\" (run tools/trainer_translate first)\"}\n");
    return 2;
  }
  if (c.steps < 0 || c.steps > kMaxSteps) {
    std::printf("{\"kind\":\"torch_result\",\"engine\":\"libtorch\",\"status\":\"error\","
                "\"error\":\"steps out of range\"}\n");
    return 2;
  }
  c.sandbox = !force_no_sandbox;

  // ---- manifest mode: N models + Byzantine aggregation ----
  const json::Value* models = spec.get("models");
  if (models && models->is_array() && !models->as_array().empty()) {
    std::vector<TrainOutcome> outs;
    std::vector<std::string> names;
    std::vector<uint64_t> seeds;
    std::vector<int> stepsv;
    for (const auto& m : models->as_array()) {
      if (!m.is_object()) continue;
      names.push_back(m.str("name", "m" + std::to_string(names.size())));
      seeds.push_back(static_cast<uint64_t>(m.num("seed", 42)));
      stepsv.push_back(static_cast<int>(m.num("steps", c.steps)));
    }
    const auto t0 = std::chrono::steady_clock::now();
    outs.resize(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
      TorchOutcome o;
      if (c.sandbox) {
        train_sandboxed(c, names[i], seeds[i], stepsv[i], o);
      } else {
        JobCfg one = c;
        one.name = names[i];
        one.seed = seeds[i];
        one.steps = stepsv[i];
        o = run_torch(one);
      }
      TrainOutcome to;
      to.model_name = names[i];
      to.model_index = static_cast<int64_t>(i);
      to.ok = o.ok;
      to.steps = o.steps;
      to.wall_s = o.wall_s;
      to.steps_per_s = o.steps_per_s;
      to.final_loss = o.final_loss;
      to.grad_values = o.grad_values;
      to.checkpoint_id = o.checkpoint_id;
      to.error = o.error;
      outs[i] = std::move(to);
    }
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    AggregateReport rep;
    try {
      rep = MultiModelTrainer::aggregate(outs);
    } catch (const std::exception& e) {
      rep.contributors = 0;
    }

    // A manifest report is only "ok" when every replica trained, or when at
    // least one did and the aggregate has something to work with. Reporting ok
    // for a job where every model failed would let the grid store an empty
    // aggregate as a success.
    int ok_total = 0;
    for (const auto& o : outs) {
      if (o.ok) ++ok_total;
    }
    const bool job_ok = !outs.empty() && ok_total == static_cast<int>(outs.size());
    std::string out_json = "{\"kind\":\"job_result\",\"engine\":\"libtorch\",\"status\":\"" +
                           std::string(job_ok ? "ok" : "error") + "\"," +
                           "\"job_id\":\"" + json::escape(c.job_id) + "\",\"wall_s\":" +
                           std::to_string(wall) + ",\"contributors\":" +
                           std::to_string(rep.contributors) + ",\"models\":[";
    bool first = true;
    int ok_count = 0;
    for (const auto& o : outs) {
      if (!first) out_json += ", ";
      first = false;
      out_json += "{\"name\":\"" + json::escape(o.model_name) + "\",\"status\":\"" +
                  (o.ok ? "ok" : "error") + "\"";
      if (o.ok) {
        ++ok_count;
        out_json += ",\"steps\":" + std::to_string(o.steps) +
                    ",\"steps_per_s\":" + std::to_string(o.steps_per_s) +
                    ",\"final_loss\":" + std::to_string(o.final_loss) +
                    ",\"grad_len\":" + std::to_string(o.grad_values.size()) +
                    ",\"checkpoint\":\"" + json::escape(o.checkpoint_id) + "\"";
      } else {
        out_json += ",\"error\":\"" + json::escape(o.error) + "\"";
      }
      out_json += "}";
    }
    out_json += "]";
    if (rep.contributors > 0) {
      double tmean = 0;
      for (double v : rep.trimmed_mean) tmean += v;
      out_json += ",\"aggregate\":\"" + json::escape(c.aggregate) +
                  "\",\"aggregate_grad_sum\":" + std::to_string(tmean) +
                  ",\"aggregate_grad_len\":" + std::to_string(rep.trimmed_mean.size());
    }
    out_json += ",\"models_ok\":" + std::to_string(ok_count) + "}";
    if (!envelope_out.empty()) {
      const std::vector<double>& chosen = c.aggregate == "mean"
                                             ? rep.mean
                                             : (c.aggregate == "median" ? rep.median
                                                                        : rep.trimmed_mean);
      TorchOutcome agg;
      agg.ok = rep.contributors > 0;
      double loss_sum = 0;
      double sps = 0;
      int counted = 0;
      for (const auto& o : outs) {
        if (!o.ok) continue;
        agg.steps += o.steps;
        loss_sum += o.final_loss;
        sps += o.steps_per_s;
        ++counted;
      }
      agg.final_loss = counted > 0 ? loss_sum / counted : 0;
      agg.steps_per_s = sps;
      agg.wall_s = wall;
      agg.grad_values = chosen;
      JobCfg jc = c;
      jc.name = c.job_id;
      write_outcome_envelope(envelope_out, jc, agg, "");
    }
    std::printf("%s\n", out_json.c_str());
    return job_ok ? 0 : 1;
  }

  // ---- single model ----
  if (c.sandbox) {
    TorchOutcome o;
    train_sandboxed(c, c.name, c.seed, c.steps, o);
    if (!envelope_out.empty()) write_outcome_envelope(envelope_out, c, o, "");
    if (!json) {
      std::printf("engine=libtorch name=%s ok=%d steps=%d loss=%.6f grad_len=%zu err=%s\n",
                  c.name.c_str(), o.ok ? 1 : 0, o.steps, o.final_loss, o.grad_values.size(),
                  o.error.c_str());
      return o.ok ? 0 : 1;
    }
    std::printf(
        "{\"kind\":\"torch_result\",\"engine\":\"libtorch\",\"status\":\"%s\","
        "\"job_id\":\"%s\",\"name\":\"%s\",\"seed\":%llu,\"steps\":%d,\"params\":%lld,"
        "\"wall_s\":%.6f,\"steps_per_s\":%.3f,\"final_loss\":%.9f,\"first_loss\":%.9f,"
        "\"grad_len\":%zu,\"sandbox\":true,\"device\":\"%s\",\"optimizer\":\"%s\","
        "\"loss\":\"%s\",\"error\":\"%s\"}\n",
        o.ok ? "ok" : "error", json::escape(c.job_id).c_str(), json::escape(c.name).c_str(),
        static_cast<unsigned long long>(c.seed), o.steps,
        static_cast<long long>(o.param_count), o.wall_s, o.steps_per_s, o.final_loss,
        o.first_loss, o.grad_values.size(), c.device.c_str(), c.optimizer.c_str(),
        c.loss.c_str(), json::escape(o.error).c_str());
    return o.ok ? 0 : 1;
  }

  const TorchOutcome o = run_torch(c);
  if (!envelope_out.empty()) write_outcome_envelope(envelope_out, c, o, "");
  if (!json) {
    std::printf("engine=libtorch name=%s ok=%d steps=%d loss=%.6f err=%s\n", c.name.c_str(),
                o.ok ? 1 : 0, o.steps, o.final_loss, o.error.c_str());
    return o.ok ? 0 : 1;
  }
  std::printf(
      "{\"kind\":\"torch_result\",\"engine\":\"libtorch\",\"status\":\"%s\","
      "\"job_id\":\"%s\",\"name\":\"%s\",\"seed\":%llu,\"steps\":%d,\"params\":%lld,"
      "\"wall_s\":%.6f,\"steps_per_s\":%.3f,\"final_loss\":%.9f,\"first_loss\":%.9f,"
      "\"grad_len\":%zu,\"sandbox\":false,\"device\":\"%s\",\"optimizer\":\"%s\","
      "\"loss\":\"%s\",\"error\":\"%s\"}\n",
      o.ok ? "ok" : "error", json::escape(c.job_id).c_str(), json::escape(c.name).c_str(),
      static_cast<unsigned long long>(c.seed), o.steps, static_cast<long long>(o.param_count),
      o.wall_s, o.steps_per_s, o.final_loss, o.first_loss, o.grad_values.size(),
      c.device.c_str(), c.optimizer.c_str(), c.loss.c_str(), json::escape(o.error).c_str());
  return o.ok ? 0 : 1;
}
