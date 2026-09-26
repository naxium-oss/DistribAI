// Multi-model training API (Feature 1).
//
// Register N named models (1K..100K scale), train them concurrently in
// native sandboxes (std::thread pool driving run_sandboxed children, no
// OpenMP+fork), and aggregate gradients over the versioned envelope.
// Aggregation methods: mean, median, trimmed mean (Byzantine-aware).
//
// Concurrency contract: each model trains inside its OWN sandbox child with
// its own rlimits and own-system namespaces; the pool caps how many children
// run at once (one heavy job at a time is honored by capping pool width).
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "envelope.hpp"
#include "parity.hpp"
#include "../sandbox/sandbox.hpp"

namespace distribai {

struct ModelSpec {
  std::string name;
  uint64_t seed;
  int steps;
};

struct TrainOutcome {
  std::string model_name;
  int64_t model_index = -1;
  bool ok = false;
  int steps = 0;
  double wall_s = 0, steps_per_s = 0, final_loss = 0;
  std::vector<double> grad_values;   // flattened, parameters() order
  std::string checkpoint_id;
  std::string error;
};

// Byzantine-aware aggregate over the contributing models' gradients.
struct AggregateReport {
  std::vector<double> mean;
  std::vector<double> median;
  std::vector<double> trimmed_mean;  // drops 1 high + 1 low per coordinate
  size_t contributors = 0;
};

class MultiModelTrainer {
 public:
  explicit MultiModelTrainer(SandboxLimits lim, size_t max_concurrent = 2)
      : lim_(lim), max_concurrent_(max_concurrent < 1 ? 1 : max_concurrent) {}

  void register_model(const ModelSpec& spec) { specs_.push_back(spec); }

  // Train every registered model in its own sandbox, up to max_concurrent
  // children at once. Outcomes return in registration order.
  std::vector<TrainOutcome> train_all() {
    std::vector<TrainOutcome> outs(specs_.size());
    std::atomic<size_t> next{0};
    std::vector<std::thread> pool;
    for (size_t t = 0; t < max_concurrent_; ++t) {
      pool.emplace_back([&, t] {
        for (;;) {
          const size_t i = next.fetch_add(1);
          if (i >= specs_.size()) break;
          train_one(specs_[i], lim_, static_cast<int64_t>(i), outs[i]);
        }
      });
    }
    for (auto& th : pool) th.join();
    return outs;
  }

  // Aggregate successful outcomes. Throws on mixed grad lengths.
  static AggregateReport aggregate(const std::vector<TrainOutcome>& outs) {
    AggregateReport rep;
    std::vector<const std::vector<double>*> grads;
    for (const auto& o : outs) {
      if (o.ok && !o.grad_values.empty()) {
        if (!grads.empty() && grads.back()->size() != o.grad_values.size()) {
          throw std::runtime_error("grad length mismatch across models");
        }
        grads.push_back(&o.grad_values);
      }
    }
    rep.contributors = grads.size();
    if (grads.empty()) return rep;

    const size_t n = grads[0]->size();
    rep.mean.assign(n, 0.0);
    rep.median.assign(n, 0.0);
    rep.trimmed_mean.assign(n, 0.0);

    std::vector<double> col;
    col.reserve(grads.size());
    for (size_t i = 0; i < n; ++i) {
      double sum = 0;
      col.clear();
      for (const auto* g : grads) {
        sum += (*g)[i];
        col.push_back((*g)[i]);
      }
      rep.mean[i] = sum / static_cast<double>(grads.size());
      std::sort(col.begin(), col.end());
      const size_t k = col.size();
      rep.median[i] = (k % 2 == 1) ? col[k / 2] : 0.5 * (col[k / 2 - 1] + col[k / 2]);
      if (k >= 3) {
        double ts = 0;
        for (size_t j = 1; j + 1 < k; ++j) ts += col[j];
        rep.trimmed_mean[i] = ts / static_cast<double>(k - 2);
      } else {
        rep.trimmed_mean[i] = rep.median[i];
      }
    }
    return rep;
  }

  const std::vector<ModelSpec>& specs() const { return specs_; }

 private:
  // The child writes: TrainPod (legacy framing) then a full grad_report
  // ENVELOPE whose TAG_GRAD_VALUES carries the flattened gradients. The
  // parent consumes the envelope, not the ad-hoc struct - the versioned
  // schema is the wire contract, the POD only frames the byte count.
  static void train_one(const ModelSpec& spec, SandboxLimits lim, int64_t index,
                        TrainOutcome& out) {
    out.model_name = spec.name;
    out.model_index = index;
    out.steps = spec.steps;

    SandboxLimits eff = lim;
    // Scale the memory floor with requested work; keep it modest because the
    // parity core really needs only ~7 MB RSS.
    const uint64_t floor_mb = 8 + static_cast<uint64_t>(spec.steps / 10000);
    if (eff.mem_mb < floor_mb) eff.mem_mb = floor_mb;

    TrainPod pod{};
    std::vector<uint8_t> env_bytes;
    SandboxResult res = run_sandboxed(
        eff, eff.cpu_sec + 30,
        [&](int wfd) {
          TrainPod p{};
          std::memset(&p, 0, sizeof(p));
          ParityMlp model(spec.seed);
          ParityAdamW opt(model.params(), 0.01f);
          Tensor x(64, 10), y(64, 1);
          std::vector<float> loss_vec;
          const auto t0 = std::chrono::steady_clock::now();
          for (int s = 0; s < spec.steps; ++s) {
            model.make_batch_external(x, y, model.rng());
            loss_vec.push_back(model.train_step_fixed_data(x, y, opt));
          }
          const double wall =
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          const auto grads = model.flat_grads();
          double gsum = 0;
          for (float g : grads) gsum += g;

          env::Envelope e(env::Kind::GradReport, static_cast<uint32_t>(index));
          e.add_str(env::TAG_MODEL_NAME, spec.name);
          e.add_u64(env::TAG_MODEL_INDEX, static_cast<uint64_t>(index));
          e.add_u64(env::TAG_SEED, spec.seed);
          e.add_u64(env::TAG_STEPS, static_cast<uint64_t>(spec.steps));
          e.add_u64(env::TAG_N_PARAMS, static_cast<uint64_t>(model.param_count()));
          e.add_f64(env::TAG_WALL_S, wall);
          e.add_f64(env::TAG_STEPS_PER_S, wall > 0 ? spec.steps / wall : 0);
          e.add_f64(env::TAG_MS_PER_STEP, spec.steps > 0 ? 1000.0 * wall / spec.steps : 0);
          e.add_f64(env::TAG_FINAL_LOSS, loss_vec.empty() ? 0 : loss_vec.back());
          e.add_u64(env::TAG_GRAD_LEN, grads.size());
          e.add_f64(env::TAG_GRAD_SUM, gsum);
          std::vector<double> grad_d(grads.begin(), grads.end());
          e.add_f64_array(env::TAG_GRAD_VALUES, grad_d);
          e.add_bool(env::TAG_SANDBOX, true);
          e.add_bool(env::TAG_OWN_SYSTEM, true);
          const auto bytes = e.encode();

          p.status = 0;
          p.steps = static_cast<uint64_t>(spec.steps);
          p.param_count = static_cast<uint64_t>(model.param_count());
          p.grad_len = grads.size();
          p.wall_s = wall;
          p.steps_per_s = wall > 0 ? spec.steps / wall : 0;
          p.final_loss = loss_vec.empty() ? 0 : loss_vec.back();
          p.grad_sum = gsum;
          p.grad_blob_len = bytes.size();  // envelope travels after the POD
          (void)write(wfd, &p, sizeof(p));
          (void)write(wfd, bytes.data(), bytes.size());
        },
        &pod, sizeof(pod), &env_bytes);

    if (!res.ok) {
      out.error = res.error.empty() ? "sandbox failed" : res.error;
      return;
    }

    // Consume the envelope (the POD is ignored beyond framing).
    env::Envelope e;
    std::string err;
    if (!env::Envelope::decode(env_bytes.data(), env_bytes.size(), e, err)) {
      out.error = "envelope decode failed: " + err;
      return;
    }
    out.ok = true;
    uint64_t u;
    double d;
    out.steps = e.get_u64(env::TAG_STEPS, u) ? static_cast<int>(u) : spec.steps;
    out.wall_s = e.get_f64(env::TAG_WALL_S, d) ? d : 0;
    out.steps_per_s = e.get_f64(env::TAG_STEPS_PER_S, d) ? d : 0;
    out.final_loss = e.get_f64(env::TAG_FINAL_LOSS, d) ? d : 0;
    e.get_f64_array(env::TAG_GRAD_VALUES, out.grad_values);
  }

  SandboxLimits lim_;
  size_t max_concurrent_;
  std::vector<ModelSpec> specs_;
};

}  // namespace distribai
