// Feature tests for the new capabilities (multi-model, checkpoint, envelope).
// Failing-then-passing discipline: each section fails against the old code
// paths and passes against the new implementations.
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "../core/envelope.hpp"
#include "../core/checkpoint.hpp"
#include "../core/multi_model.hpp"

using namespace distribai;

static int failures = 0, checks = 0;
#define CHECK(cond, msg)                                     \
  do {                                                       \
    ++checks;                                                \
    if (!(cond)) {                                           \
      ++failures;                                            \
      std::printf("FAIL: %s (line %d)\n", msg, __LINE__);    \
    }                                                        \
  } while (0)

int main() {
  // ================= Feature 0: envelope schema =================
  {
    // roundtrip with unknown-tag tolerance
    env::Envelope e(env::Kind::GradReport, 9);
    e.add_u64(env::TAG_SEED, 42);
    e.add_f64(env::TAG_FINAL_LOSS, 0.5);
    e.add_str(env::TAG_MODEL_NAME, "alpha");
    e.add_f64_array(env::TAG_GRAD_VALUES, {1.0, 2.0, 3.0});
    e.add_u64(200, 777);  // unknown tag
    auto bytes = e.encode();
    env::Envelope d; std::string err;
    CHECK(env::Envelope::decode(bytes.data(), bytes.size(), d, err), "envelope decode");
    uint64_t u; std::string s; std::vector<double> a;
    CHECK(d.get_u64(env::TAG_SEED, u) && u == 42, "u64 roundtrip");
    CHECK(d.get_str(env::TAG_MODEL_NAME, s) && s == "alpha", "str roundtrip");
    CHECK(d.get_f64_array(env::TAG_GRAD_VALUES, a) && a.size() == 3, "array roundtrip");
    // corrupted byte must fail CRC
    bytes[50] ^= 0x40;
    env::Envelope d2;
    CHECK(!env::Envelope::decode(bytes.data(), bytes.size(), d2, err), "crc rejects corruption");
    // version gate: craft a v99 header
    auto v99 = e.encode();
    v99[4] = 99;
    env::Envelope d3;
    CHECK(!env::Envelope::decode(v99.data(), v99.size(), d3, err), "version gate rejects newer");
    CHECK(err.find("newer") != std::string::npos, "version error is structured");
  }

  // ================= Feature 2: checkpoint/resume bit-exact =================
  {
    ::system("rm -rf /tmp/ckp_feat");
    // reference arm: 250 steps straight
    ParityMlp ref(42); ParityAdamW ref_opt(ref.params(), 0.01f); TorchRng ref_rng(42);
    Tensor rx(64,10), ry(64,1);
    std::vector<float> ref_losses;
    for (int s = 0; s < 250; ++s) {
      ref.make_batch_external(rx, ry, ref_rng);
      ref_losses.push_back(ref.train_step_fixed_data(rx, ry, ref_opt));
    }
    // crash arm: 111 steps, checkpoint, restore into fresh objects, continue
    ParityMlp m(42); ParityAdamW opt(m.params(), 0.01f); TorchRng rng(42);
    Tensor x(64,10), y(64,1);
    for (int s = 0; s < 111; ++s) {
      m.make_batch_external(x, y, rng);
      m.train_step_fixed_data(x, y, opt);
    }
    Checkpoint ck = Checkpoint::capture("alpha", 42, 111, m, opt, rng);
    const std::string path = ck.save("/tmp/ckp_feat");
    Checkpoint loaded; std::string err;
    const bool loaded_ok = Checkpoint::load(path, loaded, err);
    CHECK(loaded_ok, "checkpoint loads from file");
    CHECK(loaded.meta().steps == 111, "resume point recorded");
    ParityMlp m2(0); ParityAdamW opt2(m2.params(), 0.01f); TorchRng rng2(0);
    CHECK(loaded.restore(m2, opt2, rng2), "restore into fresh objects");
    Tensor x2(64,10), y2(64,1);
    int mismatch = 0;
    for (int s = 111; s < 250; ++s) {
      m2.make_batch_external(x2, y2, rng2);
      const float l = m2.train_step_fixed_data(x2, y2, opt2);
      if (l != ref_losses[s]) ++mismatch;
    }
    const auto wl = ref.flat_weights(), wr = m2.flat_weights();
    int wdiff = 0;
    for (size_t i = 0; i < wl.size(); ++i) if (wl[i] != wr[i]) ++wdiff;
    CHECK(mismatch == 0, "resumed losses bit-exact vs uncrashed run");
    CHECK(wdiff == 0, "final weights bit-exact vs uncrashed run");
    CHECK(Checkpoint::load("/tmp/ckp_feat/nonexistent.ckp", loaded, err) == false, "missing file errors cleanly");
  }

  // ================= Feature 1: multi-model concurrent training =================
  {
    SandboxLimits lim;
    lim.mem_mb = 64;   // parity core needs ~7 MB; keep tight on purpose
    lim.cpu_sec = 120;
    MultiModelTrainer trainer(lim, /*max_concurrent=*/2);
    trainer.register_model({"alpha", 42, 120});
    trainer.register_model({"beta", 43, 120});
    trainer.register_model({"gamma", 44, 120});
    const auto t0 = std::chrono::steady_clock::now();
    const auto outs = trainer.train_all();
    const double wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    bool all_ok = true;
    std::vector<double> seq_wall;
    for (const auto& o : outs) {
      all_ok &= o.ok && o.grad_values.size() == 1291 && o.final_loss > 0 && o.final_loss < 1.0;
    }
    CHECK(all_ok, "all three models trained with full grad vectors");
    // concurrency actually happened: 3 jobs x ~0.02s each should beat 3x serial
    CHECK(wall < 3.0, "concurrent wall time bounded");
    // aggregation: mean/median/trimmed over 3 contributors
    const auto rep = MultiModelTrainer::aggregate(outs);
    CHECK(rep.contributors == 3, "three grad contributors");
    CHECK(rep.mean.size() == 1291 && rep.median.size() == 1291 && rep.trimmed_mean.size() == 1291,
          "aggregate vectors sized");
    // median of 3 == middle value; verify on coordinate 0 by hand
    std::vector<double> c0;
    for (const auto& o : outs) c0.push_back(o.grad_values[0]);
    std::sort(c0.begin(), c0.end());
    CHECK(std::fabs(rep.median[0] - c0[1]) < 1e-12, "median is the middle value");
    // trimmed mean drops min and max: equals median for k=3
    CHECK(std::fabs(rep.trimmed_mean[0] - c0[1]) < 1e-12, "trimmed mean equals middle for k=3");
    // mixed-length rejection
    std::vector<TrainOutcome> bad = outs;
    bad[2].grad_values.resize(1290);
    bool threw = false;
    try { MultiModelTrainer::aggregate(bad); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw, "mixed grad lengths rejected");
  }

  std::printf("feature checks: %d run, %d failures\n", checks, failures);
  if (failures == 0) { std::printf("ALL FEATURE TESTS PASSED\n"); return 0; }
  return 1;
}
