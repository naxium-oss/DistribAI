// grid_fake_trainer - a test fixture, not a trainer.
//
// The grid gate needs a worker to run something that behaves like the LibTorch
// trainer without pulling torch into every CI job: read a job spec, "train",
// print the same JSON line, and write a real GradReport envelope. Gradients are
// deterministic functions of the seed, so the gate can also verify that two
// replicas with different seeds produce different gradients and that the
// aggregate lands between them.
//
// Never wire this into the grid's production paths. build/cpp_port/grid_fake_trainer
// exists only for tests/grid_e2e.sh and tools/cpp_port/Makefile's grid-test target.
//
//   grid_fake_trainer --spec job.json --json --envelope-out result.env
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../core/envelope.hpp"
#include "../torch/json_lite.hpp"

int main(int argc, char** argv) {
  std::string spec_path;
  std::string envelope_out;
  bool json = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
    if (a == "--spec") spec_path = next();
    else if (a == "--envelope-out") envelope_out = next();
    else if (a == "--json") json = true;
  }
  if (spec_path.empty()) {
    std::fprintf(stderr, "grid_fake_trainer: --spec is required\n");
    return 2;
  }
  FILE* f = std::fopen(spec_path.c_str(), "rb");
  if (!f) {
    std::printf("{\"status\":\"error\",\"error\":\"spec not found\"}\n");
    return 2;
  }
  std::string text;
  char buf[4096];
  size_t r;
  while ((r = std::fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, r);
  std::fclose(f);

  distribai::json::Value spec;
  std::string err;
  if (!distribai::json::parse(text, spec, err)) {
    std::printf("{\"status\":\"error\",\"error\":\"spec parse: %s\"}\n", err.c_str());
    return 2;
  }
  const int steps = static_cast<int>(spec.num("steps", 10));
  const uint64_t seed = static_cast<uint64_t>(spec.num("seed", 42));
  const std::string name = spec.str("name", "model");
  const std::string job_id = spec.str("job_id", "job");
  const int grad_len = 16;

  std::vector<double> grads(grad_len);
  double gsum = 0;
  for (int i = 0; i < grad_len; ++i) {
    grads[i] = std::sin(static_cast<double>(seed) * 0.5 + i) / (1.0 + steps);
    gsum += grads[i];
  }
  const double final_loss = 1.0 / (1.0 + steps) + 0.001 * static_cast<double>(seed % 7);
  const double wall_s = 0.01 * steps;
  const double steps_per_s = wall_s > 0 ? steps / wall_s : 0;

  if (!envelope_out.empty()) {
    distribai::env::Envelope e(distribai::env::Kind::GradReport, 0);
    e.add_str(distribai::env::TAG_MODEL_NAME, name);
    e.add_u64(distribai::env::TAG_SEED, seed);
    e.add_u64(distribai::env::TAG_STEPS, static_cast<uint64_t>(steps));
    e.add_u64(distribai::env::TAG_N_PARAMS, static_cast<uint64_t>(grad_len));
    e.add_f64(distribai::env::TAG_WALL_S, wall_s);
    e.add_f64(distribai::env::TAG_STEPS_PER_S, steps_per_s);
    e.add_f64(distribai::env::TAG_FINAL_LOSS, final_loss);
    e.add_u64(distribai::env::TAG_GRAD_LEN, static_cast<uint64_t>(grads.size()));
    e.add_f64(distribai::env::TAG_GRAD_SUM, gsum);
    e.add_f64_array(distribai::env::TAG_GRAD_VALUES, grads);
    e.add_bool(distribai::env::TAG_SANDBOX, true);
    const auto bytes = e.encode();
    FILE* out = std::fopen(envelope_out.c_str(), "wb");
    if (!out) {
      std::printf("{\"status\":\"error\",\"error\":\"cannot write the envelope\"}\n");
      return 1;
    }
    std::fwrite(bytes.data(), 1, bytes.size(), out);
    std::fclose(out);
  }

  if (json) {
    std::printf("{\"kind\":\"torch_result\",\"engine\":\"fake\",\"status\":\"ok\","
                "\"job_id\":\"%s\",\"name\":\"%s\",\"seed\":%llu,\"steps\":%d,"
                "\"params\":%d,\"wall_s\":%.6f,\"steps_per_s\":%.3f,\"final_loss\":%.9f,"
                "\"grad_len\":%d,\"grad_sum\":%.9f,\"sandbox\":true}\n",
                job_id.c_str(), name.c_str(), static_cast<unsigned long long>(seed), steps,
                grad_len, wall_s, steps_per_s, final_loss, grad_len, gsum);
  }
  return 0;
}
