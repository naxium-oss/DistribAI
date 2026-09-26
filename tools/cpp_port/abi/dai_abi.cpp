// C ABI implementation over the C++ train core (see dai_abi.h).
// Every exported function is noexcept at the boundary: C++ exceptions are
// caught and mapped to dai_status so hosts never see unwinding.
#include "dai_abi.h"

#include <cstring>
#include <string>
#include <vector>

#include "../core/envelope.hpp"
#include "../core/multi_model.hpp"

using namespace distribai;

namespace {

struct Job {
  SandboxLimits lim;
  uint32_t max_concurrent = 2;
  std::vector<ModelSpec> specs;
};

struct ResultSet {
  std::vector<TrainOutcome> outs;
  AggregateReport agg;
  std::vector<std::vector<uint8_t>> envelopes;
};

bool copy_err(char* err, size_t err_len, const std::string& msg) {
  if (err && err_len > 0) {
    std::strncpy(err, msg.c_str(), err_len - 1);
    err[err_len - 1] = '\0';
  }
  return false;
}

}  // namespace

uint32_t dai_abi_version(void) { return DAI_ABI_VERSION; }

dai_job* dai_job_new(void) {
  try {
    return reinterpret_cast<dai_job*>(new Job());
  } catch (...) { return nullptr; }
}

dai_status dai_job_add_model(dai_job* j, const char* name, uint64_t seed, uint32_t steps) {
  if (!j || !name || name[0] == '\0' || steps == 0) return DAI_ERR_INVALID;
  try {
    reinterpret_cast<Job*>(j)->specs.push_back({name, seed, static_cast<int>(steps)});
    return DAI_OK;
  } catch (...) { return DAI_ERR_INVALID; }
}

dai_status dai_job_set_limits(dai_job* j, uint32_t mem_mb, uint32_t cpu_sec, uint32_t max_concurrent) {
  if (!j || mem_mb < 8 || cpu_sec == 0 || max_concurrent == 0) return DAI_ERR_INVALID;
  Job* job = reinterpret_cast<Job*>(j);
  job->lim.mem_mb = mem_mb;
  job->lim.cpu_sec = cpu_sec;
  job->max_concurrent = max_concurrent;
  return DAI_OK;
}

dai_status dai_job_run(dai_job* j, dai_result** out, char* err, size_t err_len) {
  if (!j || !out) return DAI_ERR_INVALID;
  Job* job = reinterpret_cast<Job*>(j);
  try {
    MultiModelTrainer trainer(job->lim, job->max_concurrent);
    for (const auto& s : job->specs) trainer.register_model(s);
    auto* rs = new ResultSet();
    rs->outs = trainer.train_all();
    try {
      rs->agg = MultiModelTrainer::aggregate(rs->outs);
    } catch (const std::exception& e) {
      copy_err(err, err_len, e.what());
      delete rs;
      return DAI_ERR_INVALID;
    }
    *out = reinterpret_cast<dai_result*>(rs);
    const bool all_ok = std::all_of(rs->outs.begin(), rs->outs.end(),
                                    [](const TrainOutcome& o) { return o.ok; });
    return all_ok ? DAI_OK : DAI_ERR_SANDBOX;
  } catch (const std::exception& e) {
    copy_err(err, err_len, e.what());
    return DAI_ERR_INVALID;
  } catch (...) {
    copy_err(err, err_len, "unknown error in dai_job_run");
    return DAI_ERR_INVALID;
  }
}

void dai_result_free(dai_result* r) { delete reinterpret_cast<ResultSet*>(r); }
void dai_job_free(dai_job* j) { delete reinterpret_cast<Job*>(j); }

uint32_t dai_result_count(const dai_result* r) {
  return r ? static_cast<uint32_t>(reinterpret_cast<const ResultSet*>(r)->outs.size()) : 0;
}

int dai_result_ok(const dai_result* r, uint32_t i) {
  const auto* rs = reinterpret_cast<const ResultSet*>(r);
  if (!rs || i >= rs->outs.size()) return 0;
  return rs->outs[i].ok ? 1 : 0;
}

const char* dai_result_model_name(const dai_result* r, uint32_t i) {
  const auto* rs = reinterpret_cast<const ResultSet*>(r);
  if (!rs || i >= rs->outs.size()) return nullptr;
  return rs->outs[i].model_name.c_str();
}

double dai_result_final_loss(const dai_result* r, uint32_t i) {
  const auto* rs = reinterpret_cast<const ResultSet*>(r);
  if (!rs || i >= rs->outs.size() || !rs->outs[i].ok) return 0.0;
  return rs->outs[i].final_loss;
}

double dai_result_steps_per_s(const dai_result* r, uint32_t i) {
  const auto* rs = reinterpret_cast<const ResultSet*>(r);
  if (!rs || i >= rs->outs.size() || !rs->outs[i].ok) return 0.0;
  return rs->outs[i].steps_per_s;
}

const double* dai_result_aggregate(const dai_result* r, const char* method, size_t* out_len) {
  const auto* rs = reinterpret_cast<const ResultSet*>(r);
  if (!rs || !method || !out_len) return nullptr;
  const std::string m = method;
  const std::vector<double>* v = nullptr;
  if (m == "mean") v = &rs->agg.mean;
  else if (m == "median") v = &rs->agg.median;
  else if (m == "trimmed_mean") v = &rs->agg.trimmed_mean;
  if (!v || v->empty()) return nullptr;
  *out_len = v->size();
  return v->data();
}

const uint8_t* dai_result_envelope(const dai_result* r, uint32_t i, size_t* out_len) {
  // Envelope bytes are regenerated on demand from the outcome (cheap, keeps
  // the result set small and the ABI stable).
  const auto* rs = reinterpret_cast<const ResultSet*>(r);
  if (!rs || !out_len || i >= rs->outs.size() || !rs->outs[i].ok) return nullptr;
  const auto& o = rs->outs[i];
  auto& slot = const_cast<std::vector<std::vector<uint8_t>>&>(rs->envelopes);
  if (slot.size() <= i) slot.resize(i + 1);
  if (slot[i].empty()) {
    env::Envelope e(env::Kind::GradReport, i);
    e.add_str(env::TAG_MODEL_NAME, o.model_name);
    e.add_u64(env::TAG_MODEL_INDEX, static_cast<uint64_t>(i));
    e.add_f64(env::TAG_FINAL_LOSS, o.final_loss);
    e.add_f64(env::TAG_STEPS_PER_S, o.steps_per_s);
    e.add_u64(env::TAG_GRAD_LEN, o.grad_values.size());
    e.add_f64_array(env::TAG_GRAD_VALUES, o.grad_values);
    slot[i] = e.encode();
  }
  *out_len = slot[i].size();
  return slot[i].data();
}

dai_status dai_envelope_verify(const uint8_t* data, size_t n) {
  if (!data || n == 0) return DAI_ERR_INVALID;
  env::Envelope e;
  std::string err;
  try {
    if (!env::Envelope::decode(data, n, e, err)) return DAI_ERR_PARSE;
  } catch (...) { return DAI_ERR_PARSE; }
  return DAI_OK;
}

dai_status dai_envelope_get_loss(const uint8_t* data, size_t n, double* out, int* ok) {
  if (!data || !out || !ok) return DAI_ERR_INVALID;
  *ok = 0;
  env::Envelope e;
  std::string err;
  try {
    if (!env::Envelope::decode(data, n, e, err)) return DAI_ERR_PARSE;
  } catch (...) { return DAI_ERR_PARSE; }
  double d;
  if (e.get_f64(env::TAG_FINAL_LOSS, d)) { *out = d; *ok = 1; }
  return DAI_OK;
}
