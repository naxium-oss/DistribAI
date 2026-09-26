// distribai_orch - the native grid's coordinator.
//
// One process, one HTTP port serving three surfaces:
//   * the worker API    /v1/register, /v1/heartbeat, /v1/claim, /v1/result, /v1/bye
//   * the operator API  /v1/jobs (submit), /v1/jobs/<id>/cancel, /v1/health
//   * the read API and dashboard  /v1/summary, /v1/nodes, /v1/tasks, /
//
// Work arrives as a job directory, meaning the job.json, TorchScript module and
// data files that tools/trainer_translate produces. The orchestrator snapshots
// that directory into its own jobs directory, then hands one replica to each
// worker that asks for work. Every worker trains its replica and reports a
// GradReport envelope. Once all replicas have reported, the gradients are
// aggregated (mean, median or trimmed mean), the result is written next to the
// bundle, and credits go into the SQLite ledger.
//
// Durability: nodes, jobs, tasks and credits live in SQLite using
// runtime/db/schema.sql. Bundles, per-task envelopes and results live under the
// grid directory on disk, so a restart picks the queue back up.
//
//   distribai_orch [--host 127.0.0.1] [--port 50061] [--db runtime/db/grid.db]
//                  [--schema runtime/db/schema.sql] [--jobs-dir runtime/grid/jobs]
//                  [--tasks-dir runtime/grid/tasks] [--web-dir tools/cpp_port/grid/web]
//                  [--invite CODE] [--token TOKEN] [--node-ttl 60]
//                  [--aggregate trimmed_mean] [--submit JOB_DIR] [--replicas N]
#include <dirent.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "../core/envelope.hpp"
#include "../core/multi_model.hpp"
#include "http.hpp"
#include "store.hpp"

namespace {

using distribai::AggregateReport;
using distribai::MultiModelTrainer;
using distribai::TrainOutcome;
using distribai::grid::CreditRow;
using distribai::grid::JobRow;
using distribai::grid::NodeRow;
using distribai::grid::Store;
using distribai::grid::TaskCounts;
using distribai::grid::TaskRow;
using distribai::http::Request;
using distribai::http::Response;

namespace env = distribai::env;
namespace grid = distribai::grid;
namespace json = distribai::json;

std::atomic<bool> g_stop{false};

void on_signal(int) { g_stop.store(true); }

std::string dirname_of(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
}

// Creates every missing parent in one pass. Grid directories are few and
// shallow, so a loop over the separators is enough.
bool make_dirs(const std::string& path) {
  std::string acc;
  for (size_t i = 0; i < path.size(); ++i) {
    acc += path[i];
    if (path[i] == '/') {
      if (acc.size() > 1) ::mkdir(acc.c_str(), 0755);
    }
  }
  ::mkdir(path.c_str(), 0755);
  struct stat st{};
  return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool is_regular_file(const std::string& path) {
  struct stat st{};
  return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::vector<std::string> list_files(const std::string& dir) {
  std::vector<std::string> out;
  DIR* d = ::opendir(dir.c_str());
  if (!d) return out;
  while (dirent* e = ::readdir(d)) {
    const std::string name = e->d_name;
    if (name == "." || name == "..") continue;
    if (is_regular_file(dir + "/" + name)) out.push_back(name);
  }
  ::closedir(d);
  std::sort(out.begin(), out.end());
  return out;
}

struct Config {
  std::string host = "127.0.0.1";
  int port = 50061;
  std::string db = "runtime/db/grid.db";
  std::string schema = "runtime/db/schema.sql";
  std::string jobs_dir = "runtime/grid/jobs";
  std::string tasks_dir = "runtime/grid/tasks";
  std::string web_dir;
  std::string invite;
  std::string token;
  int64_t node_ttl = 60;
  std::string aggregate = "trimmed_mean";
  double credit_divisor = 1000.0;  // training steps per credit
  bool print_join = false;
  std::string submit_dir;
  int replicas = 0;
  std::string description;
};

// Encodes a parsed JSON value back to text. Job specs hold scalars, arrays and
// objects, so this covers every case the rewriter meets.
std::string encode_value(const json::Value& v) {
  switch (v.type()) {
    case json::Type::Null: return "null";
    case json::Type::Bool: return v.as_bool() ? "true" : "false";
    case json::Type::Number: {
      const double d = v.as_number();
      if (d == static_cast<double>(static_cast<int64_t>(d))) {
        return std::to_string(static_cast<int64_t>(d));
      }
      return std::to_string(d);
    }
    case json::Type::String: return "\"" + json::escape(v.as_string()) + "\"";
    case json::Type::Array: {
      std::string out = "[";
      const auto& arr = v.as_array();
      for (size_t i = 0; i < arr.size(); ++i) {
        if (i) out += ",";
        out += encode_value(arr[i]);
      }
      return out + "]";
    }
    case json::Type::Object: {
      std::string out = "{";
      bool first = true;
      for (const auto& kv : v.as_object()) {
        if (!first) out += ",";
        first = false;
        out += "\"" + json::escape(kv.first) + "\":" + encode_value(kv.second);
      }
      return out + "}";
    }
  }
  return "null";
}

// The dashboard assets usually live in tools/cpp_port/grid/web while the binary
// lands in build/cpp_port, so look in both places before giving up.
std::string resolve_web_dir(const std::string& configured) {
  std::vector<std::string> candidates;
  if (!configured.empty()) candidates.push_back(configured);
  candidates.push_back("tools/cpp_port/grid/web");
  candidates.push_back("grid/web");

  char exe[4096] = {0};
  const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  if (n > 0) {
    const std::string dir = dirname_of(std::string(exe, static_cast<size_t>(n)));
    candidates.push_back(dir + "/web");
    candidates.push_back(dir + "/../../tools/cpp_port/grid/web");
  }
  for (const auto& c : candidates) {
    if (is_regular_file(c + "/index.html")) return c;
  }
  return configured;
}

struct Bundle {
  std::string spec_json;
  std::string model_name = "model";
  int64_t steps = 100;
  int64_t batch_size = 32;
  bool has_manifest = false;
  std::vector<std::string> files;
};

bool read_bundle(const std::string& dir, Bundle& out, std::string& err) {
  if (!distribai::http::read_file(dir + "/job.json", out.spec_json)) {
    err = "job.json not found in " + dir + " (run tools/trainer_translate first)";
    return false;
  }
  json::Value spec;
  if (!json::parse(out.spec_json, spec, err) || !spec.is_object()) {
    err = "job.json is not a JSON object: " + err;
    return false;
  }
  out.model_name = spec.str("name", "model");
  out.steps = grid::i64_field(spec, "steps", 100);
  out.batch_size = grid::i64_field(spec, "batch_size", 32);
  const json::Value* models = spec.get("models");
  out.has_manifest = models && models->is_array() && !models->as_array().empty();
  out.files = list_files(dir);
  if (out.steps < 1) {
    err = "job.json needs a positive \"steps\"";
    return false;
  }
  return true;
}

// Rewrites a spec for one replica: a distinct seed, the tracked job id, and the
// task's step count. A manifest job already describes its own replicas, so it is
// passed through untouched.
std::string task_spec(const std::string& spec_json, const std::string& job_id, int64_t replica,
                      int64_t steps, bool has_manifest) {
  if (has_manifest) return spec_json;
  json::Value spec;
  std::string err;
  if (!json::parse(spec_json, spec, err) || !spec.is_object()) return spec_json;
  const double base_seed = spec.num("seed", 42);
  std::string out = "{";
  bool first = true;
  for (const auto& kv : spec.as_object()) {
    if (!first) out += ",";
    first = false;
    out += "\"" + json::escape(kv.first) + "\":";
    if (kv.first == "seed") {
      out += std::to_string(static_cast<int64_t>(base_seed) + replica);
    } else if (kv.first == "steps") {
      out += std::to_string(steps);
    } else if (kv.first == "job_id") {
      out += "\"" + json::escape(job_id) + "\"";
    } else {
      out += encode_value(kv.second);
    }
  }
  return out + "}";
}

// ---------------------------------------------------------------------------
// Grid: submission, reaping, aggregation
// ---------------------------------------------------------------------------

class Grid {
 public:
  Grid(Config cfg, Store* store) : cfg_(std::move(cfg)), store_(store) {}

  bool submit(const std::string& job_dir, int replicas, const std::string& description,
              std::string& job_id_out, std::string& err) {
    Bundle bundle;
    if (!read_bundle(job_dir, bundle, err)) return false;
    const int reps = bundle.has_manifest ? 1 : (replicas > 0 ? replicas : 2);

    JobRow job;
    job.job_id = "job-" + std::to_string(grid::now_s()) + "-" + grid::random_token(3);
    job.model_name = bundle.model_name;
    job.dataset_ref = cfg_.jobs_dir + "/" + job.job_id;
    job.description = description;
    job.status = "queued";
    job.steps = bundle.steps;
    job.batch_size = bundle.batch_size;
    job.total_steps = bundle.steps * reps;
    job.replicas = reps;
    // The job's own spec wins when it names an aggregate, because that value is
    // part of the job (the translator writes it). Otherwise the coordinator's
    // --aggregate decides.
    job.aggregate = cfg_.aggregate;
    {
      json::Value spec;
      std::string perr;
      if (!bundle.spec_json.empty() && json::parse(bundle.spec_json, spec, perr) &&
          spec.is_object()) {
        const std::string named = spec.str("aggregate");
        if (named == "mean" || named == "median" || named == "trimmed_mean") {
          job.aggregate = named;
        }
      }
    }

    const std::string snapshot = job.dataset_ref + "/bundle";
    if (!make_dirs(snapshot)) {
      err = "cannot create " + snapshot;
      return false;
    }
    size_t copied = 0;
    for (const auto& name : bundle.files) {
      std::string data;
      if (!distribai::http::read_file(job_dir + "/" + name, data)) {
        err = "cannot read " + job_dir + "/" + name;
        return false;
      }
      copied += data.size();
      if (copied > grid::kMaxBundleBytes) {
        err = "job directory exceeds the grid's bundle cap";
        return false;
      }
      if (!distribai::http::write_file(snapshot + "/" + name, data)) {
        err = "cannot write " + snapshot + "/" + name;
        return false;
      }
    }
    distribai::http::write_file(job.dataset_ref + "/submitted.json",
                                grid::Jw().s("source", job_dir).s("spec", bundle.spec_json).str());

    std::vector<TaskRow> tasks;
    for (int i = 0; i < reps; ++i) {
      TaskRow t;
      t.task_id = job.job_id + "-r" + std::to_string(i);
      t.job_id = job.job_id;
      t.status = "queued";
      t.steps = bundle.steps;
      t.step_offset = 0;
      t.max_attempts = 3;
      t.hparams_json = grid::Jw()
                           .n("replica", i)
                           .n("replicas", reps)
                           .s("aggregate", job.aggregate)
                           .b("manifest", bundle.has_manifest)
                           .str();
      tasks.push_back(std::move(t));
    }
    if (!store_->create_job(job, tasks, err)) return false;
    job_id_out = job.job_id;
    return true;
  }

  // Runs on the maintenance thread. Offline nodes lose their tasks, and a job
  // whose replicas have all reported gets aggregated here. A finished job is
  // normally closed by the request that reported its last replica, through
  // maybe_finalize(); this sweep is the safety net for jobs nobody is waiting
  // on any more (every worker gone, results lost).
  void reap() {
    std::string err;
    int64_t requeued = 0;
    int64_t failed = 0;
    store_->expire_nodes(cfg_.node_ttl, requeued, failed, err);

    std::vector<JobRow> jobs;
    if (!store_->jobs(jobs, 200, err)) return;
    for (const auto& job : jobs) {
      // A job that has been sitting in 'finalizing' means whoever claimed it
      // did not finish. Put it back and let this sweep close it.
      if (job.status == "finalizing") {
        if (grid::now_s() - job.updated_ts > 30) {
          JobRow retry = job;
          retry.status = "running";
          store_->update_job(retry, err);
        }
        continue;
      }
      maybe_finalize(job.job_id);
    }
  }

  // Closes a job once every replica has reported, or marks it failed when all
  // replica attempts did. Called right after a result arrives so the dashboard
  // shows the aggregate immediately instead of on the next sweep.
  void maybe_finalize(const std::string& job_id) {
    std::string err;
    JobRow job;
    bool found = false;
    if (!store_->job(job_id, job, found, err) || !found) return;
    if (job.status != "running" && job.status != "queued") return;

    TaskCounts counts;
    if (!store_->task_counts(job_id, counts, err)) return;
    if (counts.assigned > 0 || counts.queued > 0) return;

    // Two threads can reach this point for the same job: whoever flips the
    // status owns the aggregate and the credit payment.
    bool claimed = false;
    if (!store_->claim_finalization(job_id, claimed, err) || !claimed) return;

    if (counts.done > 0) {
      finalize(job, counts);
    } else {
      update_status(job, "failed", 0, counts.total,
                    "all " + std::to_string(counts.failed) + " replicas failed");
    }
  }

  void finalize(const JobRow& job, const TaskCounts& counts) {
    std::string err;
    std::vector<TaskRow> tasks;
    if (!store_->tasks_for_job(job.job_id, tasks, err)) return;
    Bundle bundle;
    read_bundle(job.dataset_ref + "/bundle", bundle, err);

    std::vector<TrainOutcome> outs;
    std::vector<std::string> contributors;
    for (const auto& t : tasks) {
      if (t.status != "done") continue;
      TrainOutcome o;
      o.model_name = bundle.model_name.empty() ? job.model_name : bundle.model_name;
      o.model_index = static_cast<int64_t>(outs.size());
      o.steps = static_cast<int>(t.steps);
      std::string raw;
      if (!t.gradient_blob_url.empty() &&
          distribai::http::read_file(t.gradient_blob_url, raw)) {
        env::Envelope e;
        std::string derr;
        if (env::Envelope::decode(reinterpret_cast<const uint8_t*>(raw.data()), raw.size(), e,
                                  derr)) {
          e.get_f64_array(env::TAG_GRAD_VALUES, o.grad_values);
          double d = 0;
          if (e.get_f64(env::TAG_FINAL_LOSS, d)) o.final_loss = d;
          if (e.get_f64(env::TAG_WALL_S, d)) o.wall_s = d;
          if (e.get_f64(env::TAG_STEPS_PER_S, d)) o.steps_per_s = d;
          uint64_t u = 0;
          if (e.get_u64(env::TAG_STEPS, u)) o.steps = static_cast<int>(u);
          std::string ck;
          if (e.get_str(env::TAG_CHECKPOINT_ID, ck)) o.checkpoint_id = ck;
          o.ok = !o.grad_values.empty();
        }
      }
      if (!o.ok) {
        o.error = "no readable envelope from " + t.task_id;
      } else if (!t.assignee_node_id.empty()) {
        contributors.push_back(t.assignee_node_id);
      }
      outs.push_back(std::move(o));
    }

    AggregateReport rep;
    try {
      rep = MultiModelTrainer::aggregate(outs);
    } catch (const std::exception&) {
      rep.contributors = 0;
    }

    std::vector<double> chosen;
    if (job.aggregate == "mean") {
      chosen = rep.mean;
    } else if (job.aggregate == "median") {
      chosen = rep.median;
    } else {
      chosen = rep.trimmed_mean;
    }
    const double agg_sum = sum_of(chosen);

    if (make_dirs(job.dataset_ref)) {
      env::Envelope agg(env::Kind::GradReport, 0);
      agg.add_str(env::TAG_MODEL_NAME, job.model_name);
      agg.add_u64(env::TAG_STEPS, static_cast<uint64_t>(counts.total));
      agg.add_u64(env::TAG_N_PARAMS, static_cast<uint64_t>(rep.contributors));
      agg.add_u64(env::TAG_GRAD_LEN, chosen.size());
      agg.add_f64(env::TAG_GRAD_SUM, agg_sum);
      agg.add_f64_array(env::TAG_GRAD_VALUES, chosen);
      const auto bytes = agg.encode();
      const std::string blob(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      distribai::http::write_file(job.dataset_ref + "/aggregate.env", blob);

      const std::string result =
          grid::Jw()
              .s("job_id", job.job_id)
              .s("model", job.model_name)
              .s("status", "completed")
              .s("aggregate", job.aggregate)
              .n("contributors", static_cast<double>(rep.contributors))
              .n("replicas", static_cast<double>(counts.total))
              .n("grad_len", static_cast<double>(chosen.size()))
              .n("aggregate_grad_sum", agg_sum)
              .n("mean_sum", sum_of(rep.mean))
              .n("median_sum", sum_of(rep.median))
              .n("trimmed_mean_sum", sum_of(rep.trimmed_mean))
              .raw("replicas_detail", replicas_json(tasks))
              .str();
      distribai::http::write_file(job.dataset_ref + "/result.json", result);
    }

    // Credits reward completed work. A worker that trained a replica is paid
    // even when a different replica crashed.
    for (const auto& node : contributors) {
      store_->add_credit(node, "job_reward",
                         static_cast<double>(job.steps) / cfg_.credit_divisor, err);
    }

    update_status(job, "completed", counts.done, counts.total,
                  "aggregated " + std::to_string(rep.contributors) + " of " +
                      std::to_string(counts.total) + " replicas with " + job.aggregate);
  }

  void update_status(const JobRow& job, const std::string& status, int64_t done, int64_t total,
                     const std::string& reason) {
    std::string err;
    JobRow updated = job;
    updated.status = status;
    updated.latest_reason = reason;
    updated.completed_ts = grid::now_s();
    updated.updated_ts = updated.completed_ts;
    updated.progress_pct =
        total > 0 ? 100.0 * static_cast<double>(done) / static_cast<double>(total) : 0.0;
    store_->update_job(updated, err);
  }

  static double sum_of(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return s;
  }

  static std::string replicas_json(const std::vector<TaskRow>& tasks) {
    std::string out = "[";
    bool first = true;
    for (const auto& t : tasks) {
      if (!first) out += ",";
      first = false;
      std::string metrics = t.output_json;
      json::Value parsed;
      std::string perr;
      if (metrics.empty() || !json::parse(metrics, parsed, perr) || !parsed.is_object()) {
        metrics = "{}";
      }
      out += grid::Jw()
                 .s("task_id", t.task_id)
                 .s("node", t.assignee_node_id)
                 .s("status", t.status)
                 .n("steps", static_cast<double>(t.steps))
                 .n("attempts", static_cast<double>(t.attempt_count))
                 .s("error", t.last_error)
                 .raw("metrics", metrics)
                 .str();
    }
    return out + "]";
  }

 private:
  Config cfg_;
  Store* store_;
};

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

struct Ctx {
  Grid* grid = nullptr;
  Store* store = nullptr;
  Config cfg;
  std::chrono::steady_clock::time_point started;
};

// Operator writes need the configured token. With no token set, only loopback
// callers may submit or cancel, so a careless public bind cannot accept jobs.
bool admin_ok(const Ctx& ctx, const Request& req) {
  if (!ctx.cfg.token.empty()) return req.header("x-grid-token") == ctx.cfg.token;
  return req.is_local();
}

Response handle_register(Ctx& ctx, const Request& req) {
  json::Value body;
  std::string err;
  if (!grid::parse_body(req.body, body, err)) return Response::error(400, err);
  const int proto = static_cast<int>(grid::num_field(body, "proto", 0));
  if (proto != grid::kProto) {
    return Response::error(400, "protocol mismatch: this grid speaks proto " +
                                    std::to_string(grid::kProto));
  }
  NodeRow node;
  node.node_id = grid::str_field(body, "node_id");
  if (node.node_id.empty()) return Response::error(400, "node_id is required");
  if (node.node_id.size() > 128) return Response::error(400, "node_id is too long");
  node.status = "idle";
  if (const json::Value* hw = body.get("hardware")) node.hardware_json = encode_value(*hw);
  if (const json::Value* bm = body.get("benchmark")) node.benchmark_json = encode_value(*bm);

  const std::string invite = grid::str_field(body, "invite");
  if (!ctx.cfg.invite.empty() && invite != ctx.cfg.invite) {
    return Response::error(403, "invite code required");
  }
  const bool created = ctx.store->upsert_node(node, err);
  if (!err.empty()) return Response::error(500, err);

  return Response::json(grid::Jw()
                            .s("type", grid::kTWelcome)
                            .n("proto", grid::kProto)
                            .s("node_id", node.node_id)
                            .s("session_token", node.session_token)
                            .b("new_node", created)
                            .n("heartbeat_s", 10)
                            .n("poll_s", 2)
                            .s("aggregate", ctx.cfg.aggregate)
                            .n("server_time", static_cast<double>(grid::now_s()))
                            .str());
}

Response handle_heartbeat(Ctx& ctx, const Request& req) {
  json::Value body;
  std::string err;
  if (!grid::parse_body(req.body, body, err)) return Response::error(400, err);
  const std::string node_id = grid::str_field(body, "node_id");
  if (!ctx.store->session_valid(node_id, grid::str_field(body, "token"), err)) {
    return Response::error(401, "unknown session; register again");
  }
  if (!ctx.store->heartbeat(node_id, grid::str_field(body, "status", "idle"),
                            grid::str_field(body, "current_task_id"), 0, err)) {
    return Response::error(500, err);
  }
  return Response::json(grid::Jw().s("type", "ack").s("node_id", node_id).str());
}

Response handle_claim(Ctx& ctx, const Request& req) {
  json::Value body;
  std::string err;
  if (!grid::parse_body(req.body, body, err)) return Response::error(400, err);
  const std::string node_id = grid::str_field(body, "node_id");
  if (!ctx.store->session_valid(node_id, grid::str_field(body, "token"), err)) {
    return Response::error(401, "unknown session; register again");
  }

  TaskRow task;
  bool found = false;
  if (!ctx.store->claim_task(node_id, 900, task, found, err)) {
    return Response::error(500, err);
  }
  if (!found) {
    return Response::json(grid::Jw().s("type", grid::kTIdle).n("poll_s", 2).str());
  }

  JobRow job;
  bool job_found = false;
  if (!ctx.store->job(task.job_id, job, job_found, err) || !job_found) {
    ctx.store->requeue_task(task.task_id, "task points at a missing job", err);
    return Response::error(500, "task references an unknown job");
  }
  Bundle bundle;
  if (!read_bundle(job.dataset_ref + "/bundle", bundle, err)) {
    ctx.store->requeue_task(task.task_id, "job bundle unreadable: " + err, err);
    return Response::error(500, err);
  }

  json::Value hparams;
  std::string herr;
  int64_t replica = 0;
  if (grid::parse_body(task.hparams_json, hparams, herr)) {
    replica = grid::i64_field(hparams, "replica", 0);
  }
  const std::string spec =
      task_spec(bundle.spec_json, job.job_id, replica, task.steps, bundle.has_manifest);

  std::string files = "[";
  bool first = true;
  size_t total = 0;
  for (const auto& name : bundle.files) {
    if (name == "job.json") continue;  // shipped as the rewritten spec
    std::string data;
    if (!distribai::http::read_file(job.dataset_ref + "/bundle/" + name, data)) continue;
    total += data.size();
    if (total > grid::kMaxBundleBytes) {
      ctx.store->requeue_task(task.task_id, "bundle exceeds the grid cap", err);
      return Response::error(413, "job bundle exceeds the grid's size cap");
    }
    if (!first) files += ",";
    first = false;
    files += grid::Jw().s("name", name).s("b64", distribai::b64::encode(data)).str();
  }
  files += "]";

  return Response::json(grid::Jw()
                            .s("type", grid::kTTask)
                            .s("task_id", task.task_id)
                            .s("job_id", job.job_id)
                            .s("model", job.model_name)
                            .n("replica", static_cast<double>(replica))
                            .n("steps", static_cast<double>(task.steps))
                            .n("attempt", static_cast<double>(task.attempt_count))
                            .n("lease_s", 900)
                            .s("aggregate", job.aggregate)
                            .s("spec", spec)
                            .raw("files", files)
                            .str());
}

Response handle_result(Ctx& ctx, const Request& req) {
  json::Value body;
  std::string err;
  if (!grid::parse_body(req.body, body, err)) return Response::error(400, err);
  const std::string node_id = grid::str_field(body, "node_id");
  if (!ctx.store->session_valid(node_id, grid::str_field(body, "token"), err)) {
    return Response::error(401, "unknown session; register again");
  }
  const std::string task_id = grid::str_field(body, "task_id");
  if (task_id.empty()) return Response::error(400, "task_id is required");

  TaskRow task;
  bool found = false;
  if (!ctx.store->task(task_id, task, found, err)) return Response::error(500, err);
  if (!found) return Response::error(404, "unknown task " + task_id);
  // A worker may only report on the replica the grid handed it. Without this a
  // registered node could finish or fail another node's task, or push its own
  // envelope into work it never claimed.
  if (task.assignee_node_id != node_id) {
    return Response::error(403, "task is assigned to another node");
  }
  if (task.status != "assigned" && task.status != "queued") {
    return Response::json(grid::Jw()
                              .s("type", grid::kTAck)
                              .s("task_id", task_id)
                              .s("task_status", task.status)
                              .s("note", "already finished")
                              .str());
  }

  // Workers send ok as a JSON boolean; accept a number too so a hand-written
  // probe using curl works the same way.
  bool claimed_ok = false;
  if (const json::Value* okv = body.get("ok")) {
    claimed_ok = okv->is_bool() ? okv->as_bool() : (okv->is_number() && okv->as_number() != 0);
  }
  std::string error = grid::str_field(body, "error");
  std::string envelope_path;
  const std::string envelope_b64 = grid::str_field(body, "envelope_b64");
  bool ok = false;
  if (claimed_ok) {
    if (envelope_b64.empty()) {
      error = "result reported without an envelope";
    } else {
      std::vector<uint8_t> bytes;
      std::string derr;
      if (!distribai::b64::decode(envelope_b64, bytes, derr)) {
        return Response::error(400, std::string("envelope_b64: ") + derr);
      }
      if (bytes.size() > grid::kMaxEnvelopeBytes) {
        return Response::error(413, "envelope exceeds the grid's size cap");
      }
      env::Envelope probe;
      std::string eerr;
      if (!env::Envelope::decode(bytes.data(), bytes.size(), probe, eerr)) {
        return Response::error(400, "envelope is not readable: " + eerr);
      }
      if (!make_dirs(ctx.cfg.tasks_dir)) {
        return Response::error(500, "cannot create the tasks directory");
      }
      envelope_path = ctx.cfg.tasks_dir + "/" + task_id + ".env";
      const std::string blob(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      if (!distribai::http::write_file(envelope_path, blob)) {
        return Response::error(500, "cannot store the result envelope");
      }
      ok = true;
    }
  }

  const std::string output =
      grid::Jw()
          .s("node_id", node_id)
          .n("steps", grid::num_field(body, "steps"))
          .n("final_loss", grid::num_field(body, "final_loss"))
          .n("wall_s", grid::num_field(body, "wall_s"))
          .n("steps_per_s", grid::num_field(body, "steps_per_s"))
          .s("checkpoint", grid::str_field(body, "checkpoint_id"))
          .s("engine", grid::str_field(body, "engine"))
          .n("attempt", static_cast<double>(task.attempt_count))
          .str();

  if (!ctx.store->finish_task(task_id, ok, output, error, envelope_path, err)) {
    return Response::error(500, err);
  }
  // This may have been the last replica. Closing the job here keeps the
  // dashboard honest the moment the work is finished.
  ctx.grid->maybe_finalize(task.job_id);
  return Response::json(grid::Jw()
                            .s("type", grid::kTAck)
                            .s("task_id", task_id)
                            .s("task_status", ok ? "done" : "failed")
                            .str());
}

Response handle_bye(Ctx& ctx, const Request& req) {
  json::Value body;
  std::string err;
  if (!grid::parse_body(req.body, body, err)) return Response::error(400, err);
  const std::string node_id = grid::str_field(body, "node_id");
  if (!ctx.store->session_valid(node_id, grid::str_field(body, "token"), err)) {
    return Response::error(401, "unknown session");
  }
  ctx.store->heartbeat(node_id, "offline", "", 0, err);
  return Response::json(grid::Jw().s("type", "ack").s("node_id", node_id).str());
}

Response handle_submit(Ctx& ctx, const Request& req) {
  if (!admin_ok(ctx, req)) return Response::error(403, "admin token required to submit jobs");
  json::Value body;
  std::string err;
  if (!grid::parse_body(req.body, body, err)) return Response::error(400, err);
  const std::string job_dir = grid::str_field(body, "job_dir");
  if (job_dir.empty()) return Response::error(400, "job_dir is required");
  const int replicas = static_cast<int>(grid::num_field(body, "replicas", 2));
  if (replicas < 1 || replicas > 64) return Response::error(400, "replicas must be 1..64");
  std::string job_id;
  if (!ctx.grid->submit(job_dir, replicas, grid::str_field(body, "description"), job_id, err)) {
    return Response::error(400, err);
  }
  return Response::json(grid::Jw()
                            .s("type", "accepted")
                            .s("job_id", job_id)
                            .n("replicas", replicas)
                            .s("aggregate", ctx.cfg.aggregate)
                            .str(),
                        202);
}

Response handle_cancel(Ctx& ctx, const Request& req, const std::string& job_id) {
  if (!admin_ok(ctx, req)) return Response::error(403, "admin token required to cancel jobs");
  JobRow job;
  bool found = false;
  std::string err;
  if (!ctx.store->job(job_id, job, found, err)) return Response::error(500, err);
  if (!found) return Response::error(404, "unknown job " + job_id);
  if (job.status != "queued" && job.status != "running") {
    return Response::error(409, "job is already " + job.status);
  }
  std::vector<TaskRow> tasks;
  ctx.store->tasks_for_job(job_id, tasks, err);
  for (const auto& t : tasks) {
    if (t.status == "queued" || t.status == "assigned") {
      ctx.store->finish_task(t.task_id, false, "{}", "job cancelled by operator", "", err);
    }
  }
  ctx.grid->update_status(job, "cancelled", 0, 1, "cancelled by operator");
  return Response::json(
      grid::Jw().s("type", "ack").s("job_id", job_id).s("status", "cancelled").str());
}

Response handle_summary(Ctx& ctx) {
  std::string err;
  std::vector<NodeRow> nodes;
  std::vector<JobRow> jobs;
  std::vector<TaskRow> tasks;
  std::vector<CreditRow> credits;
  std::vector<std::pair<std::string, double>> leaderboard;
  ctx.store->nodes(nodes, err);
  ctx.store->jobs(jobs, 50, err);
  ctx.store->tasks(tasks, 50, err);
  ctx.store->credits(credits, 25, err);
  ctx.store->credit_leaderboard(leaderboard, err);
  int64_t online = 0, total_jobs = 0, running = 0, done = 0, queued = 0;
  ctx.store->counts(online, total_jobs, running, done, queued, err);

  const double uptime = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                     ctx.started)
                            .count();

  std::string nodes_json = "[";
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (i) nodes_json += ",";
    const auto& n = nodes[i];
    nodes_json += grid::Jw()
                      .s("node_id", n.node_id)
                      .s("status", n.status)
                      .s("current_task_id", n.current_task_id)
                      .n("seconds_since_heartbeat",
                         static_cast<double>(grid::now_s() - n.last_heartbeat_ts))
                      .n("jobs_completed", static_cast<double>(n.jobs_completed))
                      .n("jobs_failed", static_cast<double>(n.jobs_failed))
                      .n("reliability_score", n.reliability_score)
                      .raw("hardware", n.hardware_json.empty() ? "{}" : n.hardware_json)
                      .raw("benchmark", n.benchmark_json.empty() ? "{}" : n.benchmark_json)
                      .str();
  }
  nodes_json += "]";

  std::string jobs_json = "[";
  for (size_t i = 0; i < jobs.size(); ++i) {
    if (i) jobs_json += ",";
    const auto& j = jobs[i];
    TaskCounts counts;
    ctx.store->task_counts(j.job_id, counts, err);
    jobs_json += grid::Jw()
                     .s("job_id", j.job_id)
                     .s("model_name", j.model_name)
                     .s("status", j.status)
                     .n("progress_pct", j.progress_pct)
                     .n("steps", static_cast<double>(j.steps))
                     .n("current_step", static_cast<double>(j.current_step))
                     .n("total_steps", static_cast<double>(j.total_steps))
                     .n("replicas", static_cast<double>(counts.total))
                     .n("done", static_cast<double>(counts.done))
                     .n("failed", static_cast<double>(counts.failed))
                     .n("active_nodes", static_cast<double>(j.active_nodes))
                     .s("aggregate", j.aggregate)
                     .s("reason", j.latest_reason)
                     .n("created_ts", static_cast<double>(j.created_ts))
                     .str();
  }
  jobs_json += "]";

  std::string tasks_json = "[";
  for (size_t i = 0; i < tasks.size(); ++i) {
    if (i) tasks_json += ",";
    const auto& t = tasks[i];
    tasks_json += grid::Jw()
                      .s("task_id", t.task_id)
                      .s("job_id", t.job_id)
                      .s("node_id", t.assignee_node_id)
                      .s("status", t.status)
                      .n("steps", static_cast<double>(t.steps))
                      .n("attempts", static_cast<double>(t.attempt_count))
                      .s("error", t.last_error)
                      .str();
  }
  tasks_json += "]";

  std::string credits_json = "[";
  for (size_t i = 0; i < credits.size(); ++i) {
    if (i) credits_json += ",";
    const auto& c = credits[i];
    credits_json += grid::Jw()
                        .s("node_id", c.node_id)
                        .s("type", c.tx_type)
                        .n("amount", c.amount)
                        .n("balance_after", c.balance_after)
                        .s("tx_hash", c.tx_hash)
                        .n("ts", static_cast<double>(c.ts))
                        .str();
  }
  credits_json += "]";

  std::string board_json = "[";
  for (size_t i = 0; i < leaderboard.size(); ++i) {
    if (i) board_json += ",";
    board_json +=
        grid::Jw().s("node_id", leaderboard[i].first).n("credits", leaderboard[i].second).str();
  }
  board_json += "]";

  return Response::json(grid::Jw()
                            .n("uptime_s", uptime)
                            .n("nodes_online", static_cast<double>(online))
                            .n("jobs_total", static_cast<double>(total_jobs))
                            .n("jobs_running", static_cast<double>(running))
                            .n("jobs_done", static_cast<double>(done))
                            .n("tasks_queued", static_cast<double>(queued))
                            .s("aggregate", ctx.cfg.aggregate)
                            .n("node_ttl_s", static_cast<double>(ctx.cfg.node_ttl))
                            .b("invite_required", !ctx.cfg.invite.empty())
                            .raw("nodes", nodes_json)
                            .raw("jobs", jobs_json)
                            .raw("tasks", tasks_json)
                            .raw("credits", credits_json)
                            .raw("leaderboard", board_json)
                            .str());
}

Response handle_health(Ctx& ctx) {
  std::string err;
  int64_t online = 0, total = 0, running = 0, done = 0, queued = 0;
  ctx.store->counts(online, total, running, done, queued, err);
  return Response::json(grid::Jw()
                            .s("status", "ok")
                            .s("engine", "native-grid")
                            .n("proto", grid::kProto)
                            .n("nodes_online", static_cast<double>(online))
                            .n("jobs_running", static_cast<double>(running))
                            .n("tasks_queued", static_cast<double>(queued))
                            .str());
}

Response route(Ctx& ctx, const Request& req) {
  const std::string& path = req.path;
  if (path == grid::kRegister && req.method == "POST") return handle_register(ctx, req);
  if (path == grid::kHeartbeat && req.method == "POST") return handle_heartbeat(ctx, req);
  if (path == grid::kClaim && req.method == "POST") return handle_claim(ctx, req);
  if (path == grid::kResult && req.method == "POST") return handle_result(ctx, req);
  if (path == grid::kBye && req.method == "POST") return handle_bye(ctx, req);
  if (path == grid::kJobs && req.method == "POST") return handle_submit(ctx, req);
  if (path == grid::kHealth && req.method == "GET") return handle_health(ctx);
  if (path == grid::kSummary && req.method == "GET") return handle_summary(ctx);
  if (path == grid::kNodes && req.method == "GET") {
    std::string err;
    std::vector<NodeRow> nodes;
    ctx.store->nodes(nodes, err);
    std::string out = "[";
    for (size_t i = 0; i < nodes.size(); ++i) {
      if (i) out += ",";
      out += grid::Jw()
                 .s("node_id", nodes[i].node_id)
                 .s("status", nodes[i].status)
                 .raw("hardware", nodes[i].hardware_json.empty() ? "{}" : nodes[i].hardware_json)
                 .str();
    }
    return Response::json(out + "]");
  }
  if (path == grid::kTasks && req.method == "GET") {
    std::string err;
    std::vector<TaskRow> tasks;
    ctx.store->tasks(tasks, 100, err);
    std::string out = "[";
    for (size_t i = 0; i < tasks.size(); ++i) {
      if (i) out += ",";
      out += grid::Jw()
                 .s("task_id", tasks[i].task_id)
                 .s("job_id", tasks[i].job_id)
                 .s("status", tasks[i].status)
                 .str();
    }
    return Response::json(out + "]");
  }
  if (path.rfind(std::string(grid::kJobs) + "/", 0) == 0) {
    const std::string rest = path.substr(std::strlen(grid::kJobs) + 1);
    const size_t slash = rest.find('/');
    const std::string job_id = slash == std::string::npos ? rest : rest.substr(0, slash);
    const std::string action = slash == std::string::npos ? "" : rest.substr(slash + 1);
    if (action == "cancel" && req.method == "POST") return handle_cancel(ctx, req, job_id);
    if (action.empty() && req.method == "GET") {
      JobRow job;
      bool found = false;
      std::string err;
      ctx.store->job(job_id, job, found, err);
      if (!found) return Response::error(404, "unknown job " + job_id);
      TaskCounts counts;
      ctx.store->task_counts(job_id, counts, err);
      std::vector<TaskRow> tasks;
      ctx.store->tasks_for_job(job_id, tasks, err);
      std::string result;
      distribai::http::read_file(job.dataset_ref + "/result.json", result);
      return Response::json(grid::Jw()
                                .s("job_id", job.job_id)
                                .s("model_name", job.model_name)
                                .s("status", job.status)
                                .s("aggregate", job.aggregate)
                                .s("reason", job.latest_reason)
                                .s("dataset_ref", job.dataset_ref)
                                .raw("result", result.empty() ? "{}" : result)
                                .raw("replicas_detail", Grid::replicas_json(tasks))
                                .n("replicas", static_cast<double>(counts.total))
                                .str());
    }
  }
  return Response::error(404, "no route for " + req.method + " " + path);
}

Response static_handler(Ctx& ctx, const Request& req) {
  if (req.method != "GET" && req.method != "HEAD") {
    return Response::error(405, "static assets are GET only");
  }
  std::string rel = req.path == "/" ? "index.html" : req.path.substr(1);
  if (rel.rfind("assets/", 0) == 0) rel = rel.substr(std::strlen("assets/"));
  if (rel.empty() || rel.find("..") != std::string::npos ||
      rel.find('/') != std::string::npos) {
    return Response::error(400, "bad asset name");
  }
  std::string body;
  if (ctx.cfg.web_dir.empty() ||
      !distribai::http::read_file(ctx.cfg.web_dir + "/" + rel, body)) {
    return Response::error(404,
                           "dashboard asset not found; start the orchestrator with --web-dir "
                           "pointing at tools/cpp_port/grid/web");
  }
  Response res = Response::text(body, 200, distribai::http::content_type_for(rel));
  res.headers["Cache-Control"] = "no-cache";
  return res;
}

struct Args {
  Config cfg;
  bool ok = true;
  std::string error;
};

void print_usage() {
  std::printf(
      "distribai_orch: native grid coordinator\n"
      "  --host H            bind address (default 127.0.0.1)\n"
      "  --port N            HTTP port for workers, API and dashboard (default 50061)\n"
      "  --db PATH           SQLite database (default runtime/db/grid.db)\n"
      "  --schema PATH       schema applied at startup (default runtime/db/schema.sql)\n"
      "  --jobs-dir PATH     job bundles, results, aggregate envelopes\n"
      "  --tasks-dir PATH    per-task result envelopes\n"
      "  --web-dir PATH      dashboard assets (default tools/cpp_port/grid/web)\n"
      "  --invite CODE       require this invite code when a worker registers\n"
      "  --token TOKEN       require X-Grid-Token on operator writes (submit, cancel)\n"
      "  --node-ttl S        seconds without a heartbeat before a node counts as offline\n"
      "  --aggregate METHOD  mean | median | trimmed_mean (default trimmed_mean)\n"
      "  --credit-divisor N  training steps per credit paid to a node (default 1000)\n"
      "  --submit JOB_DIR    submit one job and exit ( --replicas N , --description TEXT )\n"
      "  --print-join        print the worker join line at startup\n");
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
    if (flag == "--host") a.cfg.host = next();
    else if (flag == "--port") a.cfg.port = std::atoi(next().c_str());
    else if (flag == "--db") a.cfg.db = next();
    else if (flag == "--schema") a.cfg.schema = next();
    else if (flag == "--jobs-dir") a.cfg.jobs_dir = next();
    else if (flag == "--tasks-dir") a.cfg.tasks_dir = next();
    else if (flag == "--web-dir") a.cfg.web_dir = next();
    else if (flag == "--invite") a.cfg.invite = next();
    else if (flag == "--token") a.cfg.token = next();
    else if (flag == "--node-ttl") a.cfg.node_ttl = std::atoll(next().c_str());
    else if (flag == "--aggregate") a.cfg.aggregate = next();
    else if (flag == "--credit-divisor") a.cfg.credit_divisor = std::atof(next().c_str());
    else if (flag == "--submit") a.cfg.submit_dir = next();
    else if (flag == "--replicas") a.cfg.replicas = std::atoi(next().c_str());
    else if (flag == "--description") a.cfg.description = next();
    else if (flag == "--print-join") a.cfg.print_join = true;
    else if (flag == "--help" || flag == "-h") {
      print_usage();
      std::exit(0);
    } else {
      a.ok = false;
      a.error = "unknown flag: " + flag;
    }
  }
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);
  if (!args.ok) {
    std::fprintf(stderr, "distribai_orch: %s\n", args.error.c_str());
    return 2;
  }
  Config cfg = args.cfg;
  cfg.web_dir = resolve_web_dir(cfg.web_dir);

  if (!make_dirs(dirname_of(cfg.db))) {
    std::fprintf(stderr, "distribai_orch: cannot create %s\n", dirname_of(cfg.db).c_str());
    return 1;
  }
  Store store;
  std::string err;
  if (!store.open(cfg.db, cfg.schema, err)) {
    std::fprintf(stderr, "distribai_orch: %s\n", err.c_str());
    return 1;
  }
  Grid grid(cfg, &store);

  if (!cfg.submit_dir.empty()) {
    std::string job_id;
    if (!grid.submit(cfg.submit_dir, cfg.replicas, cfg.description, job_id, err)) {
      std::fprintf(stderr, "distribai_orch: submit failed: %s\n", err.c_str());
      return 1;
    }
    std::printf("%s\n", grid::Jw().s("type", "accepted").s("job_id", job_id).str().c_str());
    return 0;
  }

  Ctx ctx;
  ctx.grid = &grid;
  ctx.store = &store;
  ctx.cfg = cfg;
  ctx.started = std::chrono::steady_clock::now();

  distribai::http::Server server(cfg.host, cfg.port, [&ctx](const Request& req) {
    const std::string& p = req.path;
    if (p == "/" || p.rfind("/assets/", 0) == 0) return static_handler(ctx, req);
    if (p.rfind("/v1/", 0) == 0) return route(ctx, req);
    return Response::error(404, "no route for " + req.method + " " + p);
  });

  if (!server.start(err)) {
    std::fprintf(stderr, "distribai_orch: %s\n", err.c_str());
    return 1;
  }

  struct sigaction sa{};
  sa.sa_handler = on_signal;
  ::sigaction(SIGINT, &sa, nullptr);
  ::sigaction(SIGTERM, &sa, nullptr);

  std::printf("distribai_orch: http://%s:%d (api, worker API and dashboard)\n", cfg.host.c_str(),
              server.port());
  std::printf("distribai_orch: db=%s jobs=%s\n", cfg.db.c_str(), cfg.jobs_dir.c_str());
  if (cfg.print_join || !cfg.invite.empty()) {
    std::printf("join a worker:\n  distribai_worker --orchestrator http://%s:%d%s\n",
                cfg.host.c_str(), server.port(),
                cfg.invite.empty() ? "" : (" --invite " + cfg.invite).c_str());
  }
  std::fflush(stdout);

  std::thread reaper([](Grid& g) {
    while (!g_stop.load()) {
      for (int i = 0; i < 20 && !g_stop.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      if (g_stop.load()) break;
      g.reap();
    }
  }, std::ref(grid));

  server.serve(g_stop);
  reaper.join();
  server.close_listen();
  std::printf("distribai_orch: stopped\n");
  return 0;
}
