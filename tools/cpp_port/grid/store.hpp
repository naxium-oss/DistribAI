// store.hpp - the native grid's persistence layer.
//
// State lives in SQLite, in the tables runtime/db/schema.sql defines:
// active_nodes for registration and liveness, jobs and tasks for work, and
// credit_ledger for contributions. The schema file stays the source of truth,
// so the store applies it at open time and never invents tables. Anything that
// does not belong in those rows (job bundles, per-task result envelopes) is a
// file under the grid's own directory.
//
// One connection, one mutex. The orchestrator handles each HTTP request on its
// own thread, and SQLite in serialized mode plus this mutex keeps every write
// ordered without a connection pool.
#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "protocol.hpp"  // now_s and the shared field helpers

struct sqlite3;

namespace distribai::grid {

struct NodeRow {
  std::string node_id;
  std::string session_token;
  std::string hardware_json = "{}";
  std::string benchmark_json = "{}";
  std::string status = "idle";
  int contributing = 1;
  std::string current_task_id;
  int64_t last_heartbeat_ts = 0;
  int64_t jobs_completed = 0;
  int64_t jobs_failed = 0;
  double reliability_score = 1.0;
  int64_t created_ts = 0;
  int64_t updated_ts = 0;
};

struct JobRow {
  std::string job_id;
  std::string model_name;
  std::string job_type = "fine_tune";
  std::string base_model;
  std::string dataset_ref;  // job directory: job.json plus model and data files
  std::string description;
  std::string status = "queued";
  int64_t priority = 0;
  std::string priority_tier = "P1";
  std::string submitter_id = "distribai";
  std::string org = "DistribAI";
  int64_t created_ts = 0;
  int64_t updated_ts = 0;
  int64_t started_ts = 0;
  int64_t completed_ts = 0;
  int64_t steps = 100;
  int64_t batch_size = 32;
  double progress_pct = 0;
  int64_t current_step = 0;
  int64_t total_steps = 100;
  int64_t attempts = 0;
  std::string latest_task_id;
  std::string latest_reason;
  int64_t active_nodes = 0;
  int64_t replicas = 1;
  std::string aggregate = "trimmed_mean";
};

struct TaskRow {
  std::string task_id;
  std::string job_id;
  std::string assignee_node_id;
  std::string status = "queued";
  std::string weight_blob_url;  // path of the stored result envelope once reported
  std::string hparams_json;
  std::string weight_version;
  std::string gradient_blob_url;
  std::string output_json;
  std::string last_error;
  int64_t deadline_ts = 0;
  int64_t steps = 1;
  int64_t step_offset = 0;
  int64_t attempt_count = 0;
  int64_t max_attempts = 3;
  int64_t created_ts = 0;
  int64_t updated_ts = 0;
  int64_t started_ts = 0;
  int64_t completed_ts = 0;
};

struct TaskCounts {
  int64_t queued = 0;
  int64_t assigned = 0;
  int64_t done = 0;
  int64_t failed = 0;
  int64_t total = 0;
};

struct CreditRow {
  int64_t tx_id = 0;
  std::string node_id;
  std::string tx_type;
  double amount = 0;
  double balance_after = 0;
  std::string tx_hash;
  std::string prev_hash;
  int64_t ts = 0;
};

class Store {
 public:
  ~Store();

  bool open(const std::string& db_path, const std::string& schema_path, std::string& err);

  // ---- nodes ----
  // true when a new node row was created; false when an existing node was
  // refreshed. The session token is always rotated, so a restarted worker
  // cannot keep a stale one.
  bool upsert_node(NodeRow& node, std::string& err);
  bool session_valid(const std::string& node_id, const std::string& token, std::string& err);
  bool heartbeat(const std::string& node_id, const std::string& status,
                 const std::string& current_task_id, double progress_pct, std::string& err);
  bool nodes(std::vector<NodeRow>& out, std::string& err);
  bool node(const std::string& node_id, NodeRow& out, bool& found, std::string& err);
  // Marks nodes quiet for longer than ttl_s as offline and relearns their
  // assigned tasks. Returns the number of tasks put back in the queue.
  bool expire_nodes(int64_t ttl_s, int64_t& requeued, int64_t& failed, std::string& err);

  // ---- jobs ----
  // Writes the job and its task rows in one transaction. The caller builds the
  // tasks because it owns the per-replica seed logic.
  bool create_job(JobRow& job, const std::vector<TaskRow>& tasks, std::string& err);
  bool jobs(std::vector<JobRow>& out, size_t limit, std::string& err);
  bool job(const std::string& job_id, JobRow& out, bool& found, std::string& err);
  bool update_job(const JobRow& job, std::string& err);
  // Moves a running job to 'finalizing' and reports whether this caller won.
  // The maintenance thread and the request that reported the last replica can
  // both notice a finished job at the same time; only one of them may write the
  // aggregate and the credits.
  bool claim_finalization(const std::string& job_id, bool& claimed, std::string& err);
  bool task_counts(const std::string& job_id, TaskCounts& out, std::string& err);
  bool counts(int64_t& nodes_online, int64_t& jobs_total, int64_t& jobs_running,
              int64_t& jobs_done, int64_t& tasks_queued, std::string& err);

  // ---- tasks ----
  bool tasks_for_job(const std::string& job_id, std::vector<TaskRow>& out, std::string& err);
  bool tasks(std::vector<TaskRow>& out, size_t limit, std::string& err);
  // Hands the oldest queued task to a node, in one transaction. found==false
  // means nothing is waiting.
  bool claim_task(const std::string& node_id, int64_t lease_s, TaskRow& out, bool& found,
                  std::string& err);
  bool task(const std::string& task_id, TaskRow& out, bool& found, std::string& err);
  // `blob_path` is the stored result envelope, or empty when the task failed.
  bool finish_task(const std::string& task_id, bool ok, const std::string& output_json,
                   const std::string& error, const std::string& blob_path, std::string& err);
  bool requeue_task(const std::string& task_id, const std::string& reason, std::string& err);

  // ---- credits ----
  bool add_credit(const std::string& node_id, const std::string& tx_type, double amount,
                  std::string& err);
  bool credits(std::vector<CreditRow>& out, size_t limit, std::string& err);
  bool credit_leaderboard(std::vector<std::pair<std::string, double>>& out, std::string& err);

  // ---- maintenance ----
  bool vacuum_stale_sessions(int64_t ttl_s, std::string& err);

 private:
  bool exec(const std::string& sql, std::string& err);
  // Callers hold mu_. claim_task already owns the lock when it needs to read
  // back the row it just assigned.
  bool task_locked(const std::string& task_id, TaskRow& out, bool& found, std::string& err);
  sqlite3* db_ = nullptr;
  std::mutex mu_;
};

std::string random_token(size_t bytes);

}  // namespace distribai::grid
