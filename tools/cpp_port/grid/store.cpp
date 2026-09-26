// store.cpp - SQLite persistence for the native grid.
//
// Every statement here is prepared once per call. Client-supplied strings (node
// ids, job ids) always go through bind params, so a worker on a public tunnel
// cannot reach the SQL layer with a crafted id.
#include "store.hpp"

#include <sqlite3.h>

#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>

#include "../core/envelope.hpp"

namespace distribai::grid {

namespace {

// Prepared statement with a readable error path. Every method that can fail
// records a message; callers surface it to the client.
class Stmt {
 public:
  Stmt(sqlite3* db, const std::string& sql) : db_(db) {
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &st_, nullptr) != SQLITE_OK) {
      err_ = std::string(sqlite3_errmsg(db)) + " :: " + sql;
    }
  }
  ~Stmt() {
    if (st_) sqlite3_finalize(st_);
  }
  Stmt(const Stmt&) = delete;
  Stmt& operator=(const Stmt&) = delete;

  bool ok() const { return err_.empty(); }
  const std::string& error() const { return err_; }

  Stmt& text(int i, const std::string& v) {
    if (st_) sqlite3_bind_text(st_, i, v.c_str(), static_cast<int>(v.size()), SQLITE_TRANSIENT);
    return *this;
  }
  Stmt& i64(int i, int64_t v) {
    if (st_) sqlite3_bind_int64(st_, i, v);
    return *this;
  }
  Stmt& dbl(int i, double v) {
    if (st_) sqlite3_bind_double(st_, i, v);
    return *this;
  }

  // true when a row is available, false at end of rows or on error.
  bool row() {
    if (!st_) return false;
    const int rc = sqlite3_step(st_);
    if (rc == SQLITE_ROW) return true;
    if (rc != SQLITE_DONE) err_ = sqlite3_errmsg(db_);
    return false;
  }
  bool run() {
    if (!st_) return false;
    const int rc = sqlite3_step(st_);
    if (rc == SQLITE_DONE) return true;
    err_ = sqlite3_errmsg(db_);
    return false;
  }
  std::string col_text(int i) {
    const unsigned char* p = st_ ? sqlite3_column_text(st_, i) : nullptr;
    return p ? reinterpret_cast<const char*>(p) : std::string();
  }
  int64_t col_i64(int i) { return st_ ? sqlite3_column_int64(st_, i) : 0; }
  double col_dbl(int i) { return st_ ? sqlite3_column_double(st_, i) : 0.0; }
  int col_type(int i) { return st_ ? sqlite3_column_type(st_, i) : SQLITE_NULL; }

 private:
  sqlite3* db_;
  sqlite3_stmt* st_ = nullptr;
  std::string err_;
};

std::string hex32(uint32_t v) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%08x", v);
  return buf;
}

}  // namespace

std::string random_token(size_t bytes) {
  std::random_device rd;
  static const char* digits = "0123456789abcdef";
  std::string out;
  out.reserve(bytes * 2);
  for (size_t i = 0; i < bytes; ++i) {
    const unsigned v = rd() & 0xFF;
    out += digits[(v >> 4) & 0xF];
    out += digits[v & 0xF];
  }
  return out;
}

Store::~Store() {
  if (db_) sqlite3_close(db_);
}

bool Store::exec(const std::string& sql, std::string& err) {
  char* msg = nullptr;
  if (sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &msg) != SQLITE_OK) {
    err = msg ? msg : "sqlite error";
    if (msg) sqlite3_free(msg);
    return false;
  }
  return true;
}

bool Store::open(const std::string& db_path, const std::string& schema_path, std::string& err) {
  if (sqlite3_open_v2(db_path.c_str(), &db_, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE |
                                                  SQLITE_OPEN_FULLMUTEX,
                      nullptr) != SQLITE_OK) {
    err = "cannot open grid database " + db_path;
    return false;
  }
  if (!exec("PRAGMA journal_mode=WAL;", err)) return false;
  if (!exec("PRAGMA busy_timeout=5000;", err)) return false;
  if (!exec("PRAGMA foreign_keys=ON;", err)) return false;

  if (!schema_path.empty()) {
    std::ifstream in(schema_path, std::ios::binary);
    if (!in) {
      err = "schema not found: " + schema_path;
      return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    if (!exec(ss.str(), err)) {
      err = "applying " + schema_path + ": " + err;
      return false;
    }
  }

  // Columns added after a grid had already created its database. CREATE TABLE
  // IF NOT EXISTS leaves existing tables alone, so an older file needs the
  // column added by hand. A duplicate-column error means it is already there,
  // which is the normal case and is not a failure.
  for (const char* migration : {
           "ALTER TABLE jobs ADD COLUMN aggregate TEXT DEFAULT 'trimmed_mean'",
       }) {
    std::string ignored;
    exec(migration, ignored);
  }
  return true;
}

bool Store::upsert_node(NodeRow& node, std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  const int64_t now = now_s();
  NodeRow existing;
  bool found = false;
  {
    Stmt q(db_, "SELECT node_id, jobs_completed, jobs_failed, created_ts, reliability_score "
                "FROM active_nodes WHERE node_id=?");
    if (!q.ok()) { err = q.error(); return false; }
    q.text(1, node.node_id);
    if (q.row()) {
      found = true;
      existing.node_id = q.col_text(0);
      existing.jobs_completed = q.col_i64(1);
      existing.jobs_failed = q.col_i64(2);
      existing.created_ts = q.col_i64(3);
      existing.reliability_score = q.col_dbl(4);
    }
  }
  node.session_token = random_token(16);
  node.last_heartbeat_ts = now;
  node.updated_ts = now;
  if (found) {
    node.created_ts = existing.created_ts;
    node.jobs_completed = existing.jobs_completed;
    node.jobs_failed = existing.jobs_failed;
    node.reliability_score = existing.reliability_score;
    Stmt u(db_, "UPDATE active_nodes SET session_token=?, hardware_json=?, benchmark_json=?, "
                "status='idle', contributing=1, current_task_id='', last_heartbeat_ts=?, "
                "updated_ts=? WHERE node_id=?");
    if (!u.ok()) { err = u.error(); return false; }
    u.text(1, node.session_token).text(2, node.hardware_json).text(3, node.benchmark_json)
        .i64(4, now).i64(5, now).text(6, node.node_id);
    if (!u.run()) { err = u.error(); return false; }
    return false;
  }
  node.created_ts = now;
  Stmt i(db_, "INSERT INTO active_nodes (node_id, session_token, hardware_json, benchmark_json, "
              "status, contributing, current_task_id, last_heartbeat_ts, jobs_completed, "
              "jobs_failed, reliability_score, created_ts, updated_ts) "
              "VALUES (?,?,?,?,?,1,'',?,0,0,1.0,?,?)");
  if (!i.ok()) { err = i.error(); return false; }
  i.text(1, node.node_id).text(2, node.session_token).text(3, node.hardware_json)
      .text(4, node.benchmark_json).text(5, node.status).i64(6, now).i64(7, now).i64(8, now);
  if (!i.run()) { err = i.error(); return false; }
  return true;
}

bool Store::session_valid(const std::string& node_id, const std::string& token,
                          std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  if (token.empty()) return false;
  Stmt q(db_, "SELECT 1 FROM active_nodes WHERE node_id=? AND session_token=?");
  if (!q.ok()) { err = q.error(); return false; }
  q.text(1, node_id).text(2, token);
  return q.row();
}

bool Store::heartbeat(const std::string& node_id, const std::string& status,
                      const std::string& current_task_id, double progress_pct,
                      std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  const int64_t now = now_s();
  Stmt u(db_, "UPDATE active_nodes SET status=?, current_task_id=?, last_heartbeat_ts=?, "
              "updated_ts=? WHERE node_id=?");
  if (!u.ok()) { err = u.error(); return false; }
  u.text(1, status).text(2, current_task_id).i64(3, now).i64(4, now).text(5, node_id);
  if (!u.run()) { err = u.error(); return false; }
  (void)progress_pct;  // job progress is derived from task rows, not worker claims
  return true;
}

bool Store::nodes(std::vector<NodeRow>& out, std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  Stmt q(db_, "SELECT node_id, hardware_json, benchmark_json, status, contributing, "
              "current_task_id, last_heartbeat_ts, jobs_completed, jobs_failed, "
              "reliability_score, created_ts, updated_ts FROM active_nodes "
              "ORDER BY last_heartbeat_ts DESC");
  if (!q.ok()) { err = q.error(); return false; }
  while (q.row()) {
    NodeRow n;
    n.node_id = q.col_text(0);
    n.hardware_json = q.col_text(1);
    n.benchmark_json = q.col_text(2);
    n.status = q.col_text(3);
    n.contributing = static_cast<int>(q.col_i64(4));
    n.current_task_id = q.col_text(5);
    n.last_heartbeat_ts = q.col_i64(6);
    n.jobs_completed = q.col_i64(7);
    n.jobs_failed = q.col_i64(8);
    n.reliability_score = q.col_dbl(9);
    n.created_ts = q.col_i64(10);
    n.updated_ts = q.col_i64(11);
    out.push_back(std::move(n));
  }
  return true;
}

bool Store::node(const std::string& node_id, NodeRow& out, bool& found, std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  Stmt q(db_, "SELECT node_id, hardware_json, benchmark_json, status, contributing, "
              "current_task_id, last_heartbeat_ts, jobs_completed, jobs_failed, "
              "reliability_score FROM active_nodes WHERE node_id=?");
  if (!q.ok()) { err = q.error(); return false; }
  q.text(1, node_id);
  found = q.row();
  if (!found) return true;
  out.node_id = q.col_text(0);
  out.hardware_json = q.col_text(1);
  out.benchmark_json = q.col_text(2);
  out.status = q.col_text(3);
  out.contributing = static_cast<int>(q.col_i64(4));
  out.current_task_id = q.col_text(5);
  out.last_heartbeat_ts = q.col_i64(6);
  out.jobs_completed = q.col_i64(7);
  out.jobs_failed = q.col_i64(8);
  out.reliability_score = q.col_dbl(9);
  return true;
}

bool Store::expire_nodes(int64_t ttl_s, int64_t& requeued, int64_t& failed, std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  const int64_t now = now_s();
  const int64_t cutoff = now - ttl_s;
  if (!exec("BEGIN IMMEDIATE;", err)) return false;

  Stmt mark(db_, "UPDATE active_nodes SET status='offline', current_task_id='', updated_ts=? "
                 "WHERE status<>'offline' AND last_heartbeat_ts<?");
  if (!mark.ok()) { err = mark.error(); exec("ROLLBACK;", err); return false; }
  mark.i64(1, now).i64(2, cutoff);
  if (!mark.run()) { err = mark.error(); exec("ROLLBACK;", err); return false; }

  {
    Stmt q(db_, "SELECT t.task_id, t.attempt_count, t.max_attempts FROM tasks t "
                "JOIN active_nodes n ON n.node_id = t.assignee_node_id "
                "WHERE t.status='assigned' AND n.status='offline'");
    if (!q.ok()) { err = q.error(); exec("ROLLBACK;", err); return false; }
    struct Pending { std::string id; int64_t attempts; int64_t max; };
    std::vector<Pending> pend;
    while (q.row()) {
      pend.push_back({q.col_text(0), q.col_i64(1), q.col_i64(2)});
    }
    for (const auto& p : pend) {
      const bool retry = p.attempts < p.max;
      if (retry) {
        Stmt u(db_, "UPDATE tasks SET status='queued', assignee_node_id='', "
                    "last_error='assignee went offline', updated_ts=? WHERE task_id=?");
        if (!u.ok()) { err = u.error(); exec("ROLLBACK;", err); return false; }
        u.i64(1, now).text(2, p.id);
        if (!u.run()) { err = u.error(); exec("ROLLBACK;", err); return false; }
        ++requeued;
      } else {
        Stmt u(db_, "UPDATE tasks SET status='failed', assignee_node_id='', "
                    "last_error='assignee went offline', completed_ts=?, updated_ts=? "
                    "WHERE task_id=?");
        if (!u.ok()) { err = u.error(); exec("ROLLBACK;", err); return false; }
        u.i64(1, now).i64(2, now).text(3, p.id);
        if (!u.run()) { err = u.error(); exec("ROLLBACK;", err); return false; }
        ++failed;
      }
    }
  }

  Stmt jobs_fix(db_, "UPDATE jobs SET active_nodes=(SELECT COUNT(DISTINCT assignee_node_id) "
                     "FROM tasks WHERE tasks.job_id=jobs.job_id AND status='assigned'), "
                     "updated_ts=? WHERE status IN ('queued','running')");
  if (jobs_fix.ok()) {
    jobs_fix.i64(1, now);
    if (!jobs_fix.run()) { err = jobs_fix.error(); exec("ROLLBACK;", err); return false; }
  }
  return exec("COMMIT;", err);
}

bool Store::create_job(JobRow& job, const std::vector<TaskRow>& tasks, std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  const int64_t now = now_s();
  if (!exec("BEGIN IMMEDIATE;", err)) return false;

  Stmt i(db_, "INSERT INTO jobs (job_id, model_name, job_type, base_model, dataset_ref, "
              "description, status, priority, priority_tier, submitter_id, org, created_ts, "
              "updated_ts, steps, batch_size, progress_pct, current_step, total_steps, "
              "attempts, active_nodes, aggregate) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,0,?,0,0,?)");
  if (!i.ok()) { err = i.error(); exec("ROLLBACK;", err); return false; }
  job.created_ts = now;
  job.updated_ts = now;
  i.text(1, job.job_id).text(2, job.model_name).text(3, job.job_type).text(4, job.base_model)
      .text(5, job.dataset_ref).text(6, job.description).text(7, job.status)
      .i64(8, job.priority).text(9, job.priority_tier).text(10, job.submitter_id)
      .text(11, job.org).i64(12, now).i64(13, now).i64(14, job.steps)
      .i64(15, job.batch_size).i64(16, job.total_steps).text(17, job.aggregate);
  if (!i.run()) { err = i.error(); exec("ROLLBACK;", err); return false; }

  for (const auto& t : tasks) {
    Stmt tj(db_, "INSERT INTO tasks (task_id, job_id, assignee_node_id, status, "
                 "weight_blob_url, hparams_json, deadline_ts, steps, step_offset, "
                 "attempt_count, max_attempts, created_ts, updated_ts) "
                 "VALUES (?,?,'','queued',?,?,0,?,?,0,?,?,?)");
    if (!tj.ok()) { err = tj.error(); exec("ROLLBACK;", err); return false; }
    tj.text(1, t.task_id).text(2, t.job_id).text(3, t.weight_blob_url).text(4, t.hparams_json)
        .i64(5, t.steps).i64(6, t.step_offset).i64(7, t.max_attempts).i64(8, now).i64(9, now);
    if (!tj.run()) { err = tj.error(); exec("ROLLBACK;", err); return false; }
  }
  return exec("COMMIT;", err);
}

bool Store::jobs(std::vector<JobRow>& out, size_t limit, std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  Stmt q(db_, "SELECT job_id, model_name, job_type, base_model, dataset_ref, description, "
              "status, priority, priority_tier, submitter_id, org, created_ts, updated_ts, "
              "started_ts, completed_ts, steps, batch_size, progress_pct, current_step, "
              "total_steps, attempts, latest_task_id, latest_reason, active_nodes, "
              "COALESCE(aggregate,'trimmed_mean') "
              "FROM jobs ORDER BY created_ts DESC LIMIT ?");
  if (!q.ok()) { err = q.error(); return false; }
  q.i64(1, static_cast<int64_t>(limit));
  while (q.row()) {
    JobRow j;
    j.job_id = q.col_text(0);
    j.model_name = q.col_text(1);
    j.job_type = q.col_text(2);
    j.base_model = q.col_text(3);
    j.dataset_ref = q.col_text(4);
    j.description = q.col_text(5);
    j.status = q.col_text(6);
    j.priority = q.col_i64(7);
    j.priority_tier = q.col_text(8);
    j.submitter_id = q.col_text(9);
    j.org = q.col_text(10);
    j.created_ts = q.col_i64(11);
    j.updated_ts = q.col_i64(12);
    j.started_ts = q.col_i64(13);
    j.completed_ts = q.col_i64(14);
    j.steps = q.col_i64(15);
    j.batch_size = q.col_i64(16);
    j.progress_pct = q.col_dbl(17);
    j.current_step = q.col_i64(18);
    j.total_steps = q.col_i64(19);
    j.attempts = q.col_i64(20);
    j.latest_task_id = q.col_text(21);
    j.latest_reason = q.col_text(22);
    j.active_nodes = q.col_i64(23);
    j.aggregate = q.col_text(24);
    out.push_back(std::move(j));
  }
  return true;
}

bool Store::job(const std::string& job_id, JobRow& out, bool& found, std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  Stmt q(db_, "SELECT job_id, model_name, job_type, base_model, dataset_ref, description, "
              "status, priority, priority_tier, submitter_id, org, created_ts, updated_ts, "
              "started_ts, completed_ts, steps, batch_size, progress_pct, current_step, "
              "total_steps, attempts, latest_task_id, latest_reason, "
              "COALESCE(aggregate,'trimmed_mean') FROM jobs WHERE job_id=?");
  if (!q.ok()) { err = q.error(); return false; }
  q.text(1, job_id);
  found = q.row();
  if (!found) return true;
  out.job_id = q.col_text(0);
  out.model_name = q.col_text(1);
  out.job_type = q.col_text(2);
  out.base_model = q.col_text(3);
  out.dataset_ref = q.col_text(4);
  out.description = q.col_text(5);
  out.status = q.col_text(6);
  out.priority = q.col_i64(7);
  out.priority_tier = q.col_text(8);
  out.submitter_id = q.col_text(9);
  out.org = q.col_text(10);
  out.created_ts = q.col_i64(11);
  out.updated_ts = q.col_i64(12);
  out.started_ts = q.col_i64(13);
  out.completed_ts = q.col_i64(14);
  out.steps = q.col_i64(15);
  out.batch_size = q.col_i64(16);
  out.progress_pct = q.col_dbl(17);
  out.current_step = q.col_i64(18);
  out.total_steps = q.col_i64(19);
  out.attempts = q.col_i64(20);
  out.latest_task_id = q.col_text(21);
  out.latest_reason = q.col_text(22);
  out.aggregate = q.col_text(23);
  return true;
}

bool Store::update_job(const JobRow& job, std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  Stmt u(db_, "UPDATE jobs SET status=?, progress_pct=?, current_step=?, total_steps=?, "
              "attempts=?, active_nodes=?, latest_task_id=?, latest_reason=?, updated_ts=?, "
              "started_ts=?, completed_ts=? WHERE job_id=?");
  if (!u.ok()) { err = u.error(); return false; }
  u.text(1, job.status).dbl(2, job.progress_pct).i64(3, job.current_step)
      .i64(4, job.total_steps).i64(5, job.attempts).i64(6, job.active_nodes)
      .text(7, job.latest_task_id).text(8, job.latest_reason).i64(9, now_s())
      .i64(10, job.started_ts).i64(11, job.completed_ts).text(12, job.job_id);
  if (!u.run()) { err = u.error(); return false; }
  return true;
}

bool Store::claim_finalization(const std::string& job_id, bool& claimed, std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  claimed = false;
  Stmt u(db_, "UPDATE jobs SET status='finalizing', updated_ts=? WHERE job_id=? "
              "AND status IN ('queued','running')");
  if (!u.ok()) { err = u.error(); return false; }
  u.i64(1, now_s()).text(2, job_id);
  if (!u.run()) { err = u.error(); return false; }
  claimed = sqlite3_changes(db_) > 0;
  return true;
}

bool Store::task_counts(const std::string& job_id, TaskCounts& out, std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  Stmt q(db_, "SELECT status, COUNT(*), COALESCE(SUM(steps),0) FROM tasks WHERE job_id=? "
              "GROUP BY status");
  if (!q.ok()) { err = q.error(); return false; }
  q.text(1, job_id);
  while (q.row()) {
    const std::string status = q.col_text(0);
    const int64_t n = q.col_i64(1);
    out.total += n;
    if (status == "queued") out.queued += n;
    else if (status == "assigned") out.assigned += n;
    else if (status == "done") out.done += n;
    else if (status == "failed") out.failed += n;
  }
  return true;
}

bool Store::counts(int64_t& nodes_online, int64_t& jobs_total, int64_t& jobs_running,
                   int64_t& jobs_done, int64_t& tasks_queued, std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  {
    Stmt q(db_, "SELECT COUNT(*) FROM active_nodes WHERE status<>'offline'");
    if (!q.ok()) { err = q.error(); return false; }
    if (q.row()) nodes_online = q.col_i64(0);
  }
  {
    Stmt q(db_, "SELECT COUNT(*), SUM(status='running'), SUM(status='completed') FROM jobs");
    if (!q.ok()) { err = q.error(); return false; }
    if (q.row()) {
      jobs_total = q.col_i64(0);
      jobs_running = q.col_type(1) == SQLITE_NULL ? 0 : q.col_i64(1);
      jobs_done = q.col_type(2) == SQLITE_NULL ? 0 : q.col_i64(2);
    }
  }
  {
    Stmt q(db_, "SELECT COUNT(*) FROM tasks WHERE status='queued'");
    if (!q.ok()) { err = q.error(); return false; }
    if (q.row()) tasks_queued = q.col_i64(0);
  }
  return true;
}

bool Store::tasks_for_job(const std::string& job_id, std::vector<TaskRow>& out,
                          std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  Stmt q(db_, "SELECT task_id, job_id, assignee_node_id, status, weight_blob_url, hparams_json, "
              "deadline_ts, steps, step_offset, attempt_count, max_attempts, created_ts, "
              "updated_ts, started_ts, completed_ts, output_json, last_error, gradient_blob_url "
              "FROM tasks WHERE job_id=? ORDER BY created_ts ASC, task_id ASC");
  if (!q.ok()) { err = q.error(); return false; }
  q.text(1, job_id);
  while (q.row()) {
    TaskRow t;
    t.task_id = q.col_text(0);
    t.job_id = q.col_text(1);
    t.assignee_node_id = q.col_text(2);
    t.status = q.col_text(3);
    t.weight_blob_url = q.col_text(4);
    t.hparams_json = q.col_text(5);
    t.deadline_ts = q.col_i64(6);
    t.steps = q.col_i64(7);
    t.step_offset = q.col_i64(8);
    t.attempt_count = q.col_i64(9);
    t.max_attempts = q.col_i64(10);
    t.created_ts = q.col_i64(11);
    t.updated_ts = q.col_i64(12);
    t.started_ts = q.col_i64(13);
    t.completed_ts = q.col_i64(14);
    t.output_json = q.col_text(15);
    t.last_error = q.col_text(16);
    t.gradient_blob_url = q.col_text(17);
    out.push_back(std::move(t));
  }
  return true;
}

bool Store::tasks(std::vector<TaskRow>& out, size_t limit, std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  Stmt q(db_, "SELECT task_id, job_id, assignee_node_id, status, steps, step_offset, "
              "attempt_count, max_attempts, created_ts, updated_ts, started_ts, completed_ts, "
              "last_error FROM tasks ORDER BY created_ts DESC LIMIT ?");
  if (!q.ok()) { err = q.error(); return false; }
  q.i64(1, static_cast<int64_t>(limit));
  while (q.row()) {
    TaskRow t;
    t.task_id = q.col_text(0);
    t.job_id = q.col_text(1);
    t.assignee_node_id = q.col_text(2);
    t.status = q.col_text(3);
    t.steps = q.col_i64(4);
    t.step_offset = q.col_i64(5);
    t.attempt_count = q.col_i64(6);
    t.max_attempts = q.col_i64(7);
    t.created_ts = q.col_i64(8);
    t.updated_ts = q.col_i64(9);
    t.started_ts = q.col_i64(10);
    t.completed_ts = q.col_i64(11);
    t.last_error = q.col_text(12);
    out.push_back(std::move(t));
  }
  return true;
}

bool Store::claim_task(const std::string& node_id, int64_t lease_s, TaskRow& out, bool& found,
                       std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  const int64_t now = now_s();
  found = false;
  if (!exec("BEGIN IMMEDIATE;", err)) return false;

  std::string task_id;
  {
    Stmt q(db_, "SELECT t.task_id, t.job_id FROM tasks t JOIN jobs j ON j.job_id=t.job_id "
                "WHERE t.status='queued' AND j.status IN ('queued','running') "
                "ORDER BY j.priority DESC, t.created_ts ASC, t.task_id ASC LIMIT 1");
    if (!q.ok()) { err = q.error(); exec("ROLLBACK;", err); return false; }
    if (!q.row()) {
      return exec("COMMIT;", err);
    }
    task_id = q.col_text(0);
    out.job_id = q.col_text(1);
  }

  {
    Stmt u(db_, "UPDATE tasks SET assignee_node_id=?, status='assigned', "
                "attempt_count=attempt_count+1, started_ts=COALESCE(started_ts,?), "
                "deadline_ts=?, updated_ts=? WHERE task_id=? AND status='queued'");
    if (!u.ok()) { err = u.error(); exec("ROLLBACK;", err); return false; }
    u.text(1, node_id).i64(2, now).i64(3, now + lease_s).i64(4, now).text(5, task_id);
    if (!u.run()) { err = u.error(); exec("ROLLBACK;", err); return false; }
    if (sqlite3_changes(db_) != 1) {
      return exec("COMMIT;", err);  // lost the race, caller just sees "nothing"
    }
  }

  {
    Stmt j(db_, "UPDATE jobs SET status='running', started_ts=COALESCE(started_ts,?), "
                "latest_task_id=?, updated_ts=?, "
                "active_nodes=(SELECT COUNT(DISTINCT assignee_node_id) FROM tasks "
                "WHERE job_id=? AND status='assigned') WHERE job_id=?");
    if (!j.ok()) { err = j.error(); exec("ROLLBACK;", err); return false; }
    j.i64(1, now).text(2, task_id).i64(3, now).text(4, out.job_id).text(5, out.job_id);
    if (!j.run()) { err = j.error(); exec("ROLLBACK;", err); return false; }
  }

  if (!exec("COMMIT;", err)) return false;

  bool ok = false;
  if (!task_locked(task_id, out, ok, err)) return false;
  found = ok;
  return true;
}

bool Store::task(const std::string& task_id, TaskRow& out, bool& found, std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  return task_locked(task_id, out, found, err);
}

bool Store::task_locked(const std::string& task_id, TaskRow& out, bool& found,
                        std::string& err) {
  Stmt q(db_, "SELECT task_id, job_id, assignee_node_id, status, weight_blob_url, hparams_json, "
              "deadline_ts, steps, step_offset, attempt_count, max_attempts, created_ts, "
              "updated_ts, started_ts, completed_ts, output_json, last_error "
              "FROM tasks WHERE task_id=?");
  if (!q.ok()) { err = q.error(); return false; }
  q.text(1, task_id);
  found = q.row();
  if (!found) return true;
  out.task_id = q.col_text(0);
  out.job_id = q.col_text(1);
  out.assignee_node_id = q.col_text(2);
  out.status = q.col_text(3);
  out.weight_blob_url = q.col_text(4);
  out.hparams_json = q.col_text(5);
  out.deadline_ts = q.col_i64(6);
  out.steps = q.col_i64(7);
  out.step_offset = q.col_i64(8);
  out.attempt_count = q.col_i64(9);
  out.max_attempts = q.col_i64(10);
  out.created_ts = q.col_i64(11);
  out.updated_ts = q.col_i64(12);
  out.started_ts = q.col_i64(13);
  out.completed_ts = q.col_i64(14);
  out.output_json = q.col_text(15);
  out.last_error = q.col_text(16);
  return true;
}

bool Store::finish_task(const std::string& task_id, bool ok, const std::string& output_json,
                        const std::string& error, const std::string& blob_path,
                        std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  const int64_t now = now_s();
  if (!exec("BEGIN IMMEDIATE;", err)) return false;

  std::string job_id, node_id;
  {
    Stmt q(db_, "SELECT job_id, assignee_node_id FROM tasks WHERE task_id=?");
    if (!q.ok()) { err = q.error(); exec("ROLLBACK;", err); return false; }
    q.text(1, task_id);
    if (!q.row()) {
      err = "unknown task " + task_id;
      exec("ROLLBACK;", err);
      return false;
    }
    job_id = q.col_text(0);
    node_id = q.col_text(1);
  }

  {
    Stmt u(db_, "UPDATE tasks SET status=?, completed_ts=?, updated_ts=?, output_json=?, "
                "last_error=?, gradient_blob_url=? WHERE task_id=?");
    if (!u.ok()) { err = u.error(); exec("ROLLBACK;", err); return false; }
    u.text(1, ok ? "done" : "failed").i64(2, now).i64(3, now).text(4, output_json)
        .text(5, error).text(6, blob_path).text(7, task_id);
    if (!u.run()) { err = u.error(); exec("ROLLBACK;", err); return false; }
  }

  if (!node_id.empty()) {
    Stmt n(db_, "UPDATE active_nodes SET status='idle', current_task_id='', "
                "jobs_completed=jobs_completed+?, jobs_failed=jobs_failed+?, "
                "reliability_score=MAX(0.0, MIN(1.0, reliability_score + ?)), updated_ts=? "
                "WHERE node_id=?");
    if (!n.ok()) { err = n.error(); exec("ROLLBACK;", err); return false; }
    n.i64(1, ok ? 1 : 0).i64(2, ok ? 0 : 1).dbl(3, ok ? 0.02 : -0.05).i64(4, now)
        .text(5, node_id);
    if (!n.run()) { err = n.error(); exec("ROLLBACK;", err); return false; }
  }

  // Progress is measured in steps, not tasks: a replica that ran 200 of 400
  // steps should read 50%.
  int64_t done_steps = 0;
  int64_t total_steps = 0;
  {
    Stmt q(db_, "SELECT COALESCE(SUM(CASE WHEN status='done' THEN steps ELSE 0 END),0), "
                "COALESCE(SUM(steps),0) FROM tasks WHERE job_id=?");
    if (!q.ok()) { err = q.error(); exec("ROLLBACK;", err); return false; }
    q.text(1, job_id);
    if (q.row()) {
      done_steps = q.col_i64(0);
      total_steps = q.col_i64(1);
    }
  }
  {
    Stmt j(db_, "UPDATE jobs SET progress_pct=?, current_step=?, total_steps=?, updated_ts=?, "
                "active_nodes=(SELECT COUNT(DISTINCT assignee_node_id) FROM tasks "
                "WHERE job_id=? AND status='assigned') WHERE job_id=?");
    if (!j.ok()) { err = j.error(); exec("ROLLBACK;", err); return false; }
    const double pct = total_steps > 0 ? 100.0 * static_cast<double>(done_steps) /
                                             static_cast<double>(total_steps)
                                       : 0.0;
    j.dbl(1, pct).i64(2, done_steps).i64(3, total_steps).i64(4, now).text(5, job_id)
        .text(6, job_id);
    if (!j.run()) { err = j.error(); exec("ROLLBACK;", err); return false; }
  }

  return exec("COMMIT;", err);
}

bool Store::requeue_task(const std::string& task_id, const std::string& reason, std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  const int64_t now = now_s();
  Stmt u(db_, "UPDATE tasks SET status=CASE WHEN attempt_count<max_attempts THEN 'queued' "
              "ELSE 'failed' END, assignee_node_id='', last_error=?, updated_ts=?, "
              "completed_ts=CASE WHEN attempt_count<max_attempts THEN completed_ts "
              "ELSE ? END WHERE task_id=?");
  if (!u.ok()) { err = u.error(); return false; }
  u.text(1, reason).i64(2, now).i64(3, now).text(4, task_id);
  if (!u.run()) { err = u.error(); return false; }
  return true;
}

bool Store::add_credit(const std::string& node_id, const std::string& tx_type, double amount,
                       std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  const int64_t now = now_s();
  std::string prev_hash;
  double balance = 0;
  {
    Stmt q(db_, "SELECT balance_after, tx_hash FROM credit_ledger WHERE node_id=? "
                "ORDER BY tx_id DESC LIMIT 1");
    if (!q.ok()) { err = q.error(); return false; }
    q.text(1, node_id);
    if (q.row()) {
      balance = q.col_dbl(0);
      prev_hash = q.col_text(1);
    }
  }
  const double after = balance + amount;
  const std::string canonical = node_id + "|" + tx_type + "|" + std::to_string(amount) + "|" +
                                prev_hash + "|" + std::to_string(now);
  const std::string hash = hex32(distribai::env::crc32_ieee(
      reinterpret_cast<const uint8_t*>(canonical.data()), canonical.size()));
  Stmt i(db_, "INSERT INTO credit_ledger (node_id, tx_type, amount, balance_after, tx_hash, "
              "prev_hash, ts) VALUES (?,?,?,?,?,?,?)");
  if (!i.ok()) { err = i.error(); return false; }
  i.text(1, node_id).text(2, tx_type).dbl(3, amount).dbl(4, after).text(5, hash)
      .text(6, prev_hash).i64(7, now);
  if (!i.run()) { err = i.error(); return false; }
  return true;
}

bool Store::credits(std::vector<CreditRow>& out, size_t limit, std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  Stmt q(db_, "SELECT tx_id, node_id, tx_type, amount, balance_after, tx_hash, prev_hash, ts "
              "FROM credit_ledger ORDER BY tx_id DESC LIMIT ?");
  if (!q.ok()) { err = q.error(); return false; }
  q.i64(1, static_cast<int64_t>(limit));
  while (q.row()) {
    CreditRow c;
    c.tx_id = q.col_i64(0);
    c.node_id = q.col_text(1);
    c.tx_type = q.col_text(2);
    c.amount = q.col_dbl(3);
    c.balance_after = q.col_dbl(4);
    c.tx_hash = q.col_text(5);
    c.prev_hash = q.col_text(6);
    c.ts = q.col_i64(7);
    out.push_back(std::move(c));
  }
  return true;
}

bool Store::credit_leaderboard(std::vector<std::pair<std::string, double>>& out,
                               std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  Stmt q(db_, "SELECT node_id, SUM(amount) AS total FROM credit_ledger GROUP BY node_id "
              "ORDER BY total DESC");
  if (!q.ok()) { err = q.error(); return false; }
  while (q.row()) out.emplace_back(q.col_text(0), q.col_dbl(1));
  return true;
}

bool Store::vacuum_stale_sessions(int64_t ttl_s, std::string& err) {
  std::lock_guard<std::mutex> lock(mu_);
  Stmt u(db_, "UPDATE active_nodes SET session_token='' WHERE session_token<>'' AND "
              "last_heartbeat_ts<?");
  if (!u.ok()) { err = u.error(); return false; }
  u.i64(1, now_s() - ttl_s);
  if (!u.run()) { err = u.error(); return false; }
  return true;
}

}  // namespace distribai::grid
