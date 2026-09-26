// suite_store.cpp - unit tests for the native grid's SQLite store.
//
// Every grid decision lands in one of four tables: active_nodes, jobs, tasks,
// credit_ledger. The store is the only place that writes them, so a bug here
// shows up as lost credits, a job that never finalizes, or a task stuck
// assigned forever. This category drives the real store against a real SQLite
// file and pins the contracts the coordinator relies on:
//
//   * open: missing schema and unwritable paths fail loudly
//   * nodes: registration, token rotation, heartbeat, listing order, counts
//   * expiry: an offline node's task goes back to the queue, or to failed once
//     it is out of attempts
//   * jobs and tasks: creation, claim order and priority, step-based progress,
//     finish, requeue, and the one-winner finalization claim
//   * credits: chained ledger rows, leaderboard totals, per-node chains
//   * migrations: a database created before jobs.aggregate existed is patched up
//   * transactions: a failed create leaves no half-written task rows
//   * persistence: reopening the same file keeps nodes, jobs and tokens
//
// Run: build/cpp_port/suite_store   (or: make -C tools/cpp_port suite-store)
#include <sqlite3.h>
#include <stdlib.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "../framework.hpp"
#include "grid/store.hpp"

using distribai::grid::CreditRow;
using distribai::grid::JobRow;
using distribai::grid::NodeRow;
using distribai::grid::Store;
using distribai::grid::TaskCounts;
using distribai::grid::TaskRow;

namespace {

// The suite runs from the repo root, but `make suite-store` runs from
// tools/cpp_port, so hunt upward for the schema before giving up.
std::string g_schema = "runtime/db/schema.sql";

std::string find_schema() {
  std::string prefix;
  for (int depth = 0; depth < 6; ++depth) {
    const std::string candidate = prefix + "runtime/db/schema.sql";
    std::ifstream in(candidate);
    if (in.good()) return candidate;
    prefix += "../";
  }
  return "runtime/db/schema.sql";  // let open() report the real error
}

std::string tmp_dir() {
  const char* base = std::getenv("TMPDIR");
  std::string tpl = std::string(base ? base : "/tmp") + "/distribai_store_XXXXXX";
  std::vector<char> buf(tpl.begin(), tpl.end());
  buf.push_back('\0');
  char* made = mkdtemp(buf.data());
  return made ? std::string(made) : std::string("/tmp");
}

// Opens the store on the repo schema. Every section uses its own database file
// so a section never reads another section's counters.
bool open_store(Store& s, const std::string& db, std::string& err) {
  return s.open(db, g_schema, err);
}

bool is_hex(const std::string& s) {
  if (s.empty()) return false;
  for (char c : s) {
    const bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    if (!hex) return false;
  }
  return true;
}

// Builds a job with `replicas` tasks named <id>-r0, <id>-r1, ...
void make_bundle(JobRow& job, std::vector<TaskRow>& tasks, const std::string& id, int replicas,
                 int64_t steps, int64_t priority = 0, const std::string& aggregate = "trimmed_mean",
                 int64_t max_attempts = 3) {
  job = JobRow{};
  job.job_id = id;
  job.model_name = "m-" + id;
  job.status = "queued";
  job.priority = priority;
  job.steps = steps;
  job.batch_size = 8;
  job.total_steps = steps * replicas;
  job.aggregate = aggregate;
  tasks.clear();
  for (int i = 0; i < replicas; ++i) {
    TaskRow t;
    t.task_id = id + "-r" + std::to_string(i);
    t.job_id = id;
    t.steps = steps;
    t.step_offset = i * steps;
    t.max_attempts = max_attempts;
    t.hparams_json = "{}";
    tasks.push_back(t);
  }
}

}  // namespace

int main() {
  g_schema = find_schema();
  const std::string dir = tmp_dir();
  std::string err;

  // -------------------------------------------------------------------------
  suite::section("open");
  {
    Store s;
    CHECK(!s.open(dir + "/x.db", "runtime/db/does-not-exist.sql", err),
          "open rejects a missing schema file");
    CHECK(err.find("schema not found") != std::string::npos,
          "the missing-schema error names the file");
    CHECK(err.find("does-not-exist.sql") != std::string::npos,
          "the missing-schema error includes the path");
  }
  {
    Store s;
    err.clear();
    CHECK(!s.open(dir + "/no/such/dir/grid.db", g_schema, err),
          "open rejects a path whose directory does not exist");
    CHECK(!err.empty(), "the unwritable-path error is reported");
  }

  // -------------------------------------------------------------------------
  suite::section("nodes, sessions and credits");
  {
    const std::string db = dir + "/a.db";
    Store s;
    err.clear();
    CHECK(open_store(s, db, err), "open creates a fresh grid database");
    CHECK(err.empty(), "a clean open leaves no error");

    NodeRow n1;
    n1.node_id = "alpha";
    n1.hardware_json = "{\"cpu\":8}";
    CHECK(s.upsert_node(n1, err), "first registration reports a new node");
    CHECK(n1.session_token.size() == 32, "the session token is 32 hex characters");
    CHECK(is_hex(n1.session_token), "the session token is hex only");
    CHECK(n1.created_ts > 0, "registration stamps created_ts");
    const std::string first_token = n1.session_token;
    const int64_t first_created = n1.created_ts;

    NodeRow n1b;
    n1b.node_id = "alpha";
    n1b.hardware_json = "{\"cpu\":8}";
    CHECK(!s.upsert_node(n1b, err), "re-registration reports an existing node");
    CHECK(n1b.session_token != first_token, "re-registration rotates the session token");
    CHECK(n1b.created_ts == first_created, "re-registration keeps created_ts");

    CHECK(s.session_valid("alpha", n1b.session_token, err), "the fresh token validates");
    CHECK(!s.session_valid("alpha", first_token, err), "the rotated-out token is rejected");
    CHECK(!s.session_valid("alpha", "wrong-token", err), "a wrong token is rejected");
    CHECK(!s.session_valid("alpha", "", err), "an empty token is rejected");
    CHECK(!s.session_valid("nobody", n1b.session_token, err), "an unknown node id is rejected");

    CHECK(s.heartbeat("alpha", "busy", "t-1", 0.5, err), "heartbeat succeeds");
    NodeRow read;
    bool found = false;
    CHECK(s.node("alpha", read, found, err), "node lookup succeeds");
    CHECK(found, "the registered node is found");
    CHECK(read.status == "busy", "heartbeat stored the status");
    CHECK(read.current_task_id == "t-1", "heartbeat stored the current task");
    CHECK(read.hardware_json == "{\"cpu\":8}", "node lookup returns hardware json");

    NodeRow missing;
    found = true;
    CHECK(s.node("ghost", missing, found, err), "lookup of an unknown node is not an error");
    CHECK(!found, "an unknown node is reported as missing");

    // One second so the listing order is unambiguous.
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    NodeRow n2;
    n2.node_id = "beta";
    CHECK(s.upsert_node(n2, err), "a second node registers");
    std::vector<NodeRow> nodes;
    CHECK(s.nodes(nodes, err), "the node list loads");
    CHECK(nodes.size() == 2, "both nodes are listed");
    CHECK(nodes.size() == 2 && nodes[0].node_id == "beta",
          "the list is newest heartbeat first");

    int64_t online = 0, jobs_total = 0, jobs_running = 0, jobs_done = 0, queued = 0;
    CHECK(s.counts(online, jobs_total, jobs_running, jobs_done, queued, err), "counts load");
    CHECK(online == 2, "counts sees two online nodes");

    // Credits, one hash-chained ledger per node.
    CHECK(s.add_credit("alpha", "job_reward", 10.0, err), "the first credit is recorded");
    CHECK(s.add_credit("alpha", "job_reward", 5.0, err), "a second credit is recorded");
    CHECK(s.add_credit("beta", "job_reward", 7.0, err), "a credit on another node is recorded");
    std::vector<CreditRow> ledger;
    CHECK(s.credits(ledger, 10, err), "the ledger loads");
    CHECK(ledger.size() == 3, "the ledger holds three rows");
    const bool have_three = ledger.size() == 3;
    CHECK(have_three && ledger[0].tx_id == 3, "the ledger is newest first");
    CHECK(have_three && ledger[0].node_id == "beta", "the newest row names its node");
    CHECK(have_three && ledger[0].prev_hash.empty(), "a node's first credit starts a fresh chain");
    CHECK(have_three && ledger[0].tx_hash.size() == 8, "the transaction hash is eight characters");
    CHECK(have_three && is_hex(ledger[0].tx_hash), "the transaction hash is hex");
    CHECK(have_three && ledger[0].balance_after == 7.0, "the balance starts at the first amount");
    CHECK(have_three && ledger[1].node_id == "alpha", "the middle row belongs to alpha");
    CHECK(have_three && ledger[1].amount == 5.0, "the middle row holds the second amount");
    CHECK(have_three && ledger[1].balance_after == 15.0, "the balance accumulates");
    CHECK(have_three && ledger[1].prev_hash == ledger[2].tx_hash,
          "each row chains to the one before");
    CHECK(have_three && ledger[2].prev_hash.empty(),
          "the first row of a chain has no previous hash");
    CHECK(have_three && ledger[2].balance_after == 10.0, "the first balance is its amount");

    std::vector<std::pair<std::string, double>> board;
    CHECK(s.credit_leaderboard(board, err), "the leaderboard loads");
    CHECK(board.size() == 2, "the leaderboard has one row per node");
    const bool have_two = board.size() == 2;
    CHECK(have_two && board[0].first == "alpha", "the leaderboard is biggest first");
    CHECK(have_two && board[0].second == 15.0, "the leaderboard sums each node");
    CHECK(have_two && board[1].second == 7.0, "the second total is summed too");

    // Expiry: claim a task as alpha, then let the node go quiet.
    JobRow job;
    std::vector<TaskRow> tasks;
    make_bundle(job, tasks, "exp", 2, 10);
    CHECK(s.create_job(job, tasks, err), "a job is created for the expiry check");
    TaskRow claimed;
    bool got = false;
    CHECK(s.claim_task("alpha", 30, claimed, got, err), "claim succeeds");
    CHECK(got, "alpha gets the first queued task");
    CHECK(s.claim_task("alpha", 30, claimed, got, err), "second claim succeeds");
    CHECK(got, "alpha gets the second queued task");

    int64_t requeued = 0, failed = 0;
    CHECK(s.expire_nodes(-10, requeued, failed, err), "expiry runs");
    CHECK(requeued == 2, "both assigned tasks return to the queue");
    CHECK(failed == 0, "nothing fails on the first expiry");
    CHECK(s.node("alpha", read, found, err) && read.status == "offline",
          "the quiet node is marked offline");
    found = false;
    TaskRow back;
    CHECK(s.task("exp-r0", back, found, err), "the requeued task loads");
    CHECK(found, "the requeued task still exists");
    CHECK(back.status == "queued", "the requeued task is queued again");
    CHECK(back.assignee_node_id.empty(), "the requeued task lost its assignee");
    CHECK(back.last_error == "assignee went offline", "the requeue reason is recorded");

    online = 99;
    CHECK(s.counts(online, jobs_total, jobs_running, jobs_done, queued, err), "counts reload");
    CHECK(online == 0, "an expired node stops counting as online");

    CHECK(s.vacuum_stale_sessions(-1, err), "stale sessions are vacuumed");
    CHECK(!s.session_valid("alpha", n1b.session_token, err),
          "a vacuumed node can no longer authenticate");

    // Drain the requeued work so the next claim has exactly one candidate.
    TaskRow again;
    for (int i = 0; i < 2; ++i) {
      bool spent = false;
      CHECK(s.claim_task("alpha", 30, again, spent, err), "a requeued task is reclaimed");
      CHECK(spent, "requeued work is handed out again");
      CHECK(s.finish_task(again.task_id, true, "{}", "", "/tmp/x.env", err),
            "a requeued task finishes on the second try");
    }

    // Ran out of attempts: the next expiry fails the task instead of requeueing.
    JobRow hard;
    std::vector<TaskRow> ht;
    make_bundle(hard, ht, "hard", 1, 10, 0, "mean", 1);
    CHECK(s.create_job(hard, ht, err), "a one-attempt job is created");
    CHECK(s.claim_task("beta", 30, claimed, got, err) && got,
          "beta claimed the one-attempt task");
    CHECK(claimed.task_id == "hard-r0", "the only queued task is the one-attempt task");
    requeued = failed = 0;
    CHECK(s.expire_nodes(-10, requeued, failed, err), "second expiry runs");
    CHECK(failed == 1, "a task out of attempts fails");
    CHECK(requeued == 0, "a task out of attempts is not requeued");
    found = false;
    CHECK(s.task("hard-r0", back, found, err) && found && back.status == "failed",
          "the failed task carries the failed status");

    CHECK(distribai::grid::random_token(16) != distribai::grid::random_token(16),
          "random tokens differ");
    CHECK(distribai::grid::random_token(0).empty(), "a zero-length token is empty");
    CHECK(is_hex(distribai::grid::random_token(4)), "random_token output is hex");
  }

  // -------------------------------------------------------------------------
  suite::section("jobs, tasks, progress and finalization");
  {
    const std::string db = dir + "/b.db";
    Store s;
    CHECK(open_store(s, db, err), "second database opens");

    NodeRow node;
    node.node_id = "node-b";
    CHECK(s.upsert_node(node, err), "a worker registers on the second database");

    JobRow job;
    std::vector<TaskRow> tasks;
    make_bundle(job, tasks, "job1", 3, 100, 0, "mean");
    CHECK(s.create_job(job, tasks, err), "create_job writes the job and its tasks");
    CHECK(job.created_ts > 0, "create_job stamps created_ts");

    JobRow loaded;
    bool found = false;
    CHECK(s.job("job1", loaded, found, err), "the job loads");
    CHECK(found, "the job is found");
    CHECK(loaded.status == "queued", "a new job is queued");
    CHECK(loaded.aggregate == "mean", "the job keeps its aggregate method");
    CHECK(loaded.steps == 100, "the job keeps its step count");
    CHECK(loaded.batch_size == 8, "the job keeps its batch size");

    JobRow ghost;
    found = true;
    CHECK(s.job("nope", ghost, found, err), "an unknown job lookup is not an error");
    CHECK(!found, "an unknown job is reported missing");

    std::vector<TaskRow> job_tasks;
    CHECK(s.tasks_for_job("job1", job_tasks, err), "the job's tasks load");
    CHECK(job_tasks.size() == 3, "the job has three tasks");
    const bool three_tasks = job_tasks.size() == 3;
    CHECK(three_tasks && job_tasks[0].task_id == "job1-r0", "tasks come back in id order");
    CHECK(three_tasks && job_tasks[2].task_id == "job1-r2", "the last task is present");
    for (const auto& t : job_tasks) {
      CHECK(t.status == "queued", "each new task starts queued");
    }

    TaskCounts counts;
    CHECK(s.task_counts("job1", counts, err), "task counts load");
    CHECK(counts.queued == 3 && counts.total == 3, "three tasks wait in the queue");

    TaskRow t;
    bool got = false;
    CHECK(s.claim_task("node-b", 60, t, got, err), "the first task is claimed");
    CHECK(got, "a queued task was available");
    CHECK(t.task_id == "job1-r0", "claim hands out the oldest task");
    CHECK(t.status == "assigned", "a claimed task is assigned");
    CHECK(t.assignee_node_id == "node-b", "the assignee is recorded");
    CHECK(t.attempt_count == 1, "claiming counts an attempt");
    CHECK(t.deadline_ts > distribai::grid::now_s(), "claim sets a future deadline");

    CHECK(s.job("job1", loaded, found, err), "the job reloads after a claim");
    CHECK(loaded.status == "running", "a job with a claim is running");
    CHECK(loaded.started_ts > 0, "the first claim stamps started_ts");
    CHECK(loaded.latest_task_id == "job1-r0", "the job names its latest task");
    CHECK(loaded.progress_pct == 0.0, "an unfinished job reports no progress");

    counts = TaskCounts{};
    CHECK(s.task_counts("job1", counts, err), "counts reload after a claim");
    CHECK(counts.queued == 2, "two tasks stay queued");
    CHECK(counts.assigned == 1, "one task is assigned");

    TaskRow t2;
    CHECK(s.claim_task("node-b", 60, t2, got, err) && got, "the second task is claimed");
    CHECK(t2.task_id == "job1-r1", "the second claim takes the next task");

    CHECK(s.finish_task("job1-r0", true, "{\"loss\":0.5}", "", "/tmp/t0.env", err),
          "a successful task finishes");
    found = false;
    CHECK(s.task("job1-r0", t, found, err) && found, "the finished task loads");
    CHECK(t.status == "done", "a successful task is done");
    CHECK(t.output_json == "{\"loss\":0.5}", "the result json is stored");
    CHECK(t.completed_ts > 0, "a finished task is stamped");
    {
      std::vector<TaskRow> after;
      CHECK(s.tasks_for_job("job1", after, err), "the job's tasks reload after a finish");
      bool blob_seen = false;
      for (const auto& row : after) {
        if (row.task_id == "job1-r0") blob_seen = row.gradient_blob_url == "/tmp/t0.env";
      }
      CHECK(blob_seen, "the envelope path is stored on the task row");
    }

    CHECK(s.job("job1", loaded, found, err), "the job reloads after a finish");
    CHECK(suite::near(loaded.progress_pct, 33.33, 0.02), "progress counts finished steps");
    CHECK(loaded.current_step == 100, "the job reports the finished steps");
    CHECK(loaded.total_steps == 300, "the job reports the total steps");

    NodeRow nread;
    CHECK(s.node("node-b", nread, found, err), "the contributing node loads");
    CHECK(nread.jobs_completed == 1, "a finished task pays the node a completion");
    CHECK(nread.jobs_failed == 0, "a success is not counted as a failure");

    CHECK(s.finish_task("job1-r1", false, "", "crashed", "", err), "a failed task finishes");
    CHECK(s.task("job1-r1", t, found, err) && t.status == "failed", "the task is failed");
    CHECK(t.last_error == "crashed", "the failure reason is stored");
    CHECK(s.job("job1", loaded, found, err) && suite::near(loaded.progress_pct, 33.33, 0.02),
          "a failed task does not advance progress");
    CHECK(s.node("node-b", nread, found, err), "the node reloads after a failure");
    CHECK(nread.jobs_failed == 1, "the failure is counted");
    CHECK(suite::near(nread.reliability_score, 0.95, 1e-9), "a failure lowers reliability");

    err.clear();
    CHECK(!s.finish_task("ghost-task", true, "", "", "", err), "finishing an unknown task fails");
    CHECK(err.find("unknown task") != std::string::npos, "the unknown-task error is clear");

    found = true;
    TaskRow absent;
    CHECK(s.task("ghost-task", absent, found, err), "an unknown task lookup is not an error");
    CHECK(!found, "an unknown task is reported missing");

    CHECK(s.requeue_task("job1-r2", "worker gave up", err), "a queued task can be requeued");
    found = false;
    CHECK(s.task("job1-r2", t, found, err) && found, "the requeued task loads");
    CHECK(t.status == "queued", "a task under its attempt limit is queued again");
    CHECK(t.last_error == "worker gave up", "the requeue reason is stored");
    CHECK(t.assignee_node_id.empty(), "the requeue clears the assignee");

    std::vector<JobRow> jobs;
    CHECK(s.jobs(jobs, 10, err), "the job list loads");
    CHECK(jobs.size() == 1, "one job is listed");
    CHECK(jobs.size() == 1 && jobs[0].aggregate == "mean",
          "the job list carries the aggregate");
    std::vector<JobRow> none;
    CHECK(s.jobs(none, 0, err), "a zero limit is accepted");
    CHECK(none.empty(), "a zero limit returns nothing");

    std::vector<TaskRow> some;
    CHECK(s.tasks(some, 2, err), "the task list honours a limit");
    CHECK(some.size() == 2, "the task limit is applied");

    // Finalization is a race between the last reporter and the maintenance
    // thread: exactly one of them may claim it.
    bool claimed = false;
    CHECK(s.claim_finalization("job1", claimed, err), "the finalization claim runs");
    CHECK(claimed, "the first caller wins the finalization");
    CHECK(s.job("job1", loaded, found, err) && loaded.status == "finalizing",
          "the job moves to finalizing");
    claimed = true;
    CHECK(s.claim_finalization("job1", claimed, err), "a second finalization claim runs");
    CHECK(!claimed, "the second caller loses the finalization");
    claimed = true;
    CHECK(s.claim_finalization("ghost-job", claimed, err), "claiming a missing job is not an error");
    CHECK(!claimed, "a missing job cannot be finalized");

    // A cancelled job must not be finalizable or claimable.
    JobRow cancelled;
    std::vector<TaskRow> ct;
    make_bundle(cancelled, ct, "cancel-job", 1, 10);
    CHECK(s.create_job(cancelled, ct, err), "a job to cancel is created");
    cancelled.status = "cancelled";
    CHECK(s.update_job(cancelled, err), "the job is cancelled");
    CHECK(s.job("cancel-job", loaded, found, err) && loaded.status == "cancelled",
          "the cancellation is stored");
    claimed = true;
    CHECK(s.claim_finalization("cancel-job", claimed, err), "finalization of a cancelled job runs");
    CHECK(!claimed, "a cancelled job cannot be finalized");
  }

  // -------------------------------------------------------------------------
  suite::section("claim ordering and cancelled work");
  {
    const std::string db = dir + "/f.db";
    Store s;
    CHECK(open_store(s, db, err), "third database opens");
    NodeRow node;
    node.node_id = "node-f";
    CHECK(s.upsert_node(node, err), "a worker registers on the third database");

    // Only a cancelled job: nothing should be handed out.
    JobRow cancelled;
    std::vector<TaskRow> ct;
    make_bundle(cancelled, ct, "only-cancelled", 2, 10);
    CHECK(s.create_job(cancelled, ct, err), "the cancelled job is created");
    cancelled.status = "cancelled";
    CHECK(s.update_job(cancelled, err), "the cancelled job is marked");

    // A live job with lower priority, created first.
    JobRow live;
    std::vector<TaskRow> lt;
    make_bundle(live, lt, "live", 1, 10, 1);
    CHECK(s.create_job(live, lt, err), "a live job is created");

    TaskRow t;
    bool got = true;
    CHECK(s.claim_task("node-f", 30, t, got, err), "claim runs with cancelled work present");
    CHECK(got, "the live job has work to hand out");
    CHECK(t.task_id == "live-r0", "cancelled tasks are skipped");

    // Priority wins over arrival order.
    JobRow high;
    std::vector<TaskRow> ht;
    make_bundle(high, ht, "high", 1, 10, 9);
    CHECK(s.create_job(high, ht, err), "a high-priority job is created");
    got = false;
    CHECK(s.claim_task("node-f", 30, t, got, err) && got, "the next claim finds work");
    CHECK(t.task_id == "high-r0", "priority beats arrival order");

    // Both queued tasks are out now, so the queue is empty.
    got = true;
    CHECK(s.claim_task("node-f", 30, t, got, err), "claiming an empty queue is not an error");
    CHECK(!got, "an empty queue reports nothing found");

    // update_job round trip.
    JobRow upd;
    bool found = false;
    CHECK(s.job("live", upd, found, err) && found, "the live job loads");
    upd.progress_pct = 42.5;
    upd.current_step = 85;
    upd.total_steps = 200;
    upd.attempts = 2;
    upd.latest_reason = "halfway";
    CHECK(s.update_job(upd, err), "update_job writes the row");
    JobRow back;
    CHECK(s.job("live", back, found, err) && found, "the updated job reloads");
    CHECK(suite::near(back.progress_pct, 42.5, 1e-9), "update_job stored the progress");
    CHECK(back.current_step == 85, "update_job stored the current step");
    CHECK(back.attempts == 2, "update_job stored the attempts");
    CHECK(back.latest_reason == "halfway", "update_job stored the reason");
  }

  // -------------------------------------------------------------------------
  suite::section("transactions and duplicate keys");
  {
    const std::string db = dir + "/e.db";
    Store s;
    CHECK(open_store(s, db, err), "the transaction database opens");
    JobRow job;
    std::vector<TaskRow> tasks;
    make_bundle(job, tasks, "dup", 1, 10);
    CHECK(s.create_job(job, tasks, err), "the first job is created");

    JobRow clash;
    std::vector<TaskRow> clash_tasks;
    make_bundle(clash, clash_tasks, "dup", 2, 10);
    clash_tasks[0].task_id = "evil-0";
    clash_tasks[1].task_id = "evil-1";
    err.clear();
    CHECK(!s.create_job(clash, clash_tasks, err), "a duplicate job id is refused");
    CHECK(!err.empty(), "the duplicate-key error is reported");
    bool found = true;
    TaskRow t;
    CHECK(s.task("evil-0", t, found, err), "looking up a rolled-back task is not an error");
    CHECK(!found, "the failed create left no task rows behind");
  }

  // -------------------------------------------------------------------------
  suite::section("migration and persistence");
  {
    const std::string db = dir + "/c.db";
    // A database from before jobs.aggregate existed. The schema uses
    // CREATE TABLE IF NOT EXISTS, so only the ALTER pass can add the column.
    sqlite3* raw = nullptr;
    CHECK(sqlite3_open(db.c_str(), &raw) == SQLITE_OK, "a legacy database is created by hand");
    // The real schema as it stood before the aggregate column, so the only
    // difference the migration has to repair is that one column.
    const char* legacy =
        "CREATE TABLE jobs (job_id TEXT PRIMARY KEY, model_name TEXT NOT NULL, "
        "job_type TEXT DEFAULT 'fine_tune', base_model TEXT, dataset_ref TEXT, "
        "description TEXT, status TEXT NOT NULL DEFAULT 'queued', priority INTEGER DEFAULT 0, "
        "priority_tier TEXT DEFAULT 'P1', total_votes INTEGER DEFAULT 0, "
        "vote_weight REAL DEFAULT 1.0, submitter_id TEXT DEFAULT 'distribai', "
        "org TEXT DEFAULT 'DistribAI', created_ts INTEGER NOT NULL, "
        "updated_ts INTEGER NOT NULL, started_ts INTEGER, completed_ts INTEGER, "
        "steps INTEGER DEFAULT 100, batch_size INTEGER DEFAULT 32, "
        "queue_position INTEGER, estimated_start_hours REAL, "
        "active_nodes INTEGER DEFAULT 0, progress_pct REAL DEFAULT 0, "
        "current_step INTEGER DEFAULT 0, total_steps INTEGER DEFAULT 100, "
        "attempts INTEGER DEFAULT 0, latest_task_id TEXT, latest_reason TEXT);"
        "INSERT INTO jobs (job_id, model_name, created_ts, updated_ts) "
        "VALUES ('old-job','m',1,1);";
    char* msg = nullptr;
    CHECK(sqlite3_exec(raw, legacy, nullptr, nullptr, &msg) == SQLITE_OK,
          "the legacy jobs table is written");
    if (msg) sqlite3_free(msg);
    sqlite3_close(raw);

    std::string token;
    {
      Store s;
      CHECK(open_store(s, db, err), "the store opens a pre-aggregate database");
      JobRow old_job;
      bool found = false;
      CHECK(s.job("old-job", old_job, found, err), "the legacy row loads");
      CHECK(found, "the legacy row survived the migration");
      CHECK(old_job.aggregate == "trimmed_mean", "the new column defaults for old rows");

      JobRow fresh;
      std::vector<TaskRow> fresh_tasks;
      make_bundle(fresh, fresh_tasks, "new-job", 1, 10, 0, "median");
      CHECK(s.create_job(fresh, fresh_tasks, err), "a job is created after the migration");
      JobRow stored;
      CHECK(s.job("new-job", stored, found, err) && found, "the new job loads");
      CHECK(stored.aggregate == "median", "the migrated column stores a non-default method");

      NodeRow node;
      node.node_id = "persist-node";
      CHECK(s.upsert_node(node, err), "a node is written before the reopen");
      token = node.session_token;
    }

    // Persistence: a second store on the same file sees everything.
    {
      Store s;
      CHECK(open_store(s, db, err), "the same database reopens");
      NodeRow reread;
      bool found = false;
      CHECK(s.node("persist-node", reread, found, err), "the node loads after reopen");
      CHECK(found, "the node survived the reopen");
      CHECK(s.session_valid("persist-node", token, err), "the session survives the reopen");
      JobRow stored;
      CHECK(s.job("new-job", stored, found, err) && found, "the job survives the reopen");
      CHECK(stored.aggregate == "median", "the aggregate survives the reopen");
    }
  }

  return suite::finish("store");
}
