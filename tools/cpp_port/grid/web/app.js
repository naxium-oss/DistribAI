/* app.js - the native grid dashboard.
 *
 * Polls /v1/summary every few seconds and rebuilds the tables. Everything is
 * written with textContent rather than innerHTML, so a node id or error string
 * coming from a worker can never inject markup. Polling pauses while the tab is
 * hidden and resumes on focus.
 */

(function () {
  "use strict";

  var REFRESH_MS = 3000;
  var summary = null;
  var lastOk = 0;
  var timer = null;

  var $ = function (id) { return document.getElementById(id); };

  /* ---- helpers ---- */

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function cell(label, content) {
    var td = el("td", null);
    td.setAttribute("data-label", label);
    if (content instanceof Node) {
      td.appendChild(content);
    } else {
      td.textContent = content === null || content === undefined ? "" : String(content);
    }
    return td;
  }

  function tag(text, kind) {
    return el("span", "tag" + (kind ? " tag--" + kind : ""), text);
  }

  function num(value, digits) {
    var n = Number(value);
    if (!isFinite(n)) return "0";
    return n.toFixed(digits === undefined ? 0 : digits);
  }

  function statusKind(status) {
    if (status === "idle" || status === "done" || status === "completed") return "ok";
    if (status === "busy" || status === "running" || status === "assigned") return "busy";
    if (status === "failed" || status === "offline" || status === "cancelled") return "bad";
    return "warn";
  }

  function ago(seconds) {
    var s = Number(seconds);
    if (!isFinite(s) || s < 0) return "unknown";
    if (s < 2) return "just now";
    if (s < 60) return num(s) + "s ago";
    if (s < 3600) return num(s / 60, 1) + "m ago";
    return num(s / 3600, 1) + "h ago";
  }

  function duration(seconds) {
    var s = Number(seconds);
    if (!isFinite(s) || s < 0) return "unknown";
    if (s < 60) return num(s, 1) + "s";
    var m = Math.floor(s / 60);
    if (m < 60) return m + "m " + num(s % 60) + "s";
    return Math.floor(m / 60) + "h " + (m % 60) + "m";
  }

  function replace(tbody, rows, emptyNode) {
    tbody.textContent = "";
    rows.forEach(function (row) { tbody.appendChild(row); });
    if (emptyNode) emptyNode.hidden = rows.length > 0;
  }

  function progress(value) {
    var wrap = el("div");
    var bar = el("div", "bar");
    var fill = el("div", "bar__fill");
    var pct = Math.max(0, Math.min(100, Number(value) || 0));
    fill.style.width = pct + "%";
    bar.appendChild(fill);
    wrap.appendChild(bar);
    wrap.appendChild(el("span", "bar__label", num(pct, 1) + "%"));
    return wrap;
  }

  /* ---- rendering ---- */

  function renderTotals(data) {
    var nodes = data.nodes || [];
    var jobs = data.jobs || [];
    var running = jobs.filter(function (j) { return j.status === "running" || j.status === "queued"; });
    var live = nodes.filter(function (n) { return n.status !== "offline"; });

    $("stat-nodes").textContent = num(data.nodes_online);
    $("stat-nodes-note").textContent = live.length === nodes.length
      ? (live.length ? "all healthy" : "none yet")
      : (nodes.length - live.length) + " offline";

    $("stat-running").textContent = num(data.jobs_running);
    $("stat-running-note").textContent = running.length
      ? (running.length === 1 ? "1 in the queue" : running.length + " in the queue")
      : "queue empty";

    $("stat-done").textContent = num(data.jobs_done);
    $("stat-done-note").textContent = Number(data.jobs_total) > Number(data.jobs_done)
      ? (Number(data.jobs_total) - Number(data.jobs_done)) + " unfinished"
      : (Number(data.jobs_total) ? "all finished" : "no finished jobs");

    $("stat-queued").textContent = num(data.tasks_queued);
    $("stat-queued-note").textContent = Number(data.tasks_queued) > 0
      ? "waiting for a worker"
      : "nothing waiting";

    var board = data.leaderboard || [];
    var total = board.reduce(function (sum, row) { return sum + (Number(row.credits) || 0); }, 0);
    $("stat-credits").textContent = num(total, total % 1 ? 1 : 0);
    $("stat-credits-note").textContent = board.length
      ? board.length + (board.length === 1 ? " contributor" : " contributors")
      : "ledger empty";

    $("ttl").textContent = num(data.node_ttl_s);
    $("footer-meta").textContent = "Coordinator up " + duration(data.uptime_s) +
      ", aggregate " + (data.aggregate || "trimmed_mean") +
      (data.invite_required ? ", invite code required" : ", open registration") + ".";
  }

  function renderWorkers(data) {
    var rows = (data.nodes || []).map(function (n) {
      var tr = el("tr");
      tr.appendChild(cell("Node", el("span", "mono", n.node_id)));
      tr.appendChild(cell("Status", tag(n.status, statusKind(n.status))));
      tr.appendChild(cell("Task", n.current_task_id ? el("span", "mono", n.current_task_id) : "none"));
      tr.appendChild(cell("Done", num(n.jobs_completed)));
      tr.appendChild(cell("Failed", num(n.jobs_failed)));
      tr.appendChild(cell("Reliability", num(n.reliability_score, 2)));
      tr.appendChild(cell("Last seen", ago(n.seconds_since_heartbeat)));
      if (n.hardware && n.hardware.cpus) {
        tr.title = n.hardware.cpus + " cpus, " + num(n.hardware.mem_mb) + " MiB RAM";
      }
      return tr;
    });
    replace($("workers-table").querySelector("tbody"), rows, $("workers-empty"));
  }

  function renderJobs(data) {
    var rows = (data.jobs || []).map(function (j) {
      var tr = el("tr");
      tr.appendChild(cell("Job", el("span", "mono", j.job_id)));
      tr.appendChild(cell("Model", j.model_name));
      tr.appendChild(cell("Status", tag(j.status, statusKind(j.status))));
      var progressCell = el("div");
      progressCell.appendChild(progress(j.progress_pct));
      progressCell.appendChild(el("span", "bar__label",
        num(j.current_step) + " of " + num(j.total_steps) + " steps"));
      tr.appendChild(cell("Progress", progressCell));
      tr.appendChild(cell("Replicas", num(j.done) + " done, " + num(j.failed) + " failed, " +
        num(j.active_nodes) + " active"));
      tr.appendChild(cell("Aggregate", j.aggregate + (j.reason ? " (" + j.reason + ")" : "")));
      return tr;
    });
    replace($("jobs-table").querySelector("tbody"), rows, $("jobs-empty"));
  }

  function renderLedger(data) {
    var rows = (data.credits || []).map(function (c) {
      var tr = el("tr");
      tr.appendChild(cell("Node", el("span", "mono", c.node_id)));
      tr.appendChild(cell("Type", c.type));
      tr.appendChild(cell("Amount", num(c.amount, 3)));
      tr.appendChild(cell("Balance", num(c.balance_after, 3)));
      tr.appendChild(cell("Hash", el("span", "mono", (c.tx_hash || "").slice(0, 12))));
      return tr;
    });
    replace($("ledger-table").querySelector("tbody"), rows, $("ledger-empty"));
  }

  function renderBoard(data) {
    var rows = (data.leaderboard || []).map(function (b) {
      var tr = el("tr");
      tr.appendChild(cell("Node", el("span", "mono", b.node_id)));
      tr.appendChild(cell("Credits", num(b.credits, 3)));
      return tr;
    });
    replace($("board-table").querySelector("tbody"), rows, $("board-empty"));
  }

  function setLinkState(state, text) {
    var pill = $("link-state");
    pill.dataset.state = state;
    pill.textContent = text;
  }

  /* ---- polling ---- */

  function render(data) {
    summary = data;
    renderTotals(data);
    renderWorkers(data);
    renderJobs(data);
    renderLedger(data);
    renderBoard(data);
  }

  function load() {
    fetch("/v1/summary", { headers: { Accept: "application/json" }, cache: "no-store" })
      .then(function (res) {
        if (!res.ok) throw new Error("HTTP " + res.status);
        return res.json();
      })
      .then(function (data) {
        lastOk = Date.now();
        setLinkState("ok", "live");
        render(data);
      })
      .catch(function () {
        var stale = Date.now() - lastOk;
        setLinkState(stale > 15000 ? "down" : "stale",
          stale > 15000 ? "coordinator unreachable" : "reconnecting");
      });
  }

  function start() {
    if (timer) clearInterval(timer);
    timer = setInterval(load, REFRESH_MS);
  }

  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
  }

  /* ---- theme ---- */

  function applyTheme(name) {
    document.documentElement.setAttribute("data-theme", name);
    try { localStorage.setItem("distribai-theme", name); } catch (e) { /* private mode */ }
  }

  function initTheme() {
    var saved = null;
    try { saved = localStorage.getItem("distribai-theme"); } catch (e) { saved = null; }
    applyTheme(saved || "auto");
    $("theme").addEventListener("click", function () {
      var order = ["auto", "light", "dark"];
      var current = document.documentElement.getAttribute("data-theme") || "auto";
      applyTheme(order[(order.indexOf(current) + 1) % order.length]);
    });
  }

  /* ---- boot ---- */

  document.addEventListener("DOMContentLoaded", function () {
    $("endpoint").textContent = location.origin + "  |  proto 1";
    initTheme();
    $("refresh").addEventListener("click", load);
    document.addEventListener("visibilitychange", function () {
      if (document.hidden) { stop(); } else { load(); start(); }
    });
    load();
    start();
  });
})();
