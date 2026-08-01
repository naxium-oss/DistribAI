#!/usr/bin/env python3
"""Maintenance: replace broken orchestrator dashboard HTML tails with valid scripts."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ORCH = REPO / "worker" / "src" / "dashboard" / "static" / "orch"

ORCH_TAIL: dict[str, str] = {
    "orchestrator-nodes.html": """
    <script src="/shared/scripts.js"></script>
    <script>
        var nodesRefreshTimer = null;

        function loadNodes() {
            fetch('/admin/nodes')
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    var tbody = document.getElementById('nodesTableBody');
                    if (!data.nodes || data.nodes.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-muted);padding:40px;">No nodes connected</td></tr>';
                        return;
                    }
                    tbody.innerHTML = data.nodes.map(function(node) {
                        var statusClass = node.online ? 'online' : 'offline';
                        var statusText = node.online ? 'Online' : 'Offline';
                        var nid = String(node.node_id || 'unknown');
                        return '<tr>' +
                            '<td><code>' + escapeHtml(nid) + '</code></td>' +
                            '<td><span class="node-status ' + statusClass + '">' + statusText + '</span></td>' +
                            '<td>' + escapeHtml(node.ip || '—') + '</td>' +
                            '<td>' + escapeHtml(node.hardware_summary || '—') + '</td>' +
                            '<td>' + (Number(node.credits) || 0).toFixed(2) + '</td>' +
                            '<td>' +
                                '<button class="btn" onclick="viewNode(' + JSON.stringify(nid) + ')">View</button> ' +
                                '<button class="btn btn-danger" onclick="disconnectNode(' + JSON.stringify(nid) + ')">Disconnect</button>' +
                            '</td>' +
                        '</tr>';
                    }).join('');
                })
                .catch(function(err) { console.error('Failed to load nodes:', err); });
        }

        function viewNode(id) {
            window.location.href = '/orchestrator-node.html?id=' + encodeURIComponent(id);
        }

        function disconnectNode(id) {
            console.warn('disconnectNode not implemented for', id);
        }

        document.addEventListener('DOMContentLoaded', function() {
            loadNodes();
            if (nodesRefreshTimer) clearInterval(nodesRefreshTimer);
            nodesRefreshTimer = setInterval(loadNodes, 10000);
        });
        window.addEventListener('pagehide', function() {
            if (nodesRefreshTimer) clearInterval(nodesRefreshTimer);
        });
    </script>
</body>
</html>
""",
    "orchestrator-jobs.html": """
    <script src="/shared/scripts.js"></script>
    <script>
        var jobsRefreshTimer = null;

        function loadJobs() {
            fetch('/admin/jobs')
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    var tbody = document.getElementById('jobsTableBody');
                    if (!data.jobs || data.jobs.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-muted);padding:40px;">No jobs found</td></tr>';
                        return;
                    }
                    tbody.innerHTML = data.jobs.map(function(job) {
                        var jobId = String(job.id || job.job_id || 'unknown');
                        var priorityClass = 'priority-' + (job.priority || 'normal');
                        var statusClass = 'status-' + (job.status || 'queued');
                        return '<tr>' +
                            '<td><code>' + escapeHtml(jobId) + '</code></td>' +
                            '<td>' + escapeHtml(job.model_name || 'Unknown') + '</td>' +
                            '<td><span class="priority-indicator ' + priorityClass + '">' + escapeHtml(job.priority || 'normal') + '</span></td>' +
                            '<td><span class="status-badge-inline ' + statusClass + '">' + escapeHtml(job.status || 'queued') + '</span></td>' +
                            '<td>' + (Number(job.progress) || 0).toFixed(1) + '%</td>' +
                            '<td><button class="btn btn-danger" onclick="cancelJob(' + JSON.stringify(jobId) + ')">Cancel</button></td>' +
                        '</tr>';
                    }).join('');
                })
                .catch(function(err) { console.error('Failed to load jobs:', err); });
        }

        function cancelJob(jobId) {
            if (!confirm('Cancel job ' + jobId + '?')) return;
            fetch('/admin/jobs/' + encodeURIComponent(jobId), { method: 'DELETE' })
                .then(function(r) { return r.json(); })
                .then(function(data) { if (data.ok) loadJobs(); })
                .catch(function(err) { console.error('Failed to cancel job:', err); });
        }

        document.addEventListener('DOMContentLoaded', function() {
            loadJobs();
            if (jobsRefreshTimer) clearInterval(jobsRefreshTimer);
            jobsRefreshTimer = setInterval(loadJobs, 5000);
        });
        window.addEventListener('pagehide', function() {
            if (jobsRefreshTimer) clearInterval(jobsRefreshTimer);
        });
    </script>
</body>
</html>
""",
    "orchestrator-credits.html": """
    <script src="/shared/scripts.js"></script>
    <script>
        var creditsRefreshTimer = null;

        function loadCredits() {
            fetch('/admin/credits')
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    var tbody = document.getElementById('creditsTableBody');
                    if (!data.credits || data.credits.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="3" style="text-align:center;padding:40px;color:var(--text-muted);">No credit data available</td></tr>';
                        return;
                    }
                    tbody.innerHTML = data.credits.map(function(c) {
                        var nodeId = String(c.node_id || 'unknown');
                        return '<tr>' +
                            '<td style="padding:12px;border-bottom:1px solid var(--border);"><code>' + escapeHtml(nodeId) + '</code></td>' +
                            '<td style="padding:12px;text-align:right;border-bottom:1px solid var(--border);font-family:Geist Mono,monospace;">' + (Number(c.balance) || 0).toFixed(2) + '</td>' +
                            '<td style="padding:12px;text-align:right;border-bottom:1px solid var(--border);font-family:Geist Mono,monospace;">' + (Number(c.total_earned) || 0).toFixed(2) + '</td>' +
                        '</tr>';
                    }).join('');
                })
                .catch(function(err) { console.error('Failed to load credits:', err); });

            fetch('/admin/ledger/root')
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    document.getElementById('ledgerRoot').textContent = data.root_hash || 'None';
                })
                .catch(function(err) { console.error('Failed to load ledger root:', err); });
        }

        document.addEventListener('DOMContentLoaded', function() {
            loadCredits();
            if (creditsRefreshTimer) clearInterval(creditsRefreshTimer);
            creditsRefreshTimer = setInterval(loadCredits, 15000);
        });
        window.addEventListener('pagehide', function() {
            if (creditsRefreshTimer) clearInterval(creditsRefreshTimer);
        });
    </script>
</body>
</html>
""",
    "orchestrator.html": """
    <script src="/shared/scripts.js"></script>
    <script>
        var orchRefreshTimer = null;

        function loadOrchestratorStats() {
            fetch('/admin/stats')
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    document.getElementById('statNodes').textContent = data.active_nodes || 0;
                    document.getElementById('statJobs').textContent = data.running_jobs || 0;
                    document.getElementById('statCredits').textContent = formatNumber(data.credits_distributed || 0);
                    document.getElementById('statTFlops').textContent = (data.total_tflops || 0).toFixed(1);
                })
                .catch(function(err) { console.error('Failed to load stats:', err); });
        }

        function loadNodes() {
            fetch('/admin/nodes')
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    var tbody = document.getElementById('nodesTableBody');
                    if (!data.nodes || data.nodes.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-muted);padding:40px;">No nodes connected</td></tr>';
                        return;
                    }
                    tbody.innerHTML = data.nodes.map(function(node) {
                        var statusClass = node.online ? 'online' : 'offline';
                        var statusText = node.online ? 'Online' : 'Offline';
                        var nid = String(node.node_id || 'unknown');
                        return '<tr>' +
                            '<td><code>' + escapeHtml(nid) + '</code></td>' +
                            '<td><span class="node-status ' + statusClass + '">' + statusText + '</span></td>' +
                            '<td>' + escapeHtml(String(node.benchmark_score != null ? node.benchmark_score : '—')) + '</td>' +
                            '<td>' + escapeHtml(String(node.current_job || 'Idle')) + '</td>' +
                            '<td>' +
                                '<button class="action-btn primary" onclick="viewNode(' + JSON.stringify(nid) + ')">View</button> ' +
                                '<button class="action-btn danger" onclick="disconnectNode(' + JSON.stringify(nid) + ')">Disconnect</button>' +
                            '</td>' +
                        '</tr>';
                    }).join('');
                })
                .catch(function(err) { console.error('Failed to load nodes:', err); });
        }

        function loadJobQueue() {
            fetch('/admin/jobs?status=queued,running')
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    var container = document.getElementById('jobQueueList');
                    if (!data.jobs || data.jobs.length === 0) {
                        container.innerHTML = '<div style="text-align:center;color:var(--text-muted);padding:40px;">No jobs in queue</div>';
                        return;
                    }
                    container.innerHTML = data.jobs.map(function(job) {
                        var priorityClass = 'priority-' + (job.priority || 'normal');
                        var jobId = String(job.id || job.job_id || 'unknown');
                        return '<div style="display:flex;justify-content:space-between;align-items:center;padding:12px 0;border-bottom:1px solid var(--border);">' +
                            '<div>' +
                                '<div style="font-weight:500;">' + escapeHtml(jobId) + '</div>' +
                                '<div style="font-size:0.8rem;color:var(--text-muted);">' + escapeHtml(job.model_name || 'Unknown model') + '</div>' +
                            '</div>' +
                            '<div style="text-align:right;">' +
                                '<span class="priority-indicator ' + priorityClass + '">' + escapeHtml(job.priority || 'normal') + '</span>' +
                                '<div style="font-size:0.8rem;color:var(--text-muted);margin-top:4px;">' + escapeHtml(job.status || 'queued') + '</div>' +
                            '</div>' +
                        '</div>';
                    }).join('');
                })
                .catch(function(err) { console.error('Failed to load jobs:', err); });
        }

        function addLogEntry(level, message) {
            var container = document.getElementById('logContainer');
            var time = new Date().toLocaleTimeString();
            var entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = '<span class="log-time">' + escapeHtml(time) + '</span><span class="log-' + escapeHtml(level) + '">' + escapeHtml(message) + '</span>';
            container.insertBefore(entry, container.firstChild);
            while (container.children.length > 50) {
                container.removeChild(container.lastChild);
            }
        }

        function recalculatePriorities() {
            fetch('/admin/recalculate-priorities', { method: 'POST' })
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    if (data.ok) {
                        showToast('Priorities recalculated', 'success');
                        addLogEntry('success', 'Job priorities recalculated');
                        loadJobQueue();
                    }
                })
                .catch(function() { showToast('Failed to recalculate priorities', 'error'); });
        }

        function syncAllNodes() {
            fetch('/admin/sync-all', { method: 'POST' })
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    if (data.ok) {
                        showToast('All nodes synced', 'success');
                        addLogEntry('success', 'Synchronized all nodes');
                        loadNodes();
                    }
                })
                .catch(function() { showToast('Failed to sync nodes', 'error'); });
        }

        function clearCompletedJobs() {
            fetch('/admin/clear-completed', { method: 'POST' })
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    if (data.ok) {
                        showToast('Completed jobs cleared', 'success');
                        addLogEntry('success', 'Cleared completed jobs');
                        loadJobQueue();
                    }
                })
                .catch(function() { showToast('Failed to clear jobs', 'error'); });
        }

        function viewNode(nodeId) {
            window.location.href = '/orchestrator-node.html?id=' + encodeURIComponent(nodeId);
        }

        function disconnectNode(nodeId) {
            showConfirmModal({
                title: 'Disconnect Node?',
                message: 'Are you sure you want to disconnect node ' + nodeId + '?',
                actionText: 'Disconnect',
                actionType: 'danger',
                onConfirm: function() {
                    fetch('/admin/nodes/' + encodeURIComponent(nodeId) + '/disconnect', { method: 'POST' })
                        .then(function(r) { return r.json(); })
                        .then(function(data) {
                            if (data.ok) {
                                showToast('Node disconnected', 'success');
                                addLogEntry('info', 'Disconnected node: ' + nodeId);
                                loadNodes();
                            }
                        })
                        .catch(function() { showToast('Failed to disconnect node', 'error'); });
                }
            });
        }

        document.addEventListener('DOMContentLoaded', function() {
            loadOrchestratorStats();
            loadNodes();
            loadJobQueue();
            if (orchRefreshTimer) clearInterval(orchRefreshTimer);
            orchRefreshTimer = setInterval(function() {
                loadOrchestratorStats();
                loadNodes();
                loadJobQueue();
            }, 10000);
        });
        window.addEventListener('pagehide', function() {
            if (orchRefreshTimer) clearInterval(orchRefreshTimer);
        });
    </script>
</body>
</html>
""",
    "orchestrator-multipliers.html": """
    <script src="/shared/scripts.js"></script>
    <script>
        function loadMultipliers() {
            fetch('/admin/multipliers/stats')
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    var tbody = document.getElementById('multipliersTableBody');
                    if (!data.multipliers || data.multipliers.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="3" style="text-align:center;padding:40px;color:var(--text-muted);">No active multipliers</td></tr>';
                    } else {
                        tbody.innerHTML = data.multipliers.map(function(m) {
                            return '<tr>' +
                                '<td style="padding:12px;border-bottom:1px solid var(--border);"><code>' + escapeHtml(m.pattern || '') + '</code></td>' +
                                '<td style="padding:12px;text-align:right;border-bottom:1px solid var(--border);font-family:Geist Mono,monospace;color:var(--success);">' + escapeHtml(String(m.factor)) + 'x</td>' +
                                '<td style="padding:12px;text-align:center;border-bottom:1px solid var(--border);"><span class="node-status online">Active</span></td>' +
                            '</tr>';
                        }).join('');
                    }
                    if (data.global_boost) {
                        document.getElementById('globalBoost').textContent = Number(data.global_boost).toFixed(1) + 'x';
                    }
                })
                .catch(function(err) { console.error('Failed to load multipliers:', err); });
        }

        document.addEventListener('DOMContentLoaded', loadMultipliers);
    </script>
</body>
</html>
""",
}

MARKERS = (
    "\n    \n// Enhanced XSS Protection for Production",
    "\n    \n// Enhanced XSS",
)


def main() -> None:
    for name, tail in ORCH_TAIL.items():
        path = ORCH / name
        text = path.read_text(encoding="utf-8")
        idx = -1
        for marker in MARKERS:
            idx = text.find(marker)
            if idx != -1:
                break
        if idx == -1:
            raise SystemExit(f"marker not found in {name}")
        path.write_text(text[:idx] + tail, encoding="utf-8", newline="\n")
        print(f"fixed {name}")


if __name__ == "__main__":
    main()
