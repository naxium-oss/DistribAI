/**
 * DistribAI local developer panel — orchestrator / worker / job injection controls.
 * Same endpoints and DOM IDs; restructured for the contributor SPA remake.
 */
(function () {
    'use strict';

    var panel = {
        stream: null,
        pollHandle: null,
        lines: [],
        lineCap: 200
    };

    var STYLE = {
        success: 'var(--success)',
        warn: 'var(--warning)',
        error: 'var(--error)',
        dim: 'var(--text-muted)',
        info: 'var(--text-secondary)'
    };

    var JOB_TINT = {
        queued: 'var(--text-muted)',
        assigned: 'var(--warning)',
        running: 'var(--warning)',
        success: 'var(--success)',
        failed: 'var(--error)',
        error: 'var(--error)',
        timeout: 'var(--error)',
        cancelled: 'var(--text-muted)'
    };

    function openEventStream() {
        if (panel.stream) {
            panel.stream.close();
        }
        panel.stream = new EventSource('/api/dev/stream');
        panel.stream.onmessage = function (msg) {
            try {
                onStreamPayload(JSON.parse(msg.data));
            } catch (_err) {
                /* ignore malformed frames */
            }
        };
        panel.stream.onerror = function () {
            setTimeout(openEventStream, 3000);
        };
    }

    function onStreamPayload(ev) {
        switch (ev.type) {
            case 'dev_snapshot':
                paintOrch(ev.orchRunning, ev.orchPid);
                paintWorkers(ev.workers || []);
                break;
            case 'orch_started':
                paintOrch(true, ev.pid);
                pushLog('▶ Orchestrator started  pid=' + ev.pid, 'success');
                window.devRefreshData();
                break;
            case 'orch_stopped':
                paintOrch(false, null);
                pushLog('■ Orchestrator stopped  exit=' + ev.code, 'warn');
                break;
            case 'worker_started':
                pushLog('⊕ Worker started  ' + ev.nodeId + '  pid=' + ev.pid, 'success');
                break;
            case 'worker_stopped':
                pushLog('⊖ Worker stopped  ' + ev.nodeId + '  exit=' + ev.code, 'warn');
                reloadWorkers();
                break;
            case 'orch_log':
                pushLog(ev.msg, ev.level === 'error' ? 'error' : 'dim');
                break;
            case 'worker_log':
                pushLog('[' + ev.nodeId + '] ' + ev.msg, ev.level === 'error' ? 'error' : 'dim');
                break;
            default:
                break;
        }
    }

    function pushLog(text, tone) {
        var host = document.getElementById('dev-log');
        if (!host) {
            return;
        }
        var stamp = new Date().toLocaleTimeString('en', { hour12: false });
        panel.lines.push({ time: stamp, msg: text, style: tone });
        if (panel.lines.length > panel.lineCap) {
            panel.lines.shift();
        }

        var row = document.createElement('div');
        row.style.cssText = 'color:' + (STYLE[tone] || STYLE.info) + ';word-break:break-all;';
        row.textContent = stamp + '  ' + text;

        var lead = host.firstChild && host.firstChild.textContent;
        if (lead && (lead.indexOf('Waiting for events') === 0 || lead.indexOf('Awaiting stream') === 0)) {
            host.textContent = '';
        }
        host.appendChild(row);
        while (host.children.length > panel.lineCap) {
            host.removeChild(host.firstChild);
        }
        host.scrollTop = host.scrollHeight;
    }

    window.devClearLog = function () {
        panel.lines = [];
        var host = document.getElementById('dev-log');
        if (host) {
            host.textContent = '<span style="color:var(--text-muted)">Log cleared</span>';
        }
    };

    function paintOrch(running, pid) {
        var dot = document.getElementById('dev-stat-orch-dot');
        var label = document.getElementById('dev-stat-orch');
        var pidNode = document.getElementById('dev-orch-pid');
        var badge = document.getElementById('dev-orch-badge');
        var badgeText = document.getElementById('dev-orch-badge-text');
        var statusNode = document.getElementById('dev-orch-status');

        if (dot) {
            dot.style.background = running ? 'var(--success)' : 'var(--error)';
        }
        if (label) {
            label.textContent = running ? 'Running' : 'Stopped';
        }
        if (statusNode) {
            statusNode.textContent = running ? 'Running' : 'Stopped';
        }
        if (pidNode) {
            pidNode.textContent = pid ? 'pid ' + pid : '\u00a0';
        }
        if (badge) {
            badge.className = 'status-badge ' + (running ? 'online' : '');
        }
        if (badgeText) {
            badgeText.textContent = running ? 'Orch online' : 'Orch offline';
        }
    }

    function paintWorkers(workers) {
        var alive = workers.filter(function (w) {
            return w.alive !== false;
        });
        var list = document.getElementById('dev-worker-list');
        var dot = document.getElementById('dev-stat-workers-dot');
        var label = document.getElementById('dev-stat-workers');
        if (dot) {
            dot.style.background = alive.length ? 'var(--success)' : 'var(--text-muted)';
        }
        if (label) {
            label.textContent = alive.length + ' local running';
        }
        if (!list) {
            return;
        }
        list.textContent = alive.length
            ? alive.map(function (w) {
                  return w.nodeId + '  pid=' + w.pid;
              }).join('\n')
            : 'no local nodes running';
    }

    async function reloadWorkers() {
        try {
            var payload = await fetch('/api/dev/workers/status').then(function (res) {
                return res.json();
            });
            paintWorkers(payload.workers || []);
        } catch (_err) {
            /* offline */
        }
    }

    window.devOrchStart = async function () {
        pushLog('Starting orchestrator…', 'info');
        try {
            var payload = await fetch('/api/dev/orchestrator/start', { method: 'POST' }).then(function (res) {
                return res.json();
            });
            if (!payload.ok) {
                pushLog('Error: ' + payload.error, 'error');
            }
        } catch (err) {
            pushLog('Request failed: ' + err.message, 'error');
        }
    };

    window.devOrchStop = async function () {
        pushLog('Stopping orchestrator…', 'warn');
        try {
            await fetch('/api/dev/orchestrator/stop', { method: 'POST' });
        } catch (err) {
            pushLog('Request failed: ' + err.message, 'error');
        }
    };

    window.devWorkersStart = async function () {
        var count = parseInt(document.getElementById('dev-worker-count').value || '1', 10);
        pushLog('Starting ' + count + ' local node(s)…', 'info');
        try {
            var payload = await fetch('/api/dev/workers/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ count: count })
            }).then(function (res) {
                return res.json();
            });
            (payload.spawned || []).forEach(function (w) {
                pushLog('  + ' + w.nodeId + ' pid=' + w.pid, 'success');
            });
        } catch (err) {
            pushLog('Request failed: ' + err.message, 'error');
        }
    };

    window.devWorkersStop = async function () {
        pushLog('Stopping all local nodes…', 'warn');
        try {
            var payload = await fetch('/api/dev/workers/stop', { method: 'POST' }).then(function (res) {
                return res.json();
            });
            pushLog('  Stopped pids: ' + ((payload.stopped || []).join(', ') || 'none'), 'warn');
        } catch (err) {
            pushLog('Request failed: ' + err.message, 'error');
        }
    };

    window.devInjectJobs = async function () {
        var count = parseInt(document.getElementById('dev-job-count').value || '1', 10);
        var preset = document.getElementById('dev-job-preset').value;
        var model = document.getElementById('dev-job-model').value;
        var stepsField = document.getElementById('dev-job-steps');
        var steps = stepsField && stepsField.value ? parseInt(stepsField.value, 10) : undefined;
        var body = { count: count, preset: preset };
        if (model) {
            body.model_name = model;
        }
        if (steps) {
            body.steps = steps;
        }

        pushLog('Injecting ' + count + ' job(s) preset=' + preset + '…', 'info');
        try {
            var payload = await fetch('/api/dev/jobs/inject', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            }).then(function (res) {
                return res.json();
            });
            if (payload.ok) {
                pushLog('  ✓ Injected ' + payload.injected + ' job(s)', 'success');
                (payload.jobs || []).forEach(function (job) {
                    pushLog(
                        '    ' + job.job_id + '  model=' + job.model_name + '  steps=' + job.steps,
                        'dim'
                    );
                });
                setTimeout(window.devRefreshData, 600);
            } else {
                pushLog('  Error: ' + payload.error, 'error');
            }
        } catch (err) {
            pushLog('Request failed: ' + err.message, 'error');
        }
    };

    function paintNodes(nodes) {
        var table = document.getElementById('dev-nodes-table');
        var countEl = document.getElementById('dev-node-count');
        var dot = document.getElementById('dev-stat-workers-dot');
        if (countEl) {
            countEl.textContent = nodes.length;
        }
        if (dot) {
            dot.style.background = nodes.length ? 'var(--success)' : 'var(--text-muted)';
        }
        if (!table) {
            return;
        }
        if (!nodes.length) {
            table.textContent = '<span style="color:var(--text-muted)">No nodes connected</span>';
            return;
        }
        var statusTint = {
            idle: 'var(--success)',
            working: 'var(--warning)',
            offline: 'var(--error)',
            stale: 'var(--text-muted)'
        };
        table.textContent = nodes
            .map(function (n) {
                var hw = (n.hardware && n.hardware.gpu_model) || 'CPU';
                var tint = statusTint[n.status] || 'var(--text-muted)';
                return (
                    '<div style="display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid var(--border);">' +
                    '<div><span style="color:var(--text)">' +
                    n.node_id +
                    '</span>' +
                    '<span style="color:var(--text-muted);font-size:.68rem;margin-left:6px;">' +
                    hw +
                    '</span></div>' +
                    '<div style="display:flex;align-items:center;gap:10px;">' +
                    '<span style="font-size:.68rem;color:var(--text-muted);">✓ ' +
                    n.jobs_completed +
                    ' / ✗ ' +
                    n.jobs_failed +
                    '</span>' +
                    '<span style="font-size:.68rem;padding:2px 8px;border-radius:10px;background:' +
                    tint +
                    '22;color:' +
                    tint +
                    ';border:1px solid ' +
                    tint +
                    '44;">' +
                    n.status +
                    '</span></div></div>'
                );
            })
            .join('');
    }

    function paintJobs(jobs) {
        var tbody = document.getElementById('dev-jobs-tbody');
        var countEl = document.getElementById('dev-jobs-count');
        var jobsDot = document.getElementById('dev-stat-jobs-dot');
        var jobsLabel = document.getElementById('dev-stat-jobs');
        var active = jobs.filter(function (j) {
            return ['queued', 'assigned', 'running'].indexOf(j.status) >= 0;
        }).length;
        if (countEl) {
            countEl.textContent = jobs.length;
        }
        if (jobsLabel) {
            jobsLabel.textContent = jobs.length + ' total · ' + active + ' active';
        }
        if (jobsDot) {
            jobsDot.style.background = active
                ? 'var(--warning)'
                : jobs.length
                  ? 'var(--success)'
                  : 'var(--text-muted)';
        }
        if (!tbody) {
            return;
        }
        if (!jobs.length) {
            tbody.textContent =
                '<tr><td colspan="6" style="color:var(--text-muted);padding:12px 0;">No jobs yet</td></tr>';
            return;
        }
        tbody.textContent = jobs
            .slice(0, 30)
            .map(function (j) {
                var tint = JOB_TINT[j.status] || 'var(--text-secondary)';
                var progress = j.progress
                    ? 'step ' +
                      j.progress.step +
                      '  loss ' +
                      (j.progress.loss != null ? j.progress.loss.toFixed(4) : '')
                    : j.status === 'success' && j.result
                      ? 'loss ' +
                        (j.result.final_loss != null ? j.result.final_loss.toFixed(4) : '')
                      : '—';
                return (
                    '<tr style="border-bottom:1px solid var(--border);">' +
                    '<td style="padding:6px 10px 6px 0;color:var(--text-secondary);">' +
                    j.job_id +
                    '</td>' +
                    '<td style="padding:6px 10px;"><span style="color:' +
                    tint +
                    ';font-size:.68rem;padding:2px 7px;border-radius:9px;background:' +
                    tint +
                    '22;border:1px solid ' +
                    tint +
                    '44;">' +
                    j.status +
                    '</span></td>' +
                    '<td style="padding:6px 10px;color:var(--text-secondary);">' +
                    j.model_name +
                    '</td>' +
                    '<td style="padding:6px 10px;color:var(--text-secondary);">' +
                    j.steps +
                    '</td>' +
                    '<td style="padding:6px 10px;color:var(--text-secondary);">' +
                    (j.assigned_to || '—') +
                    '</td>' +
                    '<td style="padding:6px 10px;color:var(--text-muted);">' +
                    progress +
                    '</td></tr>'
                );
            })
            .join('');
    }

    window.devRefreshData = async function () {
        try {
            var pair = await Promise.all([
                fetch('/api/worker/nodes')
                    .then(function (r) {
                        return r.json();
                    })
                    .catch(function () {
                        return { nodes: [] };
                    }),
                fetch('/api/worker/jobs')
                    .then(function (r) {
                        return r.json();
                    })
                    .catch(function () {
                        return { jobs: [], queue_depth: 0 };
                    })
            ]);
            paintNodes(pair[0].nodes || []);
            paintJobs(pair[1].jobs || []);
        } catch (_err) {
            /* ignore */
        }
    };

    function beginPolling() {
        window.devRefreshData();
        reloadWorkers();
        if (!panel.pollHandle) {
            panel.pollHandle = setInterval(function () {
                var page = document.getElementById('page-dev');
                if (page && page.style.display !== 'none') {
                    window.devRefreshData();
                }
            }, 2500);
        }
    }

    document.querySelectorAll('nav a').forEach(function (link) {
        if (link.dataset.page === 'dev') {
            link.addEventListener('click', function () {
                beginPolling();
                fetch('/api/dev/orchestrator/status')
                    .then(function (r) {
                        return r.json();
                    })
                    .then(function (d) {
                        paintOrch(d.running, d.pid);
                    })
                    .catch(function () {
                        paintOrch(false, null);
                    });
            });
        }
    });

    openEventStream();
    fetch('/api/dev/orchestrator/status')
        .then(function (r) {
            return r.json();
        })
        .then(function (d) {
            paintOrch(d.running, d.pid);
        })
        .catch(function () {});
})();
