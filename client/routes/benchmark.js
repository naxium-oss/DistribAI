/** Benchmark start/status/SSE routes for the contributor dashboard. */
'use strict';

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

function registerBenchmarkRoutes(app, deps) {
    const { paths, nodeDataStore, findPython } = deps;
    const { loadNodeData } = nodeDataStore;
    const benchmarkState = deps.benchmarkState;

    function broadcastEvent(data) {
        const line = `data: ${JSON.stringify(data)}\n\n`;
        benchmarkState.events.push(data);
        if (benchmarkState.events.length > 500) benchmarkState.events.shift();
        for (const client of benchmarkState.clients) {
            try { client.write(line); } catch (_) { benchmarkState.clients.delete(client); }
        }
    }

    app.post('/api/benchmark/start', (req, res) => {
        if (benchmarkState.running) {
            return res.status(409).json({ error: 'Benchmark already running' });
        }

        const { skip, only, full } = req.body || {};
        const args = [];
        const safeArg = /^[a-zA-Z0-9_,]+$/;
        if (skip && typeof skip === 'string' && safeArg.test(skip) && skip.length <= 100) {
            args.push('--skip', skip);
        }
        if (only && typeof only === 'string' && safeArg.test(only) && only.length <= 100) {
            args.push('--only', only);
        } else if (full === true || full === '1' || full === 1) {
            args.push('--full');
        }
        // else: runner defaults to the 6-core dashboard suite

        const python = findPython();
        const proc = spawn(python, ['-u', paths.BENCHMARK_SCRIPT, ...args], {
            cwd: path.dirname(paths.BENCHMARK_SCRIPT),
            env: {
                ...process.env,
                PYTHONUNBUFFERED: '1',
                NET_DURATION_S: process.env.NET_DURATION_S || '2.5',
                NET_LOOPBACK_S: process.env.NET_LOOPBACK_S || '2.0',
            },
        });

        benchmarkState.running = true;
        benchmarkState.proc = proc;
        benchmarkState.events = [];
        benchmarkState.results = null;
        benchmarkState.startedAt = Date.now();

        broadcastEvent({ type: 'runner_started', timestamp: benchmarkState.startedAt });

        let stderrBuf = '';
        proc.stderr.on('data', (chunk) => { stderrBuf += chunk.toString(); });

        proc.stdout.on('data', (data) => {
            const lines = data.toString().split('\n');
            for (const line of lines) {
                const t = line.trim();
                if (!t) continue;
                try {
                    const parsed = JSON.parse(t);
                    broadcastEvent(parsed);
                    if (parsed.type === 'suite_complete') {
                        benchmarkState.results = parsed;
                        try { fs.writeFileSync(paths.RESULTS_FILE, JSON.stringify(parsed, null, 2)); } catch (_) {}
                    }
                } catch (_) {
                    broadcastEvent({ type: 'log', message: t });
                }
            }
        });

        proc.on('close', (code) => {
            benchmarkState.running = false;
            benchmarkState.proc = null;
            broadcastEvent({
                type: 'runner_stopped',
                exit_code: code,
                stderr: stderrBuf.slice(-2000),
                timestamp: Date.now(),
            });
        });

        proc.on('error', (err) => {
            benchmarkState.running = false;
            benchmarkState.proc = null;
            broadcastEvent({ type: 'runner_error', message: err.message });
        });

        res.json({ ok: true, message: 'Benchmark started' });
    });

    app.post('/api/benchmark/stop', (req, res) => {
        if (!benchmarkState.running || !benchmarkState.proc) {
            return res.status(400).json({ error: 'No benchmark running' });
        }
        benchmarkState.proc.kill('SIGTERM');
        setTimeout(() => {
            if (benchmarkState.proc) benchmarkState.proc.kill('SIGKILL');
        }, 5000);
        res.json({ ok: true, message: 'Stop signal sent' });
    });

    app.get('/api/benchmark/stream', (req, res) => {
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('X-Accel-Buffering', 'no');
        res.flushHeaders();

        for (const evt of benchmarkState.events) {
            res.write(`data: ${JSON.stringify(evt)}\n\n`);
        }

        res.write(`data: ${JSON.stringify({
            type: 'state_sync',
            running: benchmarkState.running,
            started: benchmarkState.startedAt,
        })}\n\n`);

        benchmarkState.clients.add(res);

        const keepAlive = setInterval(() => {
            try { res.write(': ping\n\n'); } catch (_) {}
        }, 25000);

        req.on('close', () => {
            clearInterval(keepAlive);
            benchmarkState.clients.delete(res);
        });
    });

    app.get('/api/benchmark/status', (req, res) => {
        res.json({
            running: benchmarkState.running,
            startedAt: benchmarkState.startedAt,
            hasResults: !!benchmarkState.results,
        });
    });

    app.get('/api/benchmark/results', (req, res) => {
        if (benchmarkState.results) {
            return res.json(benchmarkState.results);
        }
        if (fs.existsSync(paths.RESULTS_FILE)) {
            try {
                const saved = JSON.parse(fs.readFileSync(paths.RESULTS_FILE, 'utf8'));
                benchmarkState.results = saved;
                return res.json(saved);
            } catch (_) {}
        }
        res.status(404).json({ error: 'No benchmark results available' });
    });

    app.get('/api/benchmark/history', (req, res) => {
        const nodeData = loadNodeData();
        const history = [];

        if (nodeData.benchmark_results && nodeData.benchmark_results._server_validated) {
            history.push({
                timestamp: nodeData.benchmark_results.timestamp || Date.now(),
                overall_score: nodeData.benchmark_score || nodeData.benchmark_results.overall_score,
            });
        }

        if (fs.existsSync(paths.RESULTS_FILE)) {
            try {
                const saved = JSON.parse(fs.readFileSync(paths.RESULTS_FILE, 'utf8'));
                if (saved.timestamp && !history.find((h) => h.timestamp === saved.timestamp)) {
                    history.push({
                        timestamp: saved.timestamp,
                        overall_score: saved.overall_score || saved.score,
                    });
                }
            } catch (_) {}
        }

        res.json({ history: history.sort((a, b) => b.timestamp - a.timestamp) });
    });
}

module.exports = { registerBenchmarkRoutes };
