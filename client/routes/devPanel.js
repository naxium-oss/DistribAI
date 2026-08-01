/** Dev-panel and preview helper routes (local tooling only). */
'use strict';

const http = require('http');
const { spawn } = require('child_process');

function registerDevPanelRoutes(app, deps) {
    const {
        paths,
        pythonBin,
        grpcPort,
        adminPort,
        getUserNodeName,
        orchCtx,
        devState,
    } = deps;
    const { getOrchAdmin, orchAdminHeaders } = orchCtx;

    function devEmit(type, data) {
        const line = `data: ${JSON.stringify({ type, ...data, ts: Date.now() })}\n\n`;
        devState.devLog.push({ type, ...data, ts: Date.now() });
        if (devState.devLog.length > 300) devState.devLog.shift();
        for (const c of devState.devClients) {
            try {
                c.write(line);
            } catch (err) {
                devState.devClients.delete(c);
                if (process.env.DEBUG) {
                    console.debug('[DevEmit] Removed disconnected client:', err.message);
                }
            }
        }
    }

    function spawnOrchestrator() {
        if (devState.orchProc) return { ok: false, error: 'Already running' };
        const proc = spawn(
            pythonBin,
            ['-m', 'services_python.orchestrator_grpc'],
            {
                cwd: paths.REPO_ROOT,
                env: {
                    ...process.env,
                    PYTHONUNBUFFERED: '1',
                    GRPC_PORT: String(grpcPort),
                    ADMIN_PORT: String(adminPort),
                },
            }
        );
        devState.orchProc = proc;
        devEmit('orch_started', { pid: proc.pid });

        proc.stdout.on('data', (d) => d.toString().split('\n').filter(Boolean).forEach(
            (line) => devEmit('orch_log', { msg: line })
        ));
        proc.stderr.on('data', (d) => d.toString().split('\n').filter(Boolean).forEach(
            (line) => devEmit('orch_log', { msg: line, level: 'error' })
        ));
        proc.on('close', (code) => {
            devState.orchProc = null;
            devEmit('orch_stopped', { code });
        });
        return { ok: true, pid: proc.pid };
    }

    function stopOrchestrator() {
        if (!devState.orchProc) return { ok: false, error: 'Not running' };
        devState.orchProc.kill('SIGTERM');
        setTimeout(() => { if (devState.orchProc) devState.orchProc.kill('SIGKILL'); }, 3000);
        return { ok: true };
    }

    function spawnWorkers(count) {
        const spawned = [];
        for (let i = 1; i <= count; i++) {
            const idx = devState.workerProcs.length + i;
            const baseName = getUserNodeName() || 'local-worker';
            const nodeId = count > 1 ? `${baseName}-${String(i).padStart(2, '0')}` : baseName;
            const proc = spawn(
                pythonBin,
                ['-m', 'worker.src.daemon.run',
                    '--orchestrator', `localhost:${grpcPort}`,
                    '--node-id', nodeId,
                    '--worker-index', String(idx)],
                { cwd: paths.REPO_ROOT, env: { ...process.env, PYTHONUNBUFFERED: '1' } }
            );
            const entry = { pid: proc.pid, nodeId, index: idx, proc, alive: true };
            devState.workerProcs.push(entry);
            devEmit('worker_started', { pid: proc.pid, nodeId, index: idx });

            proc.stdout.on('data', (d) => d.toString().split('\n').filter(Boolean).forEach(
                (line) => devEmit('worker_log', { pid: proc.pid, nodeId, msg: line })
            ));
            proc.stderr.on('data', (d) => d.toString().split('\n').filter(Boolean).forEach(
                (line) => devEmit('worker_log', { pid: proc.pid, nodeId, msg: line, level: 'error' })
            ));
            proc.on('close', (code) => {
                entry.alive = false;
                devEmit('worker_stopped', { pid: proc.pid, nodeId, code });
                devState.workerProcs = devState.workerProcs.filter((w) => w.pid !== proc.pid);
            });
            spawned.push({ pid: proc.pid, nodeId });
        }
        return spawned;
    }

    function stopAllWorkers() {
        const pids = [];
        for (const w of devState.workerProcs) {
            if (w.alive) {
                w.proc.kill('SIGTERM');
                pids.push(w.pid);
            }
        }
        return pids;
    }

    deps.spawnOrchestrator = spawnOrchestrator;

    app.post('/api/dev/orchestrator/start', (req, res) => {
        res.json(spawnOrchestrator());
    });

    app.post('/api/dev/orchestrator/stop', (req, res) => {
        res.json(stopOrchestrator());
    });

    app.get('/api/dev/orchestrator/status', (req, res) => {
        res.json({
            running: !!devState.orchProc,
            pid: devState.orchProc ? devState.orchProc.pid : null,
        });
    });

    app.post('/api/dev/workers/start', (req, res) => {
        const count = Math.min(parseInt((req.body || {}).count || 1, 10), 10);
        const spawned = spawnWorkers(count);
        res.json({ ok: true, spawned });
    });

    app.post('/api/dev/workers/stop', (req, res) => {
        const pids = stopAllWorkers();
        res.json({ ok: true, stopped: pids });
    });

    app.post('/api/dev/workers/:pid/stop', (req, res) => {
        const pid = parseInt(req.params.pid, 10);
        const w = devState.workerProcs.find((x) => x.pid === pid);
        if (!w) return res.status(404).json({ error: 'Worker not found' });
        w.proc.kill('SIGTERM');
        res.json({ ok: true, pid });
    });

    app.get('/api/dev/workers/status', (req, res) => {
        res.json({
            workers: devState.workerProcs.map((w) => ({
                pid: w.pid,
                nodeId: w.nodeId,
                index: w.index,
                alive: w.alive,
            })),
        });
    });

    app.post('/api/dev/jobs/inject', async (req, res) => {
        const body = req.body || {};

        const countRaw = parseInt(body.count, 10);
        const count = isNaN(countRaw) ? 1 : Math.max(1, Math.min(countRaw, 50));

        const stepsRaw = body.steps ? parseInt(body.steps, 10) : undefined;
        const steps = stepsRaw !== undefined ? (isNaN(stepsRaw) ? undefined : Math.max(1, Math.min(stepsRaw, 10000))) : undefined;

        const batchSizeRaw = body.batch_size ? parseInt(body.batch_size, 10) : undefined;
        const batch_size = batchSizeRaw !== undefined ? (isNaN(batchSizeRaw) ? undefined : Math.max(1, Math.min(batchSizeRaw, 256))) : undefined;

        const deadline_s = body.deadline_s ? parseInt(body.deadline_s, 10) : undefined;

        let model_name = body.model_name;
        if (model_name && typeof model_name === 'string') {
            model_name = model_name.trim();
            if (!/^[a-zA-Z0-9_\-]+$/.test(model_name) || model_name.length > 100) {
                return res.status(400).json({ error: 'Invalid model_name format' });
            }
        }

        const preset = body.preset || 'quick';
        if (!['quick', 'standard', 'long'].includes(preset)) {
            return res.status(400).json({ error: 'Invalid preset value' });
        }

        const payload = JSON.stringify({
            preset,
            steps,
            batch_size,
            model_name,
            deadline_s,
        });

        const results = [];
        const errors = [];
        const orchAdmin = getOrchAdmin();

        for (let i = 0; i < count; i++) {
            try {
                const r = await new Promise((resolve, reject) => {
                    const url = new URL(`${orchAdmin}/admin/jobs`);
                    const opt = {
                        hostname: url.hostname,
                        port: url.port,
                        path: url.pathname,
                        method: 'POST',
                        headers: orchAdminHeaders({
                            'Content-Type': 'application/json',
                            'Content-Length': Buffer.byteLength(payload),
                        }),
                    };
                    const req2 = http.request(opt, (rsp) => {
                        let d = '';
                        rsp.on('data', (c) => { d += c; });
                        rsp.on('end', () => {
                            try {
                                resolve(JSON.parse(d));
                            } catch (parseErr) {
                                reject(new Error('Invalid JSON response from orchestrator'));
                            }
                        });
                    });
                    req2.on('error', (err) => reject(new Error(`Request failed: ${err.message}`)));
                    req2.setTimeout(4000, () => { req2.destroy(); reject(new Error('timeout')); });
                    req2.write(payload);
                    req2.end();
                });
                results.push(r);
            } catch (e) {
                errors.push(e.message);
                if (i === 0) {
                    return res.status(503).json({ error: 'Orchestrator unavailable', detail: e.message, injected: 0 });
                }
                console.error(`[JobInject] Request ${i + 1}/${count} failed:`, e.message);
            }
        }

        res.json({
            ok: errors.length === 0,
            injected: results.length,
            errors: errors.length > 0 ? errors : undefined,
            jobs: results,
        });
    });

    app.get('/api/dev/stream', (req, res) => {
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('X-Accel-Buffering', 'no');
        res.flushHeaders();

        for (const entry of devState.devLog.slice(-50)) {
            res.write(`data: ${JSON.stringify(entry)}\n\n`);
        }
        res.write(`data: ${JSON.stringify({
            type: 'dev_snapshot',
            orchRunning: !!devState.orchProc,
            orchPid: devState.orchProc ? devState.orchProc.pid : null,
            workers: devState.workerProcs.map((w) => ({ pid: w.pid, nodeId: w.nodeId, alive: w.alive })),
            ts: Date.now(),
        })}\n\n`);

        devState.devClients.add(res);
        const ka = setInterval(() => {
            try {
                res.write(': ping\n\n');
            } catch (err) {
                if (process.env.DEBUG) {
                    console.debug('[SSE] Ping failed, client likely disconnected:', err.message);
                }
            }
        }, 25000);
        req.on('close', () => { clearInterval(ka); devState.devClients.delete(res); });
    });
}

module.exports = { registerDevPanelRoutes };
