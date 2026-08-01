/** Process/status and health aggregation routes. */
'use strict';

const http = require('http');
const os = require('os');

function registerStatusRoutes(app, deps) {
    const { si, orchCtx, PORT, desktopConfig, getUserNodeName } = deps;
    const { getOrchAdmin, fetchOrchJson } = orchCtx;

    function localNodeId() {
        const cfg = desktopConfig && desktopConfig.readDesktopConfig
            ? desktopConfig.readDesktopConfig()
            : {};
        const name = (typeof getUserNodeName === 'function' && getUserNodeName())
            || cfg.node_id
            || cfg.node_name
            || '';
        return String(name || '').trim().replace(/\s+/g, '-').toLowerCase();
    }

    app.get('/api/operator/status', (req, res) => {
        orchCtx.proxyToOrch('/api/operator/status', res);
    });

    app.get('/api/worker/status', async (req, res) => {
        try {
            const orchAdmin = getOrchAdmin();
            const healthStarted = Date.now();
            const orchHealth = await new Promise((resolve) => {
                const req2 = http.get(`${orchAdmin}/admin/health`, { timeout: 1000 }, (res2) => {
                    resolve(res2.statusCode === 200);
                }).on('error', () => resolve(false));
                req2.on('timeout', () => { req2.destroy(); resolve(false); });
            });
            const orchLatencyMs = Date.now() - healthStarted;

            const [cpu, mem, graphics, cpuTemp, netStats] = await Promise.all([
                si.currentLoad(),
                si.mem(),
                si.graphics().catch(() => null),
                si.cpuTemperature().catch(() => null),
                si.networkStats().catch(() => []),
            ]);

            const controllers = graphics && graphics.controllers ? graphics.controllers : [];
            const gpuCtrl = controllers.find((c) =>
                c.vram > 0 || c.memoryTotal > 0 || c.utilizationGpu != null
            ) || controllers[0] || null;

            const gpu = gpuCtrl ? {
                vramTotal: gpuCtrl.vram || gpuCtrl.memoryTotal || 0,
                vramUsed: gpuCtrl.vramUsed != null
                    ? gpuCtrl.vramUsed
                    : (gpuCtrl.memoryUsed != null ? gpuCtrl.memoryUsed : 0),
                utilization: gpuCtrl.utilizationGpu != null
                    ? gpuCtrl.utilizationGpu
                    : (gpuCtrl.utilization != null ? gpuCtrl.utilization : null),
                temperature: gpuCtrl.temperatureGpu != null
                    ? gpuCtrl.temperatureGpu
                    : (gpuCtrl.temperatureC != null ? gpuCtrl.temperatureC : null),
            } : null;

            const iface = Array.isArray(netStats) && netStats.length
                ? (netStats.find((n) => (n.rx_sec || 0) + (n.tx_sec || 0) > 0) || netStats[0])
                : null;
            const uploadBps = iface ? Number(iface.tx_sec) || 0 : 0;
            const downloadBps = iface ? Number(iface.rx_sec) || 0 : 0;

            const nodeId = localNodeId();
            let currentJob = null;
            let recentJobs = [];
            let credits = { total_24h: 0, history: [] };
            let peers = 0;
            let earningRate = null;
            if (orchHealth && typeof fetchOrchJson === 'function') {
                try {
                    const [jobsBody, statsBody] = await Promise.all([
                        fetchOrchJson('/admin/jobs?status=queued,running,assigned&per_page=25'),
                        fetchOrchJson('/admin/stats').catch(() => null),
                    ]);
                    peers = Number(statsBody?.active_nodes) || 0;
                    const jobs = jobsBody?.jobs || [];
                    recentJobs = jobs.slice(0, 5).map((job) => ({
                        id: job.job_id || job.id,
                        model_name: job.model_name,
                        status: job.status,
                        progress: job.progress_pct || job.progress || 0,
                        credits: job.credits_earned || job.credits || 0,
                    }));
                    const mine = nodeId
                        ? jobs.find((j) =>
                            String(j.assignee_node_id || j.current_node || '') === nodeId)
                        : jobs.find((j) => j.status === 'running');
                    if (mine) {
                        currentJob = {
                            id: mine.job_id || mine.id,
                            elapsed_seconds: mine.elapsed_seconds || 0,
                            credits_earned: mine.credits_earned || mine.credits || 0,
                            progress: mine.progress_pct || mine.progress || 0,
                        };
                        earningRate = mine.earning_rate || mine.credits_per_hour || null;
                    }
                    if (nodeId) {
                        const creditBody = await fetchOrchJson(
                            `/admin/credits/${encodeURIComponent(nodeId)}`
                        );
                        credits = {
                            total_24h: creditBody?.balance || creditBody?.total_24h || 0,
                            history: creditBody?.history || [],
                        };
                    }
                } catch (_) {
                    /* orchestrator detail optional */
                }
            }

            const cpuPct = Number(cpu.currentLoad) || 0;
            const ramPct = mem.total > 0 ? (mem.active / mem.total) * 100 : 0;
            // Local resource health only — orch reachability is reported separately.
            const healthy = cpuPct < 95 && ramPct < 95;

            res.json({
                orch_connected: orchHealth,
                node_id: nodeId || null,
                platform: `${os.platform()} ${os.release()}`,
                uptime: os.uptime(),
                healthy,
                current_job: currentJob,
                recent_jobs: recentJobs,
                credits,
                network: {
                    latency_ms: orchHealth ? orchLatencyMs : null,
                    peers,
                    upload_bps: uploadBps,
                    download_bps: downloadBps,
                },
                system: {
                    cpu_percent: cpuPct,
                    ram_percent: ramPct,
                    gpu_util: gpu ? gpu.utilization : null,
                    gpu_vram_percent: gpu && gpu.vramTotal > 0
                        ? (gpu.vramUsed / gpu.vramTotal) * 100
                        : null,
                    uptime: os.uptime(),
                    cpu_temp: cpuTemp && cpuTemp.main != null ? cpuTemp.main : null,
                    gpu_temp: gpu ? gpu.temperature : null,
                },
                earning_rate: earningRate,
                timestamp: Date.now(),
            });
        } catch (err) {
            res.status(500).json({ error: err.message });
        }
    });

    app.get('/api/status', async (req, res) => {
        try {
            const orchAdmin = getOrchAdmin();
            const orchHealth = await new Promise((resolve) => {
                const req2 = http.get(`${orchAdmin}/admin/health`, { timeout: 2000 }, (res2) => {
                    resolve({ ok: res2.statusCode === 200, status: res2.statusCode });
                });
                req2.on('error', () => resolve({ ok: false, error: 'Connection failed' }));
                req2.on('timeout', () => {
                    req2.destroy();
                    resolve({ ok: false, error: 'Timeout' });
                });
            });

            res.json({
                dashboard: { ok: true, port: PORT },
                orchestrator: { url: orchAdmin, ...orchHealth },
                mode: 'real',
            });
        } catch (err) {
            res.json({
                dashboard: { ok: true },
                orchestrator: { ok: false, url: getOrchAdmin(), error: err.message },
            });
        }
    });
}

module.exports = { registerStatusRoutes };
