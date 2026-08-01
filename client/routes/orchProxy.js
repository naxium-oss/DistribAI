/** Reverse-proxy routes from the Express client to orchestrator admin. */
'use strict';

const http = require('http');

function registerOrchProxyRoutes(app, deps) {
    const {
        orchCtx,
        desktopConfig,
        getUserNodeName,
    } = deps;
    const {
        getOrchAdmin,
        orchAdminHeaders,
        fetchOrchJson,
        proxyPostJsonToOrch,
        proxyToOrch,
        proxyDeleteToOrch,
        pipeOrchStream,
    } = orchCtx;

    /** Preserve inbound query string when proxying. */
    function querySuffix(req) {
        return req.url.includes('?') ? req.url.slice(req.url.indexOf('?')) : '';
    }

    /** Normalize configured local node name into an id slug. */
    function localNodeId() {
        const cfg = desktopConfig && desktopConfig.readDesktopConfig
            ? desktopConfig.readDesktopConfig()
            : {};
        const name = (typeof getUserNodeName === 'function' && getUserNodeName()) || cfg.node_id || cfg.node_name || '';
        return String(name || '').trim().replace(/\s+/g, '-').toLowerCase();
    }

    app.post('/api/admin/distribai/registry/sync', (req, res) => {
        proxyPostJsonToOrch('/api/admin/distribai/registry/sync', req.body || {}, res);
    });

    app.post('/api/admin/public-release/publish', (req, res) => {
        proxyPostJsonToOrch('/api/admin/public-release/publish', req.body || { push: true }, res);
    });

    app.get('/api/docs/list', (req, res) => {
        proxyToOrch('/api/docs/list' + querySuffix(req), res);
    });

    app.get('/api/docs/read', (req, res) => {
        proxyToOrch('/api/docs/read' + querySuffix(req), res);
    });

    app.get('/api/worker/nodes', (req, res) => proxyToOrch('/admin/nodes' + querySuffix(req), res));
    app.get('/api/worker/jobs', (req, res) => proxyToOrch('/admin/jobs' + querySuffix(req), res));
    app.get('/api/worker/logs', (req, res) => proxyToOrch('/admin/logs' + querySuffix(req), res));
    app.get('/api/worker/health', (req, res) => proxyToOrch('/admin/health', res));

    app.post('/api/worker/nodes/:nodeId/contributing', (req, res) => {
        const nodeId = req.params.nodeId;
        if (!nodeId || !/^[a-zA-Z0-9_\-]+$/.test(nodeId) || nodeId.length > 128) {
            return res.status(400).json({ error: 'invalid node_id format' });
        }
        const url = new URL(getOrchAdmin() + `/admin/nodes/${encodeURIComponent(nodeId)}/contributing`);
        const payload = JSON.stringify(req.body);
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
        const proxyReq = http.request(opt, (orchRes) => {
            res.status(orchRes.statusCode);
            orchRes.pipe(res);
        });
        proxyReq.on('error', (err) => {
            res.status(503).json({ error: 'Orchestrator unavailable', detail: err.message });
        });
        proxyReq.setTimeout(4000, () => {
            proxyReq.destroy();
            res.status(504).json({ error: 'Orchestrator timeout' });
        });
        proxyReq.write(payload);
        proxyReq.end();
    });

    app.post('/api/worker/contribute', (req, res) => {
        const nodeId = localNodeId();
        if (!nodeId) {
            return res.status(400).json({ error: 'node_id not configured' });
        }
        const enabled = !!(req.body && (req.body.enabled ?? req.body.contributing));
        proxyPostJsonToOrch(
            `/admin/nodes/${encodeURIComponent(nodeId)}/contributing`,
            { contributing: enabled },
            res
        );
    });

    app.get('/api/worker/credits', async (req, res) => {
        try {
            const nodeId = localNodeId();
            if (nodeId) {
                const body = await fetchOrchJson(`/admin/credits/${encodeURIComponent(nodeId)}`);
                return res.json(body);
            }
            const body = await fetchOrchJson('/admin/credits');
            const credits = body?.credits || {};
            const entries = Object.entries(credits);
            if (entries.length === 1) {
                const [id, info] = entries[0];
                return res.json({
                    node_id: id,
                    balance: info.balance || 0,
                    confirmed: info.balance || 0,
                    pending: 0,
                    lifetime_earned: info.lifetime || 0,
                    lifetime_votes_cast: info.votes_cast || 0,
                    history: [],
                    transactions: [],
                    multipliers: [],
                });
            }
            const total = entries.reduce((sum, [, info]) => sum + (Number(info.balance) || 0), 0);
            const lifetime = entries.reduce((sum, [, info]) => sum + (Number(info.lifetime) || 0), 0);
            return res.json({
                balance: total,
                confirmed: total,
                pending: 0,
                lifetime_earned: lifetime,
                lifetime_votes_cast: 0,
                history: [],
                transactions: [],
                multipliers: [],
                credits,
                total_issued: body?.total_issued || lifetime,
            });
        } catch (err) {
            res.status(503).json({ error: err.message });
        }
    });
    app.get('/api/worker/credits/:nodeId', (req, res) => {
        proxyToOrch(`/admin/credits/${encodeURIComponent(req.params.nodeId)}`, res);
    });
    app.post('/api/worker/votes', (req, res) => {
        proxyPostJsonToOrch('/v1/votes', req.body || {}, res);
    });
    app.get('/api/admin/votes', (req, res) => proxyToOrch('/admin/votes', res));
    app.get('/api/admin/ledger/root', (req, res) => proxyToOrch('/admin/ledger/root', res));
    app.get('/api/admin/multipliers/stats', (req, res) => proxyToOrch('/admin/multipliers/stats', res));
    app.get('/api/admin/sybil/stats', (req, res) => proxyToOrch('/admin/sybil/stats', res));
    app.get('/api/admin/rebenchmark/stats', (req, res) => proxyToOrch('/admin/rebenchmark/stats', res));
    app.post('/api/admin/rebenchmark/trigger', (req, res) => {
        const nodeId = req.body && req.body.node_id;
        if (nodeId !== undefined && (typeof nodeId !== 'string' || nodeId.length > 128 || !/^[a-zA-Z0-9_-]*$/.test(nodeId))) {
            return res.status(400).json({ error: 'invalid node_id' });
        }
        proxyPostJsonToOrch('/api/admin/rebenchmark/trigger', req.body || {}, res);
    });
    app.get('/api/admin/transfers/stats', (req, res) => proxyToOrch('/admin/transfers/stats', res));

    app.get('/api/admin/jobs', (req, res) => proxyToOrch('/admin/jobs' + querySuffix(req), res));
    app.get('/api/admin/credits', (req, res) => proxyToOrch('/admin/credits', res));
    app.get('/api/admin/nodes', (req, res) => proxyToOrch('/admin/nodes', res));
    app.get('/api/admin/queue', async (req, res) => {
        try {
            const body = await fetchOrchJson('/admin/jobs?status=queued');
            res.json({
                ...body,
                depth: body?.queue_depth ?? (body?.jobs || []).length,
            });
        } catch (err) {
            res.status(503).json({ error: err.message });
        }
    });

    app.get('/api/jobs', (req, res) => proxyToOrch('/admin/jobs' + querySuffix(req), res));
    app.get('/api/jobs/:jobId', (req, res) => {
        const jobId = req.params.jobId;
        if (!jobId || !/^[a-zA-Z0-9_-]+$/.test(jobId) || jobId.length > 128) {
            return res.status(400).json({ error: 'invalid job_id' });
        }
        proxyToOrch(`/admin/jobs/${encodeURIComponent(jobId)}`, res);
    });
    app.post('/api/jobs', (req, res) => proxyPostJsonToOrch('/admin/jobs', req.body || {}, res));
    app.post('/api/jobs/estimate', (req, res) => proxyPostJsonToOrch('/admin/jobs/estimate', req.body || {}, res));
    app.post('/api/jobs/:jobId/cancel', (req, res) => {
        const jobId = req.params.jobId;
        if (!jobId || !/^[a-zA-Z0-9_-]+$/.test(jobId) || jobId.length > 128) {
            return res.status(400).json({ error: 'invalid job_id' });
        }
        proxyPostJsonToOrch(`/admin/jobs/${encodeURIComponent(jobId)}/cancel`, req.body || {}, res);
    });

    app.post('/admin/recalculate-priorities', (req, res) => {
        proxyPostJsonToOrch('/admin/recalculate-priorities', req.body || {}, res);
    });
    app.post('/admin/sync-all', (req, res) => proxyPostJsonToOrch('/admin/sync-all', req.body || {}, res));
    app.post('/admin/clear-completed', (req, res) => {
        proxyPostJsonToOrch('/admin/clear-completed', req.body || {}, res);
    });
    app.post('/admin/nodes/:nodeId/disconnect', (req, res) => {
        proxyPostJsonToOrch(
            `/admin/nodes/${encodeURIComponent(req.params.nodeId)}/disconnect`,
            req.body || {},
            res
        );
    });

    app.get('/admin/stats', (req, res) => proxyToOrch('/admin/stats', res));

    app.get('/api/admin/paginated-summary', async (req, res) => {
        const [jobs, nodes, credits] = await Promise.all([
            fetchOrchJson('/admin/jobs/paginated?per_page=1'),
            fetchOrchJson('/admin/nodes/paginated?per_page=1'),
            fetchOrchJson('/admin/credits/paginated?per_page=1'),
        ]);
        res.json({
            jobs: jobs?.pagination?.total ?? 0,
            nodes: nodes?.pagination?.total ?? 0,
            credits: credits?.pagination?.total ?? 0,
        });
    });

    app.post('/api/admin/votes/:voteId/cast', (req, res) => {
        const voteId = req.params.voteId;
        if (!voteId || !/^[a-zA-Z0-9_-]+$/.test(voteId) || voteId.length > 128) {
            return res.status(400).json({ error: 'invalid vote_id' });
        }
        proxyPostJsonToOrch(`/admin/votes/${encodeURIComponent(voteId)}/cast`, req.body || {}, res);
    });

    app.delete('/api/worker/jobs/:jobId', (req, res) => {
        const jobId = req.params.jobId;
        if (!jobId || !/^[a-zA-Z0-9_-]+$/.test(jobId) || jobId.length > 128) {
            return res.status(400).json({ error: 'invalid job_id' });
        }
        proxyDeleteToOrch(`/admin/jobs/${encodeURIComponent(jobId)}`, res);
    });

    function pipeSse(req, res) {
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('X-Accel-Buffering', 'no');
        res.flushHeaders();
        pipeOrchStream('/admin/stream', res, (destroyUpstream) => {
            req.on('close', destroyUpstream);
        });
    }

    app.get('/api/worker/stream', pipeSse);
    app.get('/api/stream', pipeSse);

    app.get('/api/version', (req, res) => {
        let version = '0.9.0';
        try {
            const pkg = require('../../package.json');
            if (pkg && pkg.version) version = pkg.version;
        } catch (_) {
            /* keep default */
        }
        res.json({ version, name: 'DistribAI' });
    });
}

module.exports = { registerOrchProxyRoutes };
