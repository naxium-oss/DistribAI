/** Contributor node identity, status, and local control routes. */
'use strict';

const crypto = require('crypto');
const { generateHardwareFingerprint, validateBenchmarkResults } = require('../lib/benchmarkSecurity');

function registerNodeRoutes(app, deps) {
    const { nodeDataStore, orchCtx } = deps;
    const {
        getNodeDataSecret,
        loadNodeData,
        saveNodeData,
        loadOrchSync,
        saveOrchSync,
        checkSyncConflict,
    } = nodeDataStore;
    const { fetchOrchestratorAdminNodes } = orchCtx;

    app.get('/api/node/data', (req, res) => {
        const data = loadNodeData();
        res.json(data);
    });

    app.post('/api/node/data', (req, res) => {
        const data = req.body || {};
        const current = loadNodeData();
        const merged = { ...current, ...data, last_saved: Date.now() };
        saveNodeData(merged);
        res.json({ ok: true, saved: true });
    });

    app.post('/api/node/sync', async (req, res) => {
        try {
            const localData = loadNodeData();
            const orchData = req.body || {};

            const conflicts = checkSyncConflict(localData, orchData);

            const resolved = { ...localData };
            for (const conflict of conflicts) {
                if (conflict.winner === 'local') {
                    resolved[conflict.field] = conflict.local;
                } else {
                    resolved[conflict.field] = conflict.remote;
                }
            }

            resolved.last_sync = Date.now();
            resolved.sync_conflicts = conflicts.length;
            saveNodeData(resolved);
            saveOrchSync({
                last_sync: Date.now(),
                orch_timestamp: orchData.timestamp,
                conflicts_resolved: conflicts.length,
            });

            res.json({
                ok: true,
                synced: true,
                conflicts: conflicts.length,
                data: resolved,
            });
        } catch (err) {
            res.status(500).json({ error: err.message });
        }
    });

    app.get('/api/node/sync-status', (req, res) => {
        const sync = loadOrchSync();
        const nodeData = loadNodeData();
        res.json({
            last_sync: sync.last_sync || null,
            conflicts_last_sync: nodeData.sync_conflicts || 0,
            local_timestamp: nodeData.last_saved || null,
            orch_timestamp: sync.orch_timestamp || null,
        });
    });

    app.post('/api/benchmark/results/secure', (req, res) => {
        const results = req.body || {};

        if (!validateBenchmarkResults(results)) {
            return res.status(403).json({
                error: 'Benchmark validation failed',
                message: 'Results appear tampered or invalid. Please run the benchmark again.',
            });
        }

        const benchSecret = String(process.env.BENCHMARK_SECRET || '').trim() || getNodeDataSecret();
        results._server_signature = crypto.createHash('sha256')
            .update(JSON.stringify(results) + benchSecret)
            .digest('hex')
            .substring(0, 16);
        results._server_validated = Date.now();

        const nodeData = loadNodeData();
        nodeData.benchmark_results = results;
        nodeData.benchmark_score = results.overall_score;
        saveNodeData(nodeData);

        res.json({
            ok: true,
            validated: true,
            message: 'Benchmark results validated and stored securely',
        });
    });

    app.post('/api/node/identity/secure', async (req, res) => {
        const { name, hardware_fingerprint } = req.body || {};

        if (!name || typeof name !== 'string') {
            return res.status(400).json({ error: 'Invalid name' });
        }

        const expectedFingerprint = generateHardwareFingerprint();
        if (hardware_fingerprint !== expectedFingerprint) {
            return res.status(403).json({
                error: 'Hardware mismatch',
                message: 'Identity can only be set from the original machine',
            });
        }

        const trimmed = name.trim().slice(0, 64);
        if (!/^[a-zA-Z0-9_\-. ]+$/.test(trimmed)) {
            return res.status(400).json({ error: 'Name contains invalid characters' });
        }

        const nodeData = loadNodeData();

        if (nodeData.node_name !== trimmed) {
            const nodes = await fetchOrchestratorAdminNodes();
            if (nodes !== null && nodes.some((node) => node.node_id === trimmed)) {
                const baseName = trimmed.replace(/-\d+$/, '');
                let suggestedName = `${baseName}-1`;
                let counter = 1;
                while (nodes.some((node) => node.node_id === suggestedName)) {
                    counter += 1;
                    suggestedName = `${baseName}-${counter}`;
                }
                return res.status(409).json({
                    error: 'Node name already exists',
                    message: `Node name "${trimmed}" is already in use. Suggested: "${suggestedName}"`,
                    suggested: suggestedName,
                });
            }
        }

        nodeData.node_name = trimmed;
        nodeData._identity_validated = true;
        nodeData._identity_timestamp = Date.now();
        saveNodeData(nodeData);

        res.json({ ok: true, name: trimmed, secured: true });
    });

    app.get('/api/node/identity/secure', (req, res) => {
        const nodeData = loadNodeData();
        res.json({
            name: nodeData.node_name || null,
            validated: nodeData._identity_validated || false,
            hardware_fingerprint: generateHardwareFingerprint(),
        });
    });

    app.get('/api/benchmark/results/secure', (req, res) => {
        const nodeData = loadNodeData();
        if (nodeData.benchmark_results && nodeData.benchmark_results._server_validated) {
            res.json({
                ok: true,
                validated: true,
                results: nodeData.benchmark_results,
                score: nodeData.benchmark_score,
            });
        } else {
            res.json({
                ok: true,
                validated: false,
                results: null,
                score: null,
                message: 'No validated benchmark results found',
            });
        }
    });
}

module.exports = { registerNodeRoutes };
