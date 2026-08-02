/** Desktop settings read/write API routes. */
'use strict';

const { ensureIdentity } = require('../lib/identityStore');

function registerSettingsRoutes(app, deps) {
    const { desktopConfig, getUserNodeName, setUserNodeName } = deps;
    const { readDesktopConfig, writeDesktopConfig, getPCUsername } = desktopConfig;

    app.get('/api/settings/resources', (req, res) => {
        const cfg = readDesktopConfig();
        res.json({
            cpuPercent: cfg.cpuPercent || 50,
            gpuPercent: cfg.gpuPercent || 50,
            ramPercent: cfg.ramPercent || 50,
            region: cfg.region || 'auto',
        });
    });

    app.post('/api/settings/resources', (req, res) => {
        const { cpuPercent, gpuPercent, ramPercent, region } = req.body || {};
        const cfg = readDesktopConfig();

        if (typeof cpuPercent === 'number' && cpuPercent >= 0 && cpuPercent <= 100) {
            cfg.cpuPercent = cpuPercent;
        }
        if (typeof gpuPercent === 'number' && gpuPercent >= 0 && gpuPercent <= 100) {
            cfg.gpuPercent = gpuPercent;
        }
        if (typeof ramPercent === 'number' && ramPercent >= 0 && ramPercent <= 100) {
            cfg.ramPercent = ramPercent;
        }
        if (typeof region === 'string' && region.length <= 64) {
            cfg.region = region;
        }

        writeDesktopConfig(cfg);
        res.json({
            ok: true,
            cpuPercent: cfg.cpuPercent,
            gpuPercent: cfg.gpuPercent,
            ramPercent: cfg.ramPercent,
            region: cfg.region,
        });
    });

    // Alias used by settings.html (cpu/gpu/ram percent fields).
    app.get('/api/settings/limits', (req, res) => {
        const cfg = readDesktopConfig();
        res.json({
            cpu: cfg.cpuPercent || 50,
            gpu: cfg.gpuPercent || 50,
            ram: cfg.ramPercent || 50,
        });
    });

    app.post('/api/settings/limits', (req, res) => {
        const body = req.body || {};
        const cfg = readDesktopConfig();
        const cpu = Number(body.cpu);
        const gpu = Number(body.gpu);
        const ram = Number(body.ram);
        if (Number.isFinite(cpu) && cpu >= 0 && cpu <= 100) cfg.cpuPercent = cpu;
        if (Number.isFinite(gpu) && gpu >= 0 && gpu <= 100) cfg.gpuPercent = gpu;
        if (Number.isFinite(ram) && ram >= 0 && ram <= 100) cfg.ramPercent = ram;
        writeDesktopConfig(cfg);
        res.json({
            ok: true,
            cpu: cfg.cpuPercent,
            gpu: cfg.gpuPercent,
            ram: cfg.ramPercent,
        });
    });

    app.get('/api/regions', (req, res) => {
        res.json({
            regions: [
                { id: 'auto', name: 'Auto (Closest)' },
                { id: 'us-east-1', name: 'US East (N. Virginia)' },
                { id: 'us-east-2', name: 'US East (Ohio)' },
                { id: 'us-west-1', name: 'US West (N. California)' },
                { id: 'us-west-2', name: 'US West (Oregon)' },
                { id: 'eu-west-1', name: 'Europe (Ireland)' },
                { id: 'eu-west-2', name: 'Europe (London)' },
                { id: 'eu-central-1', name: 'Europe (Frankfurt)' },
                { id: 'ap-southeast-1', name: 'Asia Pacific (Singapore)' },
                { id: 'ap-southeast-2', name: 'Asia Pacific (Sydney)' },
                { id: 'ap-northeast-1', name: 'Asia Pacific (Tokyo)' },
                { id: 'ca-central-1', name: 'Canada (Central)' },
                { id: 'sa-east-1', name: 'South America (São Paulo)' },
            ],
        });
    });

    app.post('/api/settings/node-name', (req, res) => {
        const name = (req.body || {}).name;
        if (!name || typeof name !== 'string' || name.trim().length === 0) {
            return res.status(400).json({ error: 'name required' });
        }
        const trimmed = name.trim().slice(0, 64);
        if (!/^[a-zA-Z0-9_\-. ]+$/.test(trimmed)) {
            return res.status(400).json({
                error: 'name contains invalid characters (alphanumeric, dash, underscore, period, space only)',
            });
        }
        setUserNodeName(trimmed);
        const cfg = readDesktopConfig();
        cfg.node_name = getUserNodeName();
        if (!cfg.node_id) cfg.node_id = getUserNodeName().replace(/\s+/g, '-').toLowerCase();
        writeDesktopConfig(cfg);
        res.json({ ok: true, name: getUserNodeName() });
    });

    app.get('/api/settings/node-name', (req, res) => {
        const cfg = readDesktopConfig();

        if (!cfg.node_name && !cfg.node_id && !getUserNodeName()) {
            const pcUserName = getPCUsername();
            cfg.node_name = pcUserName;
            cfg.node_id = pcUserName.toLowerCase().replace(/\s+/g, '-');
            writeDesktopConfig(cfg);
            setUserNodeName(pcUserName);
        }

        const ensured = ensureIdentity(cfg, getPCUsername);
        if (ensured.changed) {
            writeDesktopConfig(ensured.cfg);
            cfg = ensured.cfg;
        }

        const defaultName = getPCUsername();
        res.json({ name: getUserNodeName() || cfg.node_name || cfg.node_id || defaultName });
    });

    app.post('/api/settings/reset-node', (req, res) => {
        setUserNodeName(null);
        const cfg = readDesktopConfig();
        delete cfg.node_name;
        delete cfg.cpu_limit;
        delete cfg.gpu_limit;
        writeDesktopConfig(cfg);
        res.json({ ok: true });
    });

    app.post('/api/settings/unlink-node', (req, res) => {
        setUserNodeName(null);
        const cfg = readDesktopConfig();
        delete cfg.node_name;
        delete cfg.node_id;
        delete cfg.auth;
        delete cfg.orchestrator_token;
        writeDesktopConfig(cfg);
        res.json({ ok: true });
    });
}

module.exports = { registerSettingsRoutes };
