'use strict';

// Node contributor dashboard: local Express surface that proxies orch admin APIs
// and hosts the worker UI static assets via registerNodeDashboardRoutes.

const express = require('express');
const si = require('systeminformation');
const os = require('os');

const { createPaths } = require('./lib/paths');
const { createNodeDataStore } = require('./lib/nodeDataStore');
const { createDesktopConfigStore } = require('./lib/desktopConfig');
const { findPython } = require('./lib/pythonUtil');
const { createOrchestratorContext } = require('./lib/orchestratorContext');
const { createNotificationHelpers } = require('./lib/notifications');
const { registerIdentityRoutes } = require('./lib/identityStore');
const { registerNodeDashboardRoutes } = require('./routes/nodeDashboard');

const app = express();
app.use(express.json({ limit: '10mb' }));

const PORT = process.env.PORT || 3000;
const LISTEN_HOST = process.env.LISTEN_HOST || '127.0.0.1';
const GRPC_PORT = parseInt(process.env.GRPC_PORT || '50051', 10);
const ADMIN_PORT = parseInt(process.env.ADMIN_PORT || '8766', 10);

const paths = createPaths(__dirname);
const nodeDataStore = createNodeDataStore(paths);
const desktopConfig = createDesktopConfigStore(paths);
const orchCtx = createOrchestratorContext(ADMIN_PORT);
const pythonBin = findPython();

let userNodeName = null;

// Live benchmark SSE / process bookkeeping shared with route handlers.
const benchmarkState = {
    running: false,
    proc: null,
    clients: new Set(),
    events: [],
    results: null,
    startedAt: null,
};

// Local orchestrator + worker subprocess handles used by the dev harness routes.
const devState = {
    orchProc: null,
    workerProcs: [],
    devClients: new Set(),
    devLog: [],
};

const { sendNativeNotification } = createNotificationHelpers(benchmarkState);

const deps = {
    paths,
    nodeDataStore,
    desktopConfig,
    orchCtx,
    benchmarkState,
    devState,
    si,
    os,
    PORT,
    pythonBin,
    grpcPort: GRPC_PORT,
    adminPort: ADMIN_PORT,
    getUserNodeName: () => userNodeName,
    setUserNodeName: (name) => { userNodeName = name; },
    findPython,
    sendNativeNotification,
    ...orchCtx,
};

registerNodeDashboardRoutes(app, deps);
registerIdentityRoutes(app, desktopConfig);

orchCtx.startDiscovery();

if (process.env.AUTO_START_ORCH === '1') {
    setTimeout(() => {
        console.log('AUTO_START_ORCH: spawning orchestrator...');
        if (typeof deps.spawnOrchestrator === 'function') {
            deps.spawnOrchestrator();
        }
    }, 500);
}

app.listen(PORT, LISTEN_HOST, () => {
    console.log(`DistribAI Dashboard running at http://${LISTEN_HOST}:${PORT}`);
    console.log(`  Orchestrator admin proxy → ${orchCtx.getOrchAdmin()}`);
    console.log(`  Native notifications: ${os.platform() === 'win32' ? 'Windows Toast' : os.platform() === 'darwin' ? 'macOS Notification Center' : os.platform() === 'linux' ? 'notify-send' : 'Not available'}`);
});
