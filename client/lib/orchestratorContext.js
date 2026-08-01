/** Discover the local orchestrator admin URL and wrap orch-proxy helpers. */
'use strict';

const http = require('http');
const { createOrchProxy } = require('../orch-proxy');

async function discoverOrchestrator(adminPort) {
    if (process.env.ORCHESTRATOR_ADMIN_URL) {
        return process.env.ORCHESTRATOR_ADMIN_URL.replace(/\/$/, '');
    }

    const portsToTry = [8766, 50051, 8080, 3000, 8000, 9000];

    for (const port of portsToTry) {
        try {
            const found = await new Promise((resolve) => {
                const req = http.get(`http://127.0.0.1:${port}/admin/health`, { timeout: 500 }, (res) => {
                    resolve(res.statusCode === 200);
                });
                req.on('error', () => resolve(false));
                req.on('timeout', () => { req.destroy(); resolve(false); });
            });
            if (found) {
                console.log(`[Discovery] Found orchestrator at http://127.0.0.1:${port}`);
                return `http://127.0.0.1:${port}`;
            }
        } catch (err) {
            if (process.env.DEBUG) {
                console.debug(`[Discovery] Port ${port} not responding: ${err.message}`);
            }
        }
    }

    return `http://127.0.0.1:${adminPort}`;
}

function createOrchestratorContext(adminPort) {
    let orchAdmin = `http://127.0.0.1:${adminPort}`;
    const proxy = createOrchProxy(() => orchAdmin);

    function fetchOrchestratorAdminNodes() {
        return proxy.fetchOrchJson('/admin/nodes').then((body) => {
            if (body && typeof body === 'object' && Array.isArray(body.nodes)) {
                return body.nodes;
            }
            return null;
        });
    }

    async function startDiscovery() {
        try {
            const url = await discoverOrchestrator(adminPort);
            orchAdmin = url.replace(/\/$/, '');
            console.log(`[Orchestrator] Using admin API at ${orchAdmin}`);
        } catch (_) {
            console.log(`[Orchestrator] Using default admin API at ${orchAdmin}`);
        }
    }

    return {
        getOrchAdmin: () => orchAdmin,
        setOrchAdmin: (url) => { orchAdmin = url.replace(/\/$/, ''); },
        startDiscovery,
        fetchOrchestratorAdminNodes,
        ...proxy,
    };
}

module.exports = { createOrchestratorContext, discoverOrchestrator };
