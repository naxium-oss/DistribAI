'use strict';

const http = require('http');
const https = require('https');

/**
 * Factory for orchestrator admin HTTP proxy helpers.
 * @param {() => string} getOrchAdminUrl - resolves current ORCH_ADMIN base URL
 */
function createOrchProxy(getOrchAdminUrl) {
    function orchAdminUrl(orchPath) {
        return new URL(getOrchAdminUrl() + orchPath);
    }

    function httpLibForUrl(url) {
        return url.protocol === 'https:' ? https : http;
    }

    function orchAdminHeaders(extra = {}) {
        const headers = { ...extra };
        const secret = String(process.env.DISTRIBAI_ADMIN_SECRET || '').trim();
        if (secret) {
            headers.Authorization = `Bearer ${secret}`;
        }
        return headers;
    }

    function fetchOrchJson(orchPath, timeoutMs = 4000) {
        return new Promise((resolve) => {
            const url = orchAdminUrl(orchPath);
            const lib = httpLibForUrl(url);
            const req = lib.get(url.toString(), { headers: orchAdminHeaders() }, (orchRes) => {
                const chunks = [];
                orchRes.on('data', (c) => chunks.push(c));
                orchRes.on('end', () => {
                    try {
                        resolve(JSON.parse(Buffer.concat(chunks).toString('utf8')));
                    } catch {
                        resolve(null);
                    }
                });
            });
            req.on('error', () => resolve(null));
            req.setTimeout(timeoutMs, () => {
                req.destroy();
                resolve(null);
            });
        });
    }

    function proxyPostJsonToOrch(orchPath, bodyObj, res) {
        const payload = JSON.stringify(bodyObj !== undefined && bodyObj !== null ? bodyObj : {});
        const url = orchAdminUrl(orchPath);
        const opt = {
            hostname: url.hostname,
            port: url.port || (url.protocol === 'https:' ? 443 : 80),
            path: url.pathname + url.search,
            method: 'POST',
            headers: orchAdminHeaders({
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(payload),
            }),
        };
        const lib = httpLibForUrl(url);
        const proxyReq = lib.request(opt, (orchRes) => {
            res.status(orchRes.statusCode);
            orchRes.pipe(res);
        });
        proxyReq.on('error', (err) => {
            res.status(503).json({ error: 'Orchestrator unavailable', detail: err.message });
        });
        proxyReq.setTimeout(120000, () => {
            proxyReq.destroy();
            if (!res.headersSent) res.status(504).json({ error: 'Orchestrator timeout' });
        });
        proxyReq.write(payload);
        proxyReq.end();
    }

    function proxyToOrch(orchPath, res) {
        const url = orchAdminUrl(orchPath);
        const lib = httpLibForUrl(url);
        const req = lib.get(url.toString(), { headers: orchAdminHeaders() }, (orchRes) => {
            res.status(orchRes.statusCode);
            orchRes.pipe(res);
        });
        req.on('error', (err) => {
            res.status(503).json({ error: 'Orchestrator unavailable', detail: err.message });
        });
        req.setTimeout(4000, () => {
            req.destroy();
            res.status(504).json({ error: 'Orchestrator timeout' });
        });
    }

    function proxyDeleteToOrch(orchPath, res) {
        const url = orchAdminUrl(orchPath);
        const lib = httpLibForUrl(url);
        const opt = {
            hostname: url.hostname,
            port: url.port || (url.protocol === 'https:' ? 443 : 80),
            path: url.pathname + url.search,
            method: 'DELETE',
            headers: orchAdminHeaders(),
        };
        const req = lib.request(opt, (orchRes) => {
            res.status(orchRes.statusCode);
            orchRes.pipe(res);
        });
        req.on('error', (err) => {
            res.status(503).json({ error: 'Orchestrator unavailable', detail: err.message });
        });
        req.setTimeout(4000, () => {
            req.destroy();
            if (!res.headersSent) res.status(504).json({ error: 'Orchestrator timeout' });
        });
        req.end();
    }

    function pipeOrchStream(orchPath, res, onClientClose) {
        const url = orchAdminUrl(orchPath);
        const lib = httpLibForUrl(url);
        const upstream = lib.get(url.toString(), { headers: orchAdminHeaders() }, (orchRes) => {
            orchRes.pipe(res);
        });
        upstream.on('error', () => {
            try {
                res.write('data: {"type":"error","msg":"orchestrator offline"}\n\n');
            } catch (_) {
                /* client gone */
            }
        });
        if (typeof onClientClose === 'function') {
            onClientClose(() => upstream.destroy());
        }
        return upstream;
    }

    return {
        orchAdminHeaders,
        fetchOrchJson,
        proxyPostJsonToOrch,
        proxyToOrch,
        proxyDeleteToOrch,
        pipeOrchStream,
    };
}

module.exports = { createOrchProxy };
