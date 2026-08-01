// Operator orchestrator dashboard: serves orch/node/shared static HTML and
// forwards /admin and /api/admin traffic to the Python admin HTTP port.

const express = require('express');
const path = require('path');
const fs = require('fs');
const http = require('http');

const app = express();
const PORT = process.env.ORCH_PORT || 3212;

const DASHBOARD_STATIC = path.join(__dirname, '../worker/src/dashboard/static');
const ORCH_ADMIN = process.env.ADMIN_HOST ? `http://${process.env.ADMIN_HOST}:${process.env.ADMIN_PORT || 8766}` : 'http://127.0.0.1:8766';

const PROXY_TIMEOUT_MS = Number(process.env.ORCH_PROXY_TIMEOUT_MS) || 30000;

function orchAdminHeaders() {
    const headers = {};
    const secret = String(process.env.DISTRIBAI_ADMIN_SECRET || process.env.JWT_SECRET || '').trim();
    if (secret) {
        headers.Authorization = `Bearer ${secret}`;
    }
    return headers;
}

function proxyToOrch(proxyPath, res, method = 'GET', body = null) {
    const url = `${ORCH_ADMIN}${proxyPath}`;
    const options = {
        method: method,
        headers: orchAdminHeaders()
    };
    if (body && method !== 'GET') {
        options.headers['Content-Type'] = 'application/json';
    }
    const req = http.request(url, options, (orchRes) => {
        res.statusCode = orchRes.statusCode;
        orchRes.pipe(res);
    });
    if (body && method !== 'GET') {
        req.write(body);
    }
    req.on('error', (err) => {
        if (!res.headersSent) {
            res.status(502).json({ error: 'Orchestrator offline', details: err.message });
        }
    });
    req.setTimeout(PROXY_TIMEOUT_MS, () => {
        req.destroy(new Error('Orchestrator proxy timeout'));
        if (!res.headersSent) {
            res.status(504).json({ error: 'Orchestrator request timeout' });
        }
    });
    req.end();
}

function shouldProxyAdminPath(reqPath) {
    // Static .html under /admin stays on disk; JSON/API paths go to Python.
    if (!reqPath.startsWith('/admin/')) {
        return false;
    }
    return !reqPath.endsWith('.html');
}

app.use(express.json({ limit: '1mb' }));
app.use(express.urlencoded({ extended: true }));

app.use((req, res, next) => {
    if (req.path === '/') {
        return res.redirect('/orchestrator.html');
    }
    next();
});

const DASHBOARD_ORCH = path.join(DASHBOARD_STATIC, 'orch');
const DASHBOARD_SHARED = path.join(DASHBOARD_STATIC, 'shared');
const DASHBOARD_NODE = path.join(DASHBOARD_STATIC, 'node');

// Contributor pages are mounted under /node; bare /dashboard.html (etc.) redirect.
const NODE_PAGE_REDIRECTS = [
    'dashboard.html',
    'jobs.html',
    'job.html',
    'credits.html',
    'settings.html',
    'benchmark.html',
    'help.html',
    'thanks.html',
    'admin.html',
    'dev.html',
];
for (const page of NODE_PAGE_REDIRECTS) {
    app.get('/' + page, (_req, res) => {
        res.redirect(302, '/node/' + page);
    });
}

if (fs.existsSync(DASHBOARD_STATIC) && fs.existsSync(DASHBOARD_ORCH)) {
    app.use('/shared', express.static(DASHBOARD_SHARED));
    app.use('/node', express.static(DASHBOARD_NODE));
    app.use(express.static(DASHBOARD_ORCH));
} else {
    console.warn(`Dashboard orch static directory not found at ${DASHBOARD_ORCH}`);
}

app.get('/admin/*', (req, res, next) => {
    if (!shouldProxyAdminPath(req.path)) {
        return next();
    }
    proxyToOrch(req.path, res);
});
app.post('/admin/*', (req, res) => {
    const body = req.body ? JSON.stringify(req.body) : null;
    proxyToOrch(req.path, res, 'POST', body);
});
app.get('/api/admin/*', (req, res) => proxyToOrch(req.path.replace('/api', ''), res));
app.post('/api/admin/*', (req, res) => {
    const body = req.body ? JSON.stringify(req.body) : null;
    proxyToOrch(req.path.replace('/api', ''), res, 'POST', body);
});
app.get('/api/status', (req, res) => proxyToOrch('/admin/health', res));
app.get('/api/worker/status', (req, res) => proxyToOrch('/admin/health', res));
app.get('/api/operator/status', (req, res) => proxyToOrch('/api/operator/status', res));

app.listen(PORT, '127.0.0.1', () => {
    console.log(`Orchestrator Dashboard running at http://127.0.0.1:${PORT}`);
    console.log(`Static assets: ${DASHBOARD_ORCH}`);
});
