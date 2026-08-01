/** Static HTML/JS mounting for node and orch dashboards. */
'use strict';

const express = require('express');
const path = require('path');
const fs = require('fs');

function shouldServeFullSpa(req) {
    const q = req.query || {};
    return q.preview === '1' || typeof q.role === 'string';
}

function registerStaticDashboardRoutes(app, deps) {
    const { paths } = deps;
    const DASHBOARD_NODE = path.join(paths.DASHBOARD_STATIC, 'node');
    const DASHBOARD_ORCH = path.join(paths.DASHBOARD_STATIC, 'orch');
    const DASHBOARD_SHARED = path.join(paths.DASHBOARD_STATIC, 'shared');

    if (fs.existsSync(paths.DASHBOARD_STATIC) && fs.existsSync(DASHBOARD_NODE)) {
        app.use('/shared', express.static(DASHBOARD_SHARED));
        app.get('/', (req, res) => {
            if (shouldServeFullSpa(req)) {
                return res.sendFile(path.join(DASHBOARD_NODE, 'index.html'));
            }
            res.redirect('/dashboard.html');
        });
        app.get('/index.html', (req, res) => {
            if (shouldServeFullSpa(req)) {
                return res.sendFile(path.join(DASHBOARD_NODE, 'index.html'));
            }
            res.redirect('/dashboard.html');
        });
        app.get('/nodes.html', (_req, res) => {
            res.redirect(302, '/dashboard.html');
        });
        // Mount contributor pages first so `/dashboard.html` is never shadowed by orch/.
        // Orch HTML is available at `/orchestrator*.html` and also under `/orch/`.
        app.use('/node', express.static(DASHBOARD_NODE));
        app.use(express.static(DASHBOARD_NODE));
        if (fs.existsSync(DASHBOARD_ORCH)) {
            app.use('/orch', express.static(DASHBOARD_ORCH));
            app.use(express.static(DASHBOARD_ORCH));
        }
    } else {
        console.warn(`Dashboard node static directory not found at ${DASHBOARD_NODE}`);
    }
}

module.exports = { registerStaticDashboardRoutes };
