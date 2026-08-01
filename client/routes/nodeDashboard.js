/** Static/node dashboard mounting helpers. */
'use strict';

const { registerStaticDashboardRoutes } = require('./staticDashboard');
const { registerSystemRoutes } = require('./system');
const { registerSettingsRoutes } = require('./settings');
const { registerBenchmarkRoutes } = require('./benchmark');
const { registerNodeRoutes } = require('./node');
const { registerDevPanelRoutes } = require('./devPanel');
const { registerOrchProxyRoutes } = require('./orchProxy');
const { registerNotificationRoutes } = require('./notifications');
const { registerStatusRoutes } = require('./status');

/**
 * Mount contributor-facing Express routes (static, system, bench, proxy, …).
 * @param {import('express').Express} app
 * @param {object} deps - shared services, config, and mutable state
 */
function registerNodeDashboardRoutes(app, deps) {
    registerStaticDashboardRoutes(app, deps);
    registerSystemRoutes(app, deps);
    registerSettingsRoutes(app, deps);
    registerBenchmarkRoutes(app, deps);
    registerNodeRoutes(app, deps);
    registerDevPanelRoutes(app, deps);
    registerOrchProxyRoutes(app, deps);
    registerNotificationRoutes(app, deps);
    registerStatusRoutes(app, deps);
}

module.exports = { registerNodeDashboardRoutes };
