/** Stable org/node IDs persisted in ~/.distribai/desktop.json. */
'use strict';

const crypto = require('crypto');

function newOrgId() {
    return `org-${crypto.randomBytes(8).toString('hex')}`;
}

function normalizeNodeId(value) {
    return String(value || '').trim().replace(/\s+/g, '-').toLowerCase();
}

function ensureIdentity(cfg, getPCUsername) {
    const next = { ...cfg };
    let changed = false;

    if (!next.org_id) {
        next.org_id = newOrgId();
        changed = true;
    }

    if (!next.node_name && typeof getPCUsername === 'function') {
        next.node_name = getPCUsername();
        changed = true;
    }

    if (!next.node_id) {
        const base = next.node_name || (typeof getPCUsername === 'function' ? getPCUsername() : 'node');
        next.node_id = normalizeNodeId(base) || `node-${crypto.randomBytes(4).toString('hex')}`;
        changed = true;
    }

    return { cfg: next, changed };
}

function registerIdentityRoutes(app, desktopConfig) {
    const { readDesktopConfig, writeDesktopConfig, getPCUsername } = desktopConfig;

    app.get('/api/settings/org-id', (req, res) => {
        const { cfg, changed } = ensureIdentity(readDesktopConfig(), getPCUsername);
        if (changed) {
            writeDesktopConfig(cfg);
        }
        res.json({ org_id: cfg.org_id, node_id: cfg.node_id, node_name: cfg.node_name || null });
    });
}

module.exports = { ensureIdentity, newOrgId, normalizeNodeId, registerIdentityRoutes };
