/** Notification history and SSE fan-out routes. */
'use strict';

const os = require('os');

function registerNotificationRoutes(app, deps) {
    const { sendNativeNotification } = deps;

    app.post('/api/notifications/send', (req, res) => {
        const { title, message, icon, type } = req.body || {};

        if (!title || !message) {
            return res.status(400).json({ error: 'Title and message required' });
        }

        const sent = sendNativeNotification(title, message, icon || type || 'info');
        res.json({ ok: true, sent, platform: os.platform() });
    });

    app.get('/api/notifications/capability', (req, res) => {
        const platform = os.platform();
        let capable = false;
        let method = 'none';

        if (platform === 'win32') {
            capable = true;
            method = 'windows-toast';
        } else if (platform === 'darwin') {
            capable = true;
            method = 'macos-notification-center';
        } else if (platform === 'linux') {
            capable = true;
            method = 'notify-send';
        }

        res.json({ capable, platform, method });
    });
}

module.exports = { registerNotificationRoutes };
