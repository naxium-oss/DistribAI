/** In-app / SSE notification helpers shared by Express routes. */
'use strict';

function createNotificationHelpers(benchmarkState) {
    function sendDistribAINotification(title, message, icon = 'info') {
        const notification = {
            type: 'notification',
            title,
            message,
            icon,
            timestamp: Date.now(),
            source: 'DistribAI',
        };

        if (!global.notificationHistory) {
            global.notificationHistory = [];
        }
        global.notificationHistory.unshift(notification);
        if (global.notificationHistory.length > 100) {
            global.notificationHistory = global.notificationHistory.slice(0, 100);
        }

        if (benchmarkState && benchmarkState.clients) {
            benchmarkState.clients.forEach((client) => {
                try {
                    client.write(`data: ${JSON.stringify(notification)}\n\n`);
                } catch (err) {
                    console.error('[Notification] Failed to send to client:', err.message);
                }
            });
        }

        console.log(`[DistribAI Notification] ${title}: ${message}`);
        return true;
    }

    function sendNativeNotification(title, message, icon = 'info') {
        return sendDistribAINotification(title, message, icon);
    }

    return { sendDistribAINotification, sendNativeNotification };
}

module.exports = { createNotificationHelpers };
