/** Read/write `~/.distribai/desktop.json` and derive a display username. */
'use strict';

const fs = require('fs');
const os = require('os');

function createDesktopConfigStore(paths) {
    function readDesktopConfig() {
        try {
            if (!fs.existsSync(paths.DESKTOP_CONFIG)) {
                return {};
            }
            return JSON.parse(fs.readFileSync(paths.DESKTOP_CONFIG, 'utf8'));
        } catch (_) {
            return {};
        }
    }

    function writeDesktopConfig(config) {
        fs.mkdirSync(paths.CONFIG_DIR, { recursive: true });
        fs.writeFileSync(paths.DESKTOP_CONFIG, JSON.stringify(config, null, 2));
    }

    function getPCUsername() {
        const homeDir = os.homedir();
        const segments = homeDir.split(/[\\/]/);
        const leaf = segments[segments.length - 1];

        if (leaf && leaf.length > 0) {
            const cleaned = leaf.replace(/[^a-zA-Z0-9_\- .]/g, '');
            return cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
        }

        const host = os.hostname().split('.')[0];
        return host.charAt(0).toUpperCase() + host.slice(1);
    }

    return { readDesktopConfig, writeDesktopConfig, getPCUsername };
}

module.exports = { createDesktopConfigStore };
