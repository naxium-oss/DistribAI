/** Checksummed local node cache plus orchestrator sync conflict helpers. */
'use strict';

const crypto = require('crypto');
const fs = require('fs');

function createNodeDataStore(paths) {
    let cachedSecret = null;

    function getNodeDataSecret() {
        if (cachedSecret) {
            return cachedSecret;
        }
        const fromEnv = process.env.NODE_DATA_SECRET;
        if (fromEnv && String(fromEnv).trim()) {
            cachedSecret = String(fromEnv).trim();
            return cachedSecret;
        }
        try {
            if (fs.existsSync(paths.NODE_DATA_SECRET_FILE)) {
                cachedSecret = fs.readFileSync(paths.NODE_DATA_SECRET_FILE, 'utf8').trim();
                if (cachedSecret) {
                    return cachedSecret;
                }
            }
        } catch (err) {
            console.warn('[Security] Could not read node data secret file:', err.message);
        }
        cachedSecret = crypto.randomBytes(32).toString('hex');
        try {
            fs.mkdirSync(paths.CONFIG_DIR, { recursive: true });
            fs.writeFileSync(paths.NODE_DATA_SECRET_FILE, cachedSecret, { mode: 0o600 });
        } catch (err) {
            console.warn('[Security] Could not persist node data secret; checksums reset if process restarts:', err.message);
        }
        return cachedSecret;
    }

    function generateChecksum(data) {
        const withoutChecksum = { ...data };
        delete withoutChecksum._checksum;
        const canonical = JSON.stringify(withoutChecksum, Object.keys(withoutChecksum).sort());
        return crypto.createHash('sha256').update(canonical + getNodeDataSecret()).digest('hex').substring(0, 16);
    }

    function verifyChecksum(data) {
        return data._checksum === generateChecksum(data);
    }

    function loadNodeData() {
        try {
            if (!fs.existsSync(paths.NODE_DATA_FILE)) {
                return {};
            }
            const data = JSON.parse(fs.readFileSync(paths.NODE_DATA_FILE, 'utf8'));
            if (data._checksum && !verifyChecksum(data)) {
                console.warn('[Security] Node data checksum mismatch - possible tampering detected');
                return {};
            }
            return data;
        } catch (_) {
            return {};
        }
    }

    function saveNodeData(data) {
        fs.mkdirSync(paths.CONFIG_DIR, { recursive: true });
        data._checksum = generateChecksum(data);
        data._timestamp = Date.now();
        fs.writeFileSync(paths.NODE_DATA_FILE, JSON.stringify(data, null, 2));
    }

    function loadOrchSync() {
        try {
            if (!fs.existsSync(paths.ORCH_SYNC_FILE)) {
                return {};
            }
            return JSON.parse(fs.readFileSync(paths.ORCH_SYNC_FILE, 'utf8'));
        } catch (_) {
            return {};
        }
    }

    function saveOrchSync(data) {
        fs.mkdirSync(paths.CONFIG_DIR, { recursive: true });
        fs.writeFileSync(paths.ORCH_SYNC_FILE, JSON.stringify(data, null, 2));
    }

    function checkSyncConflict(localData, orchData) {
        const conflicts = [];
        const watched = ['credits', 'completed_jobs', 'benchmark_score', 'node_tier'];

        for (const field of watched) {
            if (localData[field] !== undefined && orchData[field] !== undefined) {
                if (JSON.stringify(localData[field]) !== JSON.stringify(orchData[field])) {
                    conflicts.push({
                        field,
                        local: localData[field],
                        remote: orchData[field],
                        winner: (field === 'credits' && localData[field] > orchData[field]) ? 'local' : 'remote',
                    });
                }
            }
        }

        return conflicts;
    }

    return {
        getNodeDataSecret,
        loadNodeData,
        saveNodeData,
        loadOrchSync,
        saveOrchSync,
        checkSyncConflict,
    };
}

module.exports = { createNodeDataStore };
