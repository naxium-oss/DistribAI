/** Hardware fingerprint + integrity checks for local benchmark results. */
'use strict';

const crypto = require('crypto');
const os = require('os');

function generateHardwareFingerprint() {
    const material = [
        os.hostname(),
        os.platform(),
        os.arch(),
        os.cpus().length,
        os.totalmem(),
    ].join('|');
    return crypto.createHash('sha256').update(material).digest('hex').substring(0, 32);
}

function validateBenchmarkResults(results) {
    if (!results || typeof results !== 'object') {
        return false;
    }

    const requiredFields = ['overall_score', 'test_results', 'timestamp', 'hardware_fingerprint'];
    for (const field of requiredFields) {
        if (!(field in results)) {
            return false;
        }
    }

    const score = parseFloat(results.overall_score);
    if (isNaN(score) || score < 0 || score > 1000) {
        return false;
    }

    const timestamp = parseInt(results.timestamp, 10);
    const now = Date.now();
    if (isNaN(timestamp) || timestamp > now || timestamp < now - 3600000) {
        return false;
    }

    const expectedFingerprint = generateHardwareFingerprint();
    if (results.hardware_fingerprint !== expectedFingerprint) {
        console.warn('[Security] Hardware fingerprint mismatch - possible benchmark tampering');
        return false;
    }

    return true;
}

module.exports = { generateHardwareFingerprint, validateBenchmarkResults };
