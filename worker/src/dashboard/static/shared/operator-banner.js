/**
 * Surfaces operator posture warnings from /api/operator/status (auth, TLS, CORS, secrets).
 */
(function () {
    var ELEMENT_ID = 'cai-operator-truth-banner';
    var DISMISS_STORAGE = 'cai-operator-banner-dismiss-v1';
    var SNOOZE_STORAGE_PREFIX = 'cai-operator-banner-snooze-';
    var DAY_MS = 24 * 60 * 60 * 1000;

    function safeText(value) {
        return String(value == null ? '' : value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function hostIsLoopback(host) {
        var normalized = String(host || '').trim().toLowerCase();
        return normalized === '127.0.0.1' || normalized === 'localhost' || normalized === '::1';
    }

    function fingerprintFor(warnings) {
        return warnings.join('\u241f');
    }

    function snoozeStorageKey(code) {
        return SNOOZE_STORAGE_PREFIX + code;
    }

    function warningIsSnoozed(code) {
        try {
            var until = parseInt(localStorage.getItem(snoozeStorageKey(code)) || '0', 10);
            return until > Date.now();
        } catch (err) {
            return false;
        }
    }

    function snoozeCode(code, durationMs) {
        try {
            localStorage.setItem(snoozeStorageKey(code), String(Date.now() + (durationMs || DAY_MS)));
        } catch (err) {
            /* ignore storage failures */
        }
    }

    function bannerWasDismissed(fingerprint) {
        try {
            return localStorage.getItem(DISMISS_STORAGE) === fingerprint;
        } catch (err) {
            return false;
        }
    }

    function clearBanner(fingerprint) {
        try {
            localStorage.setItem(DISMISS_STORAGE, fingerprint);
        } catch (err) {
            /* ignore */
        }
        var node = document.getElementById(ELEMENT_ID);
        if (node) {
            node.remove();
        }
    }

    function paintBanner(status) {
        var prior = document.getElementById(ELEMENT_ID);
        if (prior) {
            prior.remove();
        }
        if (!status || status.ok !== true) {
            return;
        }

        var items = [];
        if (!status.admin_auth_enforced) {
            items.push({
                code: 'admin_auth',
                text: 'Admin API is not requiring Bearer auth on this bind.'
            });
        }
        if (!status.grpc_tls) {
            items.push({ code: 'grpc_tls', text: 'gRPC is not using TLS.' });
        }
        if (status.cors_permissive) {
            items.push({ code: 'cors', text: 'CORS allows any origin.' });
        }
        if (
            status.registration_requires_poc === false &&
            status.admin_host &&
            !hostIsLoopback(status.admin_host)
        ) {
            items.push({
                code: 'open_registration',
                text: 'Open node registration is allowed without PoC on this bind.'
            });
        }
        if (!status.signing_key_from_env || !status.jwt_secret_from_env) {
            items.push({
                code: 'ephemeral_secrets',
                text: 'Signing or JWT secrets are ephemeral (not set in environment).'
            });
        }

        items = items.filter(function (item) {
            return !warningIsSnoozed(item.code);
        });
        if (!items.length) {
            return;
        }

        var texts = items.map(function (item) {
            return item.text;
        });
        var fingerprint = fingerprintFor(texts);
        if (bannerWasDismissed(fingerprint)) {
            return;
        }

        var banner = document.createElement('div');
        banner.id = ELEMENT_ID;
        banner.setAttribute('role', 'status');
        banner.style.cssText =
            'display:flex;flex-wrap:wrap;align-items:center;gap:10px;padding:10px 16px;margin:0;' +
            'background:#1a2e28;color:#b6f3e4;border-bottom:1px solid #2f5a4e;font-size:13px;line-height:1.4;';

        var copy = document.createElement('span');
        copy.innerHTML =
            '<strong>Operator notice</strong> (' +
            safeText(status.admin_host || 'unknown') +
            '): ' +
            texts.map(safeText).join(' · ');

        var dismissBtn = document.createElement('button');
        dismissBtn.type = 'button';
        dismissBtn.textContent = 'Dismiss';
        dismissBtn.setAttribute('aria-label', 'Dismiss operator notice until warnings change');
        dismissBtn.style.cssText =
            'margin-left:auto;padding:4px 10px;border:1px solid #2f5a4e;background:#0f1f1b;color:#b6f3e4;cursor:pointer;';
        dismissBtn.addEventListener('click', function () {
            clearBanner(fingerprint);
        });

        var snoozeBtn = document.createElement('button');
        snoozeBtn.type = 'button';
        snoozeBtn.textContent = 'Snooze 24h';
        snoozeBtn.setAttribute('aria-label', 'Snooze all operator notices for 24 hours');
        snoozeBtn.style.cssText = dismissBtn.style.cssText;
        snoozeBtn.addEventListener('click', function () {
            items.forEach(function (item) {
                snoozeCode(item.code, DAY_MS);
            });
            clearBanner(fingerprint);
        });

        banner.appendChild(copy);
        banner.appendChild(snoozeBtn);
        banner.appendChild(dismissBtn);
        document.body.prepend(banner);
    }

    function fetchAndShow() {
        fetch('/api/operator/status')
            .then(function (response) {
                if (!response.ok) {
                    return null;
                }
                return response.json();
            })
            .then(function (payload) {
                if (payload) {
                    paintBanner(payload);
                }
            })
            .catch(function () {
                /* static preview may not reach orchestrator */
            });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', fetchAndShow);
    } else {
        fetchAndShow();
    }
})();
