/**
 * DistribAI contributor SPA — browser hardening helpers.
 * Escaping, input scrubbing, CSP meta injection, and safe fetch wrappers.
 * Public API surface unchanged for the SPA runtime.
 */
(function (root) {
    'use strict';

    function escapeHtml(raw) {
        if (raw == null || raw === '') {
            return '';
        }
        return String(raw)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;')
            .replace(/\//g, '&#x2F;');
    }

    function sanitizeInput(value, allowTags) {
        if (value == null || value === '') {
            return '';
        }
        void allowTags;
        var cleaned = String(value)
            .replace(/[<>]/g, '')
            .replace(/javascript:/gi, '')
            .replace(/on\w+=/gi, '')
            .replace(/data:/gi, '');
        var probe = document.createElement('div');
        probe.textContent = cleaned;
        return probe.innerHTML;
    }

    function setCSP() {
        var tag = document.createElement('meta');
        tag.httpEquiv = 'Content-Security-Policy';
        tag.content =
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; " +
            "img-src 'self' data:; connect-src 'self'; font-src 'self'; object-src 'self'; " +
            "media-src 'self'; frame-src 'none';";
        document.head.appendChild(tag);
    }

    function safeSetHTML(node, markup) {
        if (!node || markup == null) {
            return;
        }
        if (node.textContent !== undefined) {
            node.textContent = markup;
            return;
        }
        node.textContent =
            typeof DOMPurify !== 'undefined' ? DOMPurify.sanitize(markup) : sanitizeInput(markup);
    }

    function secureEventHandler(handler) {
        return function secured(evt) {
            try {
                if (!evt.target || String(evt.target.tagName || '').toLowerCase() === 'script') {
                    console.warn('Blocked potentially dangerous event');
                    return;
                }
                return handler.call(this, evt);
            } catch (err) {
                console.error('Event handler error:', err);
                evt.preventDefault();
            }
        };
    }

    function validateInput(value, type) {
        if (!value) {
            return false;
        }
        var kind = type || 'text';
        var rules = {
            text: /^[^<>]*$/,
            email: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
            number: /^-?\d+(\.\d+)?$/,
            url: /^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._+~#=]{2,256}\.[a-z]{2,6}(\/[-a-zA-Z0-9@:%._+~#=]{2,256})*$/
        };
        var re = rules[kind] || rules.text;
        return re.test(value);
    }

    var rateLimiter = {
        actions: {},
        checkLimit: function (action, limit, windowMs) {
            var cap = limit == null ? 10 : limit;
            var windowSize = windowMs == null ? 60000 : windowMs;
            var now = Date.now();
            var bucket = action + '_' + Math.floor(now / windowSize);
            if (!this.actions[bucket]) {
                this.actions[bucket] = { count: 0, resetTime: now + windowSize };
            }
            if (now > this.actions[bucket].resetTime) {
                this.actions[bucket] = { count: 0, resetTime: now + windowSize };
            }
            this.actions[bucket].count += 1;
            if (this.actions[bucket].count > cap) {
                console.warn('Rate limit exceeded for ' + action);
                return false;
            }
            return true;
        }
    };

    function handleError(error, context) {
        var label = context || '';
        console.error('Error in ' + label + ':', error);
        var banner = document.getElementById('error-message');
        if (!banner) {
            return;
        }
        banner.textContent = 'Operation failed: ' + label;
        banner.style.display = 'block';
        setTimeout(function () {
            banner.style.display = 'none';
        }, 5000);
    }

    function safeAsync(fn, context) {
        var label = context || 'operation';
        return async function wrapped() {
            try {
                return await fn.apply(this, arguments);
            } catch (err) {
                handleError(err, label);
                throw err;
            }
        };
    }

    function safeFetch(url, options, context) {
        var opts = options || {};
        var label = context || 'fetch operation';
        return safeAsync(async function () {
            var response = await fetch(url, Object.assign({ timeout: 10000 }, opts));
            if (!response.ok) {
                throw new Error('HTTP ' + response.status + ': ' + response.statusText);
            }
            return response;
        }, label)();
    }

    document.addEventListener('DOMContentLoaded', function onReady() {
        setCSP();

        var nativeFetch = root.fetch.bind(root);
        root.fetch = function patchedFetch(url, options) {
            var base = options || {};
            var secured = Object.assign({}, base, {
                headers: Object.assign(
                    {
                        'X-Requested-With': 'XMLHttpRequest',
                        'Content-Security-Policy': "default-src 'self'"
                    },
                    base.headers || {}
                )
            });
            return nativeFetch(url, secured);
        };

        document.querySelectorAll('form').forEach(function (form) {
            form.addEventListener('submit', function () {
                form.querySelectorAll('input, textarea').forEach(function (field) {
                    if (field.value) {
                        field.value = sanitizeInput(field.value);
                    }
                });
            });
        });
    });

    root.escapeHtml = escapeHtml;
    root.sanitizeInput = sanitizeInput;
    root.setCSP = setCSP;
    root.safeSetHTML = safeSetHTML;
    root.secureEventHandler = secureEventHandler;
    root.validateInput = validateInput;
    root.rateLimiter = rateLimiter;
    root.handleError = handleError;
    root.safeAsync = safeAsync;
    root.safeFetch = safeFetch;
})(typeof window !== 'undefined' ? window : this);
