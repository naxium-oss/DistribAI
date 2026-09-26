/**
 * Shared browser hardening for multi-page contributor dashboards.
 * Mirrors the SPA helpers in index-security.js.
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

    function sanitizeInput(value) {
        if (value == null || value === '') {
            return '';
        }
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
        if (document.querySelector('meta[http-equiv="Content-Security-Policy"]')) {
            return;
        }
        var tag = document.createElement('meta');
        tag.httpEquiv = 'Content-Security-Policy';
        tag.content =
            "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; " +
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; " +
            "img-src 'self' data:; connect-src 'self'; font-src 'self' https://fonts.gstatic.com; " +
            "object-src 'none'; frame-src 'none';";
        document.head.appendChild(tag);
    }

    function safeFetch(url, options, context) {
        var opts = options || {};
        var label = context || 'fetch';
        return fetch(url, opts).then(function (response) {
            if (!response.ok) {
                throw new Error(label + ' HTTP ' + response.status);
            }
            return response;
        });
    }

    document.addEventListener('DOMContentLoaded', function () {
        setCSP();
    });

    root.escapeHtml = root.escapeHtml || escapeHtml;
    root.sanitizeInput = root.sanitizeInput || sanitizeInput;
    root.setCSP = setCSP;
    root.safeFetch = root.safeFetch || safeFetch;
})(typeof window !== 'undefined' ? window : this);
