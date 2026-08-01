/**
 * Contributor-side orchestrator admin URL helpers (relative proxy by default).
 */
(function (root) {
    'use strict';

    var OrchConfig = {
        adminBase: '',
        workerApiBase: '/api/worker',
        jobsApiBase: '/api/jobs',
        adminApiBase: '/api/admin',

        resolve: function (path) {
            var base = this.adminBase || '';
            if (!path) {
                return base || '/';
            }
            if (/^https?:\/\//i.test(path)) {
                return path;
            }
            if (path.charAt(0) === '/') {
                return base + path;
            }
            return base + '/' + path;
        },

        workerUrl: function (suffix) {
            return this.workerApiBase + (suffix || '');
        },

        adminUrl: function (suffix) {
            return this.adminApiBase + (suffix || '');
        }
    };

    try {
        var stored = localStorage.getItem('distribai_orch_admin');
        if (stored) {
            OrchConfig.adminBase = String(stored).replace(/\/$/, '');
        }
    } catch (err) {
        /* storage unavailable */
    }

    root.OrchConfig = OrchConfig;
})(typeof window !== 'undefined' ? window : this);
