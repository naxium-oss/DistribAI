/**
 * Operator chrome: orchestrator navigation mount point.
 */
(function () {
    var PRIMARY = [
        { id: 'dashboard', label: 'Dashboard', href: '/orchestrator.html' },
        { id: 'jobs', label: 'Jobs', href: '/orchestrator-jobs.html' },
        { id: 'nodes', label: 'Nodes', href: '/orchestrator-nodes.html' },
        { id: 'credits', label: 'Credits', href: '/orchestrator-credits.html' },
        { id: 'logs', label: 'Logs', href: '/orchestrator-logs.html' }
    ];

    var ADMIN = [
        { id: 'settings', label: 'Settings', href: '/orchestrator-settings.html', adminOnly: true },
        { id: 'multipliers', label: 'Multipliers', href: '/orchestrator-multipliers.html', adminOnly: true }
    ];

    var NODE = [
        { id: 'node-view', label: 'Node View', href: '/node/dashboard.html' },
        { id: 'help', label: 'Help', href: '/node/help.html' }
    ];

    function linkMarkup(item, activePage) {
        var classNames = [];
        if (item.id === activePage) {
            classNames.push('active');
        }
        if (item.adminOnly) {
            classNames.push('admin-only');
        }
        var attr = classNames.length ? ' class="' + classNames.join(' ') + '"' : '';
        return '<a href="' + item.href + '"' + attr + '>' + item.label + '</a>';
    }

    function mountHeader() {
        var host = document.querySelector('header[data-cai-orch-header]');
        if (!host) {
            return;
        }
        var activePage = (document.body && document.body.dataset.activePage) || 'dashboard';
        var primary = PRIMARY.map(function (item) {
            return linkMarkup(item, activePage);
        }).join('');
        var admin = ADMIN.map(function (item) {
            return linkMarkup(item, activePage);
        }).join('');
        var node = NODE.map(function (item) {
            return linkMarkup(item, activePage);
        }).join('');

        host.innerHTML =
            '<div class="container">' +
            '<div class="header-inner">' +
            '<div class="logo"><div class="logo-mark" aria-hidden="true" title="DistribAI"><svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><g fill="currentColor"><rect x="1.5" y="6.5" width="5.5" height="11" rx="1.2"/><rect x="7" y="9" width="3.2" height="1.8" rx="0.3"/><rect x="7" y="13.2" width="3.2" height="1.8" rx="0.3"/><rect x="10.6" y="11.1" width="2.8" height="1.8" rx="0.4"/><rect x="13.8" y="9" width="3.2" height="1.8" rx="0.3"/><rect x="13.8" y="13.2" width="3.2" height="1.8" rx="0.3"/><rect x="17" y="6.5" width="5.5" height="11" rx="1.2"/></g></svg></div><span>DistribAI Orchestrator</span></div>' +
            '<button type="button" class="nav-toggle" id="navToggle" aria-expanded="false" aria-controls="main-nav" aria-label="Menu">' +
            '<span></span></button>' +
            '<nav id="main-nav" role="navigation" aria-label="Primary">' +
            primary +
            '<div class="nav-divider"></div>' +
            admin +
            '<div class="nav-divider"></div>' +
            node +
            '</nav>' +
            '<div class="header-actions">' +
            '<div class="status-badge" id="orchStatusBadge">' +
            '<span class="status-dot"></span>' +
            '<span>Checking…</span>' +
            '</div></div></div></div>';

        var toggle = host.querySelector('#navToggle');
        if (toggle) {
            toggle.addEventListener('click', function () {
                var open = host.classList.toggle('nav-open');
                toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
            });
            host.querySelectorAll('#main-nav a').forEach(function (anchor) {
                anchor.addEventListener('click', function () {
                    host.classList.remove('nav-open');
                    toggle.setAttribute('aria-expanded', 'false');
                });
            });
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', mountHeader);
    } else {
        mountHeader();
    }
})();
