/**
 * Contributor chrome: role detection + primary navigation mount.
 */
(function () {
    var pathPrefix = window.location.pathname.indexOf('/node/') === 0 ? '/node' : '';

    var LINKS = [
        { id: 'dashboard', label: 'Dashboard', href: pathPrefix + '/dashboard.html' },
        { id: 'jobs', label: 'Jobs', href: pathPrefix + '/jobs.html' },
        { id: 'credits', label: 'Credits', href: pathPrefix + '/credits.html' },
        { id: 'settings', label: 'Settings', href: pathPrefix + '/settings.html' },
        { id: 'admin', label: 'Admin', href: pathPrefix + '/admin.html', roleAdmin: true },
        { id: 'benchmark', label: 'Benchmark', href: pathPrefix + '/benchmark.html' },
        { id: 'help', label: 'Help', href: pathPrefix + '/help.html' },
        { id: 'thanks', label: 'Thanks', href: pathPrefix + '/thanks.html' },
        { id: 'dev', label: 'Dev', href: pathPrefix + '/dev.html', roleAdmin: true }
    ];

    function resolveRole() {
        if (!document.body) {
            return;
        }
        var query = new URLSearchParams(window.location.search);
        var role = query.get('role');
        if (role === 'admin' || role === 'node') {
            try {
                localStorage.setItem('distribai_role', role);
            } catch (ignore) {
                /* storage blocked */
            }
        }
        var stored = role;
        if (!stored) {
            try {
                stored = localStorage.getItem('distribai_role');
            } catch (ignore) {
                stored = null;
            }
        }
        var page = document.body.dataset.activePage || '';
        if (stored === 'admin' || page === 'admin' || page === 'dev') {
            document.body.classList.add('role-admin-mode');
        }
    }

    function linkMarkup(item, activePage) {
        var classNames = [];
        if (item.id === activePage) {
            classNames.push('active');
        }
        if (item.roleAdmin) {
            classNames.push('role-admin');
        }
        var attr = classNames.length ? ' class="' + classNames.join(' ') + '"' : '';
        return '<a href="' + item.href + '"' + attr + '>' + item.label + '</a>';
    }

    function mountHeader() {
        var host = document.querySelector('header[data-cai-node-header]');
        if (!host) {
            return;
        }
        var activePage = (document.body && document.body.dataset.activePage) || 'dashboard';
        var hint = host.dataset.searchPlaceholder || 'Search nodes and jobs';
        var links = LINKS.map(function (item) {
            return linkMarkup(item, activePage);
        }).join('');

        host.innerHTML =
            '<div class="container">' +
            '<div class="header-inner">' +
            '<div class="logo"><div class="logo-mark" aria-hidden="true" title="DistribAI"><svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><g fill="currentColor"><rect x="1.5" y="6.5" width="5.5" height="11" rx="1.2"/><rect x="7" y="9" width="3.2" height="1.8" rx="0.3"/><rect x="7" y="13.2" width="3.2" height="1.8" rx="0.3"/><rect x="10.6" y="11.1" width="2.8" height="1.8" rx="0.4"/><rect x="13.8" y="9" width="3.2" height="1.8" rx="0.3"/><rect x="13.8" y="13.2" width="3.2" height="1.8" rx="0.3"/><rect x="17" y="6.5" width="5.5" height="11" rx="1.2"/></g></svg></div><span>DistribAI</span></div>' +
            '<button type="button" class="nav-toggle" id="navToggle" aria-expanded="false" aria-controls="main-nav" aria-label="Menu">' +
            '<span></span></button>' +
            '<nav id="main-nav" role="navigation" aria-label="Primary">' + links + '</nav>' +
            '<div class="header-actions">' +
            '<div class="search-box">' +
            '<div class="search-field">' +
            '<span class="dai-ico dai-ico-search search-icon" aria-hidden="true"></span>' +
            '<input type="search" class="search-input" id="searchInput" placeholder="' + hint + '" aria-label="' + hint + '" autocomplete="off">' +
            '</div>' +
            '<div class="search-results" id="searchResults" role="listbox" aria-label="Search results"></div>' +
            '</div>' +
            '<div class="status-badge" id="orchStatusBadge">' +
            '<span class="status-dot"></span><span>Orch Offline</span>' +
            '</div>' +
            '<button id="activityBtn" type="button" style="background:none;border:none;color:var(--text-secondary);cursor:pointer;padding:8px;" aria-label="Activity feed">' +
            '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
            '<path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/>' +
            '</svg></button>' +
            '</div></div></div>';

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

    function start() {
        resolveRole();
        mountHeader();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', start);
    } else {
        start();
    }
})();
