/**
 * Shared dashboard helpers — formatting, fetch retry, toasts, session, shortcuts.
 */

function escapeHtml(value) {
    if (value == null || value === '') {
        return '';
    }
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function formatDuration(totalSeconds) {
    if (totalSeconds == null || totalSeconds < 0) {
        return '—';
    }
    var hours = Math.floor(totalSeconds / 3600);
    var minutes = Math.floor((totalSeconds % 3600) / 60);
    var seconds = Math.floor(totalSeconds % 60);
    if (hours > 0) {
        return hours + 'h ' + minutes + 'm';
    }
    if (minutes > 0) {
        return minutes + 'm ' + seconds + 's';
    }
    return seconds + 's';
}

function formatBytes(byteCount) {
    if (!byteCount) {
        return '0 B';
    }
    var unit = 1024;
    var labels = ['B', 'KB', 'MB', 'GB', 'TB'];
    var index = Math.min(labels.length - 1, Math.floor(Math.log(byteCount) / Math.log(unit)));
    return (byteCount / Math.pow(unit, index)).toFixed(2) + ' ' + labels[index];
}

function formatNumber(value) {
    if (value == null) {
        return '—';
    }
    if (value >= 1e6) {
        return (value / 1e6).toFixed(1) + 'M';
    }
    if (value >= 1e3) {
        return (value / 1e3).toFixed(1) + 'K';
    }
    return String(value);
}

function formatPercent(value) {
    if (value == null) {
        return '—';
    }
    return Math.round(value) + '%';
}

function handleApiError(error, contextLabel) {
    console.error('[' + contextLabel + '] request failed:', error);
    if (typeof window.showNotification === 'function') {
        showNotification(contextLabel + ' failed: ' + (error.message || 'Unknown error'), 'error');
    }
}

function fetchWithRetry(url, options, retries) {
    var remaining = retries == null ? 3 : retries;
    var opts = options || {};
    return new Promise(function (resolve, reject) {
        var attempt = 0;
        function run() {
            attempt += 1;
            fetch(url, opts)
                .then(function (response) {
                    if (!response.ok && attempt < remaining) {
                        setTimeout(run, 1000 * attempt);
                        return;
                    }
                    resolve(response);
                })
                .catch(function (err) {
                    if (attempt < remaining) {
                        setTimeout(run, 1000 * attempt);
                        return;
                    }
                    reject(err);
                });
        }
        run();
    });
}

function showToast(message, kind, ttlMs) {
    var lifetime = ttlMs || 5000;
    var stack = document.getElementById('toastStack');
    if (!stack) {
        return;
    }
    var icons = {
        success: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>',
        error: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
        warning: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        info: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
    };
    var node = document.createElement('div');
    node.className = 'toast ' + (kind || 'info');
    node.innerHTML =
        '<div class="toast-icon">' + (icons[kind] || icons.info) + '</div>' +
        '<span>' + escapeHtml(message) + '</span>' +
        '<button type="button" class="toast-close" onclick="this.parentElement.remove()" aria-label="Dismiss">&times;</button>';
    stack.appendChild(node);
    setTimeout(function () {
        node.classList.add('removing');
        setTimeout(function () {
            if (node.parentElement) {
                node.remove();
            }
        }, 280);
    }, lifetime);
}

var activityFeed = [];
var maxActivityItems = 50;

function addActivityItem(message, kind) {
    activityFeed.unshift({
        message: message,
        type: kind || 'info',
        timestamp: Date.now()
    });
    if (activityFeed.length > maxActivityItems) {
        activityFeed = activityFeed.slice(0, maxActivityItems);
    }
    renderActivityFeed();
}

function renderActivityFeed() {
    var list = document.getElementById('activityList');
    if (!list) {
        return;
    }
    if (!activityFeed.length) {
        list.innerHTML = '<div class="activity-empty">No recent activity</div>';
        return;
    }
    list.innerHTML = activityFeed.map(function (entry) {
        var safeKind = ['success', 'error', 'warning', 'info'].indexOf(entry.type) >= 0 ? entry.type : 'info';
        return (
            '<div class="activity-item ' + safeKind + '">' +
            '<div class="activity-content">' + escapeHtml(entry.message) + '</div>' +
            '<div class="activity-time">' + formatTimeAgo(entry.timestamp) + '</div>' +
            '</div>'
        );
    }).join('');
}

function formatTimeAgo(when) {
    var delta = Math.floor((Date.now() - when) / 1000);
    if (delta < 60) {
        return 'just now';
    }
    if (delta < 3600) {
        return Math.floor(delta / 60) + 'm ago';
    }
    if (delta < 86400) {
        return Math.floor(delta / 3600) + 'h ago';
    }
    return Math.floor(delta / 86400) + 'd ago';
}

function toggleActivityFeed() {
    var panel = document.getElementById('activityFeed');
    if (panel) {
        panel.classList.toggle('show');
    }
}

function clearActivityFeed() {
    activityFeed = [];
    renderActivityFeed();
}

var confirmCallback = null;

function showConfirmModal(options) {
    var opts = options || {};
    var modal = document.getElementById('confirmModal');
    if (!modal) {
        return;
    }
    var icon = document.getElementById('confirmIcon');
    var title = document.getElementById('confirmTitle');
    var message = document.getElementById('confirmMessage');
    var actionBtn = document.getElementById('confirmActionBtn');
    if (icon) {
        icon.textContent = opts.icon || '!';
    }
    if (title) {
        title.textContent = opts.title || 'Confirm Action';
    }
    if (message) {
        message.textContent = opts.message || 'Are you sure?';
    }
    if (actionBtn) {
        actionBtn.textContent = opts.actionText || 'Confirm';
        actionBtn.className = 'confirm-btn confirm-btn-' + (opts.actionType || 'primary');
    }
    confirmCallback = opts.onConfirm;
    modal.classList.add('show');
}

function closeConfirmModal() {
    var modal = document.getElementById('confirmModal');
    if (modal) {
        modal.classList.remove('show');
    }
    confirmCallback = null;
}

function executeConfirmAction() {
    if (typeof confirmCallback === 'function') {
        confirmCallback();
    }
    closeConfirmModal();
}

function showShortcutsModal() {
    var modal = document.getElementById('shortcutsModal');
    if (modal) {
        modal.classList.add('show');
    }
}

function closeShortcutsModal() {
    var modal = document.getElementById('shortcutsModal');
    if (modal) {
        modal.classList.remove('show');
    }
}

var _wasOffline = false;

function updateOnlineStatus() {
    var offline = !navigator.onLine;
    var banner = document.getElementById('offlineBanner');
    if (!banner) {
        return;
    }
    if (offline) {
        _wasOffline = true;
        banner.classList.add('show');
        document.body.classList.add('offline');
        showToast('You are offline. Some features may be unavailable.', 'warning', 8000);
        return;
    }
    banner.classList.remove('show');
    document.body.classList.remove('offline');
    if (_wasOffline) {
        showToast('Back online!', 'success', 3000);
        _wasOffline = false;
    }
}

var contextMenuTarget = null;

function showContextMenu(x, y, target) {
    var menu = document.getElementById('contextMenu');
    if (!menu) {
        return;
    }
    contextMenuTarget = target;
    menu.style.left = x + 'px';
    menu.style.top = y + 'px';
    menu.classList.add('show');
}

function hideContextMenu() {
    var menu = document.getElementById('contextMenu');
    if (menu) {
        menu.classList.remove('show');
    }
    contextMenuTarget = null;
}

function contextAction(action) {
    console.log('Context action:', action, contextMenuTarget);
    hideContextMenu();
}

var SESSION_TIMEOUT = 30 * 60 * 1000;
var SESSION_WARNING_TIME = 5 * 60 * 1000;
var lastActivity = Date.now();
var sessionWarningShown = false;
var sessionCheckInterval = null;

function updateActivity() {
    lastActivity = Date.now();
    if (sessionWarningShown) {
        hideSessionWarning();
        sessionWarningShown = false;
    }
}

function checkSessionTimeout() {
    var idle = Date.now() - lastActivity;
    if (idle >= SESSION_TIMEOUT) {
        logout();
        return;
    }
    if (idle >= SESSION_TIMEOUT - SESSION_WARNING_TIME && !sessionWarningShown) {
        showSessionWarning();
        sessionWarningShown = true;
    }
}

function showSessionWarning() {
    var warning = document.getElementById('sessionWarning');
    if (warning) {
        warning.classList.add('show');
    }
}

function hideSessionWarning() {
    var warning = document.getElementById('sessionWarning');
    if (warning) {
        warning.classList.remove('show');
    }
}

function extendSession() {
    updateActivity();
    hideSessionWarning();
}

function logout() {
    localStorage.removeItem('distribai_session');
    localStorage.removeItem('distribai_activity');
    window.location.href = '/login?loggedout=1';
}

function initSessionManagement() {
    document.addEventListener('mousemove', updateActivity);
    document.addEventListener('keydown', updateActivity);
    document.addEventListener('click', updateActivity);
    if (sessionCheckInterval) {
        clearInterval(sessionCheckInterval);
    }
    sessionCheckInterval = setInterval(checkSessionTimeout, 60000);
}

function initGlobalSearch() {
    var input = document.getElementById('searchInput');
    var resultsEl = document.getElementById('searchResults');
    if (!input || !resultsEl) {
        return;
    }

    var pathPrefix = window.location.pathname.indexOf('/node/') === 0 ? '/node' : '';
    var timer = null;

    function hideResults() {
        resultsEl.classList.remove('show');
    }

    function showResults() {
        resultsEl.classList.add('show');
    }

    function navigate(href) {
        hideResults();
        input.value = '';
        window.location.href = href;
    }

    function pageShortcuts(query) {
        var pages = [
            { keys: ['job', 'jobs', 'queue'], title: 'Jobs', subtitle: 'Open jobs queue', href: pathPrefix + '/jobs.html' },
            { keys: ['node', 'nodes', 'mesh', 'dashboard', 'fleet'], title: 'Dashboard', subtitle: 'Nodes and fleet overview', href: pathPrefix + '/dashboard.html' },
            { keys: ['credit', 'credits', 'balance'], title: 'Credits', subtitle: 'Credit balance and ledger', href: pathPrefix + '/credits.html' },
            { keys: ['bench', 'benchmark'], title: 'Benchmark', subtitle: 'Local hardware benchmarks', href: pathPrefix + '/benchmark.html' },
            { keys: ['help', 'faq', 'support'], title: 'Help', subtitle: 'Guides and troubleshooting', href: pathPrefix + '/help.html' },
            { keys: ['setting', 'settings', 'config'], title: 'Settings', subtitle: 'Node preferences', href: pathPrefix + '/settings.html' }
        ];
        var out = [];
        pages.forEach(function (page) {
            var hit = page.keys.some(function (key) {
                return key.indexOf(query) === 0 || query.indexOf(key) === 0;
            });
            if (hit) {
                out.push({
                    type: 'page',
                    title: page.title,
                    subtitle: page.subtitle,
                    href: page.href
                });
            }
        });
        return out;
    }

    function renderResults(items, query) {
        resultsEl.textContent = '';
        if (!items.length) {
            var empty = document.createElement('div');
            empty.className = 'search-result-empty';
            empty.textContent = 'No results for "' + query + '"';
            resultsEl.appendChild(empty);
            showResults();
            return;
        }
        items.slice(0, 12).forEach(function (item) {
            var btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'search-result-item';
            btn.setAttribute('role', 'option');
            btn.setAttribute('data-type', item.type);
            var title = document.createElement('div');
            title.className = 'search-result-title';
            title.textContent = item.title;
            var sub = document.createElement('div');
            sub.className = 'search-result-sub';
            sub.textContent = item.subtitle || '';
            btn.appendChild(title);
            btn.appendChild(sub);
            btn.addEventListener('click', function () {
                navigate(item.href);
            });
            resultsEl.appendChild(btn);
        });
        showResults();
    }

    function safeJson(response) {
        if (!response || !response.ok) {
            return Promise.resolve({});
        }
        return response.json().catch(function () {
            return {};
        });
    }

    function performSearch(query) {
        var q = query.toLowerCase();
        var items = pageShortcuts(q);
        Promise.all([
            fetch('/api/worker/jobs?per_page=100').then(safeJson).catch(function () { return {}; }),
            fetch('/api/worker/nodes').then(safeJson).catch(function () { return {}; })
        ]).then(function (pair) {
            var jobs = pair[0].jobs || [];
            var nodes = pair[1].nodes || [];
            jobs.forEach(function (job) {
                var id = String(job.job_id || job.id || '');
                var model = String(job.model_name || '');
                var status = String(job.status || '');
                var hay = (id + ' ' + model + ' ' + status).toLowerCase();
                if (hay.indexOf(q) === -1) {
                    return;
                }
                items.push({
                    type: 'job',
                    title: 'Job: ' + (model || id || 'unknown'),
                    subtitle: (status || 'unknown') + (id ? ' · ' + id : ''),
                    href: id
                        ? pathPrefix + '/job.html?id=' + encodeURIComponent(id)
                        : pathPrefix + '/jobs.html'
                });
            });
            nodes.forEach(function (node) {
                var id = String(node.node_id || node.id || '');
                var status = String(node.status || '');
                var hay = (id + ' ' + status).toLowerCase();
                if (hay.indexOf(q) === -1) {
                    return;
                }
                items.push({
                    type: 'node',
                    title: 'Node: ' + (id || 'unknown'),
                    subtitle: 'Status: ' + (status || 'unknown'),
                    href: pathPrefix + '/dashboard.html'
                });
            });
            renderResults(items, query);
        });
    }

    input.addEventListener('input', function () {
        clearTimeout(timer);
        var query = input.value.trim();
        if (!query) {
            hideResults();
            resultsEl.textContent = '';
            return;
        }
        timer = setTimeout(function () {
            performSearch(query);
        }, 200);
    });

    input.addEventListener('focus', function () {
        if (input.value.trim() && resultsEl.childNodes.length) {
            showResults();
        }
    });

    input.addEventListener('keydown', function (event) {
        if (event.key === 'Escape') {
            hideResults();
            input.blur();
            return;
        }
        if (event.key === 'Enter') {
            var first = resultsEl.querySelector('.search-result-item');
            if (first && resultsEl.classList.contains('show')) {
                event.preventDefault();
                first.click();
            }
        }
    });

    document.addEventListener('click', function (event) {
        if (!event.target.closest('.search-box')) {
            hideResults();
        }
    });
}

document.addEventListener('DOMContentLoaded', function () {
    initGlobalSearch();
    var activityBtn = document.getElementById('activityBtn');
    if (activityBtn) {
        activityBtn.addEventListener('click', toggleActivityFeed);
    }
    var confirmModal = document.getElementById('confirmModal');
    if (confirmModal) {
        confirmModal.addEventListener('click', function (event) {
            if (event.target === confirmModal) {
                closeConfirmModal();
            }
        });
    }
    var confirmActionBtn = document.getElementById('confirmActionBtn');
    if (confirmActionBtn) {
        confirmActionBtn.addEventListener('click', executeConfirmAction);
    }
    var confirmCancelBtn = document.getElementById('confirmCancelBtn');
    if (confirmCancelBtn) {
        confirmCancelBtn.addEventListener('click', closeConfirmModal);
    }
    document.addEventListener('keydown', function (event) {
        if (event.key === '?' && !event.ctrlKey && !event.altKey) {
            event.preventDefault();
            showShortcutsModal();
        }
        if (event.key === 'Escape') {
            closeShortcutsModal();
            closeConfirmModal();
            hideContextMenu();
            var createModal = document.getElementById('createJobModal');
            if (createModal && createModal.style.display !== 'none' && createModal.style.display !== '') {
                if (typeof closeAccessibleModal === 'function') {
                    closeAccessibleModal(createModal);
                } else {
                    createModal.style.display = 'none';
                }
                createModal.setAttribute('aria-hidden', 'true');
            }
        }
    });
    document.addEventListener('click', function (event) {
        var menu = document.getElementById('contextMenu');
        if (menu && !menu.contains(event.target)) {
            hideContextMenu();
        }
    });
    window.addEventListener('online', updateOnlineStatus);
    window.addEventListener('offline', updateOnlineStatus);
    updateOnlineStatus();
    initSessionManagement();
    renderActivityFeed();
});

function switchPage(pageName) {
    var routes = {
        dashboard: '/dashboard.html',
        jobs: '/jobs.html',
        credits: '/credits.html',
        settings: '/settings.html',
        admin: '/admin.html',
        benchmark: '/benchmark.html',
        help: '/help.html',
        dev: '/dev.html'
    };
    if (routes[pageName]) {
        window.location.href = routes[pageName];
    }
}

/**
 * Client-side job cost heuristic (estimate only).
 * Mirrors services_python.architecture_config._rough_parameter_count for size tiering.
 * Formula: opcodes ≈ params * steps * batch / 1e6; credits ≈ opcodes * tier_rate.
 */
function roughParameterCount(config) {
    var cfg = config || {};
    var dim = parseInt(cfg.dim, 10) || 256;
    var ffnDim = parseInt(cfg.ffn_dim, 10) || (4 * dim);
    var layers = parseInt(cfg.n_unique_layers, 10) || parseInt(cfg.n_logical_layers, 10) || 8;
    var vocab = 256;
    var family = cfg.family || 'decoder_transformer';
    if (family === 'decoder_transformer') {
        return layers * (4 * dim * dim + 2 * dim * ffnDim) + 2 * vocab * dim;
    }
    if (family === 'gru') {
        var gruLayers = parseInt(cfg.gru_layers, 10) || 2;
        return gruLayers * 3 * (dim * dim + dim * dim + 2 * dim) + 2 * vocab * dim;
    }
    if (family === 'lstm') {
        var lstmLayers = parseInt(cfg.gru_layers, 10) || 2;
        return lstmLayers * 4 * (dim * dim + dim * dim + 2 * dim) + 2 * vocab * dim;
    }
    if (family === 'gated_conv') {
        var kernel = parseInt(cfg.conv_kernel, 10) || 5;
        var convLayers = parseInt(cfg.n_logical_layers, 10) || 6;
        return convLayers * 2 * dim * dim * kernel + 2 * vocab * dim;
    }
    if (family === 'resnet_lm') {
        var resKernel = parseInt(cfg.conv_kernel, 10) || 5;
        var resLayers = parseInt(cfg.n_logical_layers, 10) || 6;
        return resLayers * (dim * dim * resKernel + dim * dim) + 2 * vocab * dim;
    }
    if (family === 'hybrid_attn_rnn') {
        var hybridLayers = parseInt(cfg.n_logical_layers, 10) || 8;
        var attnLayers = Math.floor((hybridLayers + 1) / 2);
        var rnnLayers = Math.floor(hybridLayers / 2);
        return (
            attnLayers * (4 * dim * dim + 2 * dim * ffnDim) +
            rnnLayers * 3 * (dim * dim + dim * dim + 2 * dim) +
            2 * vocab * dim
        );
    }
    if (family === 'dense_ffn') {
        var ffnLayers = parseInt(cfg.n_logical_layers, 10) || 8;
        return ffnLayers * (2 * dim * ffnDim) + 2 * vocab * dim;
    }
    var experts = parseInt(cfg.num_experts, 10) || 4;
    var moeLayers = parseInt(cfg.n_logical_layers, 10) || 8;
    return moeLayers * (experts * 2 * dim * ffnDim + dim * experts) + 2 * vocab * dim;
}

function estimateJobCost(options) {
    var opts = options || {};
    var steps = Math.max(1, parseInt(opts.steps, 10) || 1);
    var batch = Math.max(1, parseInt(opts.batchSize, 10) || 1);
    var params = 0;
    try {
        params = roughParameterCount(opts.architecture || {});
    } catch (ignore) {
        params = 1e6;
    }
    var tier = 'small';
    var tierRate = 0.8;
    if (params >= 5e7) {
        tier = 'large';
        tierRate = 2.4;
    } else if (params >= 1e6) {
        tier = 'medium';
        tierRate = 1.4;
    }
    var opcodes = (params * steps * batch) / 1e6;
    var credits = Math.max(1, Math.round(opcodes * tierRate * 100) / 100);
    return {
        estimate: true,
        params: params,
        opcodes: Math.round(opcodes * 100) / 100,
        credits: credits,
        tier: tier,
        note: 'Client-side estimate from model size × steps × batch. Not a billing quote.'
    };
}

function formatJobCostEstimate(estimate) {
    if (!estimate) {
        return '—';
    }
    return (
        '~' + formatNumber(estimate.credits) + ' credits · ' +
        formatNumber(estimate.opcodes) + ' opcodes · tier ' + estimate.tier
    );
}

function renderJobCostEstimate(targetId, options) {
    var el = typeof targetId === 'string' ? document.getElementById(targetId) : targetId;
    if (!el) {
        return null;
    }
    var estimate = estimateJobCost(options);
    el.innerHTML =
        '<div class="cost-estimate" role="status" aria-live="polite">' +
        '<div class="cost-estimate-row"><span>Est. credits</span><strong>' +
        escapeHtml(formatNumber(estimate.credits)) + '</strong></div>' +
        '<div class="cost-estimate-row"><span>Opcodes</span><strong>' +
        escapeHtml(formatNumber(estimate.opcodes)) + '</strong></div>' +
        '<div class="cost-estimate-row"><span>Size tier</span><strong>' +
        escapeHtml(estimate.tier) + '</strong></div>' +
        '<p class="cost-estimate-note">' + escapeHtml(estimate.note) + '</p>' +
        '</div>';
    refreshJobCostEstimateFromApi(el, options);
    return estimate;
}

/**
 * Prefer POST /admin/jobs/estimate (or /api proxy) when reachable; keep client heuristic otherwise.
 */
function refreshJobCostEstimateFromApi(el, options) {
    if (!el || typeof fetch !== 'function') {
        return;
    }
    var opts = options || {};
    var body = {
        steps: parseInt(opts.steps, 10) || 1,
        batch_size: parseInt(opts.batchSize, 10) || 1,
        architecture_config: opts.architecture || {},
        priority_tier: opts.priorityTier || 'P1'
    };
    var endpoints = ['/api/jobs/estimate', '/admin/jobs/estimate'];
    var tryNext = function (index) {
        if (index >= endpoints.length) {
            return;
        }
        fetch(endpoints[index], {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        })
            .then(function (resp) {
                if (!resp.ok) {
                    throw new Error('HTTP ' + resp.status);
                }
                return resp.json();
            })
            .then(function (data) {
                if (!data || data.estimate !== true) {
                    throw new Error('bad estimate payload');
                }
                var tier = data.priority_tier || data.tier || 'P1';
                el.innerHTML =
                    '<div class="cost-estimate" role="status" aria-live="polite">' +
                    '<div class="cost-estimate-row"><span>Est. credits</span><strong>' +
                    escapeHtml(formatNumber(data.credits)) + '</strong></div>' +
                    '<div class="cost-estimate-row"><span>Opcodes</span><strong>' +
                    escapeHtml(formatNumber(data.opcodes)) + '</strong></div>' +
                    '<div class="cost-estimate-row"><span>Priority tier</span><strong>' +
                    escapeHtml(String(tier)) + '</strong></div>' +
                    '<p class="cost-estimate-note">' +
                    escapeHtml(data.disclaimer || 'Server estimate — not a billing quote.') +
                    '</p></div>';
            })
            .catch(function () {
                tryNext(index + 1);
            });
    };
    tryNext(0);
}

function setCreateJobWizardStep(modal, step) {
    if (!modal) {
        return;
    }
    var target = Math.max(1, Math.min(3, parseInt(step, 10) || 1));
    modal.dataset.wizardStep = String(target);
    modal.querySelectorAll('[data-wizard-pane]').forEach(function (pane) {
        var match = String(pane.getAttribute('data-wizard-pane')) === String(target);
        pane.hidden = !match;
        pane.setAttribute('aria-hidden', match ? 'false' : 'true');
    });
    modal.querySelectorAll('[data-wizard-step]').forEach(function (btn) {
        var active = String(btn.getAttribute('data-wizard-step')) === String(target);
        btn.classList.toggle('active', active);
        btn.setAttribute('aria-current', active ? 'step' : 'false');
    });
    var backBtn = modal.querySelector('[data-wizard-back]');
    var nextBtn = modal.querySelector('[data-wizard-next]');
    var submitBtn = modal.querySelector('[data-wizard-submit]');
    if (backBtn) {
        backBtn.hidden = target <= 1;
    }
    if (nextBtn) {
        nextBtn.hidden = target >= 3;
    }
    if (submitBtn) {
        submitBtn.hidden = target < 3;
    }
}

function bindCreateJobWizard(modal) {
    if (!modal || modal.dataset.wizardBound === '1') {
        return;
    }
    modal.dataset.wizardBound = '1';
    setCreateJobWizardStep(modal, 1);
    modal.querySelectorAll('[data-wizard-step]').forEach(function (btn) {
        btn.addEventListener('click', function () {
            setCreateJobWizardStep(modal, btn.getAttribute('data-wizard-step'));
        });
    });
    var backBtn = modal.querySelector('[data-wizard-back]');
    var nextBtn = modal.querySelector('[data-wizard-next]');
    if (backBtn) {
        backBtn.addEventListener('click', function () {
            var cur = parseInt(modal.dataset.wizardStep || '1', 10);
            setCreateJobWizardStep(modal, cur - 1);
        });
    }
    if (nextBtn) {
        nextBtn.addEventListener('click', function () {
            var cur = parseInt(modal.dataset.wizardStep || '1', 10);
            if (cur === 2) {
                var dataset = document.getElementById('createJobDataset');
                if (dataset && !dataset.value.trim()) {
                    var err = document.getElementById('createJobError');
                    if (err) {
                        err.textContent = 'Provide a dataset URL (s3:// or https://) before continuing.';
                        err.style.display = 'block';
                    }
                    return;
                }
            }
            var errClear = document.getElementById('createJobError');
            if (errClear) {
                errClear.style.display = 'none';
            }
            setCreateJobWizardStep(modal, cur + 1);
            if (typeof window.refreshCreateJobCostEstimate === 'function') {
                window.refreshCreateJobCostEstimate();
            }
            if (typeof window.refreshCreateJobConfirm === 'function') {
                window.refreshCreateJobConfirm();
            }
        });
    }
}

function openAccessibleModal(modal) {
    if (!modal) {
        return;
    }
    modal.style.display = 'flex';
    modal.setAttribute('aria-hidden', 'false');
    var focusable = modal.querySelector('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
    if (focusable) {
        focusable.focus();
    }
}

function closeAccessibleModal(modal) {
    if (!modal) {
        return;
    }
    modal.style.display = 'none';
    modal.setAttribute('aria-hidden', 'true');
}

function ensureSkipLink() {
    if (document.getElementById('skipToMain')) {
        return;
    }
    var main = document.querySelector('main[role="main"], main');
    if (main && !main.id) {
        main.id = 'main-content';
    }
    var targetId = (main && main.id) || 'main-content';
    var link = document.createElement('a');
    link.id = 'skipToMain';
    link.className = 'skip-link';
    link.href = '#' + targetId;
    link.textContent = 'Skip to main content';
    document.body.insertBefore(link, document.body.firstChild);
}

document.addEventListener('DOMContentLoaded', function () {
    ensureSkipLink();
});
