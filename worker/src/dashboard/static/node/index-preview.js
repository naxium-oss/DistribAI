/** Contrib SPA runtime: nav, orch poll, jobs, credits, benchmark. */
(function (global) {
'use strict';

let pickedJob = null;
        let contribEnabled = true;

        let currentRole = 'node'; // 'admin' or 'node'
        let meshOnline = false;
        let fleetNodes = [];
        let fleetJobs = [];
        let _pollHandle = null;

        function showPage(page) {
            document.querySelectorAll('nav a').forEach(link => {
                link.classList.toggle('active', link.dataset.page === page);
            });
            document.querySelectorAll('.page-section').forEach(section => {
                section.style.display = 'none';
            });
            const target = document.getElementById('page-' + page);
            if (target) target.style.display = 'block';
        }

        function setRole(role) {
            currentRole = role;
            document.body.classList.toggle('role-admin-mode', role === 'admin');
            document.body.classList.toggle('role-node-mode', role === 'node');
            
            document.querySelectorAll('.role-admin').forEach(el => el.style.display = role === 'admin' ? 'block' : 'none');
            document.querySelectorAll('.role-node').forEach(el => el.style.display = role === 'node' ? 'block' : 'none');
            
            document.querySelectorAll('#main-nav a').forEach(a => {
                if (a.getAttribute('href').substring(1) === 'dashboard') a.classList.add('active');
                else a.classList.remove('active');
            });
            
            showPage('dashboard');
        }

        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.has('role')) {
            const r = urlParams.get('role');
            if (r === 'admin' || r === 'node') {
                localStorage.setItem('distribai_role', r);
                setRole(r);
            } else {
                setRole(localStorage.getItem('distribai_role') || 'node');
            }
        } else {
            setRole(localStorage.getItem('distribai_role') || 'node');
        }

        if (urlParams.get('preview') === '1' || localStorage.getItem('distribai_setup_done') === 'true') {
            document.getElementById('setupOverlay').classList.add('hidden');
            if (urlParams.get('preview') === '1') {
                localStorage.setItem('distribai_setup_done', 'true');
            }
        }

        const btnRegistry = document.getElementById('btnDistribaiRegistrySync');
        if (btnRegistry) btnRegistry.addEventListener('click', () => syncDistribaiRegistry());
        const btnPub = document.getElementById('btnPublicRelease');
        if (btnPub) btnPub.addEventListener('click', () => runPublicRelease());

        function detectHardware() {
            const overlay = document.getElementById('setupOverlay');
            if (overlay.classList.contains('hidden')) return;

            document.getElementById('detectedCPU').textContent = 'Detecting...';
            document.getElementById('detectedRAM').textContent = 'Detecting...';
            document.getElementById('detectedGPU').textContent = 'Detecting...';

            fetch('/api/system/info')
                .then(res => res.json())
                .then(data => {
                    const cpuText = `${data.cpu.brand} (${data.cpu.physicalCores}C/${data.cpu.cores}T)`;
                    document.getElementById('detectedCPU').textContent = cpuText;
                    document.getElementById('detectedRAM').textContent = `${data.memory.totalGB} GB Total`;
                    document.getElementById('detectedGPU').textContent = data.gpu ? `${data.gpu.model} (${data.gpu.vramGB}GB VRAM)` : 'No GPU detected';

                    window.detectedHardware = data;

                    document.getElementById('ramSlider').max = data.memory.totalGB;
                    document.getElementById('ramSlider').value = Math.floor(data.memory.totalGB * 0.5);
                    document.getElementById('ramValue').textContent = `${Math.floor(data.memory.totalGB * 0.5)} GB`;

                    updateCPUOptions(data.cpu.physicalCores);

                    if (currentRole === 'node') {
                        document.querySelectorAll('.hardware-edit-btn').forEach(btn => btn.style.display = 'none');
                    }
                })
                .catch(err => {
                    console.error('Failed to get system info:', err);
                    document.getElementById('detectedCPU').textContent = 'Error detecting';
                    document.getElementById('detectedRAM').textContent = 'Error detecting';
                    document.getElementById('detectedGPU').textContent = 'Error detecting';
                });
        }

        function loadResourceSettings() {
            fetch('/api/settings/resources')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('cpuPercentSlider').value = data.cpuPercent;
                    document.getElementById('cpuPercentValue').textContent = data.cpuPercent + '%';
                    document.getElementById('gpuPercentSlider').value = data.gpuPercent;
                    document.getElementById('gpuPercentValue').textContent = data.gpuPercent + '%';
                    document.getElementById('ramPercentSlider').value = data.ramPercent;
                    document.getElementById('ramPercentValue').textContent = data.ramPercent + '%';
                })
                .catch(() => {});
        }

        function saveResourceSettings() {
            const cpuPercent = parseInt(document.getElementById('cpuPercentSlider').value);
            const gpuPercent = parseInt(document.getElementById('gpuPercentSlider').value);
            const ramPercent = parseInt(document.getElementById('ramPercentSlider').value);

            fetch('/api/settings/resources', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ cpuPercent, gpuPercent, ramPercent })
            }).catch(() => {});
        }

        function detectGPU() {
            return 'Loading...';
        }

        function updateCPUOptions(cores) {
            const container = document.getElementById('cpuSelect');
            container.textContent = '';
            const maxCores = Math.min(16, cores);
            const options = [1, 2, 4, 6, 8, 12, 16].filter(c => c <= maxCores);
            if (options.length === 0) options.push(maxCores);
            const selected = Math.min(4, maxCores);
            options.forEach((opt) => {
                const div = document.createElement('div');
                div.className = 'cpu-option' + (opt === selected ? ' selected' : '');
                div.dataset.cores = opt;
                div.textContent = opt;
                div.addEventListener('click', () => {
                    document.querySelectorAll('.cpu-option').forEach(e => e.classList.remove('selected'));
                    div.classList.add('selected');
                    document.getElementById('cpuValue').textContent = `${opt} Cores`;
                });
                container.appendChild(div);
            });
            document.getElementById('cpuValue').textContent = `${selected} Cores`;
        }

        detectHardware();
        loadResourceSettings();

        document.getElementById('cpuPercentSlider').addEventListener('input', function() {
            document.getElementById('cpuPercentValue').textContent = `${this.value}%`;
        });
        document.getElementById('gpuPercentSlider').addEventListener('input', function() {
            document.getElementById('gpuPercentValue').textContent = `${this.value}%`;
        });
        document.getElementById('ramPercentSlider').addEventListener('input', function() {
            document.getElementById('ramPercentValue').textContent = `${this.value}%`;
        });

        let currentStep = 1;
        function showStep(step) {
            document.getElementById('setupStep1').style.display = step === 1 ? 'block' : 'none';
            document.getElementById('setupStep2').style.display = step === 2 ? 'block' : 'none';
            document.getElementById('setupStep3').style.display = step === 3 ? 'block' : 'none';
            document.getElementById('step1').classList.toggle('active', step >= 1);
            document.getElementById('step2').classList.toggle('active', step >= 2);
            document.getElementById('step3').classList.toggle('active', step >= 3);
            currentStep = step;
        }

        document.getElementById('nextStep').addEventListener('click', () => showStep(2));
        document.getElementById('nextStep2').addEventListener('click', () => showStep(3));
        document.getElementById('backStep2').addEventListener('click', () => showStep(1));
        document.getElementById('backStep3').addEventListener('click', () => showStep(2));

        document.getElementById('finishSetup').addEventListener('click', async () => {
            const cpuPercent = parseInt(document.getElementById('cpuPercentSlider').value);
            const gpuPercent = parseInt(document.getElementById('gpuPercentSlider').value);
            const ramPercent = parseInt(document.getElementById('ramPercentSlider').value);
            const region = document.getElementById('regionSelect').value;

            try {
                await fetch('/api/settings/resources', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ cpuPercent, gpuPercent, ramPercent, region })
                });
            } catch (_) {}

            document.getElementById('setupOverlay').classList.add('hidden');
            localStorage.setItem('distribai_setup_done', 'true');
            showNotification('Node configured and ready to contribute', 'success');
        });

        async function syncDistribaiRegistry() {
            const statusEl = document.getElementById('distribai-registry-status');
            const resultsEl = document.getElementById('distribai-registry-results');
            if (!statusEl || !resultsEl) return;

            statusEl.textContent = '<div class="status-dot"></div> Syncing...';
            statusEl.classList.remove('online');
            
            try {
                const resp = await fetch('/api/admin/distribai/registry/sync', { method: 'POST' });
                const data = await resp.json();
                
                if (data.ok) {
                    statusEl.textContent = '<div class="status-dot"></div> Synced ✓';
                    statusEl.classList.add('online');
                    showNotification('DistribAI model registry updated successfully', 'success');
                    
                    resultsEl.textContent = `
                        <div style="font-size: 0.85rem; color: var(--text-secondary);">
                            <p>Found ${Object.keys(data.registry || {}).length} models:</p>
                            <ul style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px;">
                                ${Object.keys(data.registry || {}).map(name => `
                                    <li style="background: var(--bg-tertiary); padding: 4px 10px; border-radius: 4px; border: 1px solid var(--border);">
                                        ${name}
                                    </li>
                                `).join('')}
                            </ul>
                        </div>
                    `;
                } else {
                    throw new Error(data.error || 'Registry sync failed');
                }
            } catch (err) {
                statusEl.textContent = '<div class="status-dot" style="background: var(--error)"></div> Sync Failed';
                showNotification(`Registry sync error: ${err.message}`, 'error');
            }
        }

        async function runPublicRelease() {
            const statusEl = document.getElementById('public-release-status');
            const logEl = document.getElementById('public-release-log');
            if (!statusEl || !logEl) return;
            statusEl.textContent = '<div class="status-dot"></div> Publishing…';
            statusEl.classList.remove('online');
            logEl.style.display = 'none';
            logEl.textContent = '';
            try {
                const resp = await fetch('/api/admin/public-release/publish', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ push: true }),
                });
                const data = await resp.json();
                if (data.ok) {
                    statusEl.textContent = '<div class="status-dot"></div> Mirror updated';
                    statusEl.classList.add('online');
                    showNotification('Public mirror publish finished', 'success');
                    if (data.log) {
                        logEl.textContent = data.log;
                        logEl.style.display = 'block';
                    }
                } else {
                    throw new Error(data.error || data.detail || 'Publish failed');
                }
            } catch (err) {
                statusEl.textContent = '<div class="status-dot" style="background:var(--error)"></div> Failed';
                showNotification(`Publish: ${err.message}`, 'error');
            }
        }

        async function loadDocs() {
            const navEl = document.getElementById('docs-nav');
            const contentEl = document.getElementById('docs-content');
            
            try {
                const resp = await fetch('/api/docs/list');
                const docs = await resp.json();
                
                navEl.textContent = docs.map(doc => `
                    <div class="queue-item" onclick="viewDoc('${doc.path}')">
                        <div class="queue-details">
                            <div class="queue-title">${doc.title}</div>
                        </div>
                    </div>
                `).join('');
                
                if (docs.length > 0) viewDoc(docs[0].path);
            } catch (err) {
                navEl.textContent = '<div style="padding: 10px; color: var(--text-muted);">Failed to load docs list</div>';
            }
        }

        async function viewDoc(path) {
            const contentEl = document.getElementById('docs-content');
            contentEl.textContent = '<p style="color: var(--text-muted);">Loading document...</p>';
            
            try {
                const resp = await fetch(`/api/docs/read?path=${encodeURIComponent(path)}`);
                const data = await resp.json();
                
                contentEl.textContent = `
                    <h2 style="margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 10px;">${data.title}</h2>
                    <div class="markdown-body">
                        ${data.html}
                    </div>
                `;
            } catch (err) {
                contentEl.textContent = '<p style="color: var(--error);">Failed to load document content</p>';
            }
        }

        function showEditModal(field, currentValue, callback) {
            const modal = document.createElement('div');
            modal.className = 'edit-modal';
            modal.textContent = `
                <div class="edit-modal-title">Edit ${field}</div>
                <input type="text" id="editFieldInput" value="${currentValue}">
                <div class="edit-modal-buttons">
                    <button class="btn btn-secondary" id="cancelEdit">Cancel</button>
                    <button class="btn btn-primary" id="saveEdit">Save</button>
                </div>
            `;
            
            const overlay = document.createElement('div');
            overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.7);z-index:1000;';
            document.body.appendChild(overlay);
            document.body.appendChild(modal);
            
            const input = modal.querySelector('#editFieldInput');
            input.focus();
            input.select();
            
            modal.querySelector('#cancelEdit').onclick = () => {
                overlay.remove();
                modal.remove();
            };
            
            modal.querySelector('#saveEdit').onclick = () => {
                callback(input.value);
                overlay.remove();
                modal.remove();
            };
            
            input.onkeydown = (e) => {
                if (e.key === 'Enter') {
                    callback(input.value);
                    overlay.remove();
                    modal.remove();
                } else if (e.key === 'Escape') {
                    overlay.remove();
                    modal.remove();
                }
            };
        }

        document.getElementById('editCPU').onclick = () => {
            showEditModal('CPU Cores', 'e.g., 8 Cores', (val) => {
                document.getElementById('detectedCPU').textContent = val;
            });
        };

        document.getElementById('editRAM').onclick = () => {
            showEditModal('RAM', 'e.g., 32 GB', (val) => {
                document.getElementById('detectedRAM').textContent = val;
            });
        };

        document.getElementById('editGPU').onclick = () => {
            showEditModal('GPU', 'e.g., RTX 4080', (val) => {
                document.getElementById('detectedGPU').textContent = val;
            });
        };

        document.getElementById('contributionToggle').addEventListener('change', function() {
            var self = this;
            contribEnabled = self.checked;
            var status = document.getElementById('connectionStatus');

            setButtonLoading(self, true);

            if (contribEnabled) {
                status.classList.add('online');
                status.textContent = '<span class="status-dot"></span><span>Online</span>';
            } else {
                status.classList.remove('online');
                status.textContent = '<span class="status-dot"></span><span>Paused</span>';
            }

            var promises = [];
            if (fleetNodes.length) {
                fleetNodes.forEach(function(n) {
                    var promise = fetchWithRetry('/api/worker/nodes/' + encodeURIComponent(n.node_id) + '/contributing', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ contributing: contribEnabled }),
                    }, 3).catch(function(err) {
                        console.error('Failed to update contributing state for node:', n.node_id, err);
                    });
                    promises.push(promise);
                });
            }

            Promise.all(promises).then(function() {
                setButtonLoading(self, false);
                showToast(contribEnabled ? 'Node started contributing' : 'Node paused', 'success', 4000);
            }).catch(function() {
                setButtonLoading(self, false);
                showToast('Some nodes may not have updated. Please try again.', 'warning', 5000);
            });
        });

        function formatTime(ms) {
            const seconds = Math.floor(ms / 1000);
            const mins = Math.floor(seconds / 60);
            const secs = seconds % 60;
            return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
        }

        document.getElementById('voteList').addEventListener('click', function(e) {
            const opt = e.target.closest('.vote-option');
            if (!opt) return;
            document.querySelectorAll('.vote-option').forEach(o => o.classList.remove('selected'));
            opt.classList.add('selected');
            pickedJob = opt.dataset.job;
        });

        function showNotification(message, type) {
            const notification = document.getElementById('notification');
            notification.textContent = message;
            notification.className = `notification ${type} show`;
            setTimeout(() => notification.classList.remove('show'), 3000);

            addActivityItem(message, type);
        }

        let activityItems = JSON.parse(localStorage.getItem('distribai_activity') || '[]');

        function addActivityItem(message, type = 'info') {
            const item = {
                id: Date.now(),
                message,
                type,
                timestamp: new Date().toISOString()
            };
            activityItems.unshift(item);
            if (activityItems.length > 50) activityItems.pop();
            localStorage.setItem('distribai_activity', JSON.stringify(activityItems));
            renderActivityFeed();

            const badge = document.getElementById('activityBadge');
            if (!document.getElementById('activityFeed').classList.contains('show')) {
                badge.style.display = 'block';
            }
        }

        function renderActivityFeed() {
            const list = document.getElementById('activityList');
            if (activityItems.length === 0) {
                list.textContent = '<div style="text-align:center;padding:20px;color:var(--text-muted);font-size:0.8rem;">No activity yet</div>';
                return;
            }

            const icons = {
                success: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>',
                error: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
                warning: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
                info: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
            };

            list.textContent = activityItems.slice(0, 20).map(item => {
                const time = new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                return `
                    <div class="activity-item">
                        <div class="activity-icon ${item.type}">${icons[item.type] || icons.info}</div>
                        <div class="activity-content">
                            <div class="activity-message">${escapeHtml(item.message)}</div>
                            <div class="activity-time">${time}</div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function clearActivityFeed() {
            activityItems = [];
            localStorage.removeItem('distribai_activity');
            renderActivityFeed();
        }

        function toggleActivityFeed() {
            const feed = document.getElementById('activityFeed');
            feed.classList.toggle('show');
            if (feed.classList.contains('show')) {
                document.getElementById('activityBadge').style.display = 'none';
                renderActivityFeed();
            }
        }

        document.getElementById('activityFeedBtn').addEventListener('click', toggleActivityFeed);

        function openShortcutsModal() {
            document.getElementById('shortcutsModal').classList.add('show');
        }

        function closeShortcutsModal() {
            document.getElementById('shortcutsModal').classList.remove('show');
        }

        document.getElementById('shortcutsModal').addEventListener('click', function(e) {
            if (e.target === this) closeShortcutsModal();
        });

        document.addEventListener('keydown', function(e) {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
                if (e.key === 'Escape') {
                    e.target.blur();
                }
                return;
            }

            switch(e.key) {
                case '?':
                    e.preventDefault();
                    openShortcutsModal();
                    break;
                case 'a':
                case 'A':
                    e.preventDefault();
                    toggleActivityFeed();
                    break;
                case '1':
                    e.preventDefault();
                    navigateToPage('dashboard');
                    break;
                case '2':
                    e.preventDefault();
                    navigateToPage('jobs');
                    break;
                case '3':
                    e.preventDefault();
                    navigateToPage('credits');
                    break;
                case '4':
                    e.preventDefault();
                    navigateToPage('settings');
                    break;
                case 'Escape':
                    closeShortcutsModal();
                    document.getElementById('activityFeed').classList.remove('show');
                    break;
            }

            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                document.getElementById('globalSearch').focus();
            }
        });

        function navigateToPage(page) {
            document.querySelectorAll('#main-nav a').forEach(a => {
                if (a.dataset.page === page) {
                    a.click();
                }
            });
        }

        const searchInput = document.getElementById('globalSearch');
        const searchResults = document.getElementById('searchResults');
        let searchTimeout;

        searchInput.addEventListener('input', function() {
            clearTimeout(searchTimeout);
            const query = this.value.trim().toLowerCase();

            if (query.length < 2) {
                searchResults.classList.remove('show');
                return;
            }

            searchTimeout = setTimeout(() => performSearch(query), 300);
        });

        searchInput.addEventListener('focus', function() {
            if (this.value.trim().length >= 2) {
                searchResults.classList.add('show');
            }
        });

        searchInput.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                this.blur();
                searchResults.classList.remove('show');
            }
        });

        document.addEventListener('click', function(e) {
            if (!e.target.closest('.search-box')) {
                searchResults.classList.remove('show');
            }
        });

        async function performSearch(query) {
            const results = [];

            try {
                const jobsRes = await fetch('/api/worker/jobs').then(r => r.json()).catch(() => ({jobs: []}));
                const jobs = jobsRes.jobs || [];
                const matchingJobs = jobs.filter(j =>
                    (j.job_id && j.job_id.toLowerCase().includes(query)) ||
                    (j.model_name && j.model_name.toLowerCase().includes(query)) ||
                    (j.status && j.status.toLowerCase().includes(query))
                );

                matchingJobs.forEach(j => {
                    results.push({
                        type: 'job',
                        title: `Job: ${j.model_name || j.job_id}`,
                        subtitle: `${j.status} - ${j.steps || 0} steps`,
                        action: () => { navigateToPage('jobs'); }
                    });
                });

                const nodesRes = await fetch('/api/worker/nodes').then(r => r.json()).catch(() => ({nodes: []}));
                const nodes = nodesRes.nodes || [];
                const matchingNodes = nodes.filter(n =>
                    (n.node_id && n.node_id.toLowerCase().includes(query)) ||
                    (n.status && n.status.toLowerCase().includes(query))
                );

                matchingNodes.forEach(n => {
                    results.push({
                        type: 'node',
                        title: `Node: ${n.node_id}`,
                        subtitle: `Status: ${n.status}`,
                        action: () => { navigateToPage('dashboard'); }
                    });
                });

            } catch (err) {
                console.error('Search error:', err);
            }

            renderSearchResults(results, query);
        }

        function renderSearchResults(results, query) {
            if (results.length === 0) {
                searchResults.textContent = '<div style="padding:12px;color:var(--text-muted);font-size:0.85rem;">No results for "' + escapeHtml(query) + '"</div>';
            } else {
                searchResults.textContent = results.map(function(r) {
                    return '<div class="search-result-item" data-type="' + r.type + '">' +
                        '<div style="font-weight:500;">' + escapeHtml(r.title) + '</div>' +
                        '<div style="font-size:0.75rem;color:var(--text-muted);margin-top:2px;">' + escapeHtml(r.subtitle) + '</div>' +
                    '</div>';
                }).join('');

                searchResults.querySelectorAll('.search-result-item').forEach(function(item, index) {
                    item.addEventListener('click', function() {
                        results[index].action();
                        searchResults.classList.remove('show');
                        searchInput.value = '';
                    });
                });
            }
            searchResults.classList.add('show');
        }

        async function exportData(type) {
            try {
                var data, filename;
                var dateStr = new Date().toISOString().split('T')[0];

                if (type === 'jobs') {
                    var jobsRes = await fetch('/api/worker/jobs').then(function(r) { return r.json(); });
                    data = JSON.stringify(jobsRes.jobs || [], null, 2);
                    filename = 'distribai-jobs-' + dateStr + '.json';
                } else if (type === 'credits') {
                    var creditsRes = await fetch('/api/worker/credits').then(function(r) { return r.json(); });
                    data = JSON.stringify(creditsRes, null, 2);
                    filename = 'distribai-credits-' + dateStr + '.json';
                } else if (type === 'nodes') {
                    var nodesRes = await fetch('/api/worker/nodes').then(function(r) { return r.json(); });
                    data = JSON.stringify(nodesRes.nodes || [], null, 2);
                    filename = 'distribai-nodes-' + dateStr + '.json';
                } else {
                    return;
                }

                var blob = new Blob([data], { type: 'application/json' });
                var url = URL.createObjectURL(blob);
                var a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);

                showNotification('Exported ' + type + ' successfully', 'success');
            } catch (err) {
                showNotification('Failed to export ' + type + ': ' + err.message, 'error');
            }
        }

        renderActivityFeed();

        function showToast(message, type, duration) {
            duration = duration || 5000;
            var stack = document.getElementById('toastStack');
            var toast = document.createElement('div');
            toast.className = 'toast ' + type;

            var icons = {
                success: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>',
                error: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
                warning: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
                info: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
            };

            toast.textContent = '<div class="toast-icon">' + (icons[type] || icons.info) + '</div>' +
                '<span>' + escapeHtml(message) + '</span>' +
                '<button class="toast-close" onclick="this.parentElement.remove()">&times;</button>';

            stack.appendChild(toast);

            addActivityItem(message, type);

            setTimeout(function() {
                toast.classList.add('removing');
                setTimeout(function() {
                    if (toast.parentElement) toast.remove();
                }, 300);
            }, duration);
        }

        var confirmCallback = null;

        function showConfirmModal(options) {
            options = options || {};
            var modal = document.getElementById('confirmModal');
            var icon = document.getElementById('confirmIcon');
            var title = document.getElementById('confirmTitle');
            var message = document.getElementById('confirmMessage');
            var actionBtn = document.getElementById('confirmActionBtn');

            icon.className = 'confirm-icon ' + (options.type || 'warning');
            icon.textContent = options.icon || '!';
            title.textContent = options.title || 'Confirm Action';
            message.textContent = options.message || 'Are you sure?';
            actionBtn.textContent = options.actionText || 'Confirm';
            actionBtn.className = 'confirm-btn confirm-btn-' + (options.actionType || 'primary');

            confirmCallback = options.onConfirm;
            modal.classList.add('show');
        }

        function closeConfirmModal() {
            document.getElementById('confirmModal').classList.remove('show');
            confirmCallback = null;
        }

        function executeConfirmAction() {
            if (typeof confirmCallback === 'function') {
                confirmCallback();
            }
            closeConfirmModal();
        }

        document.getElementById('confirmModal').addEventListener('click', function(e) {
            if (e.target === this) closeConfirmModal();
        });

        var _wasOffline = false;

        function updateOnlineStatus() {
            var isOffline = !navigator.onLine;
            var banner = document.getElementById('offlineBanner');
            var body = document.body;

            if (isOffline) {
                _wasOffline = true;
                banner.classList.add('show');
                body.classList.add('offline');
                showToast('You are offline. Some features may be unavailable.', 'warning', 8000);
            } else if (_wasOffline) {
                banner.classList.remove('show');
                body.classList.remove('offline');
                showToast('Back online!', 'success', 3000);
                _wasOffline = false;
            } else {
                banner.classList.remove('show');
                body.classList.remove('offline');
            }
        }

        window.addEventListener('online', updateOnlineStatus);
        window.addEventListener('offline', updateOnlineStatus);
        updateOnlineStatus();

        function showSkeleton(containerId, type, count) {
            count = count || 1;
            var container = document.getElementById(containerId);
            if (!container) return;

            var html = '';
            for (var i = 0; i < count; i++) {
                if (type === 'card') {
                    html += '<div class="skeleton skeleton-card"></div>';
                } else if (type === 'text') {
                    html += '<div class="skeleton skeleton-text"></div>';
                } else if (type === 'title') {
                    html += '<div class="skeleton skeleton-title"></div>';
                }
            }
            container.textContent = html;
        }

        function hideSkeleton(containerId, content) {
            var container = document.getElementById(containerId);
            if (container) {
                container.textContent = content || '';
            }
        }

        function setButtonLoading(btn, loading) {
            if (loading) {
                btn.classList.add('btn-loading');
                btn.disabled = true;
            } else {
                btn.classList.remove('btn-loading');
                btn.disabled = false;
            }
        }

        function fetchWithRetry(url, options, retries) {
            retries = retries || 3;
            options = options || {};

            return new Promise(function(resolve, reject) {
                var attempt = 0;

                function tryFetch() {
                    attempt++;
                    fetch(url, options)
                        .then(function(response) {
                            if (!response.ok && attempt < retries) {
                                setTimeout(tryFetch, 1000 * attempt);
                            } else {
                                resolve(response);
                            }
                        })
                        .catch(function(error) {
                            if (attempt < retries) {
                                setTimeout(tryFetch, 1000 * attempt);
                            } else {
                                reject(error);
                            }
                        });
                }

                tryFetch();
            });
        }

        function setupFormAutoSave(formId, storageKey) {
            var form = document.getElementById(formId);
            if (!form) return;

            var saved = localStorage.getItem(storageKey);
            if (saved) {
                try {
                    var data = JSON.parse(saved);
                    Object.keys(data).forEach(function(key) {
                        var input = form.querySelector('[name="' + key + '"]');
                        if (input) input.value = data[key];
                    });
                } catch(e) {}
            }

            form.addEventListener('change', function() {
                var data = {};
                form.querySelectorAll('input, select, textarea').forEach(function(input) {
                    if (input.name) data[input.name] = input.value;
                });
                localStorage.setItem(storageKey, JSON.stringify(data));
            });

            form.addEventListener('submit', function() {
                localStorage.removeItem(storageKey);
            });
        }

        function setProgressRing(elementId, percent) {
            var circle = document.getElementById(elementId);
            if (!circle) return;

            var radius = circle.r.baseVal.value;
            var circumference = radius * 2 * Math.PI;
            var offset = circumference - (percent / 100) * circumference;

            circle.style.strokeDasharray = circumference + ' ' + circumference;
            circle.style.strokeDashoffset = offset;
        }

        function handleApiError(error, context) {
            console.error('API Error (' + context + '):', error);

            var message = 'An error occurred';
            if (error.message && error.message.includes('Failed to fetch')) {
                message = 'Network error. Please check your connection.';
            } else if (error.message) {
                message = error.message;
            }

            showToast(message, 'error', 8000);
            addActivityItem(context + ' failed: ' + message, 'error');
        }

        function scrollToElement(elementId) {
            var element = document.getElementById(elementId);
            if (element) {
                element.scrollIntoView({ behavior: 'smooth', block: 'center' });
                element.style.animation = 'none';
                element.offsetHeight; // Trigger reflow
                element.style.animation = 'badge-pulse 0.5s ease';
            }
        }

        function copyToClipboard(text, successMessage) {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(text).then(function() {
                    showToast(successMessage || 'Copied to clipboard!', 'success', 3000);
                }).catch(function() {
                    showToast('Failed to copy', 'error', 3000);
                });
            } else {
                var textarea = document.createElement('textarea');
                textarea.value = text;
                textarea.style.position = 'fixed';
                textarea.style.opacity = '0';
                document.body.appendChild(textarea);
                textarea.select();
                try {
                    document.execCommand('copy');
                    showToast(successMessage || 'Copied to clipboard!', 'success', 3000);
                } catch (e) {
                    showToast('Failed to copy', 'error', 3000);
                }
                document.body.removeChild(textarea);
            }
        }

        function debounce(func, wait) {
            var timeout;
            return function() {
                var context = this, args = arguments;
                clearTimeout(timeout);
                timeout = setTimeout(function() {
                    func.apply(context, args);
                }, wait);
            };
        }

        function throttle(func, limit) {
            var inThrottle;
            return function() {
                var context = this, args = arguments;
                if (!inThrottle) {
                    func.apply(context, args);
                    inThrottle = true;
                    setTimeout(function() { inThrottle = false; }, limit);
                }
            };
        }

        var performanceChart = null;
        var chartDataHistory = {
            labels: [],
            cpu: [],
            memory: [],
            gpu: []
        };

        function initPerformanceChart() {
            var ctx = document.getElementById('performanceChart');
            if (!ctx) return;

            performanceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: chartDataHistory.labels,
                    datasets: [{
                        label: 'CPU Usage',
                        data: chartDataHistory.cpu,
                        borderColor: getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#22c55e',
                        backgroundColor: 'rgba(34, 197, 94, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    }, {
                        label: 'Memory',
                        data: chartDataHistory.memory,
                        borderColor: getComputedStyle(document.documentElement).getPropertyValue('--success').trim() || '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    }, {
                        label: 'GPU Usage',
                        data: chartDataHistory.gpu,
                        borderColor: getComputedStyle(document.documentElement).getPropertyValue('--warning').trim() || '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            backgroundColor: 'rgba(10, 10, 10, 0.9)',
                            titleColor: '#ededed',
                            bodyColor: '#a1a1a1',
                            borderColor: '#262626',
                            borderWidth: 1,
                            padding: 10,
                            displayColors: true,
                            callbacks: {
                                label: function(context) {
                                    return context.dataset.label + ': ' + Math.round(context.parsed.y) + '%';
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            display: false
                        },
                        y: {
                            beginAtZero: true,
                            max: 100,
                            grid: {
                                color: 'rgba(38, 38, 38, 0.5)',
                                drawBorder: false
                            },
                            ticks: {
                                color: '#a1a1a1',
                                font: {
                                    size: 11
                                },
                                callback: function(value) {
                                    return value + '%';
                                }
                            }
                        }
                    }
                }
            });
        }

        function updatePerformanceChart(cpu, memory, gpu) {
            if (!performanceChart) return;

            var now = new Date();
            var timeLabel = now.getHours() + ':' + String(now.getMinutes()).padStart(2, '0') + ':' + String(now.getSeconds()).padStart(2, '0');

            chartDataHistory.labels.push(timeLabel);
            chartDataHistory.cpu.push(cpu || 0);
            chartDataHistory.memory.push(memory || 0);
            chartDataHistory.gpu.push(gpu || 0);

            if (chartDataHistory.labels.length > 20) {
                chartDataHistory.labels.shift();
                chartDataHistory.cpu.shift();
                chartDataHistory.memory.shift();
                chartDataHistory.gpu.shift();
            }

            performanceChart.data.labels = chartDataHistory.labels;
            performanceChart.data.datasets[0].data = chartDataHistory.cpu;
            performanceChart.data.datasets[1].data = chartDataHistory.memory;
            performanceChart.data.datasets[2].data = chartDataHistory.gpu;
            performanceChart.update('none');
        }

        document.querySelectorAll('.time-range-btn').forEach(function(btn) {
            btn.addEventListener('click', function() {
                document.querySelectorAll('.time-range-btn').forEach(function(b) { b.classList.remove('active'); });
                this.classList.add('active');
                var range = this.dataset.range;
                showToast('Time range changed to ' + range, 'info', 2000);
            });
        });

        var currentTheme = localStorage.getItem('distribai_theme') || 'dark';

        function applyTheme(theme) {
            if (theme === 'light') {
                document.documentElement.style.setProperty('--bg', '#ffffff');
                document.documentElement.style.setProperty('--bg-secondary', '#f5f5f5');
                document.documentElement.style.setProperty('--bg-tertiary', '#eeeeee');
                document.documentElement.style.setProperty('--bg-elevated', '#ffffff');
                document.documentElement.style.setProperty('--text', '#1a1a1a');
                document.documentElement.style.setProperty('--text-secondary', '#666666');
                document.documentElement.style.setProperty('--text-muted', '#999999');
                document.documentElement.style.setProperty('--border', '#e0e0e0');
                document.documentElement.style.setProperty('--border-hover', '#cccccc');
            } else {
                document.documentElement.style.setProperty('--bg', '#000000');
                document.documentElement.style.setProperty('--bg-secondary', '#0a0a0a');
                document.documentElement.style.setProperty('--bg-tertiary', '#111111');
                document.documentElement.style.setProperty('--bg-elevated', '#1a1a1a');
                document.documentElement.style.setProperty('--text', '#ededed');
                document.documentElement.style.setProperty('--text-secondary', '#a1a1a1');
                document.documentElement.style.setProperty('--text-muted', '#737373');
                document.documentElement.style.setProperty('--border', '#262626');
                document.documentElement.style.setProperty('--border-hover', '#3f3f3f');
            }
            currentTheme = theme;
            localStorage.setItem('distribai_theme', theme);

            if (performanceChart) {
                performanceChart.data.datasets[0].borderColor = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
                performanceChart.data.datasets[1].borderColor = getComputedStyle(document.documentElement).getPropertyValue('--success').trim();
                performanceChart.data.datasets[2].borderColor = getComputedStyle(document.documentElement).getPropertyValue('--warning').trim();
                performanceChart.update();
            }
        }

        document.getElementById('themeToggle').addEventListener('click', function() {
            var newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            applyTheme(newTheme);
            showToast('Theme changed to ' + newTheme + ' mode', 'success', 2000);
        });

        applyTheme(currentTheme);

        var currentJobFilter = { status: 'all', model: '' };
        var currentSort = { field: 'date', direction: 'desc' };
        var selectedJobs = new Set();

        function filterJobs() {
            var statusFilter = document.getElementById('jobStatusFilter').value;
            var modelFilter = document.getElementById('jobModelFilter').value.toLowerCase();

            var jobItems = document.querySelectorAll('.job-item');
            var visibleCount = 0;

            jobItems.forEach(function(item) {
                var status = item.dataset.status || '';
                var model = item.dataset.model || '';

                var statusMatch = statusFilter === 'all' || status === statusFilter;
                var modelMatch = !modelFilter || model.toLowerCase().includes(modelFilter);

                if (statusMatch && modelMatch) {
                    item.style.display = '';
                    visibleCount++;
                } else {
                    item.style.display = 'none';
                }
            });

            var container = document.getElementById('activeJobsList');
            var emptyMsg = container.querySelector('.empty-state');
            if (visibleCount === 0 && !emptyMsg) {
                showSkeleton('activeJobsList', 'text', 3);
            }
        }

        function sortJobs(field) {
            currentSort.field = field;
            currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';

            var container = document.getElementById('activeJobsList');
            var items = Array.from(container.querySelectorAll('.job-item'));

            items.sort(function(a, b) {
                var aVal, bVal;
                if (field === 'date') {
                    aVal = parseInt(a.dataset.timestamp || 0);
                    bVal = parseInt(b.dataset.timestamp || 0);
                } else if (field === 'progress') {
                    aVal = parseInt(a.dataset.progress || 0);
                    bVal = parseInt(b.dataset.progress || 0);
                }

                if (currentSort.direction === 'asc') {
                    return aVal - bVal;
                } else {
                    return bVal - aVal;
                }
            });

            items.forEach(function(item) {
                container.appendChild(item);
            });

            document.querySelectorAll('.filter-btn').forEach(function(btn) {
                btn.classList.remove('active');
            });
            document.getElementById('sortBy' + field.charAt(0).toUpperCase() + field.slice(1)).classList.add('active');
        }

        function clearJobFilters() {
            document.getElementById('jobStatusFilter').value = 'all';
            document.getElementById('jobModelFilter').value = '';
            filterJobs();
            showToast('Filters cleared', 'info', 2000);
        }

        document.getElementById('jobStatusFilter').addEventListener('change', filterJobs);
        document.getElementById('jobModelFilter').addEventListener('input', debounce(filterJobs, 300));

        function toggleJobSelection(jobId, checkbox) {
            if (checkbox.checked) {
                selectedJobs.add(jobId);
            } else {
                selectedJobs.delete(jobId);
            }
            updateBulkActionsBar();
        }

        function updateBulkActionsBar() {
            var bar = document.getElementById('bulkActionsBar');
            var countEl = document.getElementById('selectedCount');

            if (selectedJobs.size > 0) {
                bar.classList.add('show');
                countEl.textContent = selectedJobs.size;
            } else {
                bar.classList.remove('show');
            }
        }

        function clearSelection() {
            selectedJobs.clear();
            document.querySelectorAll('.select-checkbox').forEach(function(cb) {
                cb.checked = false;
            });
            updateBulkActionsBar();
        }

        function bulkCancelJobs() {
            if (selectedJobs.size === 0) return;

            showConfirmModal({
                type: 'danger',
                icon: '⚠',
                title: 'Cancel Multiple Jobs?',
                message: 'Are you sure you want to cancel ' + selectedJobs.size + ' selected jobs?',
                actionText: 'Cancel All',
                actionType: 'danger',
                onConfirm: function() {
                    var promises = [];
                    selectedJobs.forEach(function(jobId) {
                        var promise = fetchWithRetry('/api/jobs/' + jobId + '/cancel', { method: 'POST' }, 3);
                        promises.push(promise);
                    });

                    Promise.all(promises).then(function() {
                        showToast('Cancelled ' + selectedJobs.size + ' jobs', 'success', 4000);
                        clearSelection();
                        dashPollOrchestrator();
                    }).catch(function(err) {
                        handleApiError(err, 'Bulk job cancellation');
                    });
                }
            });
        }

        function bulkExportJobs() {
            if (selectedJobs.size === 0) return;

            var selectedJobIds = Array.from(selectedJobs);
            var data = JSON.stringify({ jobIds: selectedJobIds }, null, 2);
            var blob = new Blob([data], { type: 'application/json' });
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = 'selected-jobs-' + new Date().toISOString().split('T')[0] + '.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            showToast('Exported ' + selectedJobs.size + ' jobs', 'success', 3000);
        }

        function updateSystemStats() {
            fetch('/api/stats')
                .then(r => r.json())
                .then(d => {
                    var memPct = d.memory ? d.memory.usedPercent : 0;
                    document.getElementById('vramBar').style.width = memPct + '%';
                    document.getElementById('vramPercent').textContent = Math.round(memPct) + '%';

                    var cpuPct = d.cpu ? d.cpu.usage : 0;
                    document.getElementById('utilBar').style.width = cpuPct + '%';
                    document.getElementById('utilPercent').textContent = Math.round(cpuPct) + '%';

                    var gpuUsage = d.gpu && d.gpu.utilization ? d.gpu.utilization : 0;
                    updatePerformanceChart(cpuPct, memPct, gpuUsage);

                    var gpu = d.gpu;
                    if (gpu && gpu.vramTotal > 0) {
                        var vramUsed = gpu.vramUsed != null ? gpu.vramUsed : 0;
                        var vramPct = Math.round(vramUsed / gpu.vramTotal * 100);
                        document.getElementById('vramGpuBar').style.width = Math.min(vramPct, 100) + '%';
                        var vramUsedGB = (vramUsed / 1024).toFixed(1);
                        var vramTotalGB = (gpu.vramTotal / 1024).toFixed(1);
                        document.getElementById('vramGpuPercent').textContent = vramUsedGB + '/' + vramTotalGB + 'G';
                    } else {
                        document.getElementById('vramGpuBar').style.width = '0%';
                        document.getElementById('vramGpuPercent').textContent = '—';
                    }

                    if (gpu && gpu.utilization != null) {
                        document.getElementById('utilGpuBar').style.width = gpu.utilization + '%';
                        document.getElementById('utilGpuPercent').textContent = Math.round(gpu.utilization) + '%';
                    } else {
                        document.getElementById('utilGpuBar').style.width = '0%';
                        document.getElementById('utilGpuPercent').textContent = '—';
                    }

                    if (d.cpu && d.cpu.temp != null) {
                        document.getElementById('cpuTempBar').style.width = Math.min(d.cpu.temp, 100) + '%';
                        document.getElementById('cpuTempPercent').textContent = Math.round(d.cpu.temp) + 'C';
                    } else {
                        document.getElementById('cpuTempBar').style.width = '0%';
                        document.getElementById('cpuTempPercent').textContent = '—';
                    }

                    if (gpu && gpu.temperature != null) {
                        const tempPct = Math.min(gpu.temperature, 100);
                        document.getElementById('tempBar').style.width = tempPct + '%';
                        document.getElementById('tempPercent').textContent = Math.round(gpu.temperature) + 'C';
                    } else {
                        document.getElementById('tempBar').style.width = '0%';
                        document.getElementById('tempPercent').textContent = '—';
                    }
                })
                .catch(() => {});
        }

        let dashboardStatus = { orchestrator: { ok: false } };

        async function checkDashboardStatus() {
            try {
                const resp = await fetch('/api/status');
                const status = await resp.json();
                dashboardStatus = status;

                const badge = document.getElementById('connectionStatus');
                const orchUrl = status.orchestrator?.url || 'unknown';

                if (!status.orchestrator?.ok) {
                    badge.className = 'status-badge';
                    badge.innerHTML = '<span class="status-dot" style="background:var(--error)"></span><span>Orch Offline</span>';
                    badge.title = `Orchestrator not found at ${orchUrl}. Click to retry.`;
                    badge.style.cursor = 'pointer';
                    badge.onclick = () => {
                        showNotification('Retrying connection...', 'info');
                        checkDashboardStatus();
                    };
                } else {
                    badge.className = 'status-badge online';
                    badge.innerHTML = '<span class="status-dot"></span><span>Orch Online</span>';
                    badge.title = `Connected to ${orchUrl}`;
                    badge.onclick = null;
                    badge.style.cursor = 'default';
                }
            } catch (e) {
                const badge = document.getElementById('connectionStatus');
                badge.className = 'status-badge';
                badge.innerHTML = '<span class="status-dot" style="background:var(--error)"></span><span>Disconnected</span>';
            }
        }

        setInterval(checkDashboardStatus, 5000);
        checkDashboardStatus();
        setInterval(updateSystemStats, 3000);
        updateSystemStats();

        function getDashboardVoterId() {
            const nn = document.getElementById('nodeName');
            const v = nn && nn.value ? nn.value.trim() : '';
            if (v) return v;
            if (typeof fleetNodes !== 'undefined' && fleetNodes.length) return fleetNodes[0].node_id;
            return 'anonymous';
        }

        async function refreshGovernanceVotes() {
            const loadEl = document.getElementById('governanceVotesLoading');
            const listEl = document.getElementById('governanceVotesList');
            if (!loadEl || !listEl) return;
            try {
                const resp = await fetch('/api/admin/votes');
                if (!resp.ok) throw new Error('HTTP ' + resp.status);
                const data = await resp.json();
                const votes = data.votes || [];
                loadEl.style.display = votes.length ? 'none' : '';
                loadEl.textContent = votes.length ? '' : 'No active governance proposals.';
                listEl.style.display = votes.length ? 'block' : 'none';
                if (!votes.length) { listEl.textContent = ''; return; }
                listEl.textContent = votes.map(v => {
                    const vid = v.id || v.vote_id;
                    const opts = (v.options || ['yes', 'no']).map(o =>
                        `<button type="button" class="gov-vote-btn" data-vote-id="${vid}" data-option="${String(o).replace(/"/g, '&quot;')}" style="margin:4px 6px 4px 0;padding:6px 10px;border-radius:6px;border:1px solid var(--border);background:var(--bg-tertiary);cursor:pointer;font-size:0.78rem;">${o}</button>`
                    ).join('');
                    return `<div style="border:1px solid var(--border);border-radius:8px;padding:12px;margin-bottom:10px;"><div style="font-weight:600;margin-bottom:6px;">${v.title || vid}</div><div style="color:var(--text-muted);font-size:0.76rem;margin-bottom:8px;">${v.description || ''}</div><div>${opts}</div></div>`;
                }).join('');
            } catch (e) {
                loadEl.style.display = '';
                loadEl.textContent = 'Votes unavailable: ' + e.message;
                listEl.style.display = 'none';
            }
        }

        function dashPollOrchestrator() {
            Promise.all([
                fetch('/api/worker/nodes').then(r => r.ok ? r.json() : Promise.resolve(null)).catch(() => null),
                fetch('/api/worker/jobs').then(r => r.ok ? r.json() : Promise.resolve(null)).catch(() => null),
                fetch('/api/worker/health').then(r => r.ok ? r.json() : Promise.resolve(null)).catch(() => null),
                fetch('/api/worker/credits').then(r => r.ok ? r.json() : Promise.resolve(null)).catch(() => null),
            ]).then(([nodesData, jobsData, healthData, creditsData]) => {
                meshOnline = !!(healthData && healthData.ok);

                const badge = document.getElementById('connectionStatus');
                if (!meshOnline) {
                    badge.className = 'status-badge';
                    badge.querySelector('span:last-child').textContent = 'Offline';
                } else if (!contribEnabled) {
                    badge.className = 'status-badge';
                    badge.querySelector('span:last-child').textContent = 'Paused';
                } else {
                    badge.className = 'status-badge online';
                    badge.querySelector('span:last-child').textContent = 'Online';
                }

                if (!meshOnline) { fleetNodes = []; fleetJobs = []; return; }

                const nodes = nodesData ? (nodesData.nodes || []) : [];
                fleetNodes = nodes;
                document.getElementById('activeNodes').textContent = nodes.length;
                document.getElementById('activeWorkers').textContent = nodes.filter(n => n.status === 'working').length;

                const jobs = jobsData ? (jobsData.jobs || []) : [];
                fleetJobs = jobs;
                document.getElementById('queuedJobs').textContent = jobs.filter(j => j.status === 'queued').length;

                const visible = jobs.filter(j => ['queued','assigned','running'].includes(j.status));
                const ql = document.getElementById('queueList');
                if (visible.length === 0) {
                    ql.textContent = '<div style="text-align:center;padding:20px 0;color:var(--text-muted);font-size:.82rem;">No jobs queued</div>';
                } else {
                    ql.textContent = visible.slice(0, 8).map((j, i) => {
                        const isRun = j.status === 'running' || j.status === 'assigned';
                        const pct = j.progress ? Math.round((j.progress.step || 0) / j.steps * 100) : 0;
                        return `<div class="queue-item${isRun ? ' running' : ''}">
                            <div class="queue-rank ${i === 0 ? 'p0' : i < 2 ? 'p1' : 'p2'}">${i + 1}</div>
                            <div class="queue-details">
                                <div class="queue-title">${j.model_name || j.job_id}</div>
                                <div class="queue-meta">${j.assigned_to || 'unassigned'} · ${j.steps} steps${pct ? ' · ' + pct + '%' : ''}</div>
                            </div>
                        </div>`;
                    }).join('');
                }

                const myJob = jobs.find(j => j.status === 'running' || j.status === 'assigned');
                const taskHasJob = document.getElementById('taskHasJob');
                const taskNoJob = document.getElementById('taskNoJob');
                if (myJob) {
                    taskHasJob.style.display = '';
                    taskNoJob.style.display = 'none';
                    document.getElementById('taskId').textContent = myJob.job_id;
                    const pct = myJob.progress ? Math.round((myJob.progress.step || 0) / myJob.steps * 100) : 0;
                    document.getElementById('taskProgress').style.width = pct + '%';
                    document.getElementById('elapsedTime').textContent = myJob.progress ? (myJob.progress.step + ' / ' + myJob.steps) : '—';
                    document.getElementById('sessionEarnings').textContent = pct + '%';
                    document.getElementById('earningRate').textContent = (myJob.progress && myJob.progress.loss != null) ? myJob.progress.loss.toFixed(3) + ' loss' : '—';
                } else {
                    taskHasJob.style.display = 'none';
                    taskNoJob.style.display = '';
                }

                updateJobsPage(jobs);
                updateVoteList(jobs);
                refreshGovernanceVotes();
                updateCreditsDisplay(creditsData);
            });
        }

        var isFirstJobsLoad = true;

        function updateJobsPage(jobs) {
            var active = jobs.filter(function(j) { return ['assigned','running'].indexOf(j.status) !== -1; });
            var done = jobs.filter(function(j) { return ['success','failed','timeout','cancelled'].indexOf(j.status) !== -1; });
            var ajl = document.getElementById('activeJobsList');

            if (isFirstJobsLoad && active.length === 0 && jobs.length === 0) {
                setTimeout(function() {
                    if (document.getElementById('activeJobsList').innerHTML.includes('skeleton')) {
                        ajl.textContent = '<div style="text-align:center;padding:20px 0;color:var(--text-muted);font-size:.82rem;">No active jobs</div>';
                    }
                }, 500);
                isFirstJobsLoad = false;
                return;
            }

            isFirstJobsLoad = false;

            ajl.textContent = active.length === 0
                ? '<div style="text-align:center;padding:20px 0;color:var(--text-muted);font-size:.82rem;">No active jobs</div>'
                : active.map(function(j) {
                    const s = j.progress ? j.progress.step : 0;
                    return `<div class="job-item"><div class="job-info"><div class="job-name">${j.model_name || j.job_id}</div><div class="job-meta">Job ID: ${j.job_id} · Step ${s} / ${j.steps}</div></div><div style="display:flex;align-items:center;gap:10px;"><div class="job-status running">${j.status}</div><button type="button" class="job-cancel-btn" data-job-id="${j.job_id}" title="Cancel job" style="font-size:0.72rem;padding:6px 10px;border-radius:6px;border:1px solid var(--border);background:var(--bg-tertiary);color:var(--text-secondary);cursor:pointer;">Cancel</button></div></div>`;
                }).join('');
            const jhl = document.getElementById('jobHistoryList');
            jhl.textContent = done.length === 0
                ? '<div style="text-align:center;padding:20px 0;color:var(--text-muted);font-size:.82rem;">No completed jobs yet</div>'
                : done.slice(0, 10).map(j => {
                    const ago = j.completed_at ? timeAgo(j.completed_at) : '—';
                    return `<div class="history-item"><div class="history-info"><div class="history-name">${j.model_name || j.job_id}</div><div class="history-time">${ago}</div></div><div class="history-status ${j.status === 'success' ? 'completed' : ''}">${j.status}</div></div>`;
                }).join('');
        }

        function updateVoteList(jobs) {
            const queued = jobs.filter(j => j.status === 'queued');
            const vl = document.getElementById('voteList');
            if (queued.length === 0) {
                vl.textContent = '<div style="text-align:center;padding:16px 0;color:var(--text-muted);font-size:.78rem;">No queued jobs to vote on</div>';
            } else {
                vl.textContent = queued.slice(0, 6).map(j => {
                    return `<div class="vote-option" data-job="${j.model_name || j.job_id}"><div class="vote-option-icon"><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg></div><div class="vote-option-details"><div class="vote-option-title">${j.model_name || j.job_id}</div><div class="vote-option-meta">${j.steps} steps</div></div></div>`;
                }).join('');
            }
        }

        function updateCreditsDisplay(creditsData) {
            const allCredits = creditsData && creditsData.credits ? creditsData.credits : {};
            let totalBalance = 0, totalLifetime = 0, totalPending = 0, totalSpent = 0;
            const nodeIds = Object.keys(allCredits);
            nodeIds.forEach(nid => {
                const c = allCredits[nid];
                totalBalance += c.balance || 0;
                totalLifetime += c.lifetime || 0;
            });

            const bv = document.getElementById('balanceValue');
            if (bv) bv.textContent = totalBalance.toFixed(1);
            const pc = document.getElementById('pendingCredits');
            if (pc) pc.textContent = totalPending.toFixed(1);
            const lc = document.getElementById('lifetimeCredits');
            if (lc) lc.textContent = totalLifetime.toFixed(1);

            const cb = document.getElementById('creditsBalance');
            if (cb) cb.textContent = totalBalance.toFixed(1);
            const cp = document.getElementById('creditsPending');
            if (cp) cp.textContent = totalPending.toFixed(1);
            const cl = document.getElementById('creditsLifetime');
            if (cl) cl.textContent = totalLifetime.toFixed(1);
            const cs = document.getElementById('creditsSpent');
            if (cs) cs.textContent = totalSpent.toFixed(1);
            const cn = document.getElementById('creditsNet');
            if (cn) cn.textContent = (totalLifetime - totalSpent).toFixed(1);

            if (fleetNodes.length > 0) {
                const firstNode = fleetNodes[0].node_id;
                fetch(`/api/worker/credits/${encodeURIComponent(firstNode)}`)
                    .then(r => r.ok ? r.json() : null)
                    .then(data => {
                        if (!data || !data.transactions || !data.transactions.length) return;
                        const txList = document.getElementById('transactionList');
                        const txList2 = document.getElementById('creditsTransactionList');
                        const html = data.transactions.slice(0, 8).map(tx => {
                            const ago = timeAgo(tx.ts);
                            return `<div class="transaction">
                                <span class="transaction-desc">${tx.description}</span>
                                <span class="transaction-amount positive">+${tx.amount.toFixed(1)}</span>
                            </div>`;
                        }).join('');
                        if (txList) txList.textContent = html;
                        if (txList2) txList2.textContent = html;
                    })
                    .catch(() => {});
            }
        }

        function timeAgo(ts) {
            const diff = Date.now() / 1000 - ts;
            if (diff < 60) return Math.round(diff) + 's ago';
            if (diff < 3600) return Math.round(diff / 60) + 'm ago';
            if (diff < 86400) return Math.round(diff / 3600) + 'h ago';
            return Math.round(diff / 86400) + 'd ago';
        }

        window.refreshCreateJobCostEstimate = function() {
            var architecture = {};
            try { architecture = parseArchitectureConfig(); } catch (ignore) { architecture = {}; }
            var opts = {
                architecture: architecture,
                steps: document.getElementById('createJobSteps').value,
                batchSize: document.getElementById('createJobBatchSize').value
            };
            if (typeof renderJobCostEstimate === 'function') {
                renderJobCostEstimate('createJobCostGlance', opts);
                renderJobCostEstimate('createJobCostEstimate', opts);
            }
        };

        window.refreshCreateJobConfirm = function() {
            var el = document.getElementById('createJobConfirmSummary');
            if (!el) return;
            var architecture = {};
            try { architecture = parseArchitectureConfig(); } catch (ignore) { architecture = { family: 'invalid' }; }
            var dataset = (document.getElementById('createJobDataset').value || '').trim();
            var hint = document.getElementById('createJobDatasetHint');
            el.innerHTML =
                '<div><strong>Family</strong> · ' + escapeHtml(architecture.family || '—') + '</div>' +
                '<div><strong>Steps / batch</strong> · ' + escapeHtml(document.getElementById('createJobSteps').value) +
                ' / ' + escapeHtml(document.getElementById('createJobBatchSize').value) + '</div>' +
                '<div><strong>Dataset</strong> · <code>' + escapeHtml(dataset || '(missing)') + '</code></div>' +
                '<div style="margin-top:6px;color:var(--text-muted);font-size:0.8rem;">' +
                escapeHtml((hint && hint.textContent) || '') + '</div>';
            window.refreshCreateJobCostEstimate();
        };

        window.showCreateJobModal = function() {
            var modal = document.getElementById('createJobModal');
            if (typeof bindCreateJobWizard === 'function') bindCreateJobWizard(modal);
            if (typeof setCreateJobWizardStep === 'function') setCreateJobWizardStep(modal, 1);
            window.refreshCreateJobCostEstimate();
            if (typeof openAccessibleModal === 'function') openAccessibleModal(modal);
            else { modal.style.display = 'flex'; }
            document.getElementById('createJobError').style.display = 'none';
            document.getElementById('createJobSuccess').style.display = 'none';
        };

        window.closeCreateJobModal = function() {
            var modal = document.getElementById('createJobModal');
            if (typeof closeAccessibleModal === 'function') closeAccessibleModal(modal);
            else { modal.style.display = 'none'; }
        };

        function parseArchitectureConfig() {
            const raw = document.getElementById('architectureConfig').value;
            const parsed = JSON.parse(raw);
            if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed) || !parsed.family) {
                throw new Error('Architecture JSON must be an object with a family');
            }
            return parsed;
        }

        function loadArchitectureConfigFile(event) {
            const file = event.target.files && event.target.files[0];
            if (!file) return;
            if (file.size > 65536) {
                document.getElementById('architectureConfigStatus').textContent = 'Architecture files must be 64 KiB or smaller.';
                event.target.value = '';
                return;
            }
            const reader = new FileReader();
            reader.onload = () => {
                document.getElementById('architectureConfig').value = reader.result;
                document.getElementById('architectureConfigStatus').textContent = `Loaded ${file.name}. Validate when submitted.`;
                window.refreshCreateJobCostEstimate();
            };
            reader.onerror = () => {
                document.getElementById('architectureConfigStatus').textContent = 'Could not read that file.';
            };
            reader.readAsText(file);
        }

        document.getElementById('architectureConfigFile').addEventListener('change', loadArchitectureConfigFile);
        ['createJobSteps', 'createJobBatchSize', 'architectureConfig'].forEach(function(id) {
            var el = document.getElementById(id);
            if (el) el.addEventListener('input', window.refreshCreateJobCostEstimate);
        });
        var datasetFileInput = document.getElementById('createJobDatasetFile');
        if (datasetFileInput) {
            datasetFileInput.addEventListener('change', function(event) {
                var file = event.target.files && event.target.files[0];
                var hint = document.getElementById('createJobDatasetHint');
                if (!file) {
                    if (hint) hint.textContent = 'No local file selected.';
                    return;
                }
                if (hint) {
                    hint.textContent = 'Packaging hint: upload "' + file.name + '" via chunked S3, then paste the s3:// URI above.';
                }
            });
        }

        window.submitCreateJob = async function() {
            const steps = parseInt(document.getElementById('createJobSteps').value);
            const batchSize = parseInt(document.getElementById('createJobBatchSize').value);
            const datasetRef = document.getElementById('createJobDataset').value.trim();
            const priority = document.getElementById('createJobPriority').value;
            const errorEl = document.getElementById('createJobError');
            const successEl = document.getElementById('createJobSuccess');
            const fail = (message) => {
                errorEl.textContent = message;
                errorEl.style.display = 'block';
                successEl.style.display = 'none';
            };

            if (!steps || !batchSize || !datasetRef) {
                fail('Architecture, dataset URL, steps, and batch size are required');
                return;
            }
            if (steps < 1 || steps > 1000000) {
                fail('Steps must be between 1 and 1000000');
                return;
            }
            if (batchSize < 1 || batchSize > 2048) {
                fail('Batch size must be between 1 and 2048');
                return;
            }

            let architectureConfig;
            try {
                architectureConfig = parseArchitectureConfig();
            } catch (err) {
                fail('Invalid architecture JSON: ' + err.message);
                return;
            }

            const cost = typeof estimateJobCost === 'function'
                ? estimateJobCost({ architecture: architectureConfig, steps, batchSize })
                : { credits: null, opcodes: null, tier: null, estimate: true };

            try {
                const resp = await fetch('/api/jobs', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        job_type: 'fine_tune',
                        base_model: 'uploaded-architecture',
                        steps,
                        batch_size: batchSize,
                        dataset_ref: datasetRef,
                        architecture_config: architectureConfig,
                        hparams: {
                            batch_size: batchSize,
                            priority,
                            cost_estimate: {
                                credits: cost.credits,
                                opcodes: cost.opcodes,
                                tier: cost.tier,
                                estimate: true
                            }
                        },
                        submitter_id: 'node-user',
                        org: 'DistribAI'
                    })
                });
                const data = await resp.json();
                if (data.ok) {
                    errorEl.style.display = 'none';
                    successEl.style.display = 'block';
                    setTimeout(() => {
                        closeCreateJobModal();
                        successEl.style.display = 'none';
                    }, 1500);
                    dashPollOrchestrator();
                } else {
                    fail(data.error || 'Failed to create job');
                }
            } catch (err) {
                fail('Network error: ' + err.message);
            }
        };

        document.getElementById('createJobModal').addEventListener('click', function(e) {
            if (e.target === this) {
                closeCreateJobModal();
            }
        });
        if (typeof bindCreateJobWizard === 'function') {
            bindCreateJobWizard(document.getElementById('createJobModal'));
        }

        async function loadAdminSurface() {
            const endpoints = [
                ['adminVotes', '/api/admin/votes', data => `${data.count ?? (data.votes || []).length} active vote(s)`],
                ['adminLedger', '/api/admin/ledger/root', data => `Root: ${data.root || data.merkle_root || 'not sealed yet'}`],
                ['adminMultipliers', '/api/admin/multipliers/stats', data => `${data.active_nodes ?? data.node_count ?? 0} tracked node(s)`],
                ['adminSybil', '/api/admin/sybil/stats', data => `${data.suspicious_nodes ?? data.flagged_nodes ?? 0} suspicious node(s)`],
                ['adminRebenchmark', '/api/admin/rebenchmark/stats', data => `${data.pending ?? data.pending_count ?? 0} pending rebenchmark(s)`],
                ['adminTransfers', '/api/admin/transfers/stats', data => `${data.total_transfers ?? 0} transfer(s)`],
                ['adminPaginated', '/api/admin/paginated-summary', data => `jobs=${data.jobs ?? 0}, nodes=${data.nodes ?? 0}, credits=${data.credits ?? 0}`],
            ];
            await Promise.all(endpoints.map(async ([id, url, render]) => {
                const el = document.getElementById(id);
                if (!el) return;
                try {
                    const resp = await fetch(url);
                    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                    const data = await resp.json();
                    el.textContent = render(data);
                } catch (err) {
                    el.textContent = `Unavailable: ${err.message}`;
                }
            }));
        }

        const btnRefreshAdmin = document.getElementById('btnRefreshAdmin');
        if (btnRefreshAdmin) btnRefreshAdmin.addEventListener('click', () => loadAdminSurface());

        const govList = document.getElementById('governanceVotesList');
        if (govList) {
            govList.addEventListener('click', async (e) => {
                const btn = e.target.closest('.gov-vote-btn');
                if (!btn) return;
                const voteId = btn.dataset.voteId;
                const option = btn.dataset.option;
                const statusEl = document.getElementById('governanceVoteStatus');
                try {
                    const r = await fetch(`/api/admin/votes/${encodeURIComponent(voteId)}/cast`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ voter_id: getDashboardVoterId(), option }),
                    });
                    const j = await r.json().catch(() => ({}));
                    if (!r.ok) throw new Error(j.error || 'HTTP ' + r.status);
                    if (statusEl) {
                        statusEl.style.display = 'block';
                        statusEl.style.color = 'var(--success)';
                        statusEl.textContent = 'Vote recorded.';
                    }
                    refreshGovernanceVotes();
                } catch (err) {
                    if (statusEl) {
                        statusEl.style.display = 'block';
                        statusEl.style.color = 'var(--error)';
                        statusEl.textContent = err.message || 'Vote failed';
                    }
                }
            });
        }

        var activeJobsListEl = document.getElementById('activeJobsList');
        if (activeJobsListEl) {
            activeJobsListEl.addEventListener('click', function(e) {
                var btn = e.target.closest('.job-cancel-btn');
                if (!btn) return;
                var jobId = btn.dataset.jobId;

                showConfirmModal({
                    type: 'warning',
                    icon: '!',
                    title: 'Cancel Job?',
                    message: 'Are you sure you want to cancel job ' + jobId + '? This action cannot be undone.',
                    actionText: 'Cancel Job',
                    actionType: 'danger',
                    onConfirm: function() {
                        setButtonLoading(btn, true);
                        fetchWithRetry('/api/jobs/' + jobId + '/cancel', { method: 'POST' }, 3)
                            .then(function(res) { return res.json(); })
                            .then(function(data) {
                                setButtonLoading(btn, false);
                                if (data.cancelled || data.canceled) {
                                    showToast('Job cancelled successfully', 'success', 4000);
                                    dashPollOrchestrator();
                                } else {
                                    showToast(data.message || 'Failed to cancel job', 'error', 5000);
                                }
                            })
                            .catch(function(err) {
                                setButtonLoading(btn, false);
                                handleApiError(err, 'Job cancellation');
                            });
                    }
                });
            });
        }

        function dashStartPolling() {
            if (_pollHandle) return;
            dashPollOrchestrator();
            _pollHandle = setInterval(dashPollOrchestrator, 2500);
        }
        function dashStopPolling() {
            if (_pollHandle) { clearInterval(_pollHandle); _pollHandle = null; }
        }
        dashStartPolling();

        initPerformanceChart();

        var SESSION_TIMEOUT = 30 * 60 * 1000; // 30 minutes
        var SESSION_WARNING_TIME = 5 * 60 * 1000; // 5 minutes warning
        var lastActivity = Date.now();
        var sessionWarningShown = false;

        function updateActivity() {
            lastActivity = Date.now();
            if (sessionWarningShown) {
                hideSessionWarning();
                sessionWarningShown = false;
            }
        }

        document.addEventListener('mousemove', throttle(updateActivity, 1000));
        document.addEventListener('keypress', throttle(updateActivity, 1000));
        document.addEventListener('click', throttle(updateActivity, 1000));
        document.addEventListener('scroll', throttle(updateActivity, 1000));

        function checkSessionTimeout() {
            var inactive = Date.now() - lastActivity;

            if (inactive > SESSION_TIMEOUT) {
                showToast('Session expired due to inactivity', 'warning', 5000);
                storageManager.remove('distribai_session');
                setTimeout(function() {
                    window.location.href = '/login?expired=1';
                }, 2000);
            } else if (inactive > SESSION_TIMEOUT - SESSION_WARNING_TIME && !sessionWarningShown) {
                showSessionWarning();
                sessionWarningShown = true;
            }
        }

        function showSessionWarning() {
            var warning = document.getElementById('sessionWarning');
            warning.classList.add('show');
            startSessionCountdown();
        }

        function hideSessionWarning() {
            var warning = document.getElementById('sessionWarning');
            warning.classList.remove('show');
        }

        function startSessionCountdown() {
            var endTime = Date.now() + SESSION_WARNING_TIME;
            var countdownEl = document.getElementById('sessionCountdown');

            var interval = setInterval(function() {
                if (!sessionWarningShown) {
                    clearInterval(interval);
                    return;
                }

                var remaining = endTime - Date.now();
                if (remaining <= 0) {
                    clearInterval(interval);
                    countdownEl.textContent = '0:00';
                    return;
                }

                var minutes = Math.floor(remaining / 60000);
                var seconds = Math.floor((remaining % 60000) / 1000);
                countdownEl.textContent = minutes + ':' + String(seconds).padStart(2, '0');
            }, 1000);
        }

        function extendSession() {
            updateActivity();
            showToast('Session extended', 'success', 3000);
        }

        function logout() {
            showToast('Logging out...', 'info', 2000);
            storageManager.remove('distribai_session');
            storageManager.remove('distribai_activity');
            setTimeout(function() {
                window.location.href = '/login';
            }, 1500);
        }

        setInterval(checkSessionTimeout, 60000);

        var contextTargetId = null;

        document.addEventListener('contextmenu', function(e) {
            var jobItem = e.target.closest('.job-item');
            if (jobItem) {
                e.preventDefault();
                contextTargetId = jobItem.dataset.jobId;
                showContextMenu(e.clientX, e.clientY);
            } else {
                hideContextMenu();
            }
        });

        document.addEventListener('click', function() {
            hideContextMenu();
        });

        function showContextMenu(x, y) {
            var menu = document.getElementById('contextMenu');
            menu.style.left = x + 'px';
            menu.style.top = y + 'px';
            menu.classList.add('show');

            var rect = menu.getBoundingClientRect();
            if (rect.right > window.innerWidth) {
                menu.style.left = (x - rect.width) + 'px';
            }
            if (rect.bottom > window.innerHeight) {
                menu.style.top = (y - rect.height) + 'px';
            }
        }

        function hideContextMenu() {
            document.getElementById('contextMenu').classList.remove('show');
        }

        function contextAction(action) {
            if (!contextTargetId) return;

            switch(action) {
                case 'copy':
                    copyToClipboard(contextTargetId, 'Job ID copied: ' + contextTargetId);
                    break;
                case 'view':
                    showToast('Viewing job ' + contextTargetId, 'info', 2000);
                    break;
                case 'cancel':
                    showConfirmModal({
                        type: 'warning',
                        icon: '!',
                        title: 'Cancel Job?',
                        message: 'Cancel job ' + contextTargetId + '?',
                        actionText: 'Cancel Job',
                        actionType: 'danger',
                        onConfirm: function() {
                            fetchWithRetry('/api/jobs/' + contextTargetId + '/cancel', { method: 'POST' }, 3)
                                .then(function() {
                                    showToast('Job cancelled', 'success', 3000);
                                    dashPollOrchestrator();
                                })
                                .catch(function(err) {
                                    handleApiError(err, 'Job cancellation');
                                });
                        }
                    });
                    break;
            }
            hideContextMenu();
        }

        var wsConnected = false;

        function updateWSIndicator(connected) {
            wsConnected = connected;
            var indicators = document.querySelectorAll('.ws-indicator');
            indicators.forEach(function(ind) {
                if (connected) {
                    ind.classList.add('connected');
                    ind.classList.remove('disconnected');
                    ind.querySelector('.ws-text').textContent = 'Live';
                } else {
                    ind.classList.add('disconnected');
                    ind.classList.remove('connected');
                    ind.querySelector('.ws-text').textContent = 'Polling';
                }
            });
        }

        var eventSource = null;

        function initRealtimeConnection() {
            try {
                eventSource = new EventSource('/api/worker/stream');

                eventSource.onopen = function() {
                    updateWSIndicator(true);
                    appLogger.info('Realtime connection established');
                };

                eventSource.onerror = function() {
                    updateWSIndicator(false);
                    appLogger.warn('Realtime connection error');
                    if (!autoRefreshEnabled) {
                        toggleAutoRefresh();
                    }
                };

                eventSource.onmessage = function(event) {
                    try {
                        var data = JSON.parse(event.data);
                        if (data.type === 'ping') {
                            updateWSIndicator(true);
                        }
                    } catch (e) {
                    }
                };
            } catch (e) {
                updateWSIndicator(false);
                appLogger.info('Realtime not available, using polling fallback');
            }

            window.addEventListener('online', function() {
                updateWSIndicator(true);
                if (eventSource) {
                    eventSource.close();
                    initRealtimeConnection();
                }
            });

            window.addEventListener('offline', function() {
                updateWSIndicator(false);
            });
        }

        initRealtimeConnection();

        var autoRefreshEnabled = true;

        function toggleAutoRefresh() {
            autoRefreshEnabled = !autoRefreshEnabled;
            var indicators = document.querySelectorAll('.auto-refresh-indicator');

            if (autoRefreshEnabled) {
                dashStartPolling();
                indicators.forEach(function(ind) {
                    ind.classList.add('active');
                    ind.querySelector('.refresh-text').textContent = 'Auto';
                });
                showToast('Auto-refresh enabled', 'success', 2000);
            } else {
                dashStopPolling();
                indicators.forEach(function(ind) {
                    ind.classList.remove('active');
                    ind.querySelector('.refresh-text').textContent = 'Paused';
                });
                showToast('Auto-refresh paused', 'warning', 2000);
            }

            localStorage.setItem('distribai_autorefresh', autoRefreshEnabled ? '1' : '0');
        }

        var savedAutoRefresh = localStorage.getItem('distribai_autorefresh');
        if (savedAutoRefresh === '0') {
            autoRefreshEnabled = true;
            toggleAutoRefresh();
        }

        function createPagination(containerId, currentPage, totalPages, onPageChange) {
            var container = document.getElementById(containerId);
            if (!container) return;

            var html = '<div class="pagination">';

            html += '<button class="page-btn" ' + (currentPage <= 1 ? 'disabled' : '') + ' onclick="' + onPageChange + '(' + (currentPage - 1) + ')">&lt;</button>';

            var startPage = Math.max(1, currentPage - 2);
            var endPage = Math.min(totalPages, startPage + 4);

            if (startPage > 1) {
                html += '<button class="page-btn" onclick="' + onPageChange + '(1)">1</button>';
                if (startPage > 2) html += '<span class="page-ellipsis">...</span>';
            }

            for (var i = startPage; i <= endPage; i++) {
                html += '<button class="page-btn ' + (i === currentPage ? 'active' : '') + '" onclick="' + onPageChange + '(' + i + ')">' + i + '</button>';
            }

            if (endPage < totalPages) {
                if (endPage < totalPages - 1) html += '<span class="page-ellipsis">...</span>';
                html += '<button class="page-btn" onclick="' + onPageChange + '(' + totalPages + ')">' + totalPages + '</button>';
            }

            html += '<button class="page-btn" ' + (currentPage >= totalPages ? 'disabled' : '') + ' onclick="' + onPageChange + '(' + (currentPage + 1) + ')">&gt;</button>';

            html += '</div>';
            container.textContent = html;
        }

        // Data import is intentionally not exposed here. The orchestrator has no
        // transactional import contract, so the dashboard only offers live exports.

        document.addEventListener('keydown', function(e) {
            if (e.altKey && e.key === 'r') {
                e.preventDefault();
                toggleAutoRefresh();
            }

            if (e.altKey && e.key === 'f') {
                e.preventDefault();
                var filterInput = document.getElementById('jobModelFilter');
                if (filterInput) filterInput.focus();
            }

            if (e.altKey && e.key === 'e') {
                e.preventDefault();
                var activePage = document.querySelector('.page-section[style*="block"]');
                if (activePage) {
                    var pageId = activePage.id.replace('page-', '');
                    if (['jobs', 'credits', 'nodes'].indexOf(pageId) !== -1) {
                        exportData(pageId);
                    }
                }
            }

            if (e.shiftKey && e.key === '?') {
                e.preventDefault();
                showShortcutsModal();
            }
        });

        document.addEventListener('visibilitychange', function() {
            if (document.hidden) {
                if (autoRefreshEnabled) {
                    dashStopPolling();
                    window._slowPollTimer = setInterval(dashPollOrchestrator, 30000); // Every 30s
                }
            } else {
                if (window._slowPollTimer) {
                    clearInterval(window._slowPollTimer);
                    window._slowPollTimer = null;
                }
                if (autoRefreshEnabled) {
                    dashStartPolling();
                }
            }
        });

        var validators = {
            jobId: function(value) {
                return /^[a-zA-Z0-9-]+$/.test(value) && value.length >= 3;
            },

            email: function(value) {
                return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
            },

            url: function(value) {
                try {
                    new URL(value);
                    return true;
                } catch {
                    return false;
                }
            },

            numberRange: function(value, min, max) {
                var num = parseFloat(value);
                return !isNaN(num) && num >= min && num <= max;
            },

            required: function(value) {
                return value !== null && value !== undefined && String(value).trim() !== '';
            },

            isValidJSON: function(str) {
                try {
                    JSON.parse(str);
                    return true;
                } catch {
                    return false;
                }
            },

            fileType: function(filename, allowedTypes) {
                var ext = filename.split('.').pop().toLowerCase();
                return allowedTypes.indexOf(ext) !== -1;
            }
        };

        function validateForm(formId, rules) {
            var form = document.getElementById(formId);
            if (!form) return { valid: false, errors: [] };

            var errors = [];
            var valid = true;

            for (var field in rules) {
                var input = form.querySelector('[name="' + field + '"]');
                if (!input) continue;

                var value = input.value;
                var rule = rules[field];

                if (rule.required && !validators.required(value)) {
                    errors.push({ field: field, message: rule.label + ' is required' });
                    valid = false;
                    input.classList.add('error');
                } else if (rule.validator && !rule.validator(value)) {
                    errors.push({ field: field, message: rule.errorMessage || field + ' is invalid' });
                    valid = false;
                    input.classList.add('error');
                } else {
                    input.classList.remove('error');
                }
            }

            return { valid: valid, errors: errors };
        }

        function showFormErrors(formId, errors) {
            var form = document.getElementById(formId);
            if (!form) return;

            form.querySelectorAll('.error-message').forEach(function(el) { el.remove(); });

            errors.forEach(function(error) {
                var input = form.querySelector('[name="' + error.field + '"]');
                if (input) {
                    var errorEl = document.createElement('div');
                    errorEl.className = 'error-message';
                    errorEl.style.cssText = 'color:var(--error);font-size:0.75rem;margin-top:4px;';
                    errorEl.textContent = error.message;
                    input.parentNode.insertBefore(errorEl, input.nextSibling);
                }
            });
        }

        window.addEventListener('error', function(event) {
            console.error('Global error:', event.error);
            showToast('An unexpected error occurred. Please refresh the page.', 'error', 8000);
            addActivityItem('Error: ' + (event.error && event.error.message ? event.error.message : 'Unknown error'), 'error');
        });

        window.addEventListener('unhandledrejection', function(event) {
            console.error('Unhandled promise rejection:', event.reason);
            showToast('A network error occurred. Retrying...', 'warning', 5000);
        });

        function announceToScreenReader(message, priority) {
            priority = priority || 'polite';
            var announcer = document.getElementById('sr-announcer');
            if (!announcer) {
                announcer = document.createElement('div');
                announcer.id = 'sr-announcer';
                announcer.setAttribute('aria-live', priority);
                announcer.setAttribute('aria-atomic', 'true');
                announcer.style.cssText = 'position:absolute;left:-10000px;width:1px;height:1px;overflow:hidden;';
                document.body.appendChild(announcer);
            }
            announcer.textContent = message;
        }

        function trapFocus(element) {
            var focusableElements = element.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
            var firstFocusable = focusableElements[0];
            var lastFocusable = focusableElements[focusableElements.length - 1];

            element.addEventListener('keydown', function(e) {
                if (e.key === 'Tab') {
                    if (e.shiftKey && document.activeElement === firstFocusable) {
                        e.preventDefault();
                        lastFocusable.focus();
                    } else if (!e.shiftKey && document.activeElement === lastFocusable) {
                        e.preventDefault();
                        firstFocusable.focus();
                    }
                }
            });
        }

        function addSkipLink() {
            var skipLink = document.createElement('a');
            skipLink.href = '#main-content';
            skipLink.textContent = 'Skip to main content';
            skipLink.style.cssText = 'position:absolute;left:-10000px;z-index:10000;background:var(--accent);color:var(--bg);padding:8px 16px;';
            skipLink.addEventListener('focus', function() {
                this.style.left = '0';
            });
            skipLink.addEventListener('blur', function() {
                this.style.left = '-10000px';
            });
            document.body.insertBefore(skipLink, document.body.firstChild);

            var main = document.querySelector('main');
            if (main) main.id = 'main-content';
        }

        addSkipLink();

        function measurePerformance(label, fn) {
            var start = performance.now();
            var result = fn();
            var end = performance.now();
            return result;
        }

        window.addEventListener('load', function() {
            setTimeout(function() {
                var timing = performance.timing;
                var pageLoadTime = timing.loadEventEnd - timing.navigationStart;

                if (pageLoadTime > 3000) {
                    showToast('Page loaded slowly. Consider optimizing resources.', 'warning', 5000);
                }
            }, 0);
        });

        var storageManager = {
            set: function(key, value) {
                try {
                    localStorage.setItem(key, JSON.stringify(value));
                    return true;
                } catch (e) {
                    console.error('Storage error:', e);
                    return false;
                }
            },

            get: function(key, defaultValue) {
                try {
                    var item = localStorage.getItem(key);
                    return item ? JSON.parse(item) : defaultValue;
                } catch (e) {
                    return defaultValue;
                }
            },

            remove: function(key) {
                try {
                    localStorage.removeItem(key);
                    return true;
                } catch (e) {
                    return false;
                }
            },

            clear: function() {
                try {
                    localStorage.clear();
                    return true;
                } catch (e) {
                    return false;
                }
            }
        };

        var rateLimiter = {
            calls: {},

            canCall: function(key, limitMs) {
                var now = Date.now();
                var lastCall = this.calls[key] || 0;
                if (now - lastCall >= limitMs) {
                    this.calls[key] = now;
                    return true;
                }
                return false;
            },

            timeUntilNext: function(key, limitMs) {
                var lastCall = this.calls[key] || 0;
                return Math.max(0, limitMs - (Date.now() - lastCall));
            }
        };

        var appLogger = {
            level: 'info',

            setLevel: function(level) {
                this.level = level;
                localStorage.setItem('distribai_log_level', level);
            },

            log: function(level, message, data) {
                var levels = { debug: 0, info: 1, warn: 2, error: 3 };
                if (levels[level] >= levels[this.level]) {
                    var timestamp = new Date().toISOString();
                    var logEntry = { timestamp: timestamp, level: level, message: message, data: data };

                    if (console[level]) {
                        console[level](timestamp, message, data || '');
                    }

                    if (!window._appLogs) window._appLogs = [];
                    window._appLogs.push(logEntry);
                    if (window._appLogs.length > 100) window._appLogs.shift();
                }
            },

            debug: function(msg, data) { this.log('debug', msg, data); },
            info: function(msg, data) { this.log('info', msg, data); },
            warn: function(msg, data) { this.log('warn', msg, data); },
            error: function(msg, data) { this.log('error', msg, data); }
        };

        var savedLogLevel = localStorage.getItem('distribai_log_level');
        if (savedLogLevel) appLogger.setLevel(savedLogLevel);

        var featureDetection = {
            localStorage: function() {
                try {
                    localStorage.setItem('test', 'test');
                    localStorage.removeItem('test');
                    return true;
                } catch {
                    return false;
                }
            },

            fetch: function() {
                return typeof fetch !== 'undefined';
            },

            promises: function() {
                return typeof Promise !== 'undefined';
            },

            webSockets: function() {
                return typeof WebSocket !== 'undefined';
            },

            chartJs: function() {
                return typeof Chart !== 'undefined';
            },

            allRequired: function() {
                return this.localStorage() && this.fetch() && this.promises();
            }
        };

        if (!featureDetection.allRequired()) {
            showToast('Your browser may not support all features. Please update to the latest version.', 'warning', 10000);
        }

        appLogger.info('Application initialized', {
            theme: currentTheme,
            autoRefresh: autoRefreshEnabled,
            wsConnected: wsConnected,
            features: Object.keys(featureDetection).filter(function(f) {
                return typeof featureDetection[f] === 'function' && f !== 'allRequired' && featureDetection[f]();
            })
        });

document.querySelectorAll('nav a').forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                const page = this.dataset.page;
                
                document.querySelectorAll('nav a').forEach(l => l.classList.remove('active'));
                this.classList.add('active');
                
                document.querySelectorAll('.page-section').forEach(s => {
                    s.style.display = 'none';
                });
                
                const target = document.getElementById('page-' + page);
                if (target) {
                    target.style.display = 'block';
                } else {
                    console.error('Page not found: ' + page);
                }

                if (page === 'admin') loadAdminSurface();
                
                setTimeout(() => {
                    window.scrollTo(0, 0);
                    document.documentElement.scrollTop = 0;
                    document.body.scrollTop = 0;
                }, 10);
                
                const pollingPages = ['dashboard', 'jobs', 'dev'];
                if (pollingPages.includes(page)) {
                    dashStartPolling();
                } else {
                    dashStopPolling();
                }
            });
        });
        

        document.querySelectorAll('input[name="schedule"]').forEach(radio => {
            radio.addEventListener('change', function() {
                document.getElementById('scheduleCustom').style.display = 
                    this.value === 'schedule' ? 'block' : 'none';
            });
        });

        document.getElementById('maxRam').addEventListener('input', function() {
            document.getElementById('maxRamValue').textContent = this.value + ' GB';
        });

        const nodeNameInput = document.getElementById('nodeName');
        fetch('/api/settings/node-name')
            .then(r => r.json())
            .then(d => { if (d.name) nodeNameInput.value = d.name; })
            .catch(() => {});

        let _nodeNameTimer = null;
        nodeNameInput.addEventListener('input', function() {
            clearTimeout(_nodeNameTimer);
            _nodeNameTimer = setTimeout(() => {
                const name = this.value.trim();
                if (!name) return;
                fetch('/api/settings/node-name', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name }),
                })
                .then(r => r.json())
                .then(d => { if (d.ok) showNotification('Node name updated', 'success'); })
                .catch(() => {});
            }, 600);
        });

        document.querySelectorAll('#settingsCpuSelect .cpu-option').forEach(opt => {
            opt.addEventListener('click', function() {
                document.querySelectorAll('#settingsCpuSelect .cpu-option').forEach(o => o.classList.remove('selected'));
                this.classList.add('selected');
            });
        });

        document.getElementById('resetNode').addEventListener('click', () => {
            if (confirm('Reset all node settings? This cannot be undone.')) {
                fetch('/api/settings/reset-node', { method: 'POST' })
                    .then(r => r.json())
                    .then(d => {
                        if (!d.ok) throw new Error(d.error || 'reset failed');
                        nodeNameInput.value = 'gaming-rig-01';
                        showNotification('Node settings reset', 'success');
                    })
                    .catch(err => showNotification(err.message || 'Reset failed', 'error'));
            }
        });

        document.getElementById('unlinkNode').addEventListener('click', () => {
            if (confirm('Unlink this node from your account? You will need to re-register.')) {
                fetch('/api/settings/unlink-node', { method: 'POST' })
                    .then(r => r.json())
                    .then(d => {
                        if (!d.ok) throw new Error(d.error || 'unlink failed');
                        nodeNameInput.value = 'gaming-rig-01';
                        showNotification('Node unlinked', 'success');
                    })
                    .catch(err => showNotification(err.message || 'Unlink failed', 'error'));
            }
        });

        document.getElementById('rebenchFromSettings').addEventListener('click', () => {
            dashStopPolling();
            document.querySelectorAll('nav a').forEach(l => l.classList.remove('active'));
            document.querySelector('nav a[data-page="benchmark"]').classList.add('active');
            document.querySelectorAll('.page-section').forEach(s => s.style.display = 'none');
            document.getElementById('page-benchmark').style.display = 'block';
            window.scrollTo(0, 0);
            if (!benchState.running) benchStartBenchmark();
        });

        const BENCH_TESTS = [
            { name: 'pathtracing', label: 'Path Tracing',   icon: '' },
            { name: 'memory',      label: 'Memory',          icon: '' },
            { name: 'network',    label: 'Network',        icon: '' },
            { name: 'write',      label: 'Disk I/O',       icon: '' },
            { name: 'tensor',    label: 'Tensor Compute', icon: '' },
            { name: 'vram',      label: 'VRAM Capacity',  icon: '' },
        ];

        const BENCH_TIPS = [
            { icon: '', text: 'Path tracing is the same algorithm Pixar uses — you\'re rendering frames of a virtual Cornell box to stress your GPU.' },
            { icon: '', text: 'Your GPU performs trillions of floating-point operations per second. The tensor benchmark makes it work for its keep.' },
            { icon: '', text: 'Benchmarks monitor for thermal throttling — if your GPU slows itself down to stay cool, the score reflects the sustainable performance.' },
            { icon: '', text: 'Memory bandwidth is often the real bottleneck in ML training — more important than raw FLOPS for many workloads.' },
            { icon: '', text: 'Network speed determines how quickly gradients can be uploaded after each training step. Faster upload = faster iteration.' },
            { icon: '', text: 'Fast disk I/O matters for data loading. On large datasets, your GPU can sit idle waiting for the CPU to feed it data.' },
            { icon: '', text: 'The 1-parameter model in the tensor benchmark trains in microseconds. The 1M-parameter model can tell you your GPU\'s real sustained throughput.' },
            { icon: '', text: 'VRAM determines the maximum model size your node can train. More VRAM = eligible for larger, higher-priority P1 tasks.' },
            { icon: '', text: 'bf16 (bfloat16) roughly doubles the model size you can fit in VRAM vs fp32, at minimal precision cost for most training tasks.' },
            { icon: '', text: 'Your scores feed directly into the load balancer. Nodes with higher compute scores get assigned more training steps per credit cycle.' },
            { icon: '', text: 'Path tracing bounces light rays 5 times per pixel. Each bounce is a full ray-sphere intersection computation — pure parallel math.' },
            { icon: '', text: 'If you see thermal throttling warnings, try reseating your GPU heatsink, cleaning dust from fans, or adjusting your fan curve.' },
            { icon: '', text: 'Longer benchmarks = more accurate scores. Each sub-test runs for 15–50 seconds to average out variance and thermal transients.' },
            { icon: '', text: 'The overall score uses a weighted average: tensor compute (35%), path tracing (20%), VRAM (15%), memory/network/disk (10% each).' },
            { icon: '', text: 'DistribAI uses your benchmark score to assign micro-tasks sized for your VRAM. No task will exceed what your GPU can handle.' },
            { icon: '', text: 'You can re-benchmark any time — useful after driver updates, hardware changes, or if you adjust thermal limits.' },
            { icon: '', text: 'Scores use a logarithmic scale: going from 10 → 100 units of throughput bumps your score the same amount as 100 → 1000.' },
            { icon: '', text: 'Random-access memory latency (the pointer-chasing test) exposes your RAM controller and cache hierarchy — often a hidden bottleneck.' },
            { icon: '', text: 'All benchmark results are signed before upload. The orchestrator verifies the signature to prevent score manipulation.' },
            { icon: '', text: 'The network loopback test measures your OS kernel\'s TCP stack speed — unrelated to internet speed but relevant for local inter-process gradient passing.' },
        ];

        const benchState = {
            running:        false,
            evtSource:      null,
            startTime:      null,
            timerInterval:  null,
            tipInterval:    null,
            tipIndex:       0,
            testStates:     {},   // name → { status, pct, msg, score, metric }
            testsCompleted: 0,
            thermalFired:   false,
        };

        function buildTestCards() {
            const grid = document.getElementById('benchTestsGrid');
            grid.textContent = '';
            benchState.testStates = {};
            benchState.testsCompleted = 0;
            BENCH_TESTS.forEach(t => {
                benchState.testStates[t.name] = { status: 'idle', pct: 0, msg: 'Waiting…', score: null };
                grid.insertAdjacentHTML('beforeend', `
                    <div class="bench-test-card" id="btcard-${t.name}">
                        <div class="bench-test-header">
                            <span class="bench-test-name">${t.icon} ${t.label}</span>
                            <span class="bench-test-status" id="btstatus-${t.name}">—</span>
                        </div>
                        <div class="bench-test-msg" id="btmsg-${t.name}">Waiting…</div>
                        <div class="bench-test-bar-track">
                            <div class="bench-test-bar" id="btbar-${t.name}"></div>
                        </div>
                        <div class="bench-test-score" id="btscore-${t.name}"></div>
                    </div>`);
            });
        }

        function updateTestCard(name, { status, pct, msg, score, metric }) {
            const card   = document.getElementById(`btcard-${name}`);
            const bar    = document.getElementById(`btbar-${name}`);
            const msgEl  = document.getElementById(`btmsg-${name}`);
            const statEl = document.getElementById(`btstatus-${name}`);
            const scoreEl = document.getElementById(`btscore-${name}`);
            if (!card) return;

            card.className = 'bench-test-card' + (status === 'running' ? ' active' : status === 'done' ? ' done' : status === 'error' ? ' errored' : '');
            if (bar) { bar.style.width = (pct || 0) + '%'; bar.className = 'bench-test-bar' + (status === 'done' ? ' done' : status === 'error' ? ' error' : ''); }
            if (msgEl && msg) msgEl.textContent = msg;
            if (statEl) { statEl.textContent = status === 'running' ? 'Running' : status === 'done' ? 'Done' : status === 'error' ? 'Error' : '—'; statEl.className = 'bench-test-status ' + status; }
            if (scoreEl && score !== null && score !== undefined) {
                scoreEl.textContent = `${score.toFixed(1)} / 100${metric ? '  ·  ' + metric : ''}`;
            }
        }

        function showTip(idx) {
            const tip = BENCH_TIPS[idx % BENCH_TIPS.length];
            const el  = document.getElementById('tipText');
            const ic  = document.getElementById('tipIcon');
            if (el) { el.style.opacity = '0'; setTimeout(() => { el.textContent = tip.text; el.style.opacity = '1'; }, 200); }
            if (ic) ic.textContent = tip.icon;
        }
        function startTips() {
            benchState.tipIndex = Math.floor(Math.random() * BENCH_TIPS.length);
            showTip(benchState.tipIndex);
            benchState.tipInterval = setInterval(() => {
                benchState.tipIndex = (benchState.tipIndex + 1) % BENCH_TIPS.length;
                showTip(benchState.tipIndex);
            }, 7000);
        }
        function stopTips() {
            clearInterval(benchState.tipInterval);
        }

        function startTimer() {
            benchState.startTime = Date.now();
            benchState.timerInterval = setInterval(() => {
                const sec  = Math.floor((Date.now() - benchState.startTime) / 1000);
                const m    = String(Math.floor(sec / 60)).padStart(2, '0');
                const s    = String(sec % 60).padStart(2, '0');
                const el   = document.getElementById('benchElapsed');
                if (el) el.textContent = `${m}:${s}`;
            }, 1000);
        }
        function stopTimer() { clearInterval(benchState.timerInterval); }

        function updateOverall() {
            const done   = benchState.testsCompleted;
            const total  = BENCH_TESTS.length;
            const pct    = total > 0 ? (done / total) * 100 : 0;
            const bar    = document.getElementById('benchOverallBar');
            const count  = document.getElementById('benchTestCount');
            if (bar)   bar.style.width = pct + '%';
            if (count) count.textContent = `${done} / ${total}`;
        }

        function handleBenchEvent(evt) {
            let data;
            try { data = JSON.parse(evt.data); } catch { return; }

            switch (data.type) {

                case 'state_sync':
                    if (data.running) {
                    }
                    break;

                case 'benchmark_start':
                    updateTestCard(data.name, { status: 'running', pct: 0, msg: data.message || 'Starting…' });
                    const cur = document.getElementById('benchCurrentTest');
                    if (cur) cur.textContent = data.display || data.name;
                    break;

                case 'progress':
                    if (data.test && benchState.testStates[data.test]) {
                        const groupName = data.test.replace('_gpu','').replace('_cpu','');
                        const group = benchState.testStates[groupName];
                        if (group) {
                            updateTestCard(groupName, {
                                status: 'running',
                                pct: data.pct || 0,
                                msg: data.message || '',
                            });
                        }
                    }
                    break;

                case 'benchmark_end':
                    {
                        const score = data.score != null ? data.score : null;
                        const ts    = benchState.testStates[data.name];
                        if (ts) {
                            ts.status = 'done'; ts.pct = 100; ts.score = score;
                            benchState.testsCompleted++;
                        }
                        updateTestCard(data.name, { status: 'done', pct: 100, msg: 'Complete', score });
                        updateOverall();
                    }
                    break;

                case 'thermal_warning':
                    {
                        const bar = document.getElementById('benchThermalBar');
                        const msg = document.getElementById('benchThermalMsg');
                        if (bar) bar.classList.add('visible');
                        if (msg) msg.textContent = data.message || 'Thermal throttling detected.';
                        benchState.thermalFired = true;
                    }
                    break;

                case 'error':
                    if (data.name && benchState.testStates[data.name]) {
                        benchState.testStates[data.name].status = 'error';
                        benchState.testsCompleted++;
                        updateTestCard(data.name, { status: 'error', pct: 100, msg: data.message });
                        updateOverall();
                    }
                    break;

                case 'suite_complete':
                    benchState.running = false;
                    stopTimer(); stopTips();
                    showBenchResults(data);
                    document.getElementById('bench-running').style.display = 'none';
                    document.getElementById('bench-results').style.display = 'block';
                    document.getElementById('bench-results').className = 'bench-results-view';
                    document.getElementById('benchStartBtn').disabled = false;
                    benchState.evtSource && benchState.evtSource.close();
                    break;

                case 'runner_stopped':
                case 'runner_error':
                    if (benchState.running) {
                        benchState.running = false;
                        stopTimer(); stopTips();
                        document.getElementById('bench-running').style.display = 'none';
                        document.getElementById('bench-idle').style.display = 'block';
                        document.getElementById('benchStartBtn').disabled = false;
                        if (data.exit_code && data.exit_code !== 0) {
                            showNotification('Benchmark stopped unexpectedly', 'error');
                        }
                    }
                    break;
            }
        }

        function showBenchResults(data) {
            const overall = data.overall_score || 0;
            const scores  = data.individual_scores || {};
            const tier    = data.tier || {};
            const thermals = data.thermal_warnings || [];

            const circumference = 364.4;
            const ring = document.getElementById('scoreRingFg');
            if (ring) {
                setTimeout(() => {
                    ring.style.strokeDashoffset = circumference - (overall / 100) * circumference;
                }, 100);
            }
            const scoreNum = document.getElementById('resultOverallScore');
            if (scoreNum) scoreNum.textContent = overall.toFixed(1);

            const tierName = document.getElementById('resultTierName');
            const tierDesc = document.getElementById('resultTierDesc');
            const tierTasks = document.getElementById('resultTierTasks');
            if (tierName) tierName.textContent = tier.tier || '—';
            if (tierDesc) tierDesc.textContent = tier.description || '';
            if (tierTasks && tier.tasks) {
                tierTasks.textContent = tier.tasks.map(t => `<div class="bench-tier-task">${t}</div>`).join('');
            }

            const thermalEl = document.getElementById('resultThermalWarning');
            if (thermalEl && thermals.length) {
                thermalEl.style.display = 'block';
                thermalEl.textContent = `Thermal throttling detected during: ${thermals.join(', ')}. Scores reflect sustained (post-throttle) performance. Consider improving cooling.`;
            }

            const grid = document.getElementById('resultScoresGrid');
            if (!grid) return;
            grid.textContent = '';

            const LABELS = {
                pathtracing: { label: 'Path Tracing',  icon: '', metricKey: (r) => r.gpu_mrays ? `${r.gpu_mrays} MRays/s GPU` : `${r.cpu_mrays} MRays/s CPU` },
                memory:      { label: 'Memory',         icon: '', metricKey: (r) => r.read_gbs  ? `${r.read_gbs} GB/s read` : '' },
                network:     { label: 'Network',         icon: '', metricKey: (r) => r.download_mbps > 0 ? `${Math.min(r.download_mbps, 10000).toFixed(0)} Mbps` : 'No connection' },
                write:       { label: 'Disk I/O',        icon: '', metricKey: (r) => r.seq_read_mbs ? `${r.seq_read_mbs} MB/s seq` : '' },
                tensor:      { label: 'Tensor Compute',  icon: '', metricKey: (r) => r.largest_sps ? `${r.largest_sps} steps/s (1M)` : '' },
                vram:        { label: 'VRAM',            icon: '', metricKey: (r) => r.max_usable_gb ? `${r.max_usable_gb} GB usable` : '' },
            };

            const raw = data.results || {};
            Object.entries(LABELS).forEach(([key, meta]) => {
                const score  = scores[key] ?? 0;
                const result = raw[key] || {};
                const metric = meta.metricKey(result);
                grid.insertAdjacentHTML('beforeend', `
                    <div class="bench-score-card">
                        <div class="bench-score-card-label">${meta.icon} ${meta.label}</div>
                        <div class="bench-score-card-value">${score.toFixed(1)}</div>
                        <div class="bench-score-card-metric">${metric || 'No data'}</div>
                        <div class="bench-score-card-bar-track">
                            <div class="bench-score-card-bar" style="width:${score}%"></div>
                        </div>
                    </div>`);
            });

            localStorage.setItem('distribai_last_bench', JSON.stringify(data));
            loadPrevResults();
        }

        function loadPrevResults() {
            const raw = localStorage.getItem('distribai_last_bench');
            if (!raw) return;
            try {
                const data = JSON.parse(raw);
                if (!data || !data.overall_score) return;
                const prevEl = document.getElementById('bench-prev-results');
                if (prevEl) prevEl.style.display = 'block';

                const grid = document.getElementById('bench-prev-scores-grid');
                const tierEl = document.getElementById('bench-prev-tier');
                const scores = data.individual_scores || {};
                const tier   = data.tier || {};

                if (grid) {
                    const LABELS = { pathtracing:'Path Tracing', memory:'Memory', network:'Network', write:'Disk I/O', tensor:'Tensor', vram:'VRAM' };
                    const ICONS  = { pathtracing:'', memory:'', network:'', write:'', tensor:'', vram:'' };
                    grid.textContent = '';
                    Object.entries(LABELS).forEach(([k, label]) => {
                        const score = scores[k] ?? 0;
                        grid.insertAdjacentHTML('beforeend', `
                            <div class="bench-score-card">
                                <div class="bench-score-card-label">${ICONS[k]} ${label}</div>
                                <div class="bench-score-card-value">${score.toFixed(1)}</div>
                                <div class="bench-score-card-bar-track">
                                    <div class="bench-score-card-bar" style="width:${score}%"></div>
                                </div>
                            </div>`);
                    });
                }

                if (tierEl && tier.tier) {
                    tierEl.textContent = `
                        <div style="padding:16px;background:var(--bg-secondary);border:1px solid var(--border);border-radius:10px;margin-top:4px;">
                            <div style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px;">Assigned Tier</div>
                            <div style="font-size:1rem;font-weight:600;margin-bottom:4px;">${tier.tier}</div>
                            <div style="font-size:0.82rem;color:var(--text-secondary);">${tier.description || ''}</div>
                        </div>`;
                }
            } catch { /* corrupt storage — ignore */ }
        }

        function benchStartBenchmark() {
            if (benchState.running) return;
            benchState.running = true;
            benchState.thermalFired = false;

            document.getElementById('bench-idle').style.display    = 'none';
            document.getElementById('bench-results').style.display = 'none';
            document.getElementById('bench-running').style.display = 'block';
            document.getElementById('benchStartBtn').disabled = true;

            const tb = document.getElementById('benchThermalBar');
            if (tb) tb.classList.remove('visible');

            buildTestCards();
            startTips();
            startTimer();

            if (benchState.evtSource) benchState.evtSource.close();
            benchState.evtSource = new EventSource('/api/benchmark/stream');
            benchState.evtSource.onmessage = handleBenchEvent;
            benchState.evtSource.onerror = () => {};

            fetch('/api/benchmark/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
                .then(r => r.json())
                .then(d => { if (!d.ok) { showNotification(d.error || 'Failed to start', 'error'); } })
                .catch(e => { showNotification('Could not reach server', 'error'); benchState.running = false; });
        }

        function benchStopBenchmark() {
            fetch('/api/benchmark/stop', { method: 'POST' })
                .then(() => { showNotification('Benchmark stopped', 'error'); })
                .catch(() => {});
        }

        document.getElementById('benchStartBtn').addEventListener('click', benchStartBenchmark);
        document.getElementById('benchStopBtn').addEventListener('click', benchStopBenchmark);
        document.getElementById('benchRerunBtn').addEventListener('click', benchStartBenchmark);
        document.getElementById('benchRerunBtn2').addEventListener('click', () => {
            document.getElementById('bench-results').style.display = 'none';
            document.getElementById('bench-idle').style.display    = 'block';
        });

        loadPrevResults();

        fetch('/api/benchmark/status')
            .then(r => r.json())
            .then(d => {
                if (d.running) {
                    benchState.running = true;
                    document.getElementById('bench-idle').style.display    = 'none';
                    document.getElementById('bench-running').style.display = 'block';
                    document.getElementById('benchStartBtn').disabled = true;
                    buildTestCards();
                    startTips();
                    benchState.startTime = d.startedAt || Date.now();
                    startTimer();
                    if (benchState.evtSource) benchState.evtSource.close();
                    benchState.evtSource = new EventSource('/api/benchmark/stream');
                    benchState.evtSource.onmessage = handleBenchEvent;
                }
            })
            .catch(() => {});

  if (typeof showPage === 'function') global.showPage = showPage;
  if (typeof setRole === 'function') global.setRole = setRole;
  if (typeof loadAdminSurface === 'function') global.loadAdminSurface = loadAdminSurface;
  if (typeof sortJobs === 'function') global.sortJobs = sortJobs;
  if (typeof clearJobFilters === 'function') global.clearJobFilters = clearJobFilters;
  if (typeof exportData === 'function') global.exportData = exportData;
  if (typeof showCreateJobModal === 'function') global.showCreateJobModal = showCreateJobModal;
  if (typeof closeCreateJobModal === 'function') global.closeCreateJobModal = closeCreateJobModal;
  if (typeof submitCreateJob === 'function') global.submitCreateJob = submitCreateJob;
  if (typeof clearActivityFeed === 'function') global.clearActivityFeed = clearActivityFeed;
  if (typeof toggleActivityFeed === 'function') global.toggleActivityFeed = toggleActivityFeed;
  if (typeof openShortcutsModal === 'function') global.openShortcutsModal = openShortcutsModal;
  if (typeof closeShortcutsModal === 'function') global.closeShortcutsModal = closeShortcutsModal;
  if (typeof closeConfirmModal === 'function') global.closeConfirmModal = closeConfirmModal;
  if (typeof executeConfirmAction === 'function') global.executeConfirmAction = executeConfirmAction;
  if (typeof extendSession === 'function') global.extendSession = extendSession;
  if (typeof logout === 'function') global.logout = logout;
  if (typeof contextAction === 'function') global.contextAction = contextAction;
  if (typeof toggleAutoRefresh === 'function') global.toggleAutoRefresh = toggleAutoRefresh;
  if (typeof bulkCancelJobs === 'function') global.bulkCancelJobs = bulkCancelJobs;
  if (typeof bulkExportJobs === 'function') global.bulkExportJobs = bulkExportJobs;
  if (typeof clearSelection === 'function') global.clearSelection = clearSelection;
  if (typeof toggleJobSelection === 'function') global.toggleJobSelection = toggleJobSelection;
  if (typeof showToast === 'function') global.showToast = showToast;
  if (typeof showConfirmModal === 'function') global.showConfirmModal = showConfirmModal;
  if (typeof showNotification === 'function') global.showNotification = showNotification;
  if (typeof navigateToPage === 'function') global.navigateToPage = navigateToPage;
  if (typeof loadArchitectureConfigFile === 'function') global.loadArchitectureConfigFile = loadArchitectureConfigFile;
  if (typeof detectHardware === 'function') global.detectHardware = detectHardware;
  if (typeof showStep === 'function') global.showStep = showStep;
  if (typeof filterJobs === 'function') global.filterJobs = filterJobs;
  if (typeof dashPollOrchestrator === 'function') global.dashPollOrchestrator = dashPollOrchestrator;
  if (typeof dashStartPolling === 'function') global.dashStartPolling = dashStartPolling;
  if (typeof dashStopPolling === 'function') global.dashStopPolling = dashStopPolling;
  if (typeof updateSystemStats === 'function') global.updateSystemStats = updateSystemStats;
  if (typeof checkDashboardStatus === 'function') global.checkDashboardStatus = checkDashboardStatus;
  if (typeof syncDistribaiRegistry === 'function') global.syncDistribaiRegistry = syncDistribaiRegistry;
  if (typeof runPublicRelease === 'function') global.runPublicRelease = runPublicRelease;
  if (typeof benchStartBenchmark === 'function') global.benchStartBenchmark = benchStartBenchmark;
  if (typeof benchStopBenchmark === 'function') global.benchStopBenchmark = benchStopBenchmark;
  global.loadAdminSurface = typeof loadAdminSurface === 'function' ? loadAdminSurface : global.loadAdminSurface;
})(typeof window !== 'undefined' ? window : this);
