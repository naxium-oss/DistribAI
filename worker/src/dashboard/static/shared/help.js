/**
 * Help center: topic search with visible results, chip nav, FAQ accordion.
 */
(function () {
    'use strict';

    var TOPICS = [
        { title: 'Quick start', section: 'Start', anchor: 'start', topic: 'quick-start', content: 'name node settings resources benchmark contributing credits', keywords: 'setup begin first' },
        { title: 'Resource caps', section: 'Start', anchor: 'start', topic: 'resources', content: 'CPU GPU RAM percentage limit headroom', keywords: 'settings percent usage' },
        { title: 'The pipeline', section: 'How it works', anchor: 'how', topic: 'pipeline', content: 'job split piece process results credits mesh', keywords: 'overview network grid' },
        { title: 'Earning credits', section: 'How it works', anchor: 'how', topic: 'credits', content: 'base rate reliability surge hardware tier RTX', keywords: 'earn money reward' },
        { title: 'What is a node', section: 'FAQ', anchor: 'faq', topic: 'faq-node', content: 'computer worker anonymous ID volunteer', keywords: 'device machine' },
        { title: 'How do I earn credits', section: 'FAQ', anchor: 'faq', topic: 'faq-earn', content: 'complete jobs verified vote boost', keywords: 'get receive' },
        { title: 'Is my hardware safe', section: 'FAQ', anchor: 'faq', topic: 'faq-safe', content: 'caps thermal pause limits', keywords: 'damage safety' },
        { title: 'What models are trained', section: 'FAQ', anchor: 'faq', topic: 'faq-models', content: 'SLM small language 7M 135M', keywords: 'AI model' },
        { title: 'Will this slow my computer', section: 'FAQ', anchor: 'faq', topic: 'faq-slow', content: 'performance percentage work hours', keywords: 'lag slow' },
        { title: 'Can I use my GPU while contributing', section: 'FAQ', anchor: 'faq', topic: 'faq-gpu', content: 'gaming disable contributing GPU 0%', keywords: 'play share' },
        { title: 'What if I lose internet', section: 'FAQ', anchor: 'faq', topic: 'faq-offline', content: 'pause resume reconnect progress', keywords: 'offline wifi' },
        { title: 'How much can I earn', section: 'FAQ', anchor: 'faq', topic: 'faq-amount', content: 'RTX 4070 50 credits hour', keywords: 'income amount' },
        { title: 'Orch offline', section: 'Fix', anchor: 'fix', topic: 'fix-orch', content: 'orchestrator firewall 50051 8766 admin API', keywords: 'disconnected offline' },
        { title: 'Jobs not appearing', section: 'Fix', anchor: 'fix', topic: 'fix-jobs', content: 'contributing benchmark score resource 0%', keywords: 'empty no work' },
        { title: 'CUDA / GPU missing', section: 'Fix', anchor: 'fix', topic: 'fix-cuda', content: 'NVIDIA drivers rescan CUDA runtime', keywords: 'graphics not found' },
        { title: 'Out of memory', section: 'Fix', anchor: 'fix', topic: 'fix-oom', content: 'OOM VRAM lower GPU close apps benchmark', keywords: 'crash memory' },
        { title: 'Connection failures', section: 'Fix', anchor: 'fix', topic: 'fix-conn', content: 'host port 50051 firewall network', keywords: 'connect error' },
        { title: 'Benchmark', section: 'Glossary', anchor: 'glossary', topic: 'term-benchmark', content: 'hardware probe placement score', keywords: 'score test' },
        { title: 'Credits', section: 'Glossary', anchor: 'glossary', topic: 'term-credits', content: 'points vote boost', keywords: 'currency' },
        { title: 'Gradient', section: 'Glossary', anchor: 'glossary', topic: 'term-gradient', content: 'training weights improve', keywords: 'backprop' },
        { title: 'Micro-task', section: 'Glossary', anchor: 'glossary', topic: 'term-microtask', content: 'small slice training job', keywords: 'task' },
        { title: 'Node', section: 'Glossary', anchor: 'glossary', topic: 'term-node', content: 'machine anonymous ID', keywords: 'device' },
        { title: 'Orchestrator', section: 'Glossary', anchor: 'glossary', topic: 'term-orch', content: 'coordinator assign collect', keywords: 'server' },
        { title: 'SLM', section: 'Glossary', anchor: 'glossary', topic: 'term-slm', content: 'small language model', keywords: 'model' },
        { title: 'VRAM', section: 'Glossary', anchor: 'glossary', topic: 'term-vram', content: 'GPU memory', keywords: 'memory' },
        { title: 'Worker', section: 'Glossary', anchor: 'glossary', topic: 'term-worker', content: 'local process training tasks', keywords: 'daemon' },
        { title: 'GitHub Issues', section: 'Support', anchor: 'support', topic: 'github', content: 'bugs questions feature requests support', keywords: 'report issue' }
    ];

    var CONFIG = { maxResults: 8, minScore: 0.28, debounceMs: 120 };
    var debounceHandle = null;

    function editDistance(left, right) {
        var rows = [];
        var i;
        var j;
        for (i = 0; i <= right.length; i += 1) {
            rows[i] = [i];
        }
        for (j = 0; j <= left.length; j += 1) {
            rows[0][j] = j;
        }
        for (i = 1; i <= right.length; i += 1) {
            for (j = 1; j <= left.length; j += 1) {
                if (right.charAt(i - 1) === left.charAt(j - 1)) {
                    rows[i][j] = rows[i - 1][j - 1];
                } else {
                    rows[i][j] = Math.min(
                        rows[i - 1][j - 1] + 1,
                        rows[i][j - 1] + 1,
                        rows[i - 1][j] + 1
                    );
                }
            }
        }
        return rows[right.length][left.length];
    }

    function scoreQuery(query, haystack) {
        if (!query || !haystack) {
            return 0;
        }
        var needle = query.toLowerCase().trim();
        var hay = haystack.toLowerCase();
        if (hay.indexOf(needle) !== -1) {
            return 1.0 - (hay.indexOf(needle) / Math.max(hay.length, 1)) * 0.3;
        }
        var parts = needle.split(/\s+/);
        var tokens = hay.split(/\s+/);
        var hits = 0;
        parts.forEach(function (part) {
            if (part.length < 2) {
                return;
            }
            var matched = false;
            tokens.forEach(function (token) {
                if (matched) {
                    return;
                }
                if (token.indexOf(part) !== -1 || part.indexOf(token) !== -1) {
                    hits += 1;
                    matched = true;
                    return;
                }
                if (editDistance(part, token) <= Math.min(2, part.length / 3)) {
                    hits += 0.7;
                    matched = true;
                }
            });
        });
        return parts.length ? (hits / parts.length) * 0.85 : 0;
    }

    function searchHelp(query) {
        if (!query || query.trim().length < 2) {
            return [];
        }
        var ranked = [];
        TOPICS.forEach(function (item) {
            var titleScore = scoreQuery(query, item.title) * 2;
            var bodyScore = scoreQuery(query, item.content);
            var keyScore = scoreQuery(query, item.keywords || '') * 1.5;
            var sectionScore = scoreQuery(query, item.section) * 0.5;
            var best = Math.max(titleScore, bodyScore, keyScore, sectionScore);
            if (best >= CONFIG.minScore) {
                ranked.push({
                    title: item.title,
                    section: item.section,
                    content: item.content,
                    anchor: item.anchor,
                    topic: item.topic,
                    score: best
                });
            }
        });
        ranked.sort(function (a, b) {
            return b.score - a.score;
        });
        return ranked.slice(0, CONFIG.maxResults);
    }

    function escapeHtml(value) {
        if (window.escapeHtml) {
            return window.escapeHtml(value);
        }
        var node = document.createElement('div');
        node.textContent = String(value == null ? '' : value);
        return node.innerHTML;
    }

    function escapeRegex(value) {
        return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    function highlightMatches(text, query) {
        var html = escapeHtml(text);
        if (!query) {
            return html;
        }
        var words = query.toLowerCase().trim().split(/\s+/).filter(function (word) {
            return word.length >= 2;
        });
        words.forEach(function (word) {
            var pattern = new RegExp('(' + escapeRegex(word) + ')', 'gi');
            html = html.replace(pattern, '<mark class="help-search-result-match">$1</mark>');
        });
        return html;
    }

    function setSearchExpanded(open) {
        var input = document.getElementById('helpSearch');
        if (input) {
            input.setAttribute('aria-expanded', open ? 'true' : 'false');
        }
    }

    function setSearchMeta(text, hasQuery) {
        var meta = document.getElementById('helpSearchMeta');
        if (!meta) {
            return;
        }
        meta.textContent = text || '';
        meta.classList.toggle('has-query', !!hasQuery);
    }

    function clearSectionHits() {
        document.querySelectorAll('.help-section.is-hit').forEach(function (el) {
            el.classList.remove('is-hit');
        });
    }

    function updateSearchResults(query) {
        var panel = document.getElementById('helpTopicResults');
        if (!panel) {
            return;
        }
        var trimmed = (query || '').trim();
        function openPanel() {
            panel.classList.add('show');
            panel.setAttribute('data-open', '1');
            panel.style.display = 'block';
            setSearchExpanded(true);
        }
        function closePanel() {
            panel.classList.remove('show');
            panel.removeAttribute('data-open');
            panel.style.display = 'none';
            setSearchExpanded(false);
        }
        if (trimmed.length < 2) {
            closePanel();
            panel.replaceChildren();
            setSearchMeta('');
            clearSectionHits();
            return;
        }

        var hits = searchHelp(trimmed);
        panel.replaceChildren();

        if (!hits.length) {
            var empty = document.createElement('div');
            empty.className = 'help-search-empty';
            empty.textContent = 'No matches for “' + trimmed + '”. Try GPU, credits, orch, or benchmark.';
            panel.appendChild(empty);
            openPanel();
            setSearchMeta('0 results', true);
            return;
        }

        hits.forEach(function (hit) {
            var btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'help-search-result';
            btn.setAttribute('role', 'option');
            btn.dataset.anchor = hit.anchor;
            btn.dataset.topic = hit.topic || '';

            var title = document.createElement('div');
            title.className = 'help-search-result-title';
            title.innerHTML = highlightMatches(hit.title, trimmed);

            var section = document.createElement('div');
            section.className = 'help-search-result-section';
            section.textContent = hit.section;

            var preview = document.createElement('div');
            preview.className = 'help-search-result-preview';
            preview.innerHTML = highlightMatches(hit.content.slice(0, 110) + (hit.content.length > 110 ? '…' : ''), trimmed);

            btn.appendChild(title);
            btn.appendChild(section);
            btn.appendChild(preview);
            panel.appendChild(btn);
        });

        openPanel();
        setSearchMeta(hits.length + (hits.length === 1 ? ' result' : ' results') + ' — click to jump', true);
    }

    function navigateTo(anchor, topic) {
        var panel = document.getElementById('helpTopicResults');
        var input = document.getElementById('helpSearch');
        if (panel) {
            panel.classList.remove('show');
            panel.removeAttribute('data-open');
            panel.style.display = 'none';
            setSearchExpanded(false);
        }
        if (input) {
            input.blur();
        }
        clearSectionHits();

        var section = document.getElementById(anchor);
        var target = topic
            ? document.querySelector('[data-help-topic="' + topic + '"]')
            : null;
        var scrollEl = target || section;
        if (!scrollEl) {
            setSearchMeta('Topic not found on this page', true);
            return;
        }

        scrollEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        if (section) {
            section.classList.add('is-hit');
            setTimeout(function () {
                section.classList.remove('is-hit');
            }, 1600);
        }

        if (target && target.classList.contains('help-faq-item')) {
            document.querySelectorAll('.help-faq-item.open').forEach(function (item) {
                item.classList.remove('open');
            });
            target.classList.add('open');
        }

        setActiveChip(anchor);
        setSearchMeta('Jumped to ' + (target ? (target.querySelector('h3, dt, .help-faq-question') || {}).textContent || anchor : anchor), true);
    }

    function setActiveChip(anchorId) {
        document.querySelectorAll('.help-chip').forEach(function (chip) {
            var href = chip.getAttribute('href') || '';
            chip.classList.toggle('active', href === '#' + anchorId);
        });
    }

    function updateActiveChipFromScroll() {
        var scrollPos = window.scrollY + 140;
        var current = null;
        document.querySelectorAll('.help-section').forEach(function (section) {
            var top = section.offsetTop;
            var bottom = top + section.offsetHeight;
            if (scrollPos >= top && scrollPos < bottom) {
                current = section.id;
            }
        });
        if (current) {
            setActiveChip(current);
        }
    }

    function boot() {
        var searchInput = document.getElementById('helpSearch');
        var searchResults = document.getElementById('helpTopicResults');
        var chips = document.getElementById('helpChips');

        if (searchResults) {
            searchResults.addEventListener('click', function (event) {
                var row = event.target.closest('.help-search-result');
                if (row && row.dataset.anchor) {
                    navigateTo(row.dataset.anchor, row.dataset.topic);
                }
            });
        }

        if (searchInput) {
            function scheduleSearch() {
                clearTimeout(debounceHandle);
                debounceHandle = setTimeout(function () {
                    updateSearchResults(searchInput.value);
                }, CONFIG.debounceMs);
            }
            searchInput.addEventListener('input', scheduleSearch);
            searchInput.addEventListener('keyup', scheduleSearch);
            searchInput.addEventListener('search', scheduleSearch);
            searchInput.addEventListener('change', scheduleSearch);
            searchInput.addEventListener('keydown', function (event) {
                if (event.key === 'Escape' && searchResults) {
                    searchResults.classList.remove('show');
                    setSearchExpanded(false);
                    searchInput.blur();
                }
                if (event.key === 'Enter') {
                    var first = searchResults && searchResults.querySelector('.help-search-result');
                    if (first && searchResults.classList.contains('show')) {
                        event.preventDefault();
                        navigateTo(first.dataset.anchor, first.dataset.topic);
                    }
                }
            });
            document.addEventListener('click', function (event) {
                if (!event.target.closest('#helpSearchWrap') && searchResults) {
                    searchResults.classList.remove('show');
                    searchResults.removeAttribute('data-open');
                    searchResults.style.display = 'none';
                    setSearchExpanded(false);
                }
            });
        }

        if (chips) {
            chips.addEventListener('click', function (event) {
                var chip = event.target.closest('.help-chip');
                if (!chip) {
                    return;
                }
                event.preventDefault();
                var href = chip.getAttribute('href') || '';
                var id = href.replace(/^#/, '');
                if (id) {
                    navigateTo(id, null);
                    history.replaceState(null, '', '#' + id);
                }
            });
        }

        document.querySelectorAll('.help-faq-question').forEach(function (btn) {
            btn.addEventListener('click', function () {
                var item = btn.closest('.help-faq-item');
                var open = item.classList.contains('open');
                document.querySelectorAll('.help-faq-item.open').forEach(function (faq) {
                    faq.classList.remove('open');
                });
                if (!open) {
                    item.classList.add('open');
                }
            });
        });

        window.addEventListener('scroll', function () {
            requestAnimationFrame(updateActiveChipFromScroll);
        }, { passive: true });

        if (location.hash) {
            var hash = location.hash.replace(/^#/, '');
            if (document.getElementById(hash)) {
                setTimeout(function () {
                    navigateTo(hash, null);
                }, 50);
            }
        }

        fetch('/api/version')
            .then(function (response) {
                return response.json();
            })
            .then(function (payload) {
                var versionEl = document.getElementById('helpVersion');
                if (payload.version && versionEl) {
                    versionEl.textContent = payload.version;
                }
            })
            .catch(function () { /* offline */ });
    }

    window.searchHelp = searchHelp;
    window.helpIndex = TOPICS;

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', boot);
    } else {
        boot();
    }
})();
