/**
 * Smooth live metric updates for contributor + operator dashboards.
 */
(function (root) {
    'use strict';

    var barState = new WeakMap();
    var numState = new WeakMap();
    var rafQueued = false;
    var pendingBars = [];
    var pendingNums = [];

    function clamp(n, lo, hi) {
        return Math.max(lo, Math.min(hi, n));
    }

    function easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }

    function tick(now) {
        rafQueued = false;
        var i;
        var nextBars = [];
        for (i = 0; i < pendingBars.length; i += 1) {
            var b = pendingBars[i];
            if (!b.el.isConnected) {
                continue;
            }
            var bt = clamp((now - b.start) / b.duration, 0, 1);
            var bv = b.from + (b.to - b.from) * easeOutCubic(bt);
            b.el.style.width = bv.toFixed(2) + '%';
            if (bt < 1) {
                nextBars.push(b);
            } else {
                barState.set(b.el, b.to);
            }
        }
        pendingBars = nextBars;

        var nextNums = [];
        for (i = 0; i < pendingNums.length; i += 1) {
            var n = pendingNums[i];
            if (!n.el.isConnected) {
                continue;
            }
            var nt = clamp((now - n.start) / n.duration, 0, 1);
            var nv = n.from + (n.to - n.from) * easeOutCubic(nt);
            n.el.textContent = n.format(nv);
            if (nt < 1) {
                nextNums.push(n);
            } else {
                numState.set(n.el, n.to);
                n.el.textContent = n.format(n.to);
            }
        }
        pendingNums = nextNums;

        if (pendingBars.length || pendingNums.length) {
            rafQueued = true;
            root.requestAnimationFrame(tick);
        }
    }

    function ensureTick() {
        if (!rafQueued) {
            rafQueued = true;
            root.requestAnimationFrame(tick);
        }
    }

    function setBar(el, percent, opts) {
        if (!el) {
            return;
        }
        var to = clamp(Number(percent) || 0, 0, 100);
        var duration = (opts && opts.duration) || 420;
        var from = barState.has(el) ? barState.get(el) : to;
        if (!barState.has(el)) {
            el.style.width = to.toFixed(2) + '%';
            barState.set(el, to);
            return;
        }
        pendingBars = pendingBars.filter(function (b) {
            return b.el !== el;
        });
        pendingBars.push({
            el: el,
            from: from,
            to: to,
            start: performance.now(),
            duration: duration
        });
        ensureTick();
    }

    function setText(el, text) {
        if (!el) {
            return;
        }
        var next = text == null ? '—' : String(text);
        if (el.textContent !== next) {
            el.textContent = next;
        }
    }

    function setNumber(el, value, opts) {
        if (!el) {
            return;
        }
        var options = opts || {};
        var format = options.format || function (v) {
            return String(Math.round(v));
        };
        if (value == null || !Number.isFinite(Number(value))) {
            setText(el, options.empty != null ? options.empty : '—');
            numState.delete(el);
            return;
        }
        var to = Number(value);
        var duration = options.duration || 420;
        var from = numState.has(el) ? numState.get(el) : to;
        if (!numState.has(el) || Math.abs(to - from) < 0.001) {
            el.textContent = format(to);
            numState.set(el, to);
            return;
        }
        pendingNums = pendingNums.filter(function (n) {
            return n.el !== el;
        });
        pendingNums.push({
            el: el,
            from: from,
            to: to,
            start: performance.now(),
            duration: duration,
            format: format
        });
        ensureTick();
    }

    function setTempBar(el, celsius) {
        if (celsius == null || !Number.isFinite(Number(celsius))) {
            setBar(el, 0);
            return;
        }
        setBar(el, clamp(Number(celsius), 0, 110) / 110 * 100);
    }

    function createPoller(fn, intervalMs) {
        var timer = null;
        var running = false;
        var ms = intervalMs || 2000;

        function once() {
            if (running) {
                return;
            }
            running = true;
            Promise.resolve()
                .then(fn)
                .catch(function (err) {
                    console.error('Live poll failed:', err);
                })
                .then(function () {
                    running = false;
                });
        }

        return {
            start: function () {
                if (timer) {
                    return;
                }
                once();
                timer = setInterval(once, ms);
            },
            stop: function () {
                if (timer) {
                    clearInterval(timer);
                    timer = null;
                }
            },
            refresh: once
        };
    }

    function bindVisibility(poller) {
        document.addEventListener('visibilitychange', function () {
            if (document.hidden) {
                poller.stop();
            } else {
                poller.start();
            }
        });
        window.addEventListener('pagehide', function () {
            poller.stop();
        });
    }

    root.LiveMetrics = {
        setBar: setBar,
        setText: setText,
        setNumber: setNumber,
        setTempBar: setTempBar,
        createPoller: createPoller,
        bindVisibility: bindVisibility
    };
})(typeof window !== 'undefined' ? window : this);
