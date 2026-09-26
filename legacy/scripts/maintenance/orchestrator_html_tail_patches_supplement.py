#!/usr/bin/env python3
from pathlib import Path

ORCH = Path(__file__).resolve().parents[2] / "worker" / "src" / "dashboard" / "static" / "orch"
MARKERS = (
    "\n    \n// Enhanced XSS Protection for Production",
    "\n    \n// Enhanced XSS",
)

TAILS = {
    "orchestrator-logs.html": """
    <script src="/shared/scripts.js"></script>
    <script>
        var logsRefreshTimer = null;

        function loadLogs() {
            fetch('/admin/logs?n=200')
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    var view = document.getElementById('logView');
                    if (!data.logs || data.logs.length === 0) {
                        view.innerHTML = '<div style="color:var(--text-muted);">No logs found.</motion>';
                        return;
                    }
                    view.innerHTML = data.logs.map(function(line) {
                        var parts = String(line).split(' - ');
                        var ts = escapeHtml(parts[0] || '');
                        var rest = parts.slice(1).join(' - ');
                        var level = (rest.split(':')[0] || 'INFO').trim();
                        return '<div class="log-line">' +
                            '<span class="log-ts">' + ts + '</span>' +
                            '<span class="log-msg log-' + escapeHtml(level) + '">' + escapeHtml(rest) + '</span>' +
                        '</div>';
                    }).join('');
                    view.scrollTop = view.scrollHeight;
                })
                .catch(function(err) { console.error('Failed to load logs:', err); });
        }

        document.addEventListener('DOMContentLoaded', function() {
            loadLogs();
            if (logsRefreshTimer) clearInterval(logsRefreshTimer);
            logsRefreshTimer = setInterval(loadLogs, 3000);
        });
        window.addEventListener('pagehide', function() {
            if (logsRefreshTimer) clearInterval(logsRefreshTimer);
        });
    </script>
</body>
</html>
""",
    "orchestrator-settings.html": """
    <script src="/shared/scripts.js"></script>
</body>
</html>
""",
    "orchestrator-node.html": """
    <script src="/shared/scripts.js"></script>
    <script>
        var params = new URLSearchParams(window.location.search);
        var nodeId = params.get('id');

        function loadNodeDetails() {
            if (!nodeId) {
                document.getElementById('nodeIdTitle').textContent = 'Unknown node';
                return;
            }
            document.getElementById('nodeIdTitle').textContent = nodeId;
            fetch('/admin/nodes')
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    var nodes = data.nodes || [];
                    var node = nodes.find(function(n) { return n.node_id === nodeId; });
                    if (!node) {
                        console.warn('Node not found in admin list:', nodeId);
                    }
                })
                .catch(function(err) { console.error('Failed to load node:', err); });
        }

        document.addEventListener('DOMContentLoaded', loadNodeDetails);
    </script>
</body>
</html>
""",
}


def main() -> None:
    for name, tail in TAILS.items():
        path = ORCH / name
        text = path.read_text(encoding="utf-8")
        idx = -1
        for marker in MARKERS:
            idx = text.find(marker)
            if idx != -1:
                break
        if idx == -1:
            print("skip", name)
            continue
        fixed = tail.replace("</motion>", "</div>")
        path.write_text(text[:idx] + fixed, encoding="utf-8", newline="\n")
        print("fixed", name)


if __name__ == "__main__":
    main()
