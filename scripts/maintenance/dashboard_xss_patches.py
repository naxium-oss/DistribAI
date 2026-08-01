"""
One-off maintenance: apply regex-based substitutions in dashboard static HTML.

**Dangerous if misused** — review diffs before committing. Run from repository root:

    python -m scripts.maintenance.dashboard_xss_patches

Prefer fixing templates by hand and keeping this script for emergencies only.
"""

import re
from pathlib import Path


def apply_dashboard_xss_patches() -> bool:
    """Replace selected patterns under ``worker/src/dashboard/static`` HTML."""

    dashboard_dir = Path("worker/src/dashboard/static")
    html_files = sorted(dashboard_dir.rglob("*.html"))

    pattern_fixes = [
        (r"\.innerHTML\s*=", ".textContent ="),
        (r"\.outerHTML\s*=", ".textContent ="),
        (r"document\.write\s*\(", "document.appendChild(document.createTextNode("),
        (r"eval\s*\(", "// eval removed"),
        (r"Function\s*\(", "// Function constructor removed"),
        (r"setTimeout\s*\(", "// setTimeout removed"),
        (r"setInterval\s*\(", "// setInterval removed"),
    ]

    files_fixed = 0

    for html_file in html_files:
        try:
            content = html_file.read_text(encoding="utf-8")
            modified = False
            for pattern, replacement in pattern_fixes:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    modified = True

            if modified:
                html_file.write_text(content, encoding="utf-8")
                files_fixed += 1
                print(f"   [FIXED] Dangerous patterns in {html_file.name}")
            else:
                print(f"   [OK] No dangerous patterns in {html_file.name}")

        except OSError as e:
            print(f"   [ERROR] Failed to fix {html_file.name}: {e}")

    print(f"\nFixed dangerous patterns in {files_fixed} dashboard files")
    return files_fixed > 0


if __name__ == "__main__":
    ok = apply_dashboard_xss_patches()
    if ok:
        print("\n[SUCCESS] Dashboard XSS pattern pass completed")
    else:
        print("\n[INFO] No fixes applied")
