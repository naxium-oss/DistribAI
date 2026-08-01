# Bootstrap desloppify for DistribAI (local state under .desloppify/)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..\..
$py = if (Test-Path ".\venv\Scripts\python.exe") { ".\venv\Scripts\python.exe" } else { "python" }

Write-Host "[desloppify] installing package..."
& $py -m pip install --upgrade "desloppify[full]"

Write-Host "[desloppify] installing cursor skill guide..."
desloppify update-skill cursor

$exclude = @(
  "node_modules", "dist", "build", "venv", ".venv", "external",
  "test-results", "playwright-report", "htmlcov", "runtime/db",
  "runtime/smoke", "runtime/bundles", "runtime/checkpoints",
  "worker/src/distribai_proto", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache"
)
foreach ($path in $exclude) {
  if (Test-Path $path) {
    Write-Host "[desloppify] exclude $path"
    desloppify exclude $path
  }
}

Write-Host "[desloppify] scan..."
desloppify scan --path .

Write-Host "[desloppify] next item:"
desloppify next
