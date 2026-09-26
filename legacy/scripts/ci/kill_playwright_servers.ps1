# Safe Playwright teardown (delegates to Node; never kills all Chrome/Chromium).
param(
    [switch]$DryRun
)

$ErrorActionPreference = 'SilentlyContinue'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$args = @()
if ($DryRun) { $args += '--dry-run' }
& node (Join-Path $scriptDir 'kill_playwright_servers.cjs') @args
exit $LASTEXITCODE
