param(
    [int]$Port = 8765,
    [string]$HostAddress = "127.0.0.1",
    [string]$RuntimeRoot = "",
    [string]$RunsRoot = "",
    [string]$ArtifactStoreRoot = "",
    [string]$RunIndexPath = "",
    [string]$ExternalDataRoot = "",
    [ValidateSet("waitress", "flask")]
    [string]$Server = "waitress",
    [int]$Threads = 4,
    [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $RuntimeRoot) {
    $RuntimeRoot = Join-Path $RepoRoot "results\.autoplanner"
}
if (-not $ExternalDataRoot) {
    $ExternalDataRoot = Join-Path $RepoRoot "data_external"
}
if (-not $RunsRoot) {
    $RunsRoot = Join-Path $RuntimeRoot "runs"
}
if (-not $ArtifactStoreRoot) {
    $ArtifactStoreRoot = Join-Path $RuntimeRoot "artifacts"
}
if (-not $RunIndexPath) {
    $RunIndexPath = Join-Path $RuntimeRoot "run_index.sqlite3"
}

$env:AUTOPLANNER_RUNTIME_ROOT = $RuntimeRoot
$env:AUTOPLANNER_RUNS_ROOT = $RunsRoot
$env:AUTOPLANNER_ARTIFACT_STORE_ROOT = $ArtifactStoreRoot
$env:AUTOPLANNER_RUN_INDEX_PATH = $RunIndexPath
$env:AUTOPLANNER_EXTERNAL_DATA_ROOT = $ExternalDataRoot
$env:AUTOPLANNER_VENDOR_ROOT = Join-Path $RepoRoot "vendor"

$Url = "http://${HostAddress}:${Port}/v4"
Write-Host "AutoPlanner V4 workspace: $Url"
Write-Host "Runtime root: $RuntimeRoot"
Write-Host "Runs root: $RunsRoot"
Write-Host "Artifact store: $ArtifactStoreRoot"
Write-Host "External data root: $ExternalDataRoot"
Write-Host "Server: $Server ($Threads threads)"
Write-Host "Press Ctrl+C to stop."

if ($OpenBrowser) {
    Start-Process $Url -WindowStyle Hidden
}

Push-Location $RepoRoot
try {
    python -m cascade_planner serve --host $HostAddress --port $Port --server $Server --threads $Threads
}
finally {
    Pop-Location
}
