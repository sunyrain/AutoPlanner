param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 7860
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$python = $env:AUTOPLANNER_PYTHON
if (-not $python) {
    $defaultPython = "D:\conda\envs\py312\python.exe"
    if (Test-Path -LiteralPath $defaultPython) {
        $python = $defaultPython
    } else {
        $python = "python"
    }
}

Write-Host "Starting AutoPlanner Agent Workbench..."
Write-Host "URL: http://$HostName`:$Port/agent"
& $python -m cascade_planner.web.app --host $HostName --port $Port
