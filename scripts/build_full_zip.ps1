# Builds one big zip of all useful assets for cluster migration.
# Excludes: .venv_aizynth, __pycache__, *.pyc, .git, archive/, ChemEnzyRetroPlanner/, root *.log
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\build_full_zip.ps1

param(
    [string]$OutFile = "autoplanner_full_2026-04-23.zip"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$root = (Resolve-Path ".").Path
$zipPath = Join-Path $root $OutFile
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

# ---- selection ----
$includeDirs = @(
    "cascade_planner",
    "scripts",
    "aizdata",
    "results",
    "data",
    "data_external"
)
$includeRootFiles = @(
    "cascade_dataset.json",
    "cascade_dataset.normalized.uniprot.json",
    "PROPOSAL.md",
    "MIGRATION_MANIFEST.md",
    "README_CLUSTER.md",
    "README.md",
    "requirements.txt",
    "requirements_aizynth.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg"
)

# any cascade_full_snapshot_*.json at root (training inputs)
$snapshotFiles = Get-ChildItem -Path $root -File -Filter "cascade_full_snapshot_*.json" -ErrorAction SilentlyContinue | ForEach-Object { $_.Name }
$includeRootFiles += $snapshotFiles

# ---- exclusion predicate ----
$excludeNames = @('.venv_aizynth','__pycache__','.git','.idea','.vscode','node_modules','archive','ChemEnzyRetroPlanner','.pytest_cache','.mypy_cache','.ipynb_checkpoints')
$excludeExts  = @('.pyc','.pyo')

function ShouldExclude($relPath) {
    $parts = $relPath -split '[\\/]'
    foreach ($p in $parts) { if ($excludeNames -contains $p) { return $true } }
    $ext = [System.IO.Path]::GetExtension($relPath).ToLower()
    if ($excludeExts -contains $ext) { return $true }
    return $false
}

# ---- collect file list ----
Write-Host "Scanning files..." -ForegroundColor Cyan
$files = New-Object System.Collections.Generic.List[string]

foreach ($d in $includeDirs) {
    $full = Join-Path $root $d
    if (-not (Test-Path $full)) { Write-Host "  skip missing: $d" -ForegroundColor DarkGray; continue }
    Get-ChildItem -Path $full -Recurse -File -Force | ForEach-Object {
        $rel = $_.FullName.Substring($root.Length+1)
        if (-not (ShouldExclude $rel)) { $files.Add($_.FullName) }
    }
}
foreach ($f in $includeRootFiles) {
    $full = Join-Path $root $f
    if (Test-Path $full -PathType Leaf) { $files.Add($full) }
}

$totalBytes = ($files | ForEach-Object { (Get-Item $_).Length } | Measure-Object -Sum).Sum
Write-Host ("Found {0} files, {1:N1} MB raw" -f $files.Count, ($totalBytes/1MB)) -ForegroundColor Cyan

# ---- write zip ----
Write-Host "Writing $zipPath ..." -ForegroundColor Cyan
$zip = [System.IO.Compression.ZipFile]::Open($zipPath, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    $i = 0
    foreach ($f in $files) {
        $i++
        $rel = $f.Substring($root.Length+1).Replace('\','/')
        if ($i % 200 -eq 0 -or $i -eq $files.Count) {
            Write-Host ("  [{0}/{1}] {2}" -f $i, $files.Count, $rel) -ForegroundColor DarkGray
        }
        try {
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $zip, $f, $rel, [System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
        } catch {
            Write-Host ("  WARN skip {0}: {1}" -f $rel, $_.Exception.Message) -ForegroundColor Yellow
        }
    }
} finally {
    $zip.Dispose()
}

# ---- summary + sha256 ----
$zipSize = (Get-Item $zipPath).Length
$sha = (Get-FileHash -Algorithm SHA256 -Path $zipPath).Hash
Set-Content -Path "$zipPath.sha256" -Value "$sha  $OutFile"

Write-Host ""
Write-Host ("Done: {0}" -f $zipPath) -ForegroundColor Green
Write-Host ("  Size:   {0:N1} MB" -f ($zipSize/1MB)) -ForegroundColor Green
Write-Host ("  SHA256: {0}" -f $sha) -ForegroundColor Green
Write-Host ("  Files:  {0}" -f $files.Count) -ForegroundColor Green
