# PowerShell migration tarball builder
# Usage:
#   .\scripts\build_migration_tarball.ps1                 # active only (~190 MB)
#   .\scripts\build_migration_tarball.ps1 -IncludeZinc    # +ZINC stock (~820 MB)
#   .\scripts\build_migration_tarball.ps1 -IncludeArchive # +cold archive (~333 MB)
param(
    [switch]$IncludeZinc,
    [switch]$IncludeArchive,
    [string]$OutDir = "."
)

$ErrorActionPreference = "Stop"
$root = (Get-Location).Path
$stamp = Get-Date -Format "yyyy-MM-dd"
$out = Join-Path $OutDir "autoplanner_migration_$stamp.tar.gz"

Write-Host "[migrate] root: $root"
Write-Host "[migrate] out : $out"
Write-Host "[migrate] flags: zinc=$IncludeZinc archive=$IncludeArchive"

# Build the include list
$includes = @(
    "cascade_planner",
    "cascade_dataset.json",
    "cascade_dataset.normalized.uniprot.json",
    "data",
    "data_external",
    "results",
    "MIGRATION_MANIFEST.md",
    "PROPOSAL.md",
    "README_CLUSTER.md",
    "requirements.txt",
    "requirements_aizynth.txt"
)

# aizdata: include all except optionally ZINC
$aizdataAll = Get-ChildItem aizdata -File | ForEach-Object { "aizdata/$($_.Name)" }
if (-not $IncludeZinc) {
    $aizdataAll = $aizdataAll | Where-Object { $_ -notlike "*zinc_stock.hdf5" }
    Write-Host "[migrate] excluding ZINC stock (632 MB)"
}
$includes += $aizdataAll

if ($IncludeArchive) {
    $includes += "archive"
}

# Filter to only existing paths
$existing = $includes | Where-Object { Test-Path $_ }
$missing = $includes | Where-Object { -not (Test-Path $_) }
foreach ($m in $missing) { Write-Warning "[migrate] missing (skip): $m" }

# Use tar (Windows 10+ has bsdtar)
Write-Host "[migrate] packing $($existing.Count) entries ..."
& tar -czf $out --exclude='*.pyc' --exclude='__pycache__' --exclude='.venv_aizynth' $existing
if ($LASTEXITCODE -ne 0) { throw "tar failed (rc=$LASTEXITCODE)" }

$size = (Get-Item $out).Length / 1MB
Write-Host ("[migrate] OK  size = {0:N1} MB" -f $size)

# SHA256
$hash = (Get-FileHash $out -Algorithm SHA256).Hash
Write-Host "[migrate] sha256 = $hash"
Set-Content -Path "$out.sha256" -Value "$hash  $(Split-Path $out -Leaf)"
Write-Host "[migrate] checksum saved -> $out.sha256"
