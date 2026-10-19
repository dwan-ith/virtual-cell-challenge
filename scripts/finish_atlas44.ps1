$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$log = 'artifacts/atlas44_finish.log'
$h5 = 'artifacts/submission/pred_atlas44.h5ad'
$vcc = 'artifacts/submission/pred_atlas44.vcc'
function Note($m) { "$((Get-Date).ToUniversalTime().ToString('u')) $m" | Tee-Object -FilePath $log -Append }

while (-not (Test-Path $h5)) {
    $alive = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*atlas_predict*' }).Count
    if (-not $alive) { Note "build gone and no h5ad; aborting finisher"; exit 2 }
    Start-Sleep -Seconds 60
}
Note ("h5ad ready ({0:N0} MB); packing" -f ((Get-Item $h5).Length / 1MB))

& vcc prep $h5 -g data/vcc2026/gene_names.csv --perts data/vcc2026/pert_counts.csv -o $vcc --force 2>&1 |
    Tee-Object -FilePath $log -Append
if (-not (Test-Path $vcc)) {
    Note "vcc prep failed; retrying with headerless gene list"
    $genes = 'artifacts/gene_names_headerless.csv'
    Get-Content data/vcc2026/gene_names.csv | Select-Object -Skip 1 | Set-Content $genes
    & vcc prep $h5 -g $genes --perts data/vcc2026/pert_counts.csv -o $vcc --force 2>&1 |
        Tee-Object -FilePath $log -Append
}
if (-not (Test-Path $vcc)) { Note "packing FAILED"; exit 1 }
Note ("packed ({0:N0} MB); deleting h5ad to reclaim disk" -f ((Get-Item $vcc).Length / 1MB))
Remove-Item $h5 -Force -ErrorAction SilentlyContinue

while ($true) {
    $out = (& vcc submit $vcc -m atlas-transfer-v44 2>&1 | Out-String)
    $out | Tee-Object -FilePath $log -Append
    if ($out -match '✓ submitted' -or $out -match 'entry:') { Note "SUBMITTED"; break }
    if ($out -match 'allowance' -or $out -match 'quota' -or $out -match 'limit') {
        Note "quota spent; retry in 600s"; Start-Sleep -Seconds 600; continue
    }
    Note "indeterminate submit outcome; retry in 600s"; Start-Sleep -Seconds 600
}
