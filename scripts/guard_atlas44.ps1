$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$log = 'artifacts/atlas44_guard.log'
$target = 'artifacts/submission/pred_atlas44.h5ad'
$partial = 'artifacts/submission/pred_atlas44.partial.h5ad'

function Note($m) { "$((Get-Date).ToUniversalTime().ToString('u')) $m" | Tee-Object -FilePath $log -Append }

Note "guard armed; watching build + disk"
while ($true) {
    $free = (Get-PSDrive C).Free / 1GB
    if (Test-Path $target) { Note ("h5ad complete; free={0:N2}GB" -f $free); break }
    $buildAlive = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*atlas_predict*' }).Count
    if ($free -lt 0.6) {
        Note ("LOW DISK {0:N2}GB - stopping build to protect the machine" -f $free)
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object { $_.CommandLine -like '*atlas_predict*' } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
        if (Test-Path $partial) { Remove-Item $partial -Force }
        Note "build stopped; artifact removed"
        exit 1
    }
    if (-not $buildAlive) {
        Note "build process gone without producing target; check atlas44_build.err"
        exit 2
    }
    Start-Sleep -Seconds 60
}
Note "build finished; guard exiting (pack/submit handled separately)"
