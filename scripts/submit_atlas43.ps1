param(
    [int]$PollSeconds = 420
)

# Submits artifacts/submission/pred_atlas43.vcc once the team's daily
# allowance resets (00:00 UTC). Exits 0 after a successful submission.
#
# Robustness note: native-command stderr from `vcc` must not terminate this
# script, so the probe runs via Start-Process with redirected output and the
# files are parsed as text. Do not set $ErrorActionPreference='Stop' around
# the vcc call -- it converts the quota message into a terminating error.

$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$artifact = 'artifacts/submission/pred_atlas43.vcc'
$model = 'atlas-transfer-v43'
$log = 'artifacts/submit_atlas43.log'
$out = 'artifacts/.probe_out.txt'
$err = 'artifacts/.probe_err.txt'

function Write-Log([string]$msg) {
    $stamp = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    "[$stamp] $msg" | Tee-Object -FilePath $log -Append | Write-Output
}

if (-not (Test-Path $artifact)) {
    Write-Log "missing $artifact"
    exit 1
}
Write-Log "armed; artifact $([math]::Round((Get-Item $artifact).Length/1GB,2)) GB"

while ($true) {
    Remove-Item $out, $err -ErrorAction SilentlyContinue
    $p = Start-Process -FilePath 'vcc' `
        -ArgumentList @('submit', '-m', $model, $artifact) `
        -RedirectStandardOutput $out -RedirectStandardError $err `
        -NoNewWindow -PassThru
    $null = $p.WaitForExit()

    $text = ''
    foreach ($f in @($out, $err)) {
        if (Test-Path $f) { $text += (Get-Content $f -Raw) + "`n" }
    }

    if ($text -match 'entry\s+([A-Za-z0-9]{8,})') {
        $entry = $Matches[1]
        Write-Log "submitted entry=$entry"
        $so = 'artifacts/.status_out.txt'
        $se = 'artifacts/.status_err.txt'
        $sp = Start-Process -FilePath 'vcc' -ArgumentList @('status', $entry, '--wait') `
            -RedirectStandardOutput $so -RedirectStandardError $se -NoNewWindow -PassThru
        $null = $sp.WaitForExit()
        foreach ($f in @($so, $se)) {
            if (Test-Path $f) { Get-Content $f | Tee-Object -FilePath $log -Append }
        }
        exit 0
    }

    if ($text -match 'daily submission allowance') {
        Write-Log "quota still spent; retrying in $PollSeconds s"
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    # A quota-refused or interrupted run can leave a lock whose owning pid is
    # gone. The CLI documents deleting it in exactly this case, so clear it
    # rather than aborting.
    if ($text -match 'is already running on this machine \(pid (\d+)') {
        $owner = [int]$Matches[1]
        if (@(Get-Process -Id $owner -ErrorAction SilentlyContinue).Count -eq 0) {
            $lock = Join-Path $env:USERPROFILE '.config\vcc\locks\submit-default.lock'
            Remove-Item $lock -Force -ErrorAction SilentlyContinue
            Write-Log "cleared stale submit lock (pid $owner not running)"
            continue
        }
    }

    Write-Log "non-quota failure, aborting"
    Write-Output $text
    exit 2
}