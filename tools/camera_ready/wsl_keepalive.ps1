<#
Keep the WSL distribution running while a camera-ready training run is in progress (Windows side).

    powershell -ExecutionPolicy Bypass -File tools\camera_ready\wsl_keepalive.ps1 [start|status|stop] [-Distro Ubuntu-22.04]

WSL may stop a distribution when no Windows client is attached, which would kill a run in tmux.
"start" launches one hidden `wsl.exe -d <distro> --exec sleep infinity` and records its PID in
%LOCALAPPDATA%\pfn-cr\wsl-keepalive.json. It is idempotent: a live recorded process is reused.
"stop" ends only the recorded process. After a reboot, run "start" again (nothing starts by itself).
#>
param(
    [ValidateSet('start', 'status', 'stop')][string]$Action = 'start',
    [string]$Distro = 'Ubuntu-22.04'
)
$ErrorActionPreference = 'Stop'
$dir = Join-Path $env:LOCALAPPDATA 'pfn-cr'
$state = Join-Path $dir 'wsl-keepalive.json'

function Get-Alive {
    if (-not (Test-Path $state)) { return $null }
    $s = Get-Content $state -Raw | ConvertFrom-Json
    $p = Get-Process -Id $s.pid -ErrorAction SilentlyContinue
    if ($p -and $p.ProcessName -eq 'wsl') { return $s }
    return $null
}

switch ($Action) {
    'start' {
        $s = Get-Alive
        if ($s) { "already running: PID $($s.pid) since $($s.started) ($($s.distro))"; break }
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $p = Start-Process -FilePath 'wsl.exe' -ArgumentList @('-d', $Distro, '--exec', 'sleep', 'infinity') -WindowStyle Hidden -PassThru
        $rec = [ordered]@{ pid = $p.Id; distro = $Distro; started = (Get-Date).ToString('s') }
        $rec | ConvertTo-Json | Set-Content -Encoding utf8 $state
        "started: PID $($p.Id) ($Distro); state $state"
    }
    'status' {
        $s = Get-Alive
        if ($s) { "running: PID $($s.pid) since $($s.started) ($($s.distro))" } else { 'not running' }
    }
    'stop' {
        $s = Get-Alive
        if ($s) { Stop-Process -Id $s.pid; Remove-Item $state; "stopped PID $($s.pid)" } else { 'not running' }
    }
}
