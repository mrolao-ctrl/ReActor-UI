# ReActor Studio Windows launcher. No network access or model downloads here.
# Reuses the existing Python virtual environment on this PC.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$logDir = Join-Path $env:LOCALAPPDATA 'ReActor-Studio\logs'

function Show-ReActorError([string] $message) {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show(
        $message, 'ReActor Studio', 'OK', 'Error'
    ) | Out-Null
}

try {
    $candidates = @()
    if ($env:REACTOR_PYTHON) {
        $candidates += $env:REACTOR_PYTHON
    }
    $candidates += @(
        (Join-Path $env:USERPROFILE 'ReActor-UI\.venv\Scripts\pythonw.exe'),
        (Join-Path $env:USERPROFILE 'ReActor-Custom\.venv\Scripts\pythonw.exe'),
        (Join-Path $env:USERPROFILE 'ReActor-UI\venv\Scripts\pythonw.exe'),
        (Join-Path $env:USERPROFILE 'ReActor-Custom\venv\Scripts\pythonw.exe')
    )
    $python = $null
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $python = (Resolve-Path -LiteralPath $candidate).Path
            break
        }
    }

    if (-not $python) {
        $message = 'No existing ReActor Python environment was found.' +
            [Environment]::NewLine + 'This installer reuses Python and models already on your PC.' +
            [Environment]::NewLine + (Join-Path $env:USERPROFILE 'ReActor-UI\.venv\Scripts\pythonw.exe') +
            [Environment]::NewLine + 'Alternatively set REACTOR_PYTHON to your own Python executable.'
        Show-ReActorError $message
        exit 2
    }

    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
        Show-ReActorError 'FFmpeg is missing from PATH. Install it or add it to PATH.'
        exit 3
    }

    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $env:REACTOR_GUI_LAUNCH = '1'
    Start-Process -FilePath $python -ArgumentList @('run.py') -WorkingDirectory $root -ErrorAction Stop
}
catch {
    Show-ReActorError ("Could not launch ReActor Studio: $($_.Exception.Message)")
    exit 1
}
