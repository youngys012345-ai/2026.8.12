param(
    [ValidateSet("validate", "plan", "tools", "run")]
    [string]$Command = "plan",
    [string]$Config = "configs/pipeline.example.toml"
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    python -m industrial_video_detection.cli $Command -c $Config
}
finally {
    Pop-Location
}


