param(
    [ValidateSet("validate", "plan", "tools", "run")]
    [string]$Command = "plan",
    [string]$Config = "configs/pipeline.example.toml"
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    python -m awesome_llm_apps.cli $Command -c $Config
}
finally {
    Pop-Location
}

