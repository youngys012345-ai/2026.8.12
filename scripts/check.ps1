$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    python -m ruff check .
    python -m unittest discover -s tests -v
    python -m industrial_video_detection.cli validate -c configs/pipeline.example.toml
}
finally {
    Pop-Location
}

