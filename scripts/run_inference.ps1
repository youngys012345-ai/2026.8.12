param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("image", "video")]
    [string]$MediaType,
    [Parameter(Mandatory = $true)]
    [string]$InputPath,
    [string]$Config = "configs/inference.example.toml",
    [int]$FrameStride = 1
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    if ($MediaType -eq "image") {
        python scripts/infer_image.py --config $Config --input $InputPath
    }
    else {
        python scripts/infer_video.py --config $Config --input $InputPath --frame-stride $FrameStride
    }
}
finally {
    Pop-Location
}
