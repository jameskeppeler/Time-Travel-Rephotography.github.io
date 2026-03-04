param(
    [Parameter(Mandatory = $true)]
    [string]$InputDir,

    [ValidateSet("test", "1500", "3000", "6000")]
    [string]$Preset = "3000",

    [ValidateSet("all", "largest")]
    [string]$Strategy = "all",
    [double]$FaceFactor = 0.65,
    [double]$DetThreshold = 0.9,
    [int]$CropIndex = -1,
    [switch]$CropOnly,
    [switch]$UseExistingCrops,
    [switch]$UseGFPGAN,

    [ValidateSet("1.3", "1.4")]
    [string]$GFPGANVersion = "1.3",
    [string]$GFPGANEnvName   = "gfpgan_py38",
    [string]$GFPGANRoot      = $env:GFPGAN_ROOT,
    [string]$FaceCropEnvName = "facecrop_py310",
    [string]$RephotoEnvName  = "rephoto_cuda11",

[double]$GFPGANBlend = 0.35
))

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
$SingleRunner = Join-Path $RepoRoot "run_rephoto_with_facecrop.ps1"

if (-not (Test-Path -LiteralPath $SingleRunner)) {
    throw "Single-image wrapper not found: $SingleRunner"
}

$ResolvedInputDir = (Resolve-Path -LiteralPath $InputDir).Path

$Files = Get-ChildItem -LiteralPath $ResolvedInputDir -File |
    Where-Object { $_.Extension -match '^\.(jpg|jpeg|png|bmp|tif|tiff|webp)$' } |
    Sort-Object Name

if ($Files.Count -eq 0) {
    throw "No supported image files found in: $ResolvedInputDir"
}

Write-Host ""
Write-Host "=== Batch run ==="
Write-Host "Input dir: $ResolvedInputDir"
Write-Host "Image count: $($Files.Count)"
Write-Host "Preset: $Preset"
Write-Host ""

foreach ($File in $Files) {
    Write-Host "=== Batch item ==="
    Write-Host "Image: $($File.FullName)"
    Write-Host ""

$RunArgs = @{
    InputImage      = $File.FullName
    Preset          = $Preset
    Strategy        = $Strategy
    FaceFactor      = $FaceFactor
    DetThreshold    = $DetThreshold
    CropIndex       = $CropIndex
    GFPGANEnvName   = $GFPGANEnvName
    GFPGANRoot      = $GFPGANRoot
    FaceCropEnvName = $FaceCropEnvName
    RephotoEnvName  = $RephotoEnvName
    GFPGANBlend     = $GFPGANBlend
    GFPGANVersion   = $GFPGANVersion
}

    if ($CropOnly) {
        $RunArgs.CropOnly = $true
    }

    if ($UseExistingCrops) {
        $RunArgs.UseExistingCrops = $true
    }

    if ($UseGFPGAN) {
    $RunArgs.UseGFPGAN = $true
}
    & $SingleRunner @RunArgs
}