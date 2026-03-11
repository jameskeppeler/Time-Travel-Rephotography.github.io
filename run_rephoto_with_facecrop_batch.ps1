param(
    [Parameter(Mandatory = $true)]
    [string]$InputDir,

    [string]$Preset = "3000",

    [ValidateSet("all", "largest")]
    [string]$Strategy = "all",
    [double]$FaceFactor = 0.65,
    [double]$DetThreshold = 0.9,
    [int]$CropIndex = -1,
    [string]$SelectedCropIndices = "",
    [switch]$CropOnly,
    [switch]$UseExistingCrops,
    [switch]$UseGFPGAN,

    [ValidateSet("1.3", "1.4")]
    [string]$GFPGANVersion = "1.3",
    [string]$GFPGANEnvName   = "gfpgan_py38",
    [string]$GFPGANRoot      = $env:GFPGAN_ROOT,
    [string]$FaceCropEnvName = "facecrop_py310",
    [string]$FaceCropCommand = "face-crop-plus",
    [string]$RephotoEnvName  = "rephoto_cuda11",
    [string]$EncoderCkptPath = (Join-Path $PSScriptRoot "checkpoint\encoder\checkpoint_g.pt"),
    [string]$ProjectorScriptPath  = (Join-Path $PSScriptRoot "projector.py"),
    [string]$PreprocessRoot  = (Join-Path $PSScriptRoot "preprocess"),
    [string]$ResultsRoot     = (Join-Path $PSScriptRoot "results"),

    [double]$GFPGANBlend = 0.35,
    [ValidateSet("b", "gb", "g")]
    [string]$SpectralSensitivity = "b",
    [double]$Gaussian = 0.75,
    [double]$VGGFace = 0.3,
    [double]$VGG = 1.0,
    [double]$ColorTransfer = 10000000000.0,
    [double]$Eye = 0.1,
    [double]$Contextual = 0.1,
    [double]$NoiseRegularize = 50000.0,
    [double]$LR = 0.1,
    [double]$CameraLR = 0.01,
    [int]$MixLayerStart = 10,
    [int]$MixLayerEnd = 18
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot

$ConfigPath = Join-Path $RepoRoot "rephoto_wrapper.config.json"

function Test-IsDefaultPathValue([string]$CurrentValue, [string]$DefaultRelativePath) {
    if ([string]::IsNullOrWhiteSpace($CurrentValue)) {
        return $false
    }

    try {
        $DefaultFull = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $DefaultRelativePath))
        if ([System.IO.Path]::IsPathRooted($CurrentValue)) {
            $CurrentFull = [System.IO.Path]::GetFullPath($CurrentValue)
        }
        else {
            $CurrentFull = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $CurrentValue))
        }
        return [string]::Equals($CurrentFull, $DefaultFull, [System.StringComparison]::OrdinalIgnoreCase)
    }
    catch {
        return $false
    }
}

if (Test-Path -LiteralPath $ConfigPath) {
    $Config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json

    if ([string]::IsNullOrWhiteSpace($GFPGANRoot) -and $Config.GFPGANRoot) {
        $GFPGANRoot = Join-Path $RepoRoot $Config.GFPGANRoot
    }
    if ($GFPGANEnvName -eq "gfpgan_py38" -and $Config.GFPGANEnvName) {
        $GFPGANEnvName = $Config.GFPGANEnvName
    }
    if ($FaceCropEnvName -eq "facecrop_py310" -and $Config.FaceCropEnvName) {
        $FaceCropEnvName = $Config.FaceCropEnvName
    }
    if ($FaceCropCommand -eq "face-crop-plus" -and $Config.FaceCropCommand) {
        $FaceCropCommand = $Config.FaceCropCommand
    }
    if ($RephotoEnvName -eq "rephoto_cuda11" -and $Config.RephotoEnvName) {
        $RephotoEnvName = $Config.RephotoEnvName
    }
    if ((Test-IsDefaultPathValue $EncoderCkptPath "checkpoint\encoder\checkpoint_g.pt") -and $Config.EncoderCkptPath) {
        $EncoderCkptPath = Join-Path $RepoRoot $Config.EncoderCkptPath
    }
    if ((Test-IsDefaultPathValue $ProjectorScriptPath "projector.py") -and $Config.ProjectorScriptPath) {
        $ProjectorScriptPath = Join-Path $RepoRoot $Config.ProjectorScriptPath
    }
    if ((Test-IsDefaultPathValue $PreprocessRoot "preprocess") -and $Config.PreprocessRoot) {
        $PreprocessRoot = Join-Path $RepoRoot $Config.PreprocessRoot
    }
    if ((Test-IsDefaultPathValue $ResultsRoot "results") -and $Config.ResultsRoot) {
        $ResultsRoot = Join-Path $RepoRoot $Config.ResultsRoot
    }
}

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
        EncoderCkptPath = $EncoderCkptPath
        ProjectorScriptPath = $ProjectorScriptPath
        PreprocessRoot  = $PreprocessRoot
        ResultsRoot     = $ResultsRoot
        FaceCropCommand = $FaceCropCommand
        GFPGANBlend     = $GFPGANBlend
        GFPGANVersion   = $GFPGANVersion
        SpectralSensitivity = $SpectralSensitivity
        Gaussian        = $Gaussian
        VGGFace         = $VGGFace
        VGG             = $VGG
        ColorTransfer   = $ColorTransfer
        Eye             = $Eye
        Contextual      = $Contextual
        NoiseRegularize = $NoiseRegularize
        LR              = $LR
        CameraLR        = $CameraLR
        MixLayerStart   = $MixLayerStart
        MixLayerEnd     = $MixLayerEnd
    }

    if ($CropOnly) {
        $RunArgs.CropOnly = $true
    }

    if (-not [string]::IsNullOrWhiteSpace($SelectedCropIndices)) {
        $RunArgs.SelectedCropIndices = $SelectedCropIndices
    }

    if ($UseExistingCrops) {
        $RunArgs.UseExistingCrops = $true
    }

    if ($UseGFPGAN) {
        $RunArgs.UseGFPGAN = $true
    }
    & $SingleRunner @RunArgs
}
