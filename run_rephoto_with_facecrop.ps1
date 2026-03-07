param(
    [Parameter(Mandatory = $true)]
    [string]$InputImage,

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
    [string]$GFPGANEnvName = "gfpgan_py38",
    [string]$GFPGANRoot = $env:GFPGAN_ROOT,
    [string]$FaceCropEnvName = "facecrop_py310",
    [string]$FaceCropCommand = "face-crop-plus",
    [string]$RephotoEnvName  = "rephoto_cuda11",
    [string]$EncoderCkptPath = ".\checkpoint\encoder\checkpoint_g.pt",
    [string]$PreprocessRoot  = ".\preprocess",
    [string]$ProjectorScriptPath = ".\projector.py",
    [string]$ResultsRoot     = ".\results",

    [double]$GFPGANBlend = 0.35,
    [ValidateSet("b", "gb", "g")]
    [string]$SpectralSensitivity = "b",
    [double]$Gaussian = 0.75,
    [double]$VGGFace = 0.3
)

$ErrorActionPreference = "Stop"

# Use the folder the script lives in as the repo root.
$RepoRoot = $PSScriptRoot

$ConfigPath = Join-Path $RepoRoot "rephoto_wrapper.config.json"

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
    if ($EncoderCkptPath -eq (Join-Path $PSScriptRoot "checkpoint\\encoder\\checkpoint_g.pt") -and $Config.EncoderCkptPath) {
        $EncoderCkptPath = Join-Path $RepoRoot $Config.EncoderCkptPath
    }
    if ($ProjectorScriptPath -eq (Join-Path $PSScriptRoot "projector.py") -and $Config.ProjectorScriptPath) {
        $ProjectorScriptPath = Join-Path $RepoRoot $Config.ProjectorScriptPath
    }
    if ($PreprocessRoot -eq (Join-Path $PSScriptRoot "preprocess") -and $Config.PreprocessRoot) {
        $PreprocessRoot = Join-Path $RepoRoot $Config.PreprocessRoot
    }
    if ($ResultsRoot -eq (Join-Path $PSScriptRoot "results") -and $Config.ResultsRoot) {
        $ResultsRoot = Join-Path $RepoRoot $Config.ResultsRoot
    }
}

# Normalize paths: allow relative paths, but resolve them from the repo root.
function Resolve-RepoPath([string]$p) {
    if ([string]::IsNullOrWhiteSpace($p)) { return $p }
    if ([System.IO.Path]::IsPathRooted($p)) { return $p }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $p))
}

$EncoderCkptPath      = Resolve-RepoPath $EncoderCkptPath
$ProjectorScriptPath  = Resolve-RepoPath $ProjectorScriptPath
$PreprocessRoot       = Resolve-RepoPath $PreprocessRoot
$ResultsRoot          = Resolve-RepoPath $ResultsRoot

if (-not [string]::IsNullOrWhiteSpace($GFPGANRoot)) {
    $GFPGANRoot = Resolve-RepoPath $GFPGANRoot
}

# Resolve and validate input image path.
$ResolvedInput = (Resolve-Path -LiteralPath $InputImage).Path
if (-not (Test-Path -LiteralPath $ResolvedInput)) {
    throw "Input image not found: $InputImage"
}

# Safe base name for folders/files.
$OriginalBase = [System.IO.Path]::GetFileNameWithoutExtension($ResolvedInput)
$SafeBase = ($OriginalBase -replace '[^\p{L}\p{Nd}]+', '_').Trim('_')
if ([string]::IsNullOrWhiteSpace($SafeBase)) {
    $SafeBase = "input_image"
}

$InputExt = [System.IO.Path]::GetExtension($ResolvedInput).ToLower()

# Preset mapping.
$PresetNorm = $Preset.Trim().ToLower()

if ($PresetNorm -eq "test") {
    $W1 = 250
    $W2 = 750
}
else {
    if (-not ($PresetNorm -match '^\d+$')) {
        throw "Preset must be 'test' or a number (e.g., 1500, 3000, 6000, 18000). Got: $Preset"
    }

    $N = [int]$PresetNorm

    if ($N -eq 1500) {
        # allowed special-case
    }
    elseif (($N % 1000) -ne 0) {
        throw "Numeric preset must be 1500 or a multiple of 1000. Got: $N"
    }

    if ($N -lt 1000 -or $N -gt 100000) {
        throw "Numeric preset must be between 1000 and 100000 (or 1500). Got: $N"
    }

    $W1 = 250
    $W2 = $N
}

# Working folders inside the repo.
$PreRoot          = $PreprocessRoot
$TempInputDir     = Join-Path $PreRoot "facecrop_input\$SafeBase"
$CropOutDir       = Join-Path $PreRoot "face_crops\$SafeBase"

$CandidateGFPGAN = Join-Path $RepoRoot "deps\GFPGAN"

if ([string]::IsNullOrWhiteSpace($GFPGANRoot)) {
    if (Test-Path -LiteralPath $CandidateGFPGAN) {
        $GFPGANRoot = $CandidateGFPGAN
    }
}

if ($UseGFPGAN -and (-not (Test-Path -LiteralPath $GFPGANRoot))) {
    throw "GFPGANRoot not found. Set env:GFPGAN_ROOT, pass -GFPGANRoot, or place GFPGAN at: $CandidateGFPGAN"
}
$GFPGANRunRoot    = Join-Path $PreRoot "gfpgan_runs\$SafeBase"
$GFPGANOutputDir  = Join-Path $GFPGANRunRoot "gfpgan_output"
$GFPGANBlendDir   = Join-Path $GFPGANRunRoot "blended_faces"

$RunStamp         = Get-Date -Format "yyyy-MM-dd_HHmmss"
$ResultRoot       = Join-Path $ResultsRoot "integrated_$SafeBase\$RunStamp"

# By default, projector uses the raw cropped faces.
# If GFPGAN is enabled later, we will switch this to the blended faces folder.
$ProjectorInputDir = $CropOutDir

New-Item -ItemType Directory -Path $TempInputDir    -Force | Out-Null
New-Item -ItemType Directory -Path $CropOutDir      -Force | Out-Null
New-Item -ItemType Directory -Path $GFPGANRunRoot   -Force | Out-Null
New-Item -ItemType Directory -Path $GFPGANOutputDir -Force | Out-Null
New-Item -ItemType Directory -Path $GFPGANBlendDir  -Force | Out-Null
New-Item -ItemType Directory -Path $ResultRoot      -Force | Out-Null

if (-not $UseExistingCrops) {
    # Clear prior temp input and prior cropped face files so only this run's files are used.
    Get-ChildItem -LiteralPath $TempInputDir    -File -ErrorAction SilentlyContinue | Remove-Item -Force
    Get-ChildItem -LiteralPath $CropOutDir      -File -ErrorAction SilentlyContinue | Remove-Item -Force
    Get-ChildItem -LiteralPath $GFPGANOutputDir -Recurse -File -ErrorAction SilentlyContinue | Remove-Item -Force
    Get-ChildItem -LiteralPath $GFPGANBlendDir  -File -ErrorAction SilentlyContinue | Remove-Item -Force

    # Copy the chosen image into a temp input folder with a safe name.
    $TempInputFile = Join-Path $TempInputDir "$SafeBase$InputExt"
    Copy-Item -LiteralPath $ResolvedInput -Destination $TempInputFile -Force

    Write-Host ""
    Write-Host "=== Face crop step ==="
    Write-Host "Input image: $ResolvedInput"
    Write-Host "Temp crop input: $TempInputFile"
    Write-Host "Crop output dir: $CropOutDir"
    $PresetLabel = if ($Preset -eq "test") { "750" } else { "$Preset" }
    Write-Host "Preset: $PresetLabel  (wplus_step $W1 $W2)"
    Write-Host ""

    # Run face cropping in the facecrop environment.
    conda run -n $FaceCropEnvName $FaceCropCommand `
        -i $TempInputDir `
        -o $CropOutDir `
        -s 1000 `
        -f jpg `
        -st $Strategy `
        -ff $FaceFactor `
        -dt $DetThreshold

    if ($LASTEXITCODE -ne 0) {
        throw "Face crop failed (command: $FaceCropCommand, env: $FaceCropEnvName)."
    }
}
else {
    Write-Host ""
    Write-Host "=== Face crop step skipped ==="
    Write-Host "Reusing existing crops from: $CropOutDir"
    Write-Host "Preset: $Preset  (wplus_step $W1 $W2)"
    Write-Host ""
}

# Gather cropped faces.
$CropFiles = Get-ChildItem -LiteralPath $CropOutDir -File -Filter "*.jpg" | Sort-Object Name

if ($CropFiles.Count -eq 0) {
    throw "No cropped face JPGs were created in: $CropOutDir"
}

if ($CropIndex -ge 0) {
    if ($CropIndex -ge $CropFiles.Count) {
        throw "Requested CropIndex $CropIndex but only $($CropFiles.Count) cropped face(s) exist."
    }

    $CropFiles = @($CropFiles[$CropIndex])
}

Write-Host ""
Write-Host "Cropped face count: $($CropFiles.Count)"
Write-Host ""

$TotalSteps = if ($CropOnly) { 1 } else { 2 + $CropFiles.Count }
$CurrentStep = 1

Write-Progress -Activity "run_rephoto_with_facecrop" `
    -Status "Crops ready ($CurrentStep of $TotalSteps)" `
    -PercentComplete ([math]::Round(($CurrentStep / $TotalSteps) * 100, 0))

if ($CropOnly) {
    Write-Host "CropOnly requested. Skipping rephoto step."
    Write-Host "Crops are in: $CropOutDir"
    return
}


if (-not $CropOnly) {
    $CurrentStep++
    Write-Progress -Activity "run_rephoto_with_facecrop" `
        -Status "GPU pre-check ($CurrentStep of $TotalSteps)" `
        -PercentComplete ([math]::Round(($CurrentStep / $TotalSteps) * 100, 0))
}

# Optional GFPGAN enhancement + blend stage.
if ($UseGFPGAN) {
    Write-Host "=== GFPGAN step ==="
    Write-Host "GFPGAN root: $GFPGANRoot"
    Write-Host "GFPGAN version: $GFPGANVersion"
    Write-Host "GFPGAN blend: $GFPGANBlend"
    Write-Host ""

    Get-ChildItem -LiteralPath $GFPGANOutputDir -Recurse -File -ErrorAction SilentlyContinue | Remove-Item -Force
    Get-ChildItem -LiteralPath $GFPGANBlendDir  -File    -ErrorAction SilentlyContinue | Remove-Item -Force

    Push-Location -LiteralPath $GFPGANRoot
try {
    conda run -n $GFPGANEnvName python (Join-Path $GFPGANRoot "inference_gfpgan.py") `
        -i $CropOutDir `
        -o $GFPGANOutputDir `
        -v $GFPGANVersion `
        -s 1 `
        --aligned `
        --bg_upsampler none `
        --suffix gfp

    if ($LASTEXITCODE -ne 0) { throw "GFPGAN inference failed." }
}
finally {
    Pop-Location
}

    $BlendScriptPath = Join-Path $env:TEMP "rephoto_gfpgan_blend.py"

@'
from PIL import Image
import os
import sys
import glob

orig_dir = sys.argv[1]
enh_dir = sys.argv[2]
out_dir = sys.argv[3]
alpha = float(sys.argv[4])

os.makedirs(out_dir, exist_ok=True)

for f in sorted(glob.glob(os.path.join(orig_dir, "*.jpg"))):
    base = os.path.splitext(os.path.basename(f))[0]
    matches = sorted(glob.glob(os.path.join(enh_dir, base + "*_gfp.png")))
    if not matches:
        print(f"MISSING {base}")
        continue

    a = Image.open(f).convert("RGBA")
    b = Image.open(matches[0]).convert("RGBA").resize(a.size)
    out = os.path.join(out_dir, base + "_blend.png")
    Image.blend(a, b, alpha).convert("RGB").save(out, quality=95)
    print(f"WROTE {out}")
'@ | Set-Content -LiteralPath $BlendScriptPath

    conda run -n $GFPGANEnvName python $BlendScriptPath `
        $CropOutDir `
        (Join-Path $GFPGANOutputDir "restored_faces") `
        $GFPGANBlendDir `
        $GFPGANBlend

    if ($LASTEXITCODE -ne 0) {
        throw "GFPGAN blend step failed."
    }

    Write-Host "GFPGAN output: $GFPGANOutputDir"
    Write-Host "GFPGAN blended faces: $GFPGANBlendDir"
    Write-Host ""
}

if ($UseGFPGAN) {
    $ProjectorInputDir = $GFPGANBlendDir
}

$ManifestPath = Join-Path $ResultRoot "run_manifest.txt"

@(
    "InputImage=$ResolvedInput"
    "SafeBase=$SafeBase"
    "Preset=$Preset"
    "WPlusStep=$W1,$W2"
    "Strategy=$Strategy"
    "FaceFactor=$FaceFactor"
    "DetThreshold=$DetThreshold"
    "CropIndex=$CropIndex"
    "CropOnly=$CropOnly"
    "UseExistingCrops=$UseExistingCrops"
    "UseGFPGAN=$UseGFPGAN"
    "GFPGANVersion=$GFPGANVersion"
    "GFPGANEnvName=$GFPGANEnvName"
    "FaceCropEnvName=$FaceCropEnvName"
    "FaceCropCommand=$FaceCropCommand"
    "RephotoEnvName=$RephotoEnvName"
    "EncoderCkptPath=$EncoderCkptPath"
    "PreprocessRoot=$PreprocessRoot"
    "ResultsRoot=$ResultsRoot"
    "GFPGANBlend=$GFPGANBlend"
    "ProjectorInputDir=$ProjectorInputDir"
    "CropCount=$($CropFiles.Count)"
    "RunStamp=$RunStamp"
    "CropOutputDir=$CropOutDir"
    "ResultRoot=$ResultRoot"
    "ProjectorScriptPath=$ProjectorScriptPath"
) | Set-Content -LiteralPath $ManifestPath

Write-Host "Manifest: $ManifestPath"
Write-Host ""

Write-Host "=== GPU pre-check ==="

conda run -n $RephotoEnvName python -c "import sys, torch; ok = torch.cuda.is_available(); print(f'cuda_available={ok}'); print(f'device_count={torch.cuda.device_count()}'); sys.exit(0 if ok else 1)"

if ($LASTEXITCODE -ne 0) {
  throw "CUDA is not available in $RephotoEnvName. Aborting before projector run."
}

Write-Host ""

if (-not (Test-Path -LiteralPath $EncoderCkptPath)) {
    throw "Encoder checkpoint not found: $EncoderCkptPath"
}

if (-not (Test-Path -LiteralPath $ProjectorScriptPath)) {
    throw "Projector script not found: $ProjectorScriptPath"
}

# Run projector on each crop in the rephoto environment.
Push-Location -LiteralPath $RepoRoot
try {
    foreach ($Crop in $CropFiles) {
        $CropBase = [System.IO.Path]::GetFileNameWithoutExtension($Crop.Name)
        $ThisResultDir = Join-Path $ResultRoot "$CropBase`_p$Preset"

        $ProjectorImagePath = $Crop.FullName

        if ($UseGFPGAN) {
            $BlendedCandidate = Join-Path $GFPGANBlendDir "$CropBase`_blend.png"
            if (-not (Test-Path -LiteralPath $BlendedCandidate)) {
                throw "Expected GFPGAN blended face not found: $BlendedCandidate"
            }
            $ProjectorImagePath = $BlendedCandidate
        }

        New-Item -ItemType Directory -Path $ThisResultDir -Force | Out-Null

        $CurrentStep++
        Write-Progress -Activity "run_rephoto_with_facecrop" `
            -Status "Rephoto crop $CurrentStep of $TotalSteps" `
            -PercentComplete ([math]::Round(($CurrentStep / $TotalSteps) * 100, 0))

        Write-Host "=== Rephoto step ==="
        Write-Host "Crop: $($Crop.FullName)"
        Write-Host "Results: $ThisResultDir"
        Write-Host ""

        conda run --no-capture-output -n $RephotoEnvName python -u $ProjectorScriptPath `
            $ProjectorImagePath `
            --encoder_ckpt $EncoderCkptPath `
            --encoder_size 256 `
            --e4e_ckpt checkpoint/e4e_ffhq_encode.pt `
            --e4e_size 256 `
            --mix_layer_range 10 18 `
            --coarse_min 32 `
            --color_transfer 10000000000.0 `
            --contextual 0.1 `
            --cx_layers relu3_4 relu2_2 relu1_2 `
            --eye 0.1 `
            --gaussian $Gaussian `
            --spectral_sensitivity $SpectralSensitivity `
            --recon_size 256 `
            --vgg 1 `
            --vggface $VGGFace `
            --lr 0.1 `
            --noise_strength 0.0 `
            --noise_ramp 0.75 `
            --noise_regularize 50000 `
            --camera_lr 0.01 `
            --log_freq 10 `
            --log_visual_freq 1000 `
            --wplus_step $W1 $W2 `
            --results_dir $ThisResultDir

        if ($LASTEXITCODE -ne 0) {
            throw "projector.py failed for crop: $($Crop.Name)"
        }

        $FinalPng = Get-ChildItem -LiteralPath $ThisResultDir -File -Filter "*.png" |
            Where-Object {
                $_.Name -notmatch '(-init|-rand)\.png$' -and
                $_.Name -notmatch '_g\.png$'
            } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1

        if ($null -ne $FinalPng) {
            $SimpleFinal = Join-Path $ThisResultDir "final_$CropBase`_p$Preset.png"
            Copy-Item -LiteralPath $FinalPng.FullName -Destination $SimpleFinal -Force
            Write-Host "Simple final copy: $SimpleFinal"
        }
    }
}
finally {
    Pop-Location
}

Write-Progress -Activity "run_rephoto_with_facecrop" -Completed

Write-Host ""
Write-Host "Done."
Write-Host "Crops are in: $CropOutDir"
Write-Host "Rephoto results are in: $ResultRoot"



