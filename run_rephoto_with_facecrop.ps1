param(
    [Parameter(Mandatory = $true)]
    [string]$InputImage,

    [string]$Preset = "3000",

    [ValidateSet("all", "largest")]
    [string]$Strategy = "all",

    [double]$FaceFactor = 0.65,

    [double]$DetThreshold = 0.9,

    [int]$CropIndex = -1,
    [string]$SelectedCropIndices = "",
    [string]$SelectedCropNames = "",
    [string]$StopFlagPath = "",
    [string]$PauseFlagPath = "",

    [switch]$CropOnly,

    [switch]$UseExistingCrops,
    [switch]$RequireSelection,

    [switch]$UseGFPGAN,
    [switch]$RecompositeOriginalImage,

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
    [double]$VGGFace = 0.3,
    [double]$VGG = 1.0,
    [double]$ColorTransfer = 10000000000.0,
    [double]$Eye = 0.1,
    [double]$Contextual = 0.1,
    [double]$NoiseRegularize = 50000.0,
    [double]$LR = 0.1,
    [double]$CameraLR = 0.01,
    [int]$MixLayerStart = 10,
    [int]$MixLayerEnd = 18,

    # Backend optimization flags
    [switch]$UseAMP,
    [int]$EarlyStopPatience = 0,
    [double]$EarlyStopMinDelta = 0.0001,
    [double]$LRDecay = 0.0
)

$ErrorActionPreference = "Stop"

# Use the folder the script lives in as the repo root.
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
elseif ($PresetNorm -eq "375") {
    $W1 = 125
    $W2 = 375
}
else {
    if (-not ($PresetNorm -match '^\d+$')) {
        throw "Preset must be 'test', 375, or a number (e.g., 1500, 3000, 6000, 18000). Got: $Preset"
    }

    $N = [int]$PresetNorm

    if ($N -eq 1500) {
        # allowed special-case
    }
    elseif (($N % 1000) -ne 0) {
        throw "Numeric preset must be 1500, 375, or a multiple of 1000. Got: $N"
    }

    if ($N -lt 1000 -or $N -gt 100000) {
        throw "Numeric preset must be between 1000 and 100000 (or 1500, 375). Got: $N"
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
New-Item -ItemType Directory -Path $ResultRoot      -Force | Out-Null
if ($UseGFPGAN) {
    New-Item -ItemType Directory -Path $GFPGANRunRoot   -Force | Out-Null
    New-Item -ItemType Directory -Path $GFPGANOutputDir -Force | Out-Null
    New-Item -ItemType Directory -Path $GFPGANBlendDir  -Force | Out-Null
}

if (-not [string]::IsNullOrWhiteSpace($StopFlagPath)) {
    try {
        if (Test-Path -LiteralPath $StopFlagPath) {
            Remove-Item -LiteralPath $StopFlagPath -Force -ErrorAction Stop
        }
    }
    catch {
        Write-Warning "Failed to clear stop flag path '$StopFlagPath' before run start: $($_.Exception.Message)"
    }
}

if (-not [string]::IsNullOrWhiteSpace($PauseFlagPath)) {
    try {
        if (Test-Path -LiteralPath $PauseFlagPath) {
            Remove-Item -LiteralPath $PauseFlagPath -Force -ErrorAction Stop
        }
    }
    catch {
        Write-Warning "Failed to clear pause flag path '$PauseFlagPath' before run start: $($_.Exception.Message)"
    }
}

if (-not $UseExistingCrops) {
    # Clear prior temp input and prior cropped face files so only this run's files are used.
    Get-ChildItem -LiteralPath $TempInputDir    -File -ErrorAction SilentlyContinue | Remove-Item -Force
    Get-ChildItem -LiteralPath $CropOutDir      -File -ErrorAction SilentlyContinue | Remove-Item -Force
    if ($UseGFPGAN) {
        Get-ChildItem -LiteralPath $GFPGANOutputDir -Recurse -File -ErrorAction SilentlyContinue | Remove-Item -Force
        Get-ChildItem -LiteralPath $GFPGANBlendDir  -File -ErrorAction SilentlyContinue | Remove-Item -Force
    }

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
    $FaceCropStart = [System.Diagnostics.Stopwatch]::StartNew()
    conda run -n $FaceCropEnvName $FaceCropCommand `
        -i $TempInputDir `
        -o $CropOutDir `
        -s 2048 `
        -f png `
        -r 3072 `
        -st $Strategy `
        -ff $FaceFactor `
        -dt $DetThreshold
    $FaceCropStart.Stop()
    Write-Host "=== Face crop elapsed: $([math]::Round($FaceCropStart.Elapsed.TotalSeconds, 1))s ==="

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
$CropFiles = Get-ChildItem -LiteralPath $CropOutDir -File |
    Where-Object { $_.Extension -match '^\.(png|jpg|jpeg|bmp|tif|tiff|webp)$' } |
    Sort-Object @{
        Expression = {
            $stem = [System.IO.Path]::GetFileNameWithoutExtension($_.Name)
            if ($stem -match '_(\d+)$') { [int]$Matches[1] } else { [int]::MaxValue }
        }
    }, @{
        Expression = { $_.Name }
    }

if ($CropFiles.Count -eq 0) {
    throw "No cropped face image files were created in: $CropOutDir"
}

$AllCropFiles = @($CropFiles)

$ParsedSelectedCropIndices = @()
if (-not [string]::IsNullOrWhiteSpace($SelectedCropIndices)) {
    $ParsedSelectedCropIndices = @(
        $SelectedCropIndices -split '[,\s;]+' |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        ForEach-Object { [int]$_ } |
        Sort-Object -Unique
    )
}

$ParsedSelectedCropNames = @()
if (-not [string]::IsNullOrWhiteSpace($SelectedCropNames)) {
    $SeenSelectedCropNames = @{}
    foreach ($RawName in ($SelectedCropNames -split '[,\s;]+')) {
        $Name = [string]$RawName
        if ([string]::IsNullOrWhiteSpace($Name)) {
            continue
        }
        $Key = $Name.ToLowerInvariant()
        if (-not $SeenSelectedCropNames.ContainsKey($Key)) {
            $SeenSelectedCropNames[$Key] = $true
            $ParsedSelectedCropNames += $Name
        }
    }
}

if ($RequireSelection -and $ParsedSelectedCropIndices.Count -eq 0 -and $ParsedSelectedCropNames.Count -eq 0 -and $CropIndex -lt 0) {
    throw "RequireSelection was set, but no selected crops were provided."
}

if ($ParsedSelectedCropNames.Count -gt 0) {
    if ($CropIndex -ge 0) {
        throw "Use either -CropIndex or -SelectedCropNames, not both."
    }
    if ($ParsedSelectedCropIndices.Count -gt 0) {
        Write-Warning "Both SelectedCropNames and SelectedCropIndices were provided. Using SelectedCropNames and ignoring SelectedCropIndices."
    }

    $CropNameToIndex = @{}
    for ($i = 0; $i -lt $AllCropFiles.Count; $i++) {
        $File = $AllCropFiles[$i]
        $NameKey = $File.Name.ToLowerInvariant()
        if (-not $CropNameToIndex.ContainsKey($NameKey)) {
            $CropNameToIndex[$NameKey] = $i
        }
        $StemKey = [System.IO.Path]::GetFileNameWithoutExtension($File.Name).ToLowerInvariant()
        if (-not $CropNameToIndex.ContainsKey($StemKey)) {
            $CropNameToIndex[$StemKey] = $i
        }
    }

    $NormalizedNameIndices = @()
    foreach ($RequestedName in $ParsedSelectedCropNames) {
        $RequestedKeys = @()
        $RequestedKeys += $RequestedName
        $RequestedFileName = [System.IO.Path]::GetFileName($RequestedName)
        if (-not [string]::IsNullOrWhiteSpace($RequestedFileName)) {
            $RequestedKeys += $RequestedFileName
            $RequestedStem = [System.IO.Path]::GetFileNameWithoutExtension($RequestedFileName)
            if (-not [string]::IsNullOrWhiteSpace($RequestedStem)) {
                $RequestedKeys += $RequestedStem
            }
        }
        $RequestedKeys = @($RequestedKeys | ForEach-Object { $_.ToLowerInvariant() } | Sort-Object -Unique)

        $MatchIndex = $null
        foreach ($RequestedKey in $RequestedKeys) {
            if ($CropNameToIndex.ContainsKey($RequestedKey)) {
                $MatchIndex = [int]$CropNameToIndex[$RequestedKey]
                break
            }
        }

        if ($null -eq $MatchIndex) {
            throw "Requested SelectedCropNames entry '$RequestedName' did not match any cropped face in $CropOutDir."
        }
        $NormalizedNameIndices += $MatchIndex
    }

    $CropFiles = @($NormalizedNameIndices | ForEach-Object { $AllCropFiles[$_] })
    Write-Host "Selected crop names: $($ParsedSelectedCropNames -join ', ')"
}
elseif ($ParsedSelectedCropIndices.Count -gt 0) {
    if ($CropIndex -ge 0) {
        throw "Use either -CropIndex or -SelectedCropIndices, not both."
    }

    $NormalizedIndices = @($ParsedSelectedCropIndices)
    foreach ($Index in $NormalizedIndices) {
        if ($Index -lt 0 -or $Index -ge $AllCropFiles.Count) {
            throw "Requested SelectedCropIndices entry '$Index' is out of range for $($AllCropFiles.Count) cropped face(s)."
        }
    }

    $CropFiles = @($NormalizedIndices | ForEach-Object { $AllCropFiles[$_] })
    Write-Host "Selected crop indices: $($NormalizedIndices -join ', ')"
}
elseif ($CropIndex -ge 0) {
    if ($CropIndex -ge $CropFiles.Count) {
        throw "Requested CropIndex $CropIndex but only $($CropFiles.Count) cropped face(s) exist."
    }

    $CropFiles = @($CropFiles[$CropIndex])
}

Write-Host ""
Write-Host "Cropped face count: $($CropFiles.Count)"
Write-Host ""

# Batch mode: 1 crop step + 1 pre-flight/GFPGAN step + 1 batch rephoto step
$TotalSteps = if ($CropOnly) { 1 } else { 3 }
$CurrentStep = 1

Write-Progress -Activity "run_rephoto_with_facecrop" `
    -Status "Crops ready ($CurrentStep of $TotalSteps)" `
    -PercentComplete ([math]::Round(($CurrentStep / $TotalSteps) * 100, 0))

if ($CropOnly) {
    Write-Host "CropOnly requested. Skipping rephoto step."
    Write-Host "Crops are in: $CropOutDir"
    return
}


$CurrentStep++
Write-Progress -Activity "run_rephoto_with_facecrop" `
    -Status "GPU pre-check ($CurrentStep of $TotalSteps)" `
    -PercentComplete ([math]::Round(($CurrentStep / $TotalSteps) * 100, 0))

$GFPGANInputDir = $CropOutDir
$GFPGANSelectedInputDir = Join-Path $GFPGANRunRoot "selected_input"
if ($UseGFPGAN -and $CropFiles.Count -lt $AllCropFiles.Count) {
    New-Item -ItemType Directory -Path $GFPGANSelectedInputDir -Force | Out-Null
    Get-ChildItem -LiteralPath $GFPGANSelectedInputDir -File -ErrorAction SilentlyContinue | Remove-Item -Force
    foreach ($Crop in $CropFiles) {
        Copy-Item -LiteralPath $Crop.FullName -Destination (Join-Path $GFPGANSelectedInputDir $Crop.Name) -Force
    }
    $GFPGANInputDir = $GFPGANSelectedInputDir
}

# Optional GFPGAN enhancement + blend stage.
if ($UseGFPGAN) {
    Write-Host "=== GFPGAN step ==="
    Write-Host "GFPGAN root: $GFPGANRoot"
    Write-Host "GFPGAN version: $GFPGANVersion"
    Write-Host "GFPGAN blend: $GFPGANBlend"
    Write-Host "GFPGAN input: $GFPGANInputDir"
    Write-Host ""

    Get-ChildItem -LiteralPath $GFPGANOutputDir -Recurse -File -ErrorAction SilentlyContinue | Remove-Item -Force
    Get-ChildItem -LiteralPath $GFPGANBlendDir  -File    -ErrorAction SilentlyContinue | Remove-Item -Force

    # Run GFPGAN inference + blend in a single conda run call.
    # The blend is pure CPU Pillow work — running it in the same process avoids
    # a second conda env activation (~5-8s saved).
    $BlendScriptPath = Join-Path $env:TEMP "rephoto_gfpgan_infer_blend.py"

@"
import subprocess, sys, os, glob
from pathlib import Path

# --- Phase 1: GFPGAN inference ---
gfpgan_root = sys.argv[1]
input_dir   = sys.argv[2]
output_dir  = sys.argv[3]
version     = sys.argv[4]
blend_out   = sys.argv[5]
alpha       = float(sys.argv[6])

infer_script = os.path.join(gfpgan_root, "inference_gfpgan.py")
ret = subprocess.run([
    sys.executable, infer_script,
    "-i", input_dir,
    "-o", output_dir,
    "-v", version,
    "-s", "1",
    "--aligned",
    "--bg_upsampler", "none",
    "--suffix", "gfp",
], cwd=gfpgan_root)
if ret.returncode != 0:
    print("GFPGAN inference failed.", file=sys.stderr)
    sys.exit(1)

# --- Phase 2: Blend original crops with enhanced faces ---
# Use NumPy vectorized blend for ~3-5x speedup over PIL.Image.blend
import numpy as np
import cv2 as cv

enh_dir = os.path.join(output_dir, "restored_faces")
os.makedirs(blend_out, exist_ok=True)

patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
orig_files = []
for pattern in patterns:
    orig_files.extend(glob.glob(os.path.join(input_dir, pattern)))

for f in sorted(set(orig_files)):
    base = os.path.splitext(os.path.basename(f))[0]
    matches = sorted(glob.glob(os.path.join(enh_dir, base + "_*_gfp.png")))
    if not matches:
        exact = os.path.join(enh_dir, base + "_gfp.png")
        if os.path.exists(exact):
            matches = [exact]
    if not matches:
        print(f"MISSING {base}")
        continue
    match = matches[0]

    a = cv.imread(f, cv.IMREAD_COLOR)
    b = cv.imread(match, cv.IMREAD_COLOR)
    if a is None or b is None:
        print(f"MISSING {base}")
        continue
    if a.shape[:2] != b.shape[:2]:
        b = cv.resize(b, (a.shape[1], a.shape[0]), interpolation=cv.INTER_LANCZOS4)
    blended = np.clip(a.astype(np.float32) * (1.0 - alpha) + b.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    out = os.path.join(blend_out, base + "_blend.png")
    cv.imwrite(out, blended, [cv.IMWRITE_PNG_COMPRESSION, 3])
    print(f"WROTE {out}")
"@ | Set-Content -LiteralPath $BlendScriptPath

    Push-Location -LiteralPath $GFPGANRoot
    $GFPGANStart = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        conda run -n $GFPGANEnvName python $BlendScriptPath `
            $GFPGANRoot `
            $GFPGANInputDir `
            $GFPGANOutputDir `
            $GFPGANVersion `
            $GFPGANBlendDir `
            $GFPGANBlend

        if ($LASTEXITCODE -ne 0) {
            throw "GFPGAN inference + blend failed."
        }
    }
    finally {
        Pop-Location
        Remove-Item -LiteralPath $BlendScriptPath -Force -ErrorAction SilentlyContinue
    }
    $GFPGANStart.Stop()
    Write-Host "=== GFPGAN elapsed: $([math]::Round($GFPGANStart.Elapsed.TotalSeconds, 1))s ==="

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
    "SelectedCropIndices=$SelectedCropIndices"
    "SelectedCropNames=$SelectedCropNames"
    "StopFlagPath=$StopFlagPath"
    "PauseFlagPath=$PauseFlagPath"
    "CropOnly=$CropOnly"
    "UseExistingCrops=$UseExistingCrops"
    "RequireSelection=$RequireSelection"
    "UseGFPGAN=$UseGFPGAN"
    "GFPGANVersion=$GFPGANVersion"
    "GFPGANEnvName=$GFPGANEnvName"
    "FaceCropEnvName=$FaceCropEnvName"
    "FaceCropCommand=$FaceCropCommand"
    "FaceResizeSize=3072"
    "RephotoEnvName=$RephotoEnvName"
    "EncoderCkptPath=$EncoderCkptPath"
    "PreprocessRoot=$PreprocessRoot"
    "ResultsRoot=$ResultsRoot"
    "GFPGANBlend=$GFPGANBlend"
    "GFPGANInputDir=$GFPGANInputDir"
    "ProjectorInputDir=$ProjectorInputDir"
    "CropCount=$($CropFiles.Count)"
    "RunStamp=$RunStamp"
    "CropOutputDir=$CropOutDir"
    "ResultRoot=$ResultRoot"
    "ProjectorScriptPath=$ProjectorScriptPath"
    "VGG=$VGG"
) | Set-Content -LiteralPath $ManifestPath

Write-Host "Manifest: $ManifestPath"
Write-Host ""

Write-Host "=== Pre-flight checks ==="

# Skip the separate GPU pre-check conda run (saves ~5-8s of env activation + torch import).
# projector.py will fail with a clear CUDA error if the GPU is unavailable.
# We still validate the checkpoint and script exist before starting.
if (-not (Test-Path -LiteralPath $EncoderCkptPath)) {
    throw "Encoder checkpoint not found: $EncoderCkptPath"
}

$BatchScriptPathCheck = Join-Path $RepoRoot "projector_batch.py"
if (-not (Test-Path -LiteralPath $BatchScriptPathCheck)) {
    # Fall back to original projector.py check for error messaging
    if (-not (Test-Path -LiteralPath $ProjectorScriptPath)) {
        throw "Projector script not found: $ProjectorScriptPath"
    }
    throw "Batch projector script not found: $BatchScriptPathCheck"
}

Write-Host "Encoder checkpoint: OK"
Write-Host "Batch projector script: OK"
Write-Host ""

# Build batch manifest JSON for projector_batch.py
$BatchManifestEntries = @()
$CropOrdinal = 0
foreach ($Crop in $CropFiles) {
    $CropOrdinal++
    $CropBase = [System.IO.Path]::GetFileNameWithoutExtension($Crop.Name)
    $ThisResultDir = Join-Path $ResultRoot ("face_{0:D3}_p{1}" -f $CropOrdinal, $Preset)

    $ProjectorImagePath = $Crop.FullName
    if ($UseGFPGAN) {
        $BlendedCandidate = Join-Path $GFPGANBlendDir "$CropBase`_blend.png"
        if (-not (Test-Path -LiteralPath $BlendedCandidate)) {
            throw "Expected GFPGAN blended face not found: $BlendedCandidate"
        }
        $ProjectorImagePath = $BlendedCandidate
    }

    New-Item -ItemType Directory -Path $ThisResultDir -Force | Out-Null
    $BatchManifestEntries += @{ input = $ProjectorImagePath; results_dir = $ThisResultDir }
}

$BatchManifestJson = Join-Path $ResultRoot "batch_manifest.json"
# Force array serialization even for a single face (ConvertTo-Json unwraps 1-element arrays)
ConvertTo-Json -InputObject @($BatchManifestEntries) -Depth 3 | Set-Content -LiteralPath $BatchManifestJson -Encoding UTF8
Write-Host "Batch manifest: $BatchManifestJson ($($BatchManifestEntries.Count) face(s))"
Write-Host ""

# Build optional optimization flags
$ProjectorOptArgs = @()
if ($UseAMP) {
    $ProjectorOptArgs += "--use_amp"
}
if ($EarlyStopPatience -gt 0) {
    $ProjectorOptArgs += @("--early_stop_patience", $EarlyStopPatience)
    $ProjectorOptArgs += @("--early_stop_min_delta", $EarlyStopMinDelta)
}
if ($LRDecay -gt 0) {
    $ProjectorOptArgs += @("--lr_decay", $LRDecay)
}

$ProjectorStopFlagArgs = @()
$ProjectorPauseFlagArgs = @()
if (-not [string]::IsNullOrWhiteSpace($StopFlagPath)) {
    $ProjectorStopFlagArgs = @("--stop_flag", $StopFlagPath)
}
if (-not [string]::IsNullOrWhiteSpace($PauseFlagPath)) {
    $ProjectorPauseFlagArgs = @("--pause_flag", $PauseFlagPath)
}

$BatchScriptPath = Join-Path $RepoRoot "projector_batch.py"

# Run batch projector — loads models once, processes all faces sequentially.
Push-Location -LiteralPath $RepoRoot
try {
    $CurrentStep++
    Write-Progress -Activity "run_rephoto_with_facecrop" `
        -Status "Rephoto batch: $($CropFiles.Count) face(s)" `
        -PercentComplete ([math]::Round(($CurrentStep / $TotalSteps) * 100, 0))

    $RephotoStart = [System.Diagnostics.Stopwatch]::StartNew()

    conda run --no-capture-output -n $RephotoEnvName python -u $BatchScriptPath `
        --manifest $BatchManifestJson `
        --encoder_ckpt $EncoderCkptPath `
        --encoder_size 256 `
        --e4e_ckpt checkpoint/e4e_ffhq_encode.pt `
        --e4e_size 256 `
        --mix_layer_range $MixLayerStart $MixLayerEnd `
        --coarse_min 32 `
        --color_transfer $ColorTransfer `
        --contextual $Contextual `
        --cx_layers relu3_4 relu2_2 relu1_2 `
        --eye $Eye `
        --gaussian $Gaussian `
        --spectral_sensitivity $SpectralSensitivity `
        --recon_size 256 `
        --vgg $VGG `
        --vggface $VGGFace `
        --lr $LR `
        --noise_strength 0.0 `
        --noise_ramp 0.75 `
        --noise_regularize $NoiseRegularize `
        --camera_lr $CameraLR `
        --log_freq 10 `
        --log_visual_freq 1000 `
        --wplus_step $W1 $W2 `
        $ProjectorStopFlagArgs `
        $ProjectorPauseFlagArgs `
        $ProjectorOptArgs `
        $(if ($RecompositeOriginalImage) { "--recomposite_original_image" } else { "" })

    $RephotoStart.Stop()
    Write-Host "=== Batch rephoto elapsed: $([math]::Round($RephotoStart.Elapsed.TotalSeconds, 1))s ==="

    if ($LASTEXITCODE -ne 0) {
        throw "projector_batch.py failed (exit code $LASTEXITCODE)"
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



