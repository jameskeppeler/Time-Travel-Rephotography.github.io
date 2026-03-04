param(
    # Optional now. If provided and -Mode is auto/copy, assets are copied from here.
    [string]$SourceRepo,

    [ValidateSet("auto", "copy", "download")]
    [string]$Mode = "auto",

    # "hf" uses a community mirror on Hugging Face (often avoids Google Drive quota).
    # "gdrive" uses gdown with the official Google Drive IDs.
    [ValidateSet("hf", "gdrive")]
    [string]$DownloadProvider = "hf",

    # Overwrite existing files
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }
}

function Download-File([string]$Url, [string]$DestPath) {
    Ensure-Dir (Split-Path -Parent $DestPath)

    if (-not $Force -and (Test-Path -LiteralPath $DestPath)) {
        Write-Host "Exists, skipping: $DestPath"
        return
    }

    Write-Host "Downloading:"
    Write-Host "  $Url"
    Write-Host "  -> $DestPath"

    # Prefer BITS on Windows (more resilient for large files)
    if (Get-Command Start-BitsTransfer -ErrorAction SilentlyContinue) {
        Start-BitsTransfer -Source $Url -Destination $DestPath -ErrorAction Stop
    }
    else {
        Invoke-WebRequest -Uri $Url -OutFile $DestPath -UseBasicParsing
    }
}

function Ensure-GDown() {
    # Try: python -m gdown
    $ok = $false
    try {
        & python -m gdown --help *> $null
        $ok = $true
    }
    catch { $ok = $false }

    if ($ok) { return }

    Write-Host "gdown not available; attempting to install it with: python -m pip install --user gdown"
    & python -m pip install --user gdown
}

function GDown-ById([string]$DriveId, [string]$DestPath) {
    Ensure-Dir (Split-Path -Parent $DestPath)

    if (-not $Force -and (Test-Path -LiteralPath $DestPath)) {
        Write-Host "Exists, skipping: $DestPath"
        return
    }

    Ensure-GDown
    $url = "https://drive.google.com/uc?id=$DriveId"

    Write-Host "Downloading (gdown):"
    Write-Host "  $url"
    Write-Host "  -> $DestPath"

    & python -m gdown $url -O $DestPath
    if ($LASTEXITCODE -ne 0) {
        throw "gdown failed for id=$DriveId"
    }
}

function Copy-FromSource([string]$SrcRoot) {
    if ([string]::IsNullOrWhiteSpace($SrcRoot)) {
        throw "Copy mode requires -SourceRepo."
    }
    if (-not (Test-Path -LiteralPath $SrcRoot)) {
        throw "SourceRepo not found: $SrcRoot"
    }

    Write-Host "Copying local assets from: $SrcRoot"

    Ensure-Dir (Join-Path $RepoRoot "dataset")
    Copy-Item -LiteralPath (Join-Path $SrcRoot "dataset\Abraham Lincoln_01.png") `
              -Destination (Join-Path $RepoRoot "dataset") -Force

    Ensure-Dir (Join-Path $RepoRoot "checkpoint")
    Copy-Item -LiteralPath (Join-Path $SrcRoot "checkpoint\*") `
              -Destination (Join-Path $RepoRoot "checkpoint") -Recurse -Force

    Ensure-Dir (Join-Path $RepoRoot "third_party\face_parsing\res\cp")
    Copy-Item -LiteralPath (Join-Path $SrcRoot "third_party\face_parsing\res\cp\79999_iter.pth") `
              -Destination (Join-Path $RepoRoot "third_party\face_parsing\res\cp") -Force

    Write-Host "Local assets copied successfully."
}

function Download-Assets([string]$Provider) {
    # Test image (small) from upstream repo
    $testImgUrl  = "https://raw.githubusercontent.com/Time-Travel-Rephotography/Time-Travel-Rephotography.github.io/main/dataset/Abraham%20Lincoln_01.png"
    $testImgDest = Join-Path $RepoRoot "dataset\Abraham Lincoln_01.png"
    Download-File $testImgUrl $testImgDest

    # Required dirs
    Ensure-Dir (Join-Path $RepoRoot "checkpoint")
    Ensure-Dir (Join-Path $RepoRoot "checkpoint\encoder")
    Ensure-Dir (Join-Path $RepoRoot "third_party\face_parsing\res\cp")

    if ($Provider -eq "hf") {
        # Community mirror containing all needed weights (names differ; we map them into expected paths)
        $hfBase = "https://huggingface.co/trysem/Time-Travel-Rephotograph/resolve/main"

        Download-File "$hfBase/e4e_ffhq_encode.pt"              (Join-Path $RepoRoot "checkpoint\e4e_ffhq_encode.pt")
        Download-File "$hfBase/stylegan2-ffhq-config-f.pt"      (Join-Path $RepoRoot "checkpoint\stylegan2-ffhq-config-f.pt")
        Download-File "$hfBase/vgg_face_dag.pt"                 (Join-Path $RepoRoot "checkpoint\vgg_face_dag.pt")

        Download-File "$hfBase/encoder_checkpoint_b.pt"         (Join-Path $RepoRoot "checkpoint\encoder\checkpoint_b.pt")
        Download-File "$hfBase/encoder_checkpoint_g.pt"         (Join-Path $RepoRoot "checkpoint\encoder\checkpoint_g.pt")
        Download-File "$hfBase/encoder_checkpoint_gb.pt"        (Join-Path $RepoRoot "checkpoint\encoder\checkpoint_gb.pt")

        Download-File "$hfBase/79999_iter.pth"                  (Join-Path $RepoRoot "third_party\face_parsing\res\cp\79999_iter.pth")
    }
    else {
        # Official Google Drive IDs from upstream scripts/download_checkpoints.sh
        GDown-ById "1hWc2JLM58_PkwfLG23Q5IH3Ysj2Mo1nr" (Join-Path $RepoRoot "checkpoint\e4e_ffhq_encode.pt")
        GDown-ById "1hvAAql9Jo0wlmLBSHRIGrtXHcKQE-Whn" (Join-Path $RepoRoot "checkpoint\stylegan2-ffhq-config-f.pt")
        GDown-ById "1mbGWbjivZxMGxZqyyOHbE310aOkYe2BR" (Join-Path $RepoRoot "checkpoint\vgg_face_dag.pt")

        GDown-ById "1ha4WXsaIpZfMHsqNLvqOPlUXsgh9VawU" (Join-Path $RepoRoot "checkpoint\encoder\checkpoint_b.pt")
        GDown-ById "1hfxDLujRIGU0G7pOdW9MMSBRzxZBmSKJ" (Join-Path $RepoRoot "checkpoint\encoder\checkpoint_g.pt")
        GDown-ById "1htekHopgxaW-MIjs6pYy7pyIK0v7Q0iS" (Join-Path $RepoRoot "checkpoint\encoder\checkpoint_gb.pt")

        # Face-parsing pretrained model (Drive ID from face-parsing.PyTorch README)
        GDown-ById "154JgKpzCPW82qINcVieuPH3fZ2e0P812" (Join-Path $RepoRoot "third_party\face_parsing\res\cp\79999_iter.pth")
    }

    Write-Host "Download complete."
}

# Decide mode
switch ($Mode) {
    "copy" {
        Copy-FromSource $SourceRepo
        return
    }
    "download" {
        Download-Assets $DownloadProvider
        return
    }
    "auto" {
        if (-not [string]::IsNullOrWhiteSpace($SourceRepo)) {
            Copy-FromSource $SourceRepo
        }
        else {
            Download-Assets $DownloadProvider
        }
        return
    }
}