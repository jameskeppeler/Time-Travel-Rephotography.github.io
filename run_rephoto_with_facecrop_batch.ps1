param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [string[]]$Presets = @("test", "1500", "3000", "6000"),

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

    [double]$GFPGANBlend = 0.35,

    [switch]$Recurse
)

$ErrorActionPreference = "Stop"

function Test-ValidPreset {
    param([string]$Value)

    if ($Value -eq "test") {
        return
    }

    $n = 0
    if (-not [int]::TryParse($Value, [ref]$n)) {
        throw "Preset '$Value' is invalid. Use 'test', '1500', or a multiple of 1000 from 1000 through 100000."
    }

    if ($n -eq 1500) {
        return
    }

    if ($n -ge 1000 -and $n -le 100000 -and ($n % 1000 -eq 0)) {
        return
    }

    throw "Preset '$Value' is invalid. Use 'test', '1500', or a multiple of 1000 from 1000 through 100000."
}

function Get-ProjectedRuntimeMinutes {
    param([int]$Iterations)

    switch ($Iterations) {
        750  { return 10 }
        1500 { return 20 }
        3000 { return 38 }
        6000 { return 136 }
    }

    if ($Iterations -lt 6000) {
        return [math]::Round((38.0 / 3000.0) * $Iterations, 0)
    }

    $Exponent = [math]::Log(467.0 / 136.0) / [math]::Log(18000.0 / 6000.0)
    $K = 136.0 / [math]::Pow(6000.0, $Exponent)

    return [math]::Round($K * [math]::Pow($Iterations, $Exponent), 0)
}

function Format-Minutes {
    param([double]$Minutes)

    $Hours = [math]::Floor($Minutes / 60)
    $Remain = [math]::Round($Minutes - ($Hours * 60), 0)

    if ($Remain -eq 60) {
        $Hours++
        $Remain = 0
    }

    if ($Hours -gt 0) {
        return "{0}h {1}m" -f $Hours, $Remain
    }

    return "{0}m" -f $Remain
}

# Validate presets first.
foreach ($Preset in $Presets) {
    Test-ValidPreset -Value $Preset
}

$RepoRoot = $PSScriptRoot
$WrapperPath = Join-Path $RepoRoot "run_rephoto_with_facecrop.ps1"

if (-not (Test-Path -LiteralPath $WrapperPath)) {
    throw "Wrapper not found: $WrapperPath"
}

$ResolvedInputPath = (Resolve-Path -LiteralPath $InputPath).Path

# Build image list.
$AllowedExt = @(".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")

if (Test-Path -LiteralPath $ResolvedInputPath -PathType Leaf) {
    $ImageFiles = @(
        Get-Item -LiteralPath $ResolvedInputPath
    )
}
elseif (Test-Path -LiteralPath $ResolvedInputPath -PathType Container) {
    if ($Recurse) {
        $ImageFiles = @(
            Get-ChildItem -LiteralPath $ResolvedInputPath -File -Recurse |
            Where-Object { $AllowedExt -contains $_.Extension.ToLower() } |
            Sort-Object FullName
        )
    }
    else {
        $ImageFiles = @(
            Get-ChildItem -LiteralPath $ResolvedInputPath -File |
            Where-Object { $AllowedExt -contains $_.Extension.ToLower() } |
            Sort-Object FullName
        )
    }
}
else {
    throw "InputPath not found: $InputPath"
}

if ($ImageFiles.Count -eq 0) {
    throw "No supported image files found at: $ResolvedInputPath"
}

$TotalJobs = $ImageFiles.Count * $Presets.Count
$JobIndex = 0

Write-Host ""
Write-Host "=== Batch run start ==="
Write-Host "Input path: $ResolvedInputPath"
Write-Host "Image count: $($ImageFiles.Count)"
Write-Host "Preset count: $($Presets.Count)"
Write-Host "Total runs: $TotalJobs"
Write-Host ""

foreach ($Image in $ImageFiles) {
    foreach ($Preset in $Presets) {
        $JobIndex++

        $PresetIterations = if ($Preset -eq "test") { 750 } else { [int]$Preset }
        $EstimatedMinutes = Get-ProjectedRuntimeMinutes -Iterations $PresetIterations
        $EstimatedText = Format-Minutes -Minutes $EstimatedMinutes

        Write-Progress -Activity "run_rephoto_with_facecrop_batch" `
            -Status ("Run {0} of {1}: {2} -> {3}" -f $JobIndex, $TotalJobs, $Image.Name, $Preset) `
            -PercentComplete ([math]::Round(($JobIndex / $TotalJobs) * 100, 0))

        Write-Host "=================================================="
        Write-Host ("Run {0} of {1}" -f $JobIndex, $TotalJobs)
        Write-Host ("Image:   {0}" -f $Image.FullName)
        Write-Host ("Preset:  {0}" -f $Preset)
        Write-Host ("Est. RTX 3060 laptop runtime: {0}" -f $EstimatedText)
        Write-Host "=================================================="
        Write-Host ""

        $RunArgs = @{
            InputImage     = $Image.FullName
            Preset         = $Preset
            Strategy       = $Strategy
            FaceFactor     = $FaceFactor
            DetThreshold   = $DetThreshold
            CropIndex      = $CropIndex
            GFPGANVersion  = $GFPGANVersion
            GFPGANEnvName  = $GFPGANEnvName
            GFPGANBlend    = $GFPGANBlend
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

        try {
            & $WrapperPath @RunArgs
        }
        catch {
            throw "Batch failed on image '$($Image.FullName)' with preset '$Preset'. Inner error: $($_.Exception.Message)"
        }

        Write-Host ""
    }
}

Write-Progress -Activity "run_rephoto_with_facecrop_batch" -Completed

Write-Host "=== Batch run complete ==="
Write-Host ""