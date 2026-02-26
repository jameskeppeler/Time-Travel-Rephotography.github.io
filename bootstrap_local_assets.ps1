param(
    [Parameter(Mandatory = $true)]
    [string]$SourceRepo
)

$ErrorActionPreference = "Stop"

Write-Host "Using source repo: $SourceRepo"

New-Item -ItemType Directory -Force ".\dataset" | Out-Null
Copy-Item "$SourceRepo\dataset\Abraham Lincoln_01.png" ".\dataset\" -Force

New-Item -ItemType Directory -Force ".\checkpoint" | Out-Null
Copy-Item "$SourceRepo\checkpoint\*" ".\checkpoint\" -Recurse -Force

New-Item -ItemType Directory -Force ".\third_party\face_parsing\res\cp" | Out-Null
Copy-Item "$SourceRepo\third_party\face_parsing\res\cp\79999_iter.pth" ".\third_party\face_parsing\res\cp\" -Force

Write-Host "Local assets copied successfully."
