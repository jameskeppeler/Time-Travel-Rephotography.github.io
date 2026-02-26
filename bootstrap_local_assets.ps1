param(
    [string]$SourceRepo = "C:\Users\james\Projects\Time-Travel-Rephotography.github.io"
)

$ErrorActionPreference = "Stop"

Write-Host "Using source repo: $SourceRepo"

# 1) Dataset image used for the verified test
New-Item -ItemType Directory -Force ".\dataset" | Out-Null
Copy-Item "$SourceRepo\dataset\Abraham Lincoln_01.png" ".\dataset\" -Force

# 2) Main checkpoint tree
New-Item -ItemType Directory -Force ".\checkpoint" | Out-Null
Copy-Item "$SourceRepo\checkpoint\*" ".\checkpoint\" -Recurse -Force

# 3) Face parsing checkpoint required by third_party/face_parsing/test.py
New-Item -ItemType Directory -Force ".\third_party\face_parsing\res\cp" | Out-Null
Copy-Item "$SourceRepo\third_party\face_parsing\res\cp\79999_iter.pth" ".\third_party\face_parsing\res\cp\" -Force

Write-Host "Local assets copied successfully."
