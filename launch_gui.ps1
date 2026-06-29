$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$guiApp = Join-Path $repoRoot "gui\app.py"

if (-not (Test-Path $guiApp)) {
    throw "GUI app not found: $guiApp"
}

conda run -n rephoto_gui python $guiApp
