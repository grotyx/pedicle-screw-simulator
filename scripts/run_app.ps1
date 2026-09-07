param([switch]$Check, [switch]$Test, [switch]$WithTotalseg)
# Usage: .\scripts\run_app.ps1 [--check] [--test] [--with-totalseg]   (also accepts the PowerShell switch forms)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$venv = Join-Path $root "venv"
$python = Join-Path $venv "Scripts\python.exe"
foreach ($arg in $args) { if ($arg -eq "--check") { $Check = $true }; if ($arg -eq "--test") { $Test = $true }; if ($arg -eq "--with-totalseg") { $WithTotalseg = $true } }
if (-not (Test-Path $python)) {
    Write-Host "Creating virtual environment..."
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) { py -3.12 -m venv $venv } else { python -m venv $venv }
    if ($LASTEXITCODE -ne 0) { Write-Error "Virtual environment creation failed"; exit $LASTEXITCODE }
}
& $python -m pip install --quiet --upgrade pip
if ($LASTEXITCODE -ne 0) { Write-Error "pip upgrade failed"; exit $LASTEXITCODE }
& $python -m pip install --quiet -r (Join-Path $root "requirements.txt")
if ($LASTEXITCODE -ne 0) { Write-Error "requirements.txt install failed"; exit $LASTEXITCODE }
if ($Test -or $Check) {
    & $python -m pip install --quiet -r (Join-Path $root "requirements-dev.txt")
    if ($LASTEXITCODE -ne 0) { Write-Error "requirements-dev.txt install failed"; exit $LASTEXITCODE }
}
if ($WithTotalseg) {
    & $python -m pip install --quiet "TotalSegmentator==2.12.0"
    if ($LASTEXITCODE -ne 0) { Write-Error "TotalSegmentator install failed"; exit $LASTEXITCODE }
}
if ($Check) { & $python -c "import vtk, PyQt6, SimpleITK, pydicom, numpy, scipy; print('Dependencies OK')"; exit $LASTEXITCODE }
if ($Test) { $env:QT_QPA_PLATFORM = "offscreen"; & $python -m pytest (Join-Path $root "tests") -q; exit $LASTEXITCODE }
& $python (Join-Path $root "main.py")
