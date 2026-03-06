# Corn Cob Classification System - PowerShell Startup Script
# Run this in PowerShell: .\run.ps1

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Corn Cob Classification System - Startup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python installed: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "  Please install Python 3.8+ from https://www.python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Get the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = $scriptDir

Write-Host "Project directory: $projectDir" -ForegroundColor Gray
Write-Host "Data directory: $projectDir\20250319" -ForegroundColor Gray

# Check data directory
if (-not (Test-Path "$projectDir\20250319")) {
    Write-Host ""
    Write-Host "⚠ Warning: 20250319 directory not found!" -ForegroundColor Yellow
    Write-Host "  Please ensure your radar data CSV files are in the 20250319 folder." -ForegroundColor Yellow
}

# Check if virtual environment exists
$venvPath = "$projectDir\venv"
if (Test-Path $venvPath) {
    Write-Host "✓ Virtual environment found" -ForegroundColor Green
    Write-Host "  Activating virtual environment..." -ForegroundColor Gray
    & "$venvPath\Scripts\Activate.ps1"
}

# Check dependencies
Write-Host ""
Write-Host "Checking dependencies..." -ForegroundColor Cyan

$dependencies = @('flask', 'tensorflow', 'pandas', 'numpy', 'sklearn')
$allInstalled = $true

foreach ($package in $dependencies) {
    try {
        python -c "import $package" 2>$null
        Write-Host "✓ $package installed" -ForegroundColor Green
    } catch {
        Write-Host "✗ $package not found" -ForegroundColor Red
        $allInstalled = $false
    }
}

if (-not $allInstalled) {
    Write-Host ""
    Write-Host "Installing missing dependencies..." -ForegroundColor Yellow
    Write-Host "Running: pip install -r requirements.txt" -ForegroundColor Gray
    Write-Host ""
    
    pip install -r "$projectDir\requirements.txt"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "✗ Failed to install dependencies" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Starting Flask Web Server..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Server will be available at: http://localhost:5000" -ForegroundColor Yellow
Write-Host ""
Write-Host "Initial startup may take 2-5 minutes while training the model..." -ForegroundColor Gray
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

# Start the Flask app
Set-Location $projectDir
python app.py

# If we get here, the app was stopped
Write-Host ""
Write-Host "Flask server stopped" -ForegroundColor Yellow
