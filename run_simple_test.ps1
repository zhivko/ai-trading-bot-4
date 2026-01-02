# Simple Chart Render Test - PowerShell Version
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Simple Chart Render Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Node.js is installed
try {
    $nodeVersion = node --version 2>$null
    if (-not $nodeVersion) {
        throw "Node.js not found"
    }
    Write-Host "Node.js detected: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Node.js is not installed!" -ForegroundColor Red
    Write-Host "Please install Node.js from https://nodejs.org/" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Alternative: Open test_chart_standalone.html in your browser" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Install puppeteer if not installed
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing Puppeteer (one-time setup)..." -ForegroundColor Yellow
    npm install puppeteer
    Write-Host ""
}

# Start Flask in background
Write-Host "Starting Flask app..." -ForegroundColor Cyan
$flaskProcess = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "app.py" -PassThru -WindowStyle Hidden

# Wait for Flask to start
Write-Host "Waiting for Flask to start..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 0
$flaskReady = $false

while ($attempt -lt $maxAttempts -and -not $flaskReady) {
    $attempt++
    Start-Sleep -Seconds 1
    Write-Host "." -NoNewline

    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5000" -TimeoutSec 2 -UseBasicParsing -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $flaskReady = $true
            Write-Host ""
            Write-Host "Flask is ready! (took $attempt seconds)" -ForegroundColor Green
        }
    } catch {
        # Keep waiting
    }
}

if (-not $flaskReady) {
    Write-Host ""
    Write-Host "Warning: Flask might not be fully ready" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Running quick chart test..." -ForegroundColor Cyan
Write-Host ""

# Run simple test
$testResult = node test_simple.js
$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host "Stopping Flask..." -ForegroundColor Cyan

# Stop Flask process
if ($flaskProcess) {
    Stop-Process -Id $flaskProcess.Id -Force -ErrorAction SilentlyContinue
}

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Test PASSED - Chart is rendering!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Check quick_test.png for visual confirmation" -ForegroundColor Cyan
} else {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Test FAILED" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Check quick_test_error.png for details" -ForegroundColor Yellow
}

Write-Host ""
Read-Host "Press Enter to exit"
exit $exitCode
