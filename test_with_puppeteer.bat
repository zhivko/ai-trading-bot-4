@echo off
echo ========================================
echo Flask Web Monitor - Puppeteer Test
echo ========================================
echo.

REM Check if Node.js is installed
where node >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Node.js is not installed!
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist .venv (
    echo Error: Virtual environment not found!
    echo Please create one first with: python -m venv .venv
    pause
    exit /b 1
)

REM Install Node dependencies if needed
if not exist node_modules (
    echo Installing Node.js dependencies...
    call npm install
    echo.
)

REM Start Flask app in background
echo Starting Flask app...
start "Flask App" cmd /c ".venv\Scripts\python.exe app.py"

REM Wait for Flask to start
echo Waiting for Flask to start (10 seconds)...
timeout /t 10 /nobreak >nul

REM Check if Flask is running
echo Checking if Flask is responding...
curl -s http://localhost:5000 >nul
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Flask might not be responding yet
    echo Waiting additional 5 seconds...
    timeout /t 5 /nobreak >nul
)

echo.
echo ========================================
echo Running Puppeteer tests...
echo ========================================
echo.

REM Run Puppeteer test
node test_charts_puppeteer.js

REM Save test result
set TEST_RESULT=%ERRORLEVEL%

echo.
echo ========================================
echo Stopping Flask app...
echo ========================================

REM Kill Flask process
taskkill /FI "WINDOWTITLE eq Flask App*" /F >nul 2>&1

echo.
if %TEST_RESULT% EQU 0 (
    echo ========================================
    echo Tests PASSED!
    echo ========================================
) else (
    echo ========================================
    echo Tests FAILED!
    echo ========================================
)

echo.
pause
exit /b %TEST_RESULT%
