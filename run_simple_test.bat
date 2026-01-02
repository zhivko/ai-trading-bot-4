@echo off
echo ========================================
echo Simple Chart Render Test
echo ========================================
echo.

REM Check if Node.js is installed
where node >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Node.js is not installed!
    echo Please install Node.js from https://nodejs.org/
    echo.
    echo Alternative: Open test_chart_standalone.html in your browser
    pause
    exit /b 1
)

REM Install puppeteer if not installed
if not exist node_modules (
    echo Installing Puppeteer (one-time setup)...
    call npm install puppeteer
    echo.
)

REM Start Flask in background
echo Starting Flask app...
start "Flask App" cmd /c ".venv\Scripts\python.exe app.py"

REM Wait for Flask
echo Waiting for Flask to start...
timeout /t 10 /nobreak >nul

echo.
echo Running quick chart test...
echo.

REM Run simple test
node test_simple.js

REM Save result
set TEST_RESULT=%ERRORLEVEL%

echo.
echo Stopping Flask...
taskkill /FI "WINDOWTITLE eq Flask App*" /F >nul 2>&1

if %TEST_RESULT% EQU 0 (
    echo.
    echo ========================================
    echo Test PASSED - Chart is rendering!
    echo ========================================
    echo.
    echo Check quick_test.png for visual confirmation
) else (
    echo.
    echo ========================================
    echo Test FAILED
    echo ========================================
    echo.
    echo Check quick_test_error.png for details
)

echo.
pause
exit /b %TEST_RESULT%
