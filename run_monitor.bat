@echo off
echo ========================================
echo Crypto Price Monitor - Starting...
echo ========================================
echo.

REM Check if virtual environment exists
if not exist .venv (
    echo Virtual environment not found!
    echo Please create one first with: python -m venv .venv
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install requirements if needed
echo Checking dependencies...
pip install -q -r requirements.txt

echo.
echo ========================================
echo Starting Flask Web Monitor...
echo Open your browser to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Run the Flask app
python app.py

pause
