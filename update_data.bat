@echo off
echo ========================================
echo Crypto Price Monitor - Manual Data Update
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

echo.
echo Starting data update...
echo This will fetch latest data from Binance for all CSV files in data/ directory
echo.

REM Run the data updater
python data_updater.py

echo.
echo ========================================
echo Update complete!
echo ========================================
echo.

pause
