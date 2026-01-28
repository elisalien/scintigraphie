@echo off
echo ========================================
echo  SCINTIGRAPHY BLOB TRACKER
echo  Medical Scifi Performance Tool
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [!] Virtual environment not found!
    echo [*] Creating virtual environment...
    python -m venv venv
    echo [*] Installing dependencies...
    call venv\Scripts\activate.bat
    pip install --upgrade pip
    pip install -r requirements.txt
    echo [+] Setup complete!
    echo.
) else (
    call venv\Scripts\activate.bat
)

echo [*] Starting Scintigraphy Tracker...
echo.
python scintigraphy_tracker.py

echo.
echo [*] Tracker closed.
pause
