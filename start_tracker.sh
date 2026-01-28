#!/bin/bash

echo "========================================"
echo " SCINTIGRAPHY BLOB TRACKER"
echo " Medical Scifi Performance Tool"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[!] Virtual environment not found!"
    echo "[*] Creating virtual environment..."
    python3 -m venv venv
    echo "[*] Installing dependencies..."
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "[+] Setup complete!"
    echo ""
else
    source venv/bin/activate
fi

echo "[*] Starting Scintigraphy Tracker..."
echo ""
python scintigraphy_tracker.py

echo ""
echo "[*] Tracker closed."
