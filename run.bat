@echo off
REM Corn Cob Classification Web Server - Startup Script for Windows
REM This script will start the Flask application server

echo.
echo ============================================================
echo Corn Cob Classification System - Startup
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org
    pause
    exit /b 1
)

echo Python is installed
python --version

REM Check if Flask is installed
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Error: Flask is not installed
    echo Installing required dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install dependencies
        pause
        exit /b 1
    )
)

echo ✓ Dependencies verified

echo.
echo ============================================================
echo Starting Flask Web Server...
echo ============================================================
echo.
echo Server will be available at: http://localhost:5000
echo.
echo Initial startup may take 2-5 minutes while training the model...
echo Press Ctrl+C to stop the server
echo.

REM Start the Flask app
python app.py

pause
