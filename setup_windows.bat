@echo off
echo ==============================================
echo Video Annotator Pro - Windows Setup
echo ==============================================

:: Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not added to your system PATH. 
    echo Please download Python 3.9 or newer from https://www.python.org/downloads/
    echo WARNING: Make sure to check the box "Add Python to PATH" during installation!
    pause
    exit /b
)

echo Creating virtual environment (.venv)...
python -m venv .venv
IF %ERRORLEVEL% NEQ 0 (
    echo Failed to create virtual environment. 
    echo Please ensure you have permission to write in this folder.
    pause
    exit /b
)

echo.
echo Activating virtual environment and installing dependencies...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    echo Failed to install some dependencies. Please check the error messages above.
    pause
    exit /b
)

echo.
echo ==============================================
echo Setup Complete! 
echo You can now close this window and double-click "run_windows.bat" to start the application.
echo ==============================================
pause
