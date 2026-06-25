@echo off
echo Starting Video Annotator Pro...

echo Checking for updates...
git pull || echo Warning: Could not pull latest updates from GitHub. Proceeding anyway.

IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo Virtual environment not found!
    echo Please double-click "setup_windows.bat" to run the setup first.
    pause
    exit /b
)

call .venv\Scripts\activate.bat

:: Start the server in a new window that stays open if it crashes
echo Server is starting up...
start "Video Annotator Pro Server" cmd /k "python main.py"

echo Opening browser in 2 seconds...
timeout /t 2 /nobreak >nul

:: Open browser to the localhost address
start http://127.0.0.1:8000

echo The application is running. 
echo Keep the new "Video Annotator Pro Server" window open!
pause
