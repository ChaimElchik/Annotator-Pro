@echo off
echo Starting Video Annotator Pro...

IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo Virtual environment not found!
    echo Please double-click "setup_windows.bat" to run the setup first.
    pause
    exit /b
)

call .venv\Scripts\activate.bat

:: Start the server in the background and capture output
echo Server is starting up...
start "Video Annotator Pro Server" python main.py

echo Opening browser in 2 seconds...
timeout /t 2 /nobreak >nul

:: Open browser to the localhost address
start http://localhost:8000

echo The application is running. 
echo Keep the new "Video Annotator Pro Server" window open!
pause
