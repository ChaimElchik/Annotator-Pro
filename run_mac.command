#!/bin/bash
cd "$(dirname "$0")"

echo "Starting Video Annotator Pro..."

if [ ! -f ".venv/bin/activate" ]; then
    echo "Virtual environment not found!"
    echo "Please double-click 'setup_mac.command' to run the setup first."
    echo "Press any key to exit..."
    read -n 1
    exit 1
fi

source .venv/bin/activate

# Start the server and record its PID
python3 main.py &
SERVER_PID=$!

echo "Server is starting up (PID: $SERVER_PID)."
echo "Opening browser in 2 seconds..."
sleep 2

# Attempt to open the browser
if command -v open > /dev/null; then
    open "http://127.0.0.1:8000"
elif command -v xdg-open > /dev/null; then
    xdg-open "http://127.0.0.1:8000"
else
    echo "Could not detect web browser command. Please open a browser manually and go to http://127.0.0.1:8000"
fi

echo "The application is running. Keep this window open."
echo "Press Ctrl+C to stop the server."

# Wait for the server process to finish so the wrapper script doesn't exit immediately
wait $SERVER_PID
