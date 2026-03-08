#!/usr/bin/env bash
# Start both Flask (5000) and FastAPI (8000) servers.
# Generate Legal Fiction requires BOTH to be running.
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate

echo "Starting Flask (port 5000) and FastAPI (port 8000)..."
echo "Flask:  http://127.0.0.1:5000/"
echo "FastAPI: http://127.0.0.1:8000/"
echo ""

# Start FastAPI in background
uvicorn app.main:app --host 127.0.0.1 --port 8000 &
FASTAPI_PID=$!

# Give FastAPI a moment to start
sleep 2

# Start Flask in foreground (so we see logs; Ctrl+C stops both)
./run_flask.sh &
FLASK_PID=$!

# Wait for either to exit
wait $FLASK_PID $FASTAPI_PID 2>/dev/null
kill $FLASK_PID $FASTAPI_PID 2>/dev/null
