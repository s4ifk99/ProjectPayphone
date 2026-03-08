#!/usr/bin/env bash
# Run Flask (case browser) with Generate button pointing to a REMOTE FastAPI backend.
# Use this on your weak computer when the LLM runs on a different (powerful) machine.
#
# Usage:
#   GENERATE_BACKEND_URL=http://192.168.1.100:8000 ./run_flask_remote.sh
#   GENERATE_BACKEND_URL=https://abc123.ngrok-free.app ./run_flask_remote.sh
#   ./run_flask_remote.sh   # uses default from env or 192.168.1.100:8000
#
# See REMOTE_GENERATION.md for full setup instructions.
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
export GENERATE_BACKEND_URL="${GENERATE_BACKEND_URL:-http://192.168.1.100:8000}"
DB="${1:-oldbailey.sqlite}"
echo "Using DB: $DB"
echo "Generate backend: $GENERATE_BACKEND_URL"
echo "Templates from: $(pwd)/src/oldbailey/web/templates/"
echo "Open http://127.0.0.1:5000/"
python3 -m oldbailey.cli serve --db "$DB" --backend-url "$GENERATE_BACKEND_URL"
