#!/usr/bin/env bash
# Run Flask app using LOCAL source (not installed package).
# This ensures template changes in src/oldbailey/web/templates/ are used.
#
# Usage:
#   ./run_flask.sh [db_path]
#   ./run_flask.sh [db_path] [backend_url]
#   GENERATE_BACKEND_URL=http://host:8000 ./run_flask.sh [db_path]
#
# For remote generation, see REMOTE_GENERATION.md or use run_flask_remote.sh.
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
DB="${1:-oldbailey.sqlite}"
BACKEND_URL="${2:-$GENERATE_BACKEND_URL}"
echo "Using DB: $DB"
echo "Templates from: $(pwd)/src/oldbailey/web/templates/"
[ -n "$BACKEND_URL" ] && echo "Generate backend: $BACKEND_URL"
echo "Open http://127.0.0.1:5000/"
if [ -n "$BACKEND_URL" ]; then
  python3 -m oldbailey.cli serve --db "$DB" --backend-url "$BACKEND_URL"
else
  python3 -m oldbailey.cli serve --db "$DB"
fi
