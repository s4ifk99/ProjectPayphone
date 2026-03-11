#!/usr/bin/env bash
# Ingest Old Bailey XML into oldbailey.sqlite for the Flask case browser.
#
# IMPORTANT: Stop the Flask/FastAPI servers (Ctrl+C on run_local.sh) before running.
# The ingest holds a lock on the database; running while servers are up will fail.
#
# Usage:
#   ./scripts/ingest_flask_data.sh
#
# Takes a few minutes (~60k XML files). Then restart: ./run_local.sh
cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

echo "Ingesting Old Bailey XML into oldbailey.sqlite..."
echo "Stop run_local.sh first if it is running."
echo ""

python3 -m oldbailey.cli ingest \
  --obo-xml data/oldbailey_xml/sessionsPapers \
  --db oldbailey.sqlite

echo ""
echo "Done. Restart with: ./run_local.sh"
