#!/usr/bin/env bash
# Ingest both databases: old_bailey.db (FastAPI) and oldbailey.sqlite (Flask).
#
# IMPORTANT: Stop run_local.sh first — Flask locks oldbailey.sqlite.
#
# Usage:
#   ./scripts/ingest_all.sh
cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate

echo "=== Ingest all Old Bailey data ==="
echo "Stop run_local.sh first if it is running."
echo ""

echo "1. Ingest into old_bailey.db (FastAPI + Generate Legal Fiction)"
python ingest_old_bailey.py \
  --xml-dir data/oldbailey_xml/sessionsPapers \
  --db old_bailey.db
echo ""

echo "2. Ingest into oldbailey.sqlite (Flask case browser)"
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
python3 -m oldbailey.cli ingest \
  --obo-xml data/oldbailey_xml/sessionsPapers \
  --db oldbailey.sqlite
echo ""

echo "Done. Restart with: ./run_local.sh"
