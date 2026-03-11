#!/usr/bin/env bash
# Ingest Old Bailey XML into old_bailey.db for FastAPI (case list, generate, stories).
# Can run while servers are up — uses a different DB than Flask.
#
# Usage:
#   ./scripts/ingest_fastapi_data.sh
cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate

echo "Ingesting Old Bailey XML into old_bailey.db (FastAPI)..."
echo ""

python ingest_old_bailey.py \
  --xml-dir data/oldbailey_xml/sessionsPapers \
  --db old_bailey.db

echo ""
echo "Done. Generate Legal Fiction will now find cases."
