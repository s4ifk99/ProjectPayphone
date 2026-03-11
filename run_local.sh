#!/usr/bin/env bash
# Run the full stack locally (Flask + FastAPI + requires Ollama).
# Checks that Ollama is reachable before starting, then runs run_both.sh.
#
# Usage:
#   ./run_local.sh
#
# Prerequisites:
#   - ollama serve (in another terminal)
#   - ollama pull smollm2:360m  (or your model)
#
# See README.md for full single-machine setup.
cd "$(dirname "$0")"

OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
if ! curl -s -f --connect-timeout 2 "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
  echo "Ollama is not reachable at ${OLLAMA_URL}"
  echo "Start Ollama first: ollama serve"
  echo "Then: ollama pull smollm2:360m"
  exit 1
fi

echo "Ollama OK. Starting Flask + FastAPI..."
exec ./run_both.sh
