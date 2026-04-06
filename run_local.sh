#!/usr/bin/env bash
# Run the full stack locally (Flask + FastAPI + requires Ollama).
# Checks that Ollama is reachable before starting, then runs run_both.sh.
#
# Usage:
#   ./run_local.sh
#
# Prerequisites:
#   - ollama serve (in another terminal)
#   - A model in Ollama (e.g. smollm2:360m, or fine-tuned payphone-story)
#
# Payphone GGUF: put payphone-story.gguf in this directory (or symlink from ~/Downloads),
# then: ollama create payphone-story -f Modelfile
# Run app with: OLLAMA_MODEL=payphone-story ./run_local.sh  (or set in .env)
#
# See README.md for full single-machine setup.
cd "$(dirname "$0")"

GGUF="$(pwd)/payphone-story.gguf"
if [[ ! -e "$GGUF" ]]; then
  echo "Note: $GGUF not found. For the fine-tuned model, symlink or copy your GGUF:"
  echo "  ln -sf \"\$HOME/Downloads/payphone-story.gguf\" \"$GGUF\""
  echo "Then: ollama create payphone-story -f Modelfile"
  echo ""
fi

OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
if ! curl -s -f --connect-timeout 2 "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
  echo "Ollama is not reachable at ${OLLAMA_URL}"
  echo "Start Ollama first: ollama serve"
  echo "Then: ollama pull smollm2:360m  (or create payphone-story from Modelfile)"
  exit 1
fi

echo "Ollama OK. Starting Flask + FastAPI..."
exec ./run_both.sh
