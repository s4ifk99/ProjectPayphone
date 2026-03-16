#!/usr/bin/env bash
# Convert merged HuggingFace model to GGUF for Ollama.
#
# Prereq: Clone llama.cpp and install its deps:
#   git clone https://github.com/ggerganov/llama.cpp
#   cd llama.cpp && pip install -r requirements.txt
#
# Usage:
#   ./scripts/convert_to_gguf.sh [merged_dir] [out_gguf]
#
# Default: merged_payphone -> payphone-story.gguf (q8_0)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MERGED="${1:-$PROJECT_ROOT/merged_payphone}"
OUTPUT="${2:-$PROJECT_ROOT/payphone-story.gguf}"
LLAMA_CPP="${LLAMA_CPP_PATH:-$PROJECT_ROOT/llama.cpp}"

if [ ! -d "$MERGED" ]; then
  echo "Error: Merged model not found: $MERGED"
  echo "Run: python scripts/merge_payphone_lora.py --output merged_payphone"
  exit 1
fi

if [ ! -f "$LLAMA_CPP/convert_hf_to_gguf.py" ]; then
  echo "Error: llama.cpp not found at $LLAMA_CPP"
  echo "Clone it: git clone https://github.com/ggerganov/llama.cpp"
  echo "Or set LLAMA_CPP_PATH=/path/to/llama.cpp"
  exit 1
fi

echo "Converting $MERGED -> $OUTPUT (q8_0)"
python "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGED" \
  --outfile "$OUTPUT" \
  --outtype q8_0

echo "Done. Create Ollama model: ollama create payphone-story -f Modelfile"
