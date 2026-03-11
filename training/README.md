# Project Payphone – LoRA Fine-tuning

Fine-tune Mistral-7B or Qwen2.5-7B for Old Bailey legal fiction generation.

## Prerequisites

- Python 3.10+
- GPU with ~16GB VRAM (e.g. RTX 4080, A100)
- CUDA for PyTorch

## Setup

```bash
cd training
pip install -r requirements.txt
```

## Prepare Training Data

First, generate stories using the app (or ensure `old_bailey.db` has case-story pairs). Then export:

```bash
python ../scripts/export_training_pairs.py --output ../data/training/case_story_pairs.jsonl
```

Optional filters:
- `--target-length 400-600` – only 400–600 word stories
- `--format instruction` – HuggingFace-style instruction/input/output
- `--limit 1000` – cap number of pairs

## Train

```bash
python train.py --data ../data/training/case_story_pairs.jsonl --output output/payphone-storyteller-lora
```

Options:
- `--model Qwen/Qwen2.5-7B-Instruct` – use Qwen instead of Mistral
- `--epochs 5` – more epochs
- `--batch-size 1` – reduce if OOM

## Use the Fine-tuned Model

After training, the adapter is saved to `output/payphone-storyteller-lora`. To use with Ollama:

1. Merge the LoRA adapter with the base model (use `merge_and_unload()` or export to GGUF)
2. Create an Ollama Modelfile or convert to GGUF and import
3. Set `OLLAMA_MODEL=payphone-storyteller` in your `.env`

See HuggingFace PEFT and Ollama docs for merging/import steps.
