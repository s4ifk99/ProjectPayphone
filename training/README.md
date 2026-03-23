# Project Payphone – LoRA Fine-tuning

Fine-tune **Qwen2.5-7B-Instruct** for Old Bailey legal fiction generation using QLoRA (Unsloth).

## Recommended: Google Colab T4 (full pipeline)

The supported workflow uses **Colab’s free T4 GPU** for both training and merge/GGUF—no local GPU required.

| Phase | Where | Doc / artifact |
|--------|--------|----------------|
| Validate + convert JSONL | Local | This file (steps 1–2 below) |
| QLoRA training | Colab T4 | [TRAINING_COLAB.md](../TRAINING_COLAB.md), `notebooks/train_payphone_colab.ipynb` |
| Merge + GGUF | Colab T4 | [IMPORT_OLLAMA.md](../IMPORT_OLLAMA.md), `notebooks/colab_full_import.ipynb` |
| Ollama + app | Local | `ollama create`, `OLLAMA_MODEL=payphone-story ./run_local.sh` |

See [notebooks/README.md](../notebooks/README.md) for a short notebook index.

---

## Payphone pipeline (Colab path)

### 1. Validate and clean dataset (local)

```bash
python scripts/validate_training_dataset.py
# Output: training_data_1/stories_dataset_cleaned.csv (~400 examples)
```

### 2. Convert to JSONL (chat format, local)

```bash
python scripts/convert_training_dataset.py
# Requires: old_bailey.db
# Output: training_data_1/training_payphone.jsonl
```

### 3. Train on Colab T4

Follow **[TRAINING_COLAB.md](../TRAINING_COLAB.md)**. Upload `training_payphone.jsonl`, run the training notebook, download `payphone-storyteller-lora.zip`.

### 4. Merge + GGUF on Colab T4

Follow **[IMPORT_OLLAMA.md](../IMPORT_OLLAMA.md)** with `colab_full_import.ipynb`. Download `payphone-story.gguf`, then create the Ollama model.

### 5. Test (optional, local)

After you have a merged/HF checkout or use base+adapter in dev:

```bash
python scripts/test_payphone_model.py
# Saves outputs to training_data_1/test_outputs/
```

*(Note: `test_payphone_model.py` expects local paths; adjust if you only have GGUF in Ollama.)*

---

## Optional: local development (GPU required)

For contributors with **~16GB VRAM** and CUDA, you can train without Colab. This is **not** the documented user path.

**Prerequisites**

- Python 3.10+
- GPU with ~16GB VRAM (e.g. RTX 4080, A100)
- CUDA for PyTorch

**Setup**

```bash
cd training
pip install -r requirements.txt
```

**Train (QLoRA)**

```bash
python scripts/train_payphone_model.py --data training_data_1/training_payphone.jsonl --output training/output/payphone-storyteller-lora
```

Options:

- `--model Qwen/Qwen2.5-7B-Instruct` – base model
- `--epochs 3` – training epochs
- `--batch-size 2` – reduce to 1 if OOM

**Export for Ollama (local, advanced)**

After training, the LoRA adapter is under `training/output/payphone-storyteller-lora`. To merge and convert without Colab you must handle merge + GGUF yourself (e.g. Unsloth merge, or `scripts/merge_payphone_lora.py` + `scripts/convert_to_gguf.sh`). Prefer **Colab T4** ([IMPORT_OLLAMA.md](../IMPORT_OLLAMA.md)) unless you know you need local export.

Example merge snippet (Unsloth, illustrative):

```python
from unsloth import FastLanguageModel
from peft import PeftModel
model, tokenizer = FastLanguageModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct", load_in_4bit=False)
model = PeftModel.from_pretrained(model, "training/output/payphone-storyteller-lora")
model = model.merge_and_unload()
model.save_pretrained("merged_payphone")
tokenizer.save_pretrained("merged_payphone")
```

Then convert to GGUF (your choice of tool / llama.cpp) and `ollama create payphone-story -f Modelfile`.

---

## Legacy: app-generated stories

For case-story pairs from the app's stories table:

```bash
python scripts/export_training_pairs.py --output data/training/case_story_pairs.jsonl
python train.py --data data/training/case_story_pairs.jsonl
```

See [TRAINING_ITERATION.md](../TRAINING_ITERATION.md) for the iterative improvement pipeline.
