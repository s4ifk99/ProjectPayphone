# Project Payphone – LoRA Fine-tuning

Fine-tune Qwen2.5-7B-Instruct for Old Bailey legal fiction generation using QLoRA (Unsloth).

## No GPU? Use Google Colab

If you have **16GB RAM, no GPU** (e.g. Intel i7 laptop): train in the cloud with the free Colab T4 GPU.

See **[TRAINING_COLAB.md](../TRAINING_COLAB.md)** and `notebooks/train_payphone_colab.ipynb`.

## Prerequisites (local training)

- Python 3.10+
- GPU with ~16GB VRAM (e.g. RTX 4080, A100)
- CUDA for PyTorch

## Setup

```bash
cd training
pip install -r requirements.txt
```

## Payphone Storytelling Pipeline

### 1. Validate and clean dataset

```bash
python scripts/validate_training_dataset.py
# Output: training_data_1/stories_dataset_cleaned.csv (~400 examples)
```

### 2. Convert to JSONL (chat format)

```bash
python scripts/convert_training_dataset.py
# Requires: old_bailey.db
# Output: training_data_1/training_payphone.jsonl
```

### 3. Train (QLoRA)

```bash
python scripts/train_payphone_model.py --data training_data_1/training_payphone.jsonl --output training/output/payphone-storyteller-lora
```

Options:
- `--model Qwen/Qwen2.5-7B-Instruct` – base model
- `--epochs 3` – training epochs
- `--batch-size 2` – reduce to 1 if OOM

### 4. Test

```bash
python scripts/test_payphone_model.py
# Saves outputs to training_data_1/test_outputs/
```

## Export for Ollama

After training, the LoRA adapter is saved to `training/output/payphone-storyteller-lora`. To use with Ollama:

1. **Merge** LoRA adapter into base model:
   ```python
   from unsloth import FastLanguageModel
   from peft import PeftModel
   model, tokenizer = FastLanguageModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct", load_in_4bit=False)
   model = PeftModel.from_pretrained(model, "training/output/payphone-storyteller-lora")
   model = model.merge_and_unload()
   model.save_pretrained("merged_payphone")
   tokenizer.save_pretrained("merged_payphone")
   ```

2. **Convert to GGUF** using `llama.cpp` or `ctranslate2`:
   ```bash
   python -m transformers.convert_hf_to_gguf merged_payphone --outfile payphone-story.Q4_K_M.gguf
   ```

3. **Create Ollama Modelfile**:
   ```
   FROM ./payphone-story.Q4_K_M.gguf
   PARAMETER temperature 0.85
   PARAMETER top_p 0.9
   PARAMETER repeat_penalty 1.1
   ```

4. **Create model**: `ollama create payphone-story -f Modelfile`

5. **Set** `OLLAMA_MODEL=payphone-story` in `.env` or run script

## Legacy: app-generated stories

For case-story pairs from the app's stories table:

```bash
python scripts/export_training_pairs.py --output data/training/case_story_pairs.jsonl
python train.py --data data/training/case_story_pairs.jsonl
```

See [TRAINING_ITERATION.md](../TRAINING_ITERATION.md) for the iterative improvement pipeline.
