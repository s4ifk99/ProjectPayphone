# Train Payphone Model on Google Colab (No GPU Needed Locally)

Use Colab's **free T4 GPU** to fine-tune the Payphone storytelling model, then use it locally with Ollama on your 16GB RAM / Intel i7 machine.

## Step 1: Prepare data locally

```bash
# Validate and convert (run on your machine)
python scripts/validate_training_dataset.py
python scripts/convert_training_dataset.py
# Produces: training_data_1/training_payphone.jsonl (~2MB)
```

## Step 2: Open Colab notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. **File → Upload notebook** → select `notebooks/train_payphone_colab.ipynb`  
   - If upload fails with a JSON error: try **File → Open notebook** and paste your repo URL (e.g. `https://github.com/s4ifk99/ProjectPayphone`), then open `notebooks/train_payphone_colab.ipynb`  
   - Or use a fresh Colab notebook and copy the code cells from the .ipynb file
3. **Runtime → Change runtime type → T4 GPU** (required)

## Step 3: Run training in Colab

1. Run cell 1 (install deps)
2. Run cell 2: click **Choose Files** and upload `training_data_1/training_payphone.jsonl`
3. Run cells 3–5: load model, train, zip and download

Training takes ~2–4 hours on T4. When done, `payphone-storyteller-lora.zip` downloads automatically.

## Step 4: Import to Ollama (Colab T4 full pipeline)

**Recommended:** Use the [**colab_full_import.ipynb**](notebooks/colab_full_import.ipynb) notebook. It merges your LoRA and converts to GGUF entirely on Colab T4, then downloads a ready-to-use `payphone-story.gguf` file. No heavy work on your machine.

1. Upload `notebooks/colab_full_import.ipynb` to Colab (or copy cells from `notebooks/colab_merge_lora.py`)
2. **Runtime → T4 GPU**, run all cells, upload `payphone-storyteller-lora.zip` when prompted
3. Download `payphone-story.gguf`
4. Locally: `ollama create payphone-story -f Modelfile` then `OLLAMA_MODEL=payphone-story ./run_local.sh`

See [IMPORT_OLLAMA.md](IMPORT_OLLAMA.md) for full details.

## Troubleshooting

- **Out of memory in Colab**: The notebook uses `MAX_SEQ_LENGTH=512`, `BATCH_SIZE=1`, `r=4`, attention-only LoRA for T4 15GB. If still OOM: try `MAX_SEQ_LENGTH=256`, or request **A100** (Runtime → Change runtime type → A100, if available).
- **Colab disconnects**: Save the adapter to Drive periodically; Colab free tier may disconnect after ~12 hours
- **Merge OOM locally**: Use [colab_full_import.ipynb](notebooks/colab_full_import.ipynb) – merge + GGUF conversion on Colab T4, download GGUF directly
