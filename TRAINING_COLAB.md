# Train Payphone Model on Google Colab (No GPU Needed Locally)

Use Colab's **free T4 GPU** to fine-tune the Payphone storytelling model, then use it locally with Ollama on your 16GB RAM / Intel i7 machine.

**Full pipeline:** prepare data locally → **train on Colab T4** → **merge + GGUF on Colab T4** → `ollama create` locally. See [IMPORT_OLLAMA.md](IMPORT_OLLAMA.md) for the import step.

**Where training runs:** GPU work happens **in your browser** on [Google Colab](https://colab.research.google.com)—not from a terminal in this repo. Push `notebooks/train_payphone_colab.ipynb` to GitHub and use **File → Open notebook**, or upload the `.ipynb` from your machine.

## Colab T4 checklist

| Step | Action |
|------|--------|
| 1 | **Local:** `python scripts/validate_training_dataset.py` and `python scripts/convert_training_dataset.py` → `training_data_1/training_payphone.jsonl` |
| 2 | **Colab:** Open the training notebook; **Runtime → Change runtime type → T4 GPU**; run the **GPU pre-flight** cell (after install)—it must show a CUDA device |
| 3 | Run the rest of the training notebook top-to-bottom; on CUDA OOM set `MAX_SEQ_LENGTH = 256` in the **T4 config** cell, **Restart session**, re-run from that cell |
| 4 | **New Colab session:** [colab_full_import.ipynb](notebooks/colab_full_import.ipynb), T4, upload `payphone-storyteller-lora.zip`, download `payphone-story.gguf` |
| 5 | **Local:** `ollama create payphone-story -f Modelfile` — see [IMPORT_OLLAMA.md](IMPORT_OLLAMA.md) |

## Step 1: Prepare data locally

```bash
# Validate and convert (run on your machine)
python scripts/validate_training_dataset.py
python scripts/convert_training_dataset.py
# Produces: training_data_1/training_payphone.jsonl (~2MB)
```

## Step 2: Open Colab notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. Either:
   - **File → Open notebook** → paste your repo URL (e.g. `https://github.com/s4ifk99/ProjectPayphone`) → open `notebooks/train_payphone_colab.ipynb`, or  
   - **File → Upload notebook** → select `notebooks/train_payphone_colab.ipynb`  
   - If upload fails with a JSON error, prefer **Open notebook** + repo URL, or copy cells from `notebooks/colab_training_script.py`
3. **Runtime → Change runtime type → T4 GPU** (required)

**Direct link (if the repo is on GitHub):**  
`https://colab.research.google.com/github/s4ifk99/ProjectPayphone/blob/main/notebooks/train_payphone_colab.ipynb`

## Step 3: Run training in Colab

1. Read the **T4 runbook** markdown at the top of the notebook.
2. Run **Install dependencies**, then the **GPU pre-flight** cell (must pass—no CPU-only training).
3. Run **Upload training data** and choose `training_data_1/training_payphone.jsonl`.
4. Run **Load model and train** (T4 config + load JSONL, then model, then trainer cells).
5. Run **Zip and download** (optional: uncomment Drive backup in that cell).

Or use **Runtime → Run all** after T4 is selected and you’ve uploaded the JSONL in the upload step.

Training takes ~2–4 hours on T4. When done, `payphone-storyteller-lora.zip` downloads automatically.

## Step 4: Import to Ollama (Colab T4 only)

Merge the adapter and build **payphone-story.gguf** entirely on Colab—same T4 workflow as training:

1. Open [notebooks/colab_full_import.ipynb](notebooks/colab_full_import.ipynb) in Colab (upload, repo URL, or  
   `https://colab.research.google.com/github/s4ifk99/ProjectPayphone/blob/main/notebooks/colab_full_import.ipynb`)
2. **Runtime → T4 GPU**
3. Run all cells; upload `payphone-storyteller-lora.zip` when prompted; download `payphone-story.gguf`
4. Follow **Register locally with Ollama** in [IMPORT_OLLAMA.md](IMPORT_OLLAMA.md)

## Troubleshooting

- **Out of memory in Colab (training):** The notebook uses `MAX_SEQ_LENGTH=512`, `BATCH_SIZE=1`, `r=4`, attention-only LoRA for T4 15GB. If still OOM: try `MAX_SEQ_LENGTH=256`, or request **A100** (Runtime → Change runtime type → A100, if available).
- **Colab disconnects:** Save the adapter to Drive periodically; Colab free tier may disconnect after ~12 hours.
- **Merge / GGUF errors:** Use **Runtime → Restart session**, then run [colab_full_import.ipynb](notebooks/colab_full_import.ipynb) from the top. See [IMPORT_OLLAMA.md](IMPORT_OLLAMA.md) troubleshooting—do not rely on local merge as the default fix.

See also [notebooks/README.md](notebooks/README.md) for a quick map of notebooks.
