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

## Step 4: Import to Ollama locally

### Option A: Merge in Colab first (recommended)

Add this cell **before** the zip cell in the notebook (after `trainer.save_model()`):

```python
# Merge LoRA into base model (requires ~16GB RAM in Colab)
model = model.merge_and_unload()
model.save_pretrained("/content/merged_payphone")
tokenizer.save_pretrained("/content/merged_payphone")
!cd /content && zip -r merged_payphone.zip merged_payphone
files.download("/content/merged_payphone.zip")
```

Then convert to GGUF locally (see Option A below).

### Option B: Download LoRA adapter only

If you download just the LoRA zip, you need a machine with enough RAM (~16GB) to merge. You can do the merge in a second Colab session:

1. Upload `payphone-storyteller-lora.zip` to Colab
2. Run merge script (load base + adapter, merge, save)
3. Download merged model
4. Convert to GGUF locally

### Convert to GGUF (local)

```bash
# Extract the adapter or merged model
unzip payphone-storyteller-lora.zip -d payphone-lora

# Convert to GGUF (requires llama.cpp or convert_hf_to_gguf)
# See: https://github.com/ggerganov/llama.cpp#convert-hugging-face-models
python -m llama_cpp.convert payphone-lora --outfile payphone-story.Q4_K_M.gguf
# Or use convert_hf_to_gguf from transformers/llama.cpp
```

### Create Ollama model

Create `Modelfile`:

```
FROM ./payphone-story.Q4_K_M.gguf
PARAMETER temperature 0.85
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
```

```bash
ollama create payphone-story -f Modelfile
```

### Run Project Payphone

```bash
OLLAMA_MODEL=payphone-story ./run_local.sh
```

## Troubleshooting

- **Out of memory in Colab**: The notebook uses `MAX_SEQ_LENGTH=512`, `BATCH_SIZE=1`, `r=4`, attention-only LoRA for T4 15GB. If still OOM: try `MAX_SEQ_LENGTH=256`, or request **A100** (Runtime → Change runtime type → A100, if available).
- **Colab disconnects**: Save the adapter to Drive periodically; Colab free tier may disconnect after ~12 hours
- **Merge OOM locally**: Do the merge in Colab (or another GPU machine) and download the merged model
