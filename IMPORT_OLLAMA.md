# Import Payphone LoRA to Ollama

Use your trained LoRA adapter with Ollama. **For 16GB RAM / no GPU:** Use the Colab T4 full pipeline below. No heavy work runs locally.

## Primary: Colab T4 Full Pipeline (merge + GGUF)

Merge and convert entirely on Colab. Download a single **payphone-story.gguf** file (~7.5GB).

1. Open [Google Colab](https://colab.research.google.com)
2. **If you had OOM before:** Runtime → Restart session (clears GPU), then run all from top
3. **Runtime → Change runtime type → T4 GPU**
4. **File → Upload notebook** → select `notebooks/colab_full_import.ipynb`  
   - Or copy the cells from `notebooks/colab_merge_lora.py` into a new notebook
5. Run all cells. When prompted, upload `payphone-storyteller-lora.zip` (from training download)
6. Download `payphone-story.gguf` when the last cell finishes

## Local Steps (after download)

1. Place `payphone-story.gguf` in the project root
2. Create Ollama model:
   ```bash
   ollama create payphone-story -f Modelfile
   ```
3. Run the app:
   ```bash
   ollama serve   # if not already running
   OLLAMA_MODEL=payphone-story ./run_local.sh
   ```

---

## Fallback: Local Merge (GPU or 24GB+ RAM only)

If you have a GPU or 24GB+ RAM, you can merge locally:

```bash
pip install transformers peft bitsandbytes accelerate
python scripts/merge_payphone_lora.py --adapter payphone-storyteller-lora --output merged_payphone
```

Then convert to GGUF:
```bash
./scripts/convert_to_gguf.sh
```
