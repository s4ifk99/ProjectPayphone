# Import Payphone LoRA to Ollama

Merge your trained LoRA into the base model and export **GGUF** on **Google Colab (T4 GPU)**. Your local machine only needs the finished `payphone-story.gguf` plus Ollama.

Train the adapter first using [TRAINING_COLAB.md](TRAINING_COLAB.md).

---

## Primary path: Colab T4 (merge + GGUF)

The notebook merges the adapter with Qwen2.5-7B-Instruct and writes a GGUF via Unsloth—optimized for a **15GB T4** (`max_seq_length=256`, 4-bit load, CPU offload for load). Merge uses **`save_pretrained_merged`** (not `merge_and_unload` + `save_pretrained`) to avoid Hugging Face `NotImplementedError` / `reverse_op` on offloaded quantized checkpoints.

### Open the notebook

1. Go to [Google Colab](https://colab.research.google.com).
2. Either:
   - **File → Open notebook** → paste your repo URL (e.g. `https://github.com/s4ifk99/ProjectPayphone`) → open `notebooks/colab_full_import.ipynb`, or  
   - **File → Upload notebook** → choose `notebooks/colab_full_import.ipynb` from this project, or  
   - Open directly (if the repo is on GitHub):  
     `https://colab.research.google.com/github/s4ifk99/ProjectPayphone/blob/main/notebooks/colab_full_import.ipynb`
3. **Runtime → Change runtime type → T4 GPU** (required).

### Run the pipeline

1. Run all cells from the top.
2. When prompted, upload **`payphone-storyteller-lora.zip`** (from Colab training download).
3. When the last cell finishes, download **`payphone-story.gguf`** (~several GB).

**Copy-paste option:** If you prefer cells in a fresh notebook, use [notebooks/colab_merge_lora.py](notebooks/colab_merge_lora.py) (one cell per `CELL` block).

### Register locally with Ollama

Place `payphone-story.gguf` in the project root (next to `Modelfile`), then:

```bash
ollama create payphone-story -f Modelfile
OLLAMA_MODEL=payphone-story ./run_local.sh
```

---

## Troubleshooting (Colab)

- **Load or save errors / odd failures:** **Runtime → Restart session**, then **Run all** from the first cell (fresh GPU state).
- **Wrong runtime:** Confirm **T4 GPU** is selected (Runtime → Change runtime type).
- **Out of memory:** After a restart, run again; avoid running other heavy notebooks in the same session. The notebook uses T4-oriented settings (`MAX_SEQ = 256`, 4-bit, offload); do not raise `MAX_SEQ` on T4 unless you know you have headroom.
- **Upload notebook fails (JSON error):** Use **File → Open notebook** with the repo URL, or copy cells from `colab_merge_lora.py`.
- **No GGUF / download cell errors:** Check the merge cell (Section 3) for tracebacks; the download step only runs if a `.gguf` file exists under `/content/gguf_output`.
- **`NotImplementedError` / `reverse_op` during save:** You are on an old notebook copy that used `merge_and_unload()` + `model.save_pretrained()`. **Re-fetch** [colab_full_import.ipynb](notebooks/colab_full_import.ipynb) from this repo (or use [colab_merge_lora.py](notebooks/colab_merge_lora.py)), **Runtime → Restart session**, run all. The current flow uses Unsloth **`save_pretrained_merged`** and pins `transformers==4.47.1` / `accelerate==1.1.1` in the install cell (use `4.47.1` if `4.47.2` is not found on PyPI). Run `pip show transformers` in Colab if issues persist.
- **`ValueError: Some modules are dispatched on the CPU or the disk`:** The T4 GPU is too small to hold the whole 4-bit 7B model. The notebook passes **`max_memory={0: "11GiB", "cpu": "80GiB"}`** and **`offload_folder="/content/offload"`** into `FastLanguageModel.from_pretrained` so Accelerate may place some layers on CPU. Refresh the notebook from the repo if your copy lacks those arguments.
- **`RuntimeError: Failed to save model` / `NotImplementedError` inside `save_pretrained_gguf`:** Unsloth sometimes ends up calling **`save_pretrained`** on a **dispatched** in-memory model (empty traceback is common). The notebook **(1)** uses **`save_pretrained_merged`**, **`del model`**, then **reloads** the merged 16-bit folder before GGUF instead of calling GGUF on the 4-bit + PEFT object. **(2)** After reload, merged **`config.json`** often keeps **`_name_or_path`** as the **Hub id** (`Qwen/...`); Unsloth then does not treat weights as on-disk and falls back to **`save_pretrained`** again — the merge cell **sets `model.config._name_or_path`** (and **`name_or_path`** if present) to the **absolute `MERGED_LOAD_DIR`** right before **`save_pretrained_gguf`**. Re-fetch [colab_full_import.ipynb](notebooks/colab_full_import.ipynb) if your copy lacks this patch.
- **`Unsloth: No config file found` when reloading `MERGED_DIR`:** Unsloth’s fast **`merged_16bit`** merge expects a **`PeftModelForCausalLM`**. Loading the adapter with plain **`PeftModel.from_pretrained`** can make Unsloth fall back to a generic save that only writes adapter-style files (often **no `config.json`**). The notebook uses **`PeftModelForCausalLM.from_pretrained`**, then **`MERGED_LOAD_DIR`** (nested search + optional **`config.json`** download from the base model id) before **`FastLanguageModel.from_pretrained(..., trust_remote_code=True)`**.
- **`OSError: no file named model.safetensors ... found in directory .../merged_16bit`:** The merge step did not leave loadable weights next to `config.json` (nested subfolder, **disk full** on `/content`, or partial save). The notebook **searches under `merged_16bit` for sharded `model-*-of-*.safetensors` / index files** and reloads that folder; if nothing is found, it **falls back to `save_pretrained_gguf` on the in-memory PEFT model** (Unsloth can merge inside GGUF). Check **`!df -h /content`** and **`!ls -laR /content/merged_16bit | head -80`** if errors persist.

---

## Appendix: Other platforms (unsupported)

These are **not** part of the documented workflow; kept for advanced users or if Colab is unavailable.

### Kaggle

Some users run merge + GGUF on Kaggle (more RAM, different stack). See [notebooks/kaggle_full_import.ipynb](notebooks/kaggle_full_import.ipynb)—marked legacy; no guarantee it matches current training exports. The Kaggle notebook mirrors the Colab **install** cell: **`PYTORCH_CUDA_ALLOC_CONF` / `PYTORCH_ALLOC_CONF`** with **`expandable_segments:True`**, plus **Session → Restart** if you hit GPU OOM. Merge uses **`max_memory` + `offload_folder`** and **`PeftModelForCausalLM`**, then **llama.cpp** `convert_hf_to_gguf.py` (not Unsloth). **Internet** must be ON in notebook settings.

### Local merge (CPU or GPU)

Requires enough RAM/VRAM and manual `llama.cpp` / conversion steps. Scripts still in repo for maintainers:

- `scripts/merge_payphone_lora.py` (`--cpu` or GPU)
- `scripts/convert_to_gguf.sh`

On **16GB RAM without GPU**, local merge often needs **swap** and is fragile; **prefer Colab T4** above.
