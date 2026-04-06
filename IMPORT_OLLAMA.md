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

- **`unsloth_zoo` 2026.x installed while you pinned `unsloth==2025.11.1`:** Pip can still pull **latest `unsloth_zoo`**, which expects **newer `transformers`** and **`torch<2.11`**, causing **`quantizer_finegrained_fp8`** import errors and version fights. The install cell pins **`unsloth_zoo==2025.11.2`** alongside **`unsloth==2025.11.1`** — re-fetch [colab_full_import.ipynb](notebooks/colab_full_import.ipynb) / [colab_merge_lora.py](notebooks/colab_merge_lora.py).
- **`ResolutionImpossible: trl==0.11.4 and unsloth==2025.11.1`:** Pip cannot install those in **one** transaction. The install cell uses **two steps**: **`unsloth` + `unsloth_zoo` + peft + bitsandbytes**, then **`transformers` / `accelerate`**, then **`pip install --force-reinstall --no-deps trl==0.10.1`** so **trl** does not pull a newer **`transformers`**.
- **`AttributeError: TrainingArguments has no attribute _VALID_DICT_FIELDS`** (often inside **`trl`**): **TRL 0.23+** targets **`transformers`** newer than **`4.47.1`**. After the pins above you should have **`trl==0.10.1`**. If **`import unsloth` still pulls a broken trl**, run **`!pip install -q --force-reinstall --no-deps trl==0.10.1`** and **Restart session**. Fallback: raise **`transformers`** into the **4.51.x** band (per Unsloth’s stated range, skipping excluded patch releases) and **re-test `save_pretrained_merged`** for **`reverse_op`** regressions.
- **`PackageNotFoundError: trl`** when printing versions: A failed pip step can leave **trl** uninstalled; the notebook uses **`_pkg_ver()`** so the install cell does not crash. Fix the pip lines above, then re-run.
- **`OSError: libnvJitLink.so.13` / bitsandbytes load error:** Pip-installed **CUDA** wheels (e.g. with **`torch+cu130`**) are not always on the loader path. The **merge** cell prepends **`.../site-packages/nvidia/*/lib`** to **`LD_LIBRARY_PATH`** before **`import unsloth`**. If it still fails, **Runtime → Restart session** and run from the top.
- **“Unsloth should be imported before transformers”:** Do **`not`** run **`import transformers`** in the **install** cell. Use **`importlib.metadata.version("transformers")`** for a version print only. The first **`import transformers`** in the session should happen **after** **`import unsloth`** in the merge cell (or follow the same rule in training cells).
- **Long red `dependency conflicts` after `pip install`:** Colab pre-installs **cudf**, **tensorflow**, **google-colab**, etc. Upgrading **torch**/**torchvision**/**transformers** for this notebook often prints scary warnings; if the **first cell ends with a clean version print** and **`import unsloth` works**, you can ignore most of them.
- **`ImportError: Unsloth: torch==... requires torchvision>=0.26.0`:** **torch** is often **2.11** while **torchvision** stays **0.25**. Run **after** all other installs: `!pip install -q --upgrade --no-deps "torchvision>=0.26.0"` (`--no-deps` avoids pip downgrading **torch** to satisfy an old torchvision). Re-fetch the repo notebook: it pins **Unsloth 2025.11.1** and ends with this line.
- **`unsloth … requires torch<2.11.0` / `requires transformers … >=4.51.3`:** You installed **Unsloth 2026.x**, which conflicts with **Colab’s torch 2.11** and this notebook’s **`transformers==4.47.1`** pin. Use **`unsloth==2025.11.1`** in the install cell (as in current `colab_full_import.ipynb`).
- **Load or save errors / odd failures:** **Runtime → Restart session**, then **Run all** from the first cell (fresh GPU state).
- **Wrong runtime:** Confirm **T4 GPU** is selected (Runtime → Change runtime type).
- **Out of memory:** After a restart, run again; avoid running other heavy notebooks in the same session. The notebook uses T4-oriented settings (`MAX_SEQ = 256`, 4-bit, offload); do not raise `MAX_SEQ` on T4 unless you know you have headroom.
- **Upload notebook fails (JSON error):** Use **File → Open notebook** with the repo URL, or copy cells from `colab_merge_lora.py`.
- **No GGUF / download cell errors:** Check the merge cell (Section 3) for tracebacks; the download step only runs if a `.gguf` file exists under `/content/gguf_output`.
- **`NotImplementedError` / `reverse_op` during save (including inside `save_pretrained_merged`):** Usually **too new a `transformers`** after `pip install unsloth` overwrote the pin. **Re-fetch** [colab_full_import.ipynb](notebooks/colab_full_import.ipynb) — the install cell installs **unsloth first**, then **`--force-reinstall transformers==4.47.1 accelerate==1.1.1`**. **Runtime → Restart session**, run all from the top; the first cell should print `transformers 4.47.1`. If it still shows 4.5x, run manually: `!pip install -q --force-reinstall "transformers==4.47.1" "accelerate==1.1.1"` then **Restart** and re-run. Legacy cause: an old copy used `merge_and_unload()` + `model.save_pretrained()` instead of **`save_pretrained_merged`** — refresh from the repo if your notebook lacks that call.
- **`ValueError: Some modules are dispatched on the CPU or the disk`:** The T4 GPU is too small to hold the whole 4-bit 7B model. The notebook passes **`max_memory={0: "11GiB", "cpu": "80GiB"}`** and **`offload_folder="/content/offload"`** into `FastLanguageModel.from_pretrained` so Accelerate may place some layers on CPU. Refresh the notebook from the repo if your copy lacks those arguments.
- **`RuntimeError: Failed to save model` / `NotImplementedError` inside `save_pretrained_gguf`:** Unsloth sometimes ends up calling **`save_pretrained`** on a **dispatched** in-memory model (empty traceback is common). The notebook **(1)** uses **`save_pretrained_merged`**, **`del model`**, then **reloads** the merged 16-bit folder before GGUF instead of calling GGUF on the 4-bit + PEFT object. **(2)** After reload, merged **`config.json`** often keeps **`_name_or_path`** as the **Hub id** (`Qwen/...`); Unsloth then does not treat weights as on-disk and falls back to **`save_pretrained`** again — the merge cell **sets `model.config._name_or_path`** (and **`name_or_path`** if present) to the **absolute `MERGED_LOAD_DIR`** right before **`save_pretrained_gguf`**. Re-fetch [colab_full_import.ipynb](notebooks/colab_full_import.ipynb) if your copy lacks this patch.
- **`Unsloth: No config file found` when reloading `MERGED_DIR`:** Unsloth’s fast **`merged_16bit`** merge expects a **`PeftModelForCausalLM`**. Loading the adapter with plain **`PeftModel.from_pretrained`** can make Unsloth fall back to a generic save that only writes adapter-style files (often **no `config.json`**). The notebook uses **`PeftModelForCausalLM.from_pretrained`**, then **`MERGED_LOAD_DIR`** (nested search + optional **`config.json`** download from the base model id) before **`FastLanguageModel.from_pretrained(..., trust_remote_code=True)`**.
- **`OSError: no file named model.safetensors ... found in directory .../merged_16bit`:** The merge step did not leave loadable weights next to `config.json` (nested subfolder, **disk full** on `/content`, or partial save). The notebook **searches under `merged_16bit` for sharded `model-*-of-*.safetensors` / index files** and reloads that folder; if nothing is found, it **falls back to `save_pretrained_gguf` on the in-memory PEFT model** (Unsloth can merge inside GGUF). Check **`!df -h /content`** and **`!ls -laR /content/merged_16bit | head -80`** if errors persist.

---

## Appendix: Other platforms

**Primary import** remains **Colab T4** above. **Kaggle** is supported for **merge-only** (HF zip); **local** scripts are for maintainers or offline conversion.

### Kaggle (merge only → download zip)

Use [notebooks/kaggle_full_import.ipynb](notebooks/kaggle_full_import.ipynb) when you want **Kaggle’s GPU** to **merge LoRA into a Hugging Face checkpoint** and download **`payphone-merged-hf.zip`**. For **GGUF on Kaggle**, use [kaggle_hf_to_gguf.ipynb](notebooks/kaggle_hf_to_gguf.ipynb) (separate notebook).

**Do not run `colab_full_import.ipynb` on Kaggle** — it uses **`google.colab.files`** and **`/content/`**, which Kaggle does not provide. Attach your LoRA as a **Kaggle dataset** (**Add data**), then use **`kaggle_full_import.ipynb`** Section 2 to scan **`/kaggle/input`** (no upload widget).

**Two-step workflow:**

1. **Kaggle:** Run all cells → download **`payphone-merged-hf.zip`** (unzip to a folder with `config.json`, tokenizer files, and `model*.safetensors`).
2. **GGUF:** Either use [colab_full_import.ipynb](notebooks/colab_full_import.ipynb) with your original **`payphone-storyteller-lora.zip`** for a **one-shot** `payphone-story.gguf`, **or** unzip the merged folder locally and run **`convert_hf_to_gguf.py`** (see [scripts/convert_to_gguf.sh](scripts/convert_to_gguf.sh)).

Requirements: **Internet** ON, **GPU** accelerator. Install cell sets **`PYTORCH_CUDA_ALLOC_CONF` / `PYTORCH_ALLOC_CONF`** to **`expandable_segments:True`**; **Session → Restart** after GPU OOM.

- **Kaggle pip red `dependency conflict`:** Ignore if **`OK: transformers …`** prints after install.
- **Kaggle stderr (cuFFT/cuDNN / computation placer):** Harmless if shard load completes.
- **Merge / save:** The notebook **`delattr`s `quantization_config` / `pre_quantization_dtype`** before **`save_pretrained`** and removes them from **`config.json`** (do not set those fields to **`None`**).
- **`TypeError: Object of type dtype is not JSON serializable`** on `model.save_pretrained(...)`: merged configs can still contain dtype objects (`torch.dtype` / numpy dtype). The Kaggle merge cell now sanitizes dtype values in `model.config.__dict__` before save and removes `_pre_quantization_dtype` from `config.json`.

### Local merge (CPU or GPU)

Requires enough RAM/VRAM and manual `llama.cpp` / conversion steps. Scripts still in repo for maintainers:

- `scripts/merge_payphone_lora.py` (`--cpu` or GPU)
- `scripts/convert_to_gguf.sh`

On **16GB RAM without GPU**, local merge often needs **swap** and is fragile; **prefer Colab T4** above.
