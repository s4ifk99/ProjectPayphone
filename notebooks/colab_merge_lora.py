"""
PAYPHONE COLAB FULL IMPORT (merge + GGUF)
Run in Colab (Runtime → T4 GPU). Merges LoRA, converts to GGUF, downloads ready-to-use payphone-story.gguf.
No heavy work on your 16GB machine.

1. Upload payphone-storyteller-lora.zip (from training download)
2. Run all cells → download payphone-story.gguf (~7.5GB)
3. Locally: ollama create payphone-story -f Modelfile && OLLAMA_MODEL=payphone-story ./run_local.sh
"""

# -------- CELL 1 --------
# Reduce CUDA fragmentation; run FIRST, then Runtime → Restart if you had OOM
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

!pip install -q unsloth transformers peft bitsandbytes accelerate

# -------- CELL 2 --------
from google.colab import files
import zipfile
import os

print("Upload payphone-storyteller-lora.zip (from Colab training download)")
uploaded = files.upload()
zip_path = list(uploaded.keys())[0] if uploaded else None
if not zip_path:
    raise FileNotFoundError("Upload payphone-storyteller-lora.zip")

# Extract
os.makedirs("/content/lora", exist_ok=True)
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall("/content/lora")
ADAPTER_PATH = "/content/lora/payphone-storyteller-lora"
if not os.path.exists(ADAPTER_PATH):
    ADAPTER_PATH = "/content/lora"  # maybe extracted structure differs
print(f"Adapter at: {ADAPTER_PATH}")

# -------- CELL 3 --------
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()

from unsloth import FastLanguageModel
from peft import PeftModel

BASE = "Qwen/Qwen2.5-7B-Instruct"
GGUF_DIR = "/content/gguf_output"
os.makedirs(GGUF_DIR, exist_ok=True)

print("Loading base model (4-bit)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE,
    max_seq_length=512,
    load_in_4bit=True,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("Merging...")
model = model.merge_and_unload()

print("Saving to GGUF (q8_0) via Unsloth - handles bitsandbytes...")
model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q8_0")
print("Done.")

# -------- CELL 4 --------
# Download GGUF (Unsloth saved it in Cell 3; no llama.cpp needed)
from google.colab import files
import glob
import shutil
gguf_files = glob.glob("/content/gguf_output/*.gguf")
if gguf_files:
    shutil.copy(gguf_files[0], "/content/payphone-story.gguf")
    files.download("/content/payphone-story.gguf")
    print("Download started. Place in project root, then: ollama create payphone-story -f Modelfile")
else:
    print("ERROR: GGUF not found in /content/gguf_output. Check Cell 3 for errors.")
