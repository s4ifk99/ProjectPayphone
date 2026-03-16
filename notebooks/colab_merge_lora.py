"""
PAYPHONE COLAB FULL IMPORT (merge + GGUF)
Run in Colab (Runtime → T4 GPU). Merges LoRA, converts to GGUF, downloads ready-to-use payphone-story.gguf.
No heavy work on your 16GB machine.

1. Upload payphone-storyteller-lora.zip (from training download)
2. Run all cells → download payphone-story.gguf (~7.5GB)
3. Locally: ollama create payphone-story -f Modelfile && OLLAMA_MODEL=payphone-story ./run_local.sh
"""

# -------- CELL 1 --------
!pip install -q transformers peft bitsandbytes accelerate

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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT = "/content/merged_payphone"
os.makedirs(OUTPUT, exist_ok=True)

print("Loading base model (4-bit)...")
bnb = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("Merging...")
model = model.merge_and_unload()

print("Saving...")
model.save_pretrained(OUTPUT)
tokenizer.save_pretrained(OUTPUT)
print("Done.")

# -------- CELL 4 --------
# Clone llama.cpp (script only; avoid pip deps that conflict with Colab)
!git clone -q --depth 1 https://github.com/ggerganov/llama.cpp /content/llama.cpp
# Use PyPI gguf to avoid Colab package conflicts; skip llama.cpp requirements.txt
!pip install -q gguf

# -------- CELL 5 --------
# Convert merged model to GGUF (q8_0 ~7.5GB)
# NO_LOCAL_GGUF=1 uses PyPI gguf instead of gguf-py (avoids dep conflicts)
import os
os.environ["NO_LOCAL_GGUF"] = "1"
!cd /content/llama.cpp && python convert_hf_to_gguf.py /content/merged_payphone \
  --outfile /content/payphone-story.gguf \
  --outtype q8_0

# -------- CELL 6 --------
from google.colab import files
import os
GGUF_PATH = "/content/payphone-story.gguf"
if os.path.exists(GGUF_PATH):
    files.download(GGUF_PATH)
    print("Download started. Place in project root, then: ollama create payphone-story -f Modelfile")
else:
    print("ERROR: GGUF file not created. Check Cell 5 for errors. Merged model at: /content/merged_payphone")
