"""
PAYPHONE COLAB FULL IMPORT (merge + GGUF) - Optimized for T4 15GB
Run in Colab (Runtime → T4 GPU). Merges LoRA, converts to GGUF, downloads payphone-story.gguf.

1. Upload payphone-storyteller-lora.zip (from training download)
2. Run all cells → download payphone-story.gguf (~7.5GB)
3. Locally: ollama create payphone-story -f Modelfile && OLLAMA_MODEL=payphone-story ./run_local.sh

Uses Unsloth save_pretrained_merged (not merge_and_unload + save_pretrained) to avoid
transformers NotImplementedError (reverse_op) on offloaded 4-bit models. After merge, the
checkpoint is reloaded from disk before save_pretrained_gguf (GGUF path calls save_pretrained
on the live model; PEFT/offloaded 4-bit objects break that).
Pinned: unsloth==2025.11.1, unsloth_zoo==2025.11.2, then transformers==4.47.1, accelerate==1.1.1, then trl==0.10.1 (--no-deps; not same pip line as unsloth).
"""

# -------- CELL 1 --------
# Reduce CUDA fragmentation; run FIRST, then Runtime → Restart if you had OOM
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

!pip install -q --upgrade pyarrow
!pip install -q "unsloth==2025.11.1" "unsloth_zoo==2025.11.2" peft bitsandbytes
!pip install -q --force-reinstall "transformers==4.47.1" "accelerate==1.1.1"
!pip install -q --force-reinstall --no-deps "trl==0.10.1"
!pip install -q --upgrade --no-deps "torchvision>=0.26.0"
import importlib.metadata as im
import torch, torchvision


def _pkg_ver(distribution_name):
    try:
        return im.version(distribution_name)
    except im.PackageNotFoundError:
        return "not installed"


print(
    "torch",
    torch.__version__,
    "torchvision",
    torchvision.__version__,
    "transformers",
    _pkg_ver("transformers"),
    "trl",
    _pkg_ver("trl"),
)
# Do not import transformers here — CELL 3 must import unsloth first.

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
import gc
import glob
import os
import site

_nv = []
for _root in site.getsitepackages():
    _nv.extend(glob.glob(os.path.join(_root, "nvidia", "*", "lib")))
if _nv:
    os.environ["LD_LIBRARY_PATH"] = ":".join(_nv) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import unsloth  # Must be before transformers, peft
from unsloth import FastLanguageModel
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
from peft import PeftModelForCausalLM
from transformers import BitsAndBytesConfig

BASE = "Qwen/Qwen2.5-7B-Instruct"
MERGED_DIR = "/content/merged_16bit"
GGUF_DIR = "/content/gguf_output"
OFFLOAD_DIR = "/content/offload"
MAX_SEQ = 256  # T4: 256 saves VRAM vs 512; fine for merge/export
os.makedirs(MERGED_DIR, exist_ok=True)
os.makedirs(GGUF_DIR, exist_ok=True)
os.makedirs(OFFLOAD_DIR, exist_ok=True)

# T4 cannot fit full 4-bit 7B on GPU; allow CPU spill via max_memory + offload_folder or transformers raises:
# ValueError: Some modules are dispatched on the CPU or the disk...
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)
print("Loading base model (4-bit, GPU+CPU offload)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE,
    max_seq_length=MAX_SEQ,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory={0: "11GiB", "cpu": "80GiB"},
    offload_folder=OFFLOAD_DIR,
    trust_remote_code=True,
)

# Use PeftModelForCausalLM so Unsloth's merged_16bit saver takes the fast merge path.
# Generic PeftModel can fall back to model.save_pretrained() and only write adapter files
# (no config.json) → reload from MERGED_DIR fails with "No config file found".
print("Loading LoRA adapter...")
model = PeftModelForCausalLM.from_pretrained(model, ADAPTER_PATH)

# Avoid merge_and_unload() + model.save_pretrained() — NotImplementedError (reverse_op) on offloaded 4-bit.
print("Merging to 16-bit HF (Unsloth save_pretrained_merged)...")
model.save_pretrained_merged(
    MERGED_DIR,
    tokenizer,
    save_method="merged_16bit",
    maximum_memory_usage=0.5,
)


def _dir_has_hf_weights(files):
    fs = set(files)
    if "model.safetensors" in fs or "pytorch_model.bin" in fs:
        return True
    if any(n.endswith(".safetensors.index.json") for n in fs):
        return True
    if any(
        n.startswith("model-") and n.endswith(".safetensors") and "of-" in n
        for n in fs
    ):
        return True
    return False


def resolve_merged_checkpoint(root):
    """Find a folder under root that has both config.json and model weight files."""
    best = None  # (depth, path)
    for dirpath, _, files in os.walk(root):
        if "config.json" not in files:
            continue
        if not _dir_has_hf_weights(files):
            continue
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel in (".", "") else rel.count(os.sep) + 1
        if best is None or depth < best[0]:
            best = (depth, dirpath)
    return best[1] if best else None


MERGED_LOAD_DIR = resolve_merged_checkpoint(MERGED_DIR)

if MERGED_LOAD_DIR is None:
    raise RuntimeError(
        f"No merged HF weights found under {MERGED_DIR}. "
        "Merge did not complete to disk; check free space with `!df -h /content` and re-run Section 3."
    )
else:
    print(f"Merged checkpoint for reload: {MERGED_LOAD_DIR}")

    # save_pretrained_gguf on the live 4-bit+PEFT object can error; reload plain HF checkpoint from disk.
    print("Releasing quantized+PEFT model; reloading merged 16-bit from disk for GGUF export...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MERGED_LOAD_DIR,
        max_seq_length=MAX_SEQ,
        dtype=_dtype,
        load_in_4bit=False,
        device_map="auto",
        max_memory={0: "11GiB", "cpu": "80GiB"},
        offload_folder=OFFLOAD_DIR,
        trust_remote_code=True,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Unsloth GGUF: if config._name_or_path is still the Hub id (from merged config.json),
    # os.path.isdir fails and Unsloth calls self.save_pretrained → NotImplementedError on dispatched models.
    _abs_merged = os.path.abspath(MERGED_LOAD_DIR)
    assert os.path.isdir(_abs_merged), f"Merged checkpoint path missing: {_abs_merged}"
    model.config._name_or_path = _abs_merged
    if hasattr(model.config, "name_or_path"):
        model.config.name_or_path = _abs_merged
    print(f"Unsloth GGUF source checkpoint (patched config): {_abs_merged}")

    print("Converting to GGUF (q8_0)...")
    model.save_pretrained_gguf(
        GGUF_DIR, tokenizer, quantization_method="q8_0", maximum_memory_usage=0.5
    )
    print("Done.")

# -------- CELL 4 --------
from google.colab import files
import glob
import shutil
gguf_files = glob.glob("/content/gguf_output/*.gguf")
if gguf_files:
    shutil.copy(gguf_files[0], "/content/payphone-story.gguf")
    files.download("/content/payphone-story.gguf")
    print("Download started. Place in project root, then: ollama create payphone-story -f Modelfile")
else:
    print("ERROR: GGUF not found in /content/gguf_output. Check the merge cell (Section 3) for errors.")
