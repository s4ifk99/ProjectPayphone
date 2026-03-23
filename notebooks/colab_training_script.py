"""
PAYPHONE COLAB TRAINING
Copy each "CELL" block into a separate cell in a new Colab notebook.
Runtime → Change runtime type → T4 GPU first.

T4 runbook (markdown cell — paste above CELL 1):
  1. Runtime → Change runtime type → T4 GPU before any code.
  2. Errors after partial run → Restart session → Run all from top.
  3. CUDA OOM → in T4 config cell set MAX_SEQ_LENGTH=256, restart, re-run from that cell.
  4. After training → colab_full_import.ipynb (new session, T4) → IMPORT_OLLAMA.md
"""

# -------- CELL 1 --------
# Reduce CUDA fragmentation (helps with T4 15GB limit)
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

!pip install -q unsloth bitsandbytes transformers datasets trl accelerate peft

# -------- CELL 2 --------
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU. In Colab: Runtime → Change runtime type → T4 GPU, then re-run from the top."
    )

name = torch.cuda.get_device_name(0)
print(f"GPU: {name}")

try:
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"VRAM: {free_b / 1e9:.2f} GB free / {total_b / 1e9:.2f} GB total")
except Exception as e:
    print(f"(Could not read VRAM info: {e})")

if "T4" not in name and "A100" not in name and "L4" not in name:
    print("Warning: This notebook is tuned for T4; other GPUs may work or OOM.")

# -------- CELL 3 --------
from google.colab import files
import os

print("Click 'Choose Files' and select training_payphone.jsonl")
uploaded = files.upload()
DATA_PATH = list(uploaded.keys())[0] if uploaded else None
if not DATA_PATH:
    raise FileNotFoundError("Upload training_payphone.jsonl")
print(f"Using {DATA_PATH}")

# -------- CELL 4 --------
import json
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# --- T4 config (CUDA OOM: set MAX_SEQ_LENGTH=256, Runtime → Restart session, re-run from this cell) ---
MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "/content/payphone-storyteller-lora"
MAX_SEQ_LENGTH = 512  # T4: 512 default; 1024 OOMs; use 256 if you still OOM
EPOCHS = 3
BATCH_SIZE = 1
LORA_R = 4  # r=8 OOMs on T4; do not raise without A100 / more VRAM
# --- end T4 config ---

def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

records = load_jsonl(DATA_PATH)
print(f"Loaded {len(records)} examples")

# -------- CELL 5 --------
print("Loading model (4-bit QLoRA)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,  # set in T4 config cell above
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention only (drop MLP to save VRAM)
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
print("Model ready")

# -------- CELL 6 --------
import os

def format_chat(record, tokenizer):
    messages = record.get("messages", [])
    if not messages:
        return ""
    try:
        out = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return out if isinstance(out, str) else ""
    except Exception as e:
        print(f"Warning: format_chat failed: {e}")
        return ""

texts = [format_chat(r, tokenizer) for r in records]
texts = [t for t in texts if t and t.strip()]
if not texts:
    raise ValueError("No valid training examples after formatting")

# Pre-truncate to avoid Unsloth cross_entropy shape mismatch (1024 vs 512)
def truncate_to_tokens(text, tokenizer, max_len):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_len:
        return text
    return tokenizer.decode(ids[:max_len], skip_special_tokens=False)

# Leave ~8 tokens headroom for special tokens trainer may add
texts = [truncate_to_tokens(t, tokenizer, MAX_SEQ_LENGTH - 8) for t in texts]
dropped = 0  # We truncate, don't drop
print(f"Formatted {len(texts)} examples" + (f" (dropped {dropped} invalid)" if dropped else ""))
dataset = Dataset.from_dict({"text": texts})
print(f"Dataset: {len(dataset)} rows")
print("\nSample (first 400 chars):")
print(dataset["text"][0][:400] + "..." if len(dataset["text"][0]) > 400 else dataset["text"][0])

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=8,  # Higher to compensate batch_size=1 (effective batch=8)
    num_train_epochs=EPOCHS,
    learning_rate=2e-4,
    fp16=not __import__("torch").cuda.is_bf16_supported(),
    bf16=__import__("torch").cuda.is_bf16_supported(),
    logging_steps=10,
    save_strategy="epoch",
    warmup_steps=15,  # ~10% of 147 steps; warmup_ratio deprecated in v5.2
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    packing=False,  # Avoid packing long sequences; safer for T4 VRAM
)

print("Training... (~2-4 hours on T4)")
print("Trainer ready. Starting...")
try:
    trainer.train()
except Exception as e:
    print(f"TRAINING FAILED: {e}")
    raise

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Verify files were saved before zipping
saved_files = os.listdir(OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else []
if not saved_files:
    raise RuntimeError(
        "Save produced no files! Check for errors above. "
        "Training may have failed silently or OUTPUT_DIR is wrong."
    )
print(f"Saved: {saved_files}")

# -------- CELL 7 --------
!cd /content && zip -r payphone-storyteller-lora.zip payphone-storyteller-lora

# Optional: backup zip to Drive if Colab disconnects (uncomment):
# from google.colab import drive
# drive.mount('/content/drive')
# !cp /content/payphone-storyteller-lora.zip "/content/drive/MyDrive/payphone-storyteller-lora.zip"

from google.colab import files
files.download("/content/payphone-storyteller-lora.zip")
print("Download started. Next: colab_full_import.ipynb + IMPORT_OLLAMA.md (merge + GGUF + Ollama).")
