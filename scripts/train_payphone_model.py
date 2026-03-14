#!/usr/bin/env python3
"""
QLoRA fine-tuning for Payphone storytelling LLM.

Uses Unsloth for efficient 4-bit training with Qwen2.5-7B-Instruct.
Expects training_payphone.jsonl with messages format (user/assistant).

Usage:
  python scripts/train_payphone_model.py [--data training_data_1/training_payphone.jsonl] [--output training/output/payphone-storyteller-lora]

Requires: GPU with ~16GB VRAM (or reduce batch_size).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Unsloth patches transformers at import; import before other ML libs
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DATA = PROJECT_ROOT / "training_data_1" / "training_payphone.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "training" / "output" / "payphone-storyteller-lora"


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def format_chat_to_text(record: dict, tokenizer) -> str:
    """Convert messages to tokenizer's chat format string."""
    messages = record.get("messages", [])
    if not messages:
        return ""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Payphone storytelling model")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Base model (HuggingFace ID)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Training JSONL (messages format)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for LoRA adapter",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Max sequence length",
    )
    args = parser.parse_args()

    data_path = args.data if args.data.is_absolute() else PROJECT_ROOT / args.data
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output

    if not data_path.exists():
        print(f"Data not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(data_path)
    if not records:
        print("No training records found.")
        sys.exit(1)

    print(f"Loading model {args.model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    texts = [format_chat_to_text(r, tokenizer) for r in records]
    dataset = Dataset.from_dict({"text": texts})

    training_args = TrainingArguments(
        output_dir=str(output_path),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        fp16=not __import__("torch").cuda.is_bf16_supported(),
        bf16=__import__("torch").cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.1,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_path)
    print(f"Saved adapter to {output_path}")


if __name__ == "__main__":
    main()
