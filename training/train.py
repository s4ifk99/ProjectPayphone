#!/usr/bin/env python3
"""
LoRA fine-tuning for Project Payphone storytelling LLM.

Usage:
  python train.py --data ../data/training/case_story_pairs.jsonl --output output/payphone-storyteller-lora

Requires: GPU with ~16GB VRAM for Mistral-7B, or use --model Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_DATA = "data/training/case_story_pairs.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def format_instruction_sample(rec: dict) -> str:
    """Format a record for chat-style training."""
    instruction = rec.get("instruction", "")
    input_text = rec.get("input", "")
    output_text = rec.get("output", rec.get("story", ""))

    if not input_text and "case" in rec:
        input_text = json.dumps(rec["case"], ensure_ascii=False, indent=2)
    if not instruction:
        instruction = (
            "Convert this Old Bailey case into a 400-600 word historical crime fiction story. "
            "Use a Hero's Journey arc internally. Output continuous prose only."
        )

    return (
        f"<s>[INST] {instruction}\n\n{input_text} [/INST]\n{output_text}</s>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model name")
    parser.add_argument("--data", type=Path, default=Path(DEFAULT_DATA), help="JSONL training data")
    parser.add_argument("--output", type=Path, default=Path("output/payphone-storyteller-lora"))
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    data_path = args.data
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent.parent / data_path

    records = load_jsonl(data_path)
    if not records:
        print("No training records found.")
        return

    texts = [format_instruction_sample(r) for r in records]
    dataset = Dataset.from_dict({"text": texts})

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output)
    print(f"Saved adapter to {args.output}")


if __name__ == "__main__":
    main()
