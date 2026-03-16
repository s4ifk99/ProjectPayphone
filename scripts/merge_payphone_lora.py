#!/usr/bin/env python3
"""
Merge Payphone LoRA adapter with base Qwen2.5-7B model.

Saves merged model in HuggingFace format for GGUF conversion.
Needs ~10–12 GB RAM (4-bit base + merge).

Usage:
  python scripts/merge_payphone_lora.py [--adapter payphone-storyteller-lora] [--output merged_payphone]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=PROJECT_ROOT / "payphone-storyteller-lora",
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "merged_payphone",
        help="Output path for merged model",
    )
    parser.add_argument(
        "--base",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run on CPU (no GPU). Uses fp16, needs ~14GB RAM. If OOM, merge in Colab instead.",
    )
    args = parser.parse_args()

    adapter_path = args.adapter if args.adapter.is_absolute() else PROJECT_ROOT / args.adapter
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output

    if not adapter_path.exists():
        print(f"Adapter not found: {adapter_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading transformers, peft...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    import torch

    if args.cpu:
        print(f"Loading base model: {args.base} (fp16, CPU – needs ~14GB RAM)")
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        print(f"Loading base model: {args.base} (4-bit)")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))

    print("Merging LoRA into base...")
    model = model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("Done.")
    print(f"Next: convert to GGUF, then ollama create payphone-story -f Modelfile")
    print(f"  See IMPORT_OLLAMA.md for full steps.")


if __name__ == "__main__":
    main()
