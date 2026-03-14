#!/usr/bin/env python3
"""
Test the fine-tuned Payphone storytelling model.

Loads adapter + base model, generates stories for 20-30 unseen cases,
saves to training_data_1/test_outputs/, and reports quality metrics.

Usage:
  python scripts/test_payphone_model.py [--model training/output/payphone-storyteller-lora] [--db old_bailey.db] [--limit 25]
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.prompts import validate_story_prose


def word_count(text: str) -> int:
    return len((text or "").split())


def build_case_summary(card: dict, full_text: str, max_summary_chars: int = 1000) -> str:
    """Build case summary from card + full_text."""
    year = card.get("year")
    year_str = str(year) if year is not None else "unknown"
    offence = "unknown"
    offences = card.get("offences") or []
    if offences and isinstance(offences[0], dict):
        o = offences[0]
        offence = o.get("offence_text") or o.get("offenceCategory") or o.get("offenceSubcategory") or offence
        if isinstance(offence, list):
            offence = offence[0] if offence else "unknown"
        offence = str(offence).strip() or "unknown"
    victim = "unknown"
    victims = card.get("victims") or []
    if victims and isinstance(victims[0], dict):
        v = victims[0].get("display_name")
        if v:
            victim = str(v).strip()
    verdict = "unknown"
    verdicts = card.get("verdicts") or []
    if verdicts and isinstance(verdicts[0], dict):
        vt = verdicts[0].get("verdict_text")
        if vt:
            verdict = str(vt).strip()
    punishment = "unknown"
    punishments = card.get("punishments") or []
    if punishments and isinstance(punishments[0], dict):
        pt = punishments[0].get("punishment_text")
        if pt:
            punishment = str(pt).strip()
    summary_text = (full_text or "").strip()[:max_summary_chars]
    if summary_text and len((full_text or "").strip()) > max_summary_chars:
        summary_text += "..."
    return f"""Year: {year_str}
Offence: {offence}
Victim: {victim}
Verdict: {verdict}
Punishment: {punishment}

Case Summary: {summary_text}"""


USER_PROMPT = """Write a 400–600 word dark historical legal fiction story based on the following Old Bailey case. The story should follow the thematic structure of the Hero's Journey but must not explicitly mention the stages.

CASE RECORD:
{case_summary}"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Test fine-tuned Payphone model")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "training" / "output" / "payphone-storyteller-lora",
        help="Path to LoRA adapter",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=PROJECT_ROOT / "old_bailey.db",
        help="Path to old_bailey.db",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Number of test cases",
    )
    args = parser.parse_args()

    adapter_path = args.model if args.model.is_absolute() else PROJECT_ROOT / args.model
    db_path = args.db if args.db.is_absolute() else PROJECT_ROOT / args.db

    if not adapter_path.exists():
        print(f"Model not found: {adapter_path}", file=sys.stderr)
        print("Run train_payphone_model.py first.", file=sys.stderr)
        sys.exit(1)
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = PROJECT_ROOT / "training_data_1" / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model: base + adapter (requires unsloth, transformers, etc.)
    try:
        from unsloth import FastLanguageModel

        # Adapter dir has base_model_name_or_path in adapter_config.json
        adapter_config_path = adapter_path / "adapter_config.json"
        base_name = "Qwen/Qwen2.5-7B-Instruct"
        if adapter_config_path.exists():
            cfg = json.loads(adapter_config_path.read_text())
            base_name = cfg.get("base_model_name_or_path", base_name)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_name,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        # Load LoRA adapter
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    # Fetch unseen cases (not in training set)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Get case_ids from training data to exclude
    training_csv = PROJECT_ROOT / "training_data_1" / "stories_dataset_cleaned.csv"
    seen = set()
    if training_csv.exists():
        import csv

        with open(training_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen.add(row.get("case_id", ""))

    rows = conn.execute(
        """
        SELECT case_id, full_text, card_json
        FROM cases
        WHERE length(full_text) > 500
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (args.limit * 2,),  # fetch extra in case some are in training
    ).fetchall()

    cases = []
    for row in rows:
        if row["case_id"] not in seen and len(cases) < args.limit:
            card = {}
            if row["card_json"]:
                try:
                    card = json.loads(row["card_json"])
                except json.JSONDecodeError:
                    pass
            cases.append({
                "case_id": row["case_id"],
                "full_text": row["full_text"] or "",
                "card": card,
            })
    conn.close()

    if not cases:
        print("No unseen cases found.")
        sys.exit(0)

    print(f"Generating {len(cases)} test stories...")

    results = []
    for i, case in enumerate(cases):
        case_id = case["case_id"]
        summary = build_case_summary(case["card"], case["full_text"])
        prompt = USER_PROMPT.format(case_summary=summary)

        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)

        outputs = model.generate(
            inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        reply = tokenizer.decode(outputs[0][inputs.shape[1] :], skip_special_tokens=True)
        story = reply.strip()

        words = word_count(story)
        ok, reason = validate_story_prose(story)
        in_range = 400 <= words <= 600

        rec = {
            "case_id": case_id,
            "story": story,
            "word_count": words,
            "in_range": in_range,
            "prose_ok": ok,
            "prose_reason": reason,
        }
        results.append(rec)

        out_path = output_dir / f"{case_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

        print(f"  [{i+1}/{len(cases)}] {case_id}: {words} words, prose={'ok' if ok else reason}")

    # Summary
    in_range = sum(1 for r in results if r["in_range"])
    prose_ok = sum(1 for r in results if r["prose_ok"])
    print(f"\nSummary: {in_range}/{len(results)} in 400-600 words, {prose_ok}/{len(results)} prose-valid")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
