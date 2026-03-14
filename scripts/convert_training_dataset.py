#!/usr/bin/env python3
"""
Convert cleaned CSV dataset to JSONL training format (chat messages).

Reads case summaries from old_bailey.db and outputs JSONL with user/assistant messages.

Usage:
  python scripts/convert_training_dataset.py [--input training_data_1/stories_dataset_cleaned.csv] [--output training_data_1/training_payphone.jsonl] [--db old_bailey.db]
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

USER_PROMPT_TEMPLATE = """Write a 400–600 word dark historical legal fiction story based on the following Old Bailey case. The story should follow the thematic structure of the Hero's Journey but must not explicitly mention the stages.

CASE RECORD:
{case_summary}"""


def build_case_summary(card: dict, full_text: str, max_summary_chars: int = 1000) -> str:
    """Build a structured case summary from card_json and full_text."""
    year = card.get("year")
    year_str = str(year) if year is not None else "unknown"

    offence = "unknown"
    offences = card.get("offences") or []
    if offences and isinstance(offences[0], dict):
        o = offences[0]
        offence = (
            o.get("offence_text")
            or o.get("offenceCategory")
            or o.get("offenceSubcategory")
            or offence
        )
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CSV to JSONL training format")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "training_data_1" / "stories_dataset_cleaned.csv",
        help="Input CSV (cleaned)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "training_data_1" / "training_payphone.jsonl",
        help="Output JSONL",
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
        default=None,
        help="Max records to convert (default: all)",
    )
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    db_path = args.db if args.db.is_absolute() else PROJECT_ROOT / args.db

    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    count = 0
    skipped = 0

    with open(input_path, encoding="utf-8") as f_in, open(
        output_path, "w", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            if args.limit is not None and count >= args.limit:
                break

            case_id = row.get("case_id", "")
            story = row.get("story_output", "")

            if not case_id or not story:
                skipped += 1
                continue

            db_row = conn.execute(
                "SELECT full_text, card_json FROM cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()

            if not db_row:
                skipped += 1
                continue

            full_text = db_row["full_text"] or ""
            card = {}
            if db_row["card_json"]:
                try:
                    card = json.loads(db_row["card_json"])
                except json.JSONDecodeError:
                    pass

            case_summary = build_case_summary(card, full_text)
            user_content = USER_PROMPT_TEMPLATE.format(case_summary=case_summary)

            record = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": story.strip()},
                ]
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    conn.close()
    print(f"Wrote {count} records to {output_path}" + (f" (skipped {skipped})" if skipped else ""))


if __name__ == "__main__":
    main()
