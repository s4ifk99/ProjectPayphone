#!/usr/bin/env python3
"""
Export (case, story) pairs from old_bailey.db for LoRA fine-tuning.

Output: data/training/case_story_pairs.jsonl

Each line is JSON:
  {"case": {...}, "story": "...", "mode": "...", "target_length": "..."}

Optional: --format instruction for HuggingFace-style {"instruction": ..., "input": ..., "output": ...}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root for app imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db


INSTRUCTION_TEMPLATE = (
    "Convert this Old Bailey case into a 400-600 word historical crime fiction story. "
    "Use a Hero's Journey arc internally. Output continuous prose only."
)


def main() -> None:
    # Run from project root
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    parser = argparse.ArgumentParser(description="Export case-story pairs for training")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("old_bailey.db"),
        help="Path to old_bailey.db",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/training/case_story_pairs.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--target-length",
        type=str,
        default=None,
        help="Filter by target_length (e.g. 400-600). Default: all",
    )
    parser.add_argument(
        "--format",
        choices=["pair", "instruction"],
        default="pair",
        help="Output format: pair (case+story) or instruction (instruction/input/output)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max pairs to export (default: all)",
    )
    args = parser.parse_args()

    conn = db.connect(args.db)
    db.init_stories_table(conn)

    # Get stories with their cases
    rows = conn.execute(
        """SELECT s.story_id, s.case_id, s.mode, s.target_length, s.story_markdown,
                  c.card_json, c.full_text, c.doc_id
           FROM stories s
           JOIN cases c ON s.case_id = c.case_id
           ORDER BY s.created_at DESC"""
    ).fetchall()

    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            if args.limit is not None and count >= args.limit:
                break

            target_length = row["target_length"]
            if args.target_length and target_length != args.target_length:
                continue

            card = {}
            if row["card_json"]:
                try:
                    card = json.loads(row["card_json"])
                except json.JSONDecodeError:
                    pass

            card["case_id"] = row["case_id"]
            card["doc_id"] = row["doc_id"]
            card["full_text"] = row["full_text"] or ""

            case_dict = {
                "case_id": row["case_id"],
                "doc_id": row["doc_id"],
                "year": card.get("year"),
                "offences": card.get("offences") or [],
                "defendants": card.get("defendants") or [],
                "victims": card.get("victims") or [],
                "verdicts": card.get("verdicts") or [],
                "punishments": card.get("punishments") or [],
                "places": card.get("places") or [],
                "full_text": (row["full_text"] or "")[:5000],
            }

            story_text = (row["story_markdown"] or "").strip()

            if args.format == "instruction":
                input_text = json.dumps(case_dict, ensure_ascii=False, indent=2)
                rec = {
                    "instruction": INSTRUCTION_TEMPLATE,
                    "input": input_text,
                    "output": story_text,
                }
            else:
                rec = {
                    "case": case_dict,
                    "story": story_text,
                    "mode": row["mode"],
                    "target_length": target_length,
                }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    conn.close()
    print(f"Exported {count} pairs to {args.output}")


if __name__ == "__main__":
    main()
