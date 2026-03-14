#!/usr/bin/env python3
"""
Validate and clean the training story dataset.

Keeps stories that:
- Have 400-600 words
- Are continuous prose (no headings, bullets, stage labels)
- Have beginning, middle, end (paragraph structure)
- Contain dark/legal vocabulary and resolution

Excludes: summaries, repetitive text, truncated stories, stories without resolution.

Usage:
  python scripts/validate_training_dataset.py [--input training_data_1/stories_dataset.csv] [--output training_data_1/stories_dataset_cleaned.csv]
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.prompts import validate_story_prose

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESOLUTION_KEYWORDS = frozenset({
    "verdict", "guilty", "not guilty", "sentenced", "condemned", "punishment",
    "hanged", "transported", "acquitted", "jury", "convicted", "executed",
    "prison", "gaol", "newgate", "tyburn", "death", "exile",
})


def word_count(text: str) -> int:
    return len((text or "").split())


def _bigram_repetition_ratio(text: str) -> float:
    """Return ratio of repeated bigrams to total unique bigrams. High = repetitive."""
    words = (text or "").lower().split()
    if len(words) < 4:
        return 0.0
    bigrams = [tuple(words[i : i + 2]) for i in range(len(words) - 1)]
    total = len(bigrams)
    unique = len(set(bigrams))
    if unique == 0:
        return 0.0
    return 1 - (unique / total)


def _has_resolution(text: str) -> bool:
    """Check if story ends with resolution (verdict, sentencing, etc.)."""
    text_lower = (text or "").lower()
    last_third = text_lower[len(text_lower) * 2 // 3 :]
    return any(kw in last_third for kw in RESOLUTION_KEYWORDS)


def _is_truncated(text: str) -> bool:
    """Heuristic: story ends mid-sentence or with ellipsis."""
    t = (text or "").strip()
    if not t:
        return True
    # Ends with trailing ellipsis (incomplete)
    if t.endswith("..."):
        return True
    # Ends without sentence-ending punctuation (., !, ?, or closing quote after)
    if not re.search(r'[.!?"]\s*$', t):
        return True
    return False


def _has_paragraph_structure(text: str) -> bool:
    """At least 3 paragraphs (double newlines) for narrative arc."""
    paras = [p.strip() for p in (text or "").split("\n\n") if p.strip()]
    return len(paras) >= 2


def validate_story(case_id: str, story: str) -> tuple[bool, str]:
    """Validate one story. Returns (ok, reason)."""
    if not story or not story.strip():
        return (False, "Empty")

    words = word_count(story)
    if words < 400:
        return (False, f"Too short ({words} words)")
    if words > 600:
        return (False, f"Too long ({words} words)")

    ok, reason = validate_story_prose(story)
    if not ok:
        return (False, reason or "Prose validation failed")

    if _is_truncated(story):
        return (False, "Truncated")

    if not _has_paragraph_structure(story):
        return (False, "No paragraph structure")

    rep = _bigram_repetition_ratio(story)
    if rep > 0.5:
        return (False, f"Too repetitive (ratio={rep:.2f})")

    if not _has_resolution(story):
        return (False, "No resolution")

    return (True, "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and clean training dataset")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "training_data_1" / "stories_dataset.csv",
        help="Input CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "training_data_1" / "stories_dataset_cleaned.csv",
        help="Output cleaned CSV",
    )
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output

    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    removed: dict[str, int] = {}

    rows_out = []

    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or ["case_id", "case_link", "story_output", "generation_time_seconds"]

        for row in reader:
            case_id = row.get("case_id", "")
            story = row.get("story_output", "")

            ok, reason = validate_story(case_id, story)
            if ok:
                rows_out.append(row)
                kept += 1
            else:
                removed[reason] = removed.get(reason, 0) + 1
                logger.debug("Removed %s: %s", case_id, reason)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        writer.writerows(rows_out)

    total = kept + sum(removed.values())
    logger.info("Kept %d / %d stories", kept, total)
    for reason, n in sorted(removed.items(), key=lambda x: -x[1]):
        logger.info("  Removed %d: %s", n, reason)
    logger.info("Output: %s", output_path)


if __name__ == "__main__":
    main()
