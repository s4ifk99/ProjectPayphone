#!/usr/bin/env python3
"""
Generate 500 dark historical legal fiction stories from Old Bailey cases.

Uses Ollama (llama3.1:8b) with custom sampling options, validates word count
(400-600), records generation time, and saves to training_data_1/stories_dataset.csv.

Usage:
  python scripts/generate_training_stories.py [--db old_bailey.db] [--workers 4] [--limit 500]

Prerequisites:
  - ollama serve
  - ollama pull llama3.1:8b
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import httpx

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_TIMEOUT = 1200.0  # 20 min per story
CASE_LINK_BASE = "https://www.oldbaileyonline.org/browse.jsp?id="

PROMPT_TEMPLATE = """Write a 400–600 word historical legal fiction story.

Requirements:
• dark tone
• continuous prose
• inspired by the Hero's Journey narrative arc
• preserve the historical facts

CASE RECORD: {case_summary}"""


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


def fetch_random_cases(db_path: Path, limit: int) -> list[dict]:
    """Query random cases with full_text > 500 chars."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT case_id, full_text, card_json
        FROM cases
        WHERE length(full_text) > 500
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()

    out = []
    for row in rows:
        card = {}
        if row["card_json"]:
            try:
                card = json.loads(row["card_json"])
            except json.JSONDecodeError:
                pass
        out.append({
            "case_id": row["case_id"],
            "full_text": row["full_text"] or "",
            "card": card,
        })
    return out


def generate_story(prompt: str) -> str:
    """Call Ollama /api/generate with custom options."""
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.85,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_ctx": 2048,
        },
    }
    with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
        r = client.post(url, json=payload)
        if r.status_code == 404:
            raise RuntimeError(
                f"Model {OLLAMA_MODEL} not found. Run: ollama pull {OLLAMA_MODEL}"
            )
        r.raise_for_status()
        data = r.json()
    response = data.get("response")
    if response is None:
        raise RuntimeError("Ollama returned no response text.")
    return str(response).strip()


def word_count(text: str) -> int:
    """Count words in text."""
    return len((text or "").split())


def worker(
    case: dict,
    csv_path: Path,
    csv_lock: threading.Lock,
    header_written: list[bool],
) -> tuple[str, float | None, str | None]:
    """
    Generate one story and append to CSV.
    Returns (case_id, generation_time_seconds, error_message).
    """
    case_id = case["case_id"]
    case_link = f"{CASE_LINK_BASE}{case_id}"
    summary = build_case_summary(case["card"], case["full_text"])
    prompt = PROMPT_TEMPLATE.format(case_summary=summary)

    for attempt in range(2):  # initial + one retry if word count fails
        try:
            start = time.perf_counter()
            story = generate_story(prompt)
            elapsed = time.perf_counter() - start

            words = word_count(story)
            if 400 <= words <= 600:
                with csv_lock:
                    write_header = not header_written[0]
                    with open(csv_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
                        if write_header:
                            writer.writerow(
                                [
                                    "case_id",
                                    "case_link",
                                    "story_output",
                                    "generation_time_seconds",
                                ]
                            )
                            header_written[0] = True
                        writer.writerow([case_id, case_link, story, round(elapsed, 2)])
                logger.info(
                    "Case %s: %d words, %.1fs",
                    case_id,
                    words,
                    elapsed,
                )
                return (case_id, elapsed, None)
            else:
                logger.warning(
                    "Case %s: word count %d outside 400-600, retrying (%s)",
                    case_id,
                    words,
                    "retry" if attempt == 0 else "giving up",
                )
                if attempt == 1:
                    # Use anyway for dataset (or skip); plan says regenerate once
                    with csv_lock:
                        write_header = not header_written[0]
                        with open(csv_path, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
                            if write_header:
                                writer.writerow(
                                    [
                                        "case_id",
                                        "case_link",
                                        "story_output",
                                        "generation_time_seconds",
                                    ]
                                )
                                header_written[0] = True
                            writer.writerow(
                                [case_id, case_link, story, round(elapsed, 2)]
                            )
                    logger.info(
                        "Case %s: kept with %d words after retry",
                        case_id,
                        words,
                    )
                    return (case_id, elapsed, None)

        except Exception as e:
            logger.exception("Case %s failed: %s", case_id, e)
            return (case_id, None, str(e))

    return (case_id, None, "Word count validation failed after retry")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training stories from Old Bailey cases"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=PROJECT_ROOT / "old_bailey.db",
        help="Path to old_bailey.db",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of stories to generate",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    if not args.db.exists():
        logger.error("Database not found: %s", args.db)
        sys.exit(1)

    # Check Ollama
    try:
        r = httpx.get(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags",
            timeout=5.0,
        )
        r.raise_for_status()
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        logger.error("Ollama not reachable: %s. Start with: ollama serve", e)
        sys.exit(1)

    cases = fetch_random_cases(args.db, args.limit)
    if not cases:
        logger.error("No cases found with full_text > 500")
        sys.exit(1)

    logger.info("Fetched %d random cases", len(cases))

    output_dir = PROJECT_ROOT / "training_data_1"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "stories_dataset.csv"

    # Truncate CSV for fresh run
    if csv_path.exists():
        csv_path.unlink()

    csv_lock = threading.Lock()
    header_written = [False]

    ok = 0
    failed = 0
    total_time = 0.0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                worker,
                case,
                csv_path,
                csv_lock,
                header_written,
            ): case
            for case in cases
        }
        for future in as_completed(futures):
            case_id, elapsed, err = future.result()
            if err:
                failed += 1
            else:
                ok += 1
                if elapsed is not None:
                    total_time += elapsed

    logger.info(
        "Done: %d succeeded, %d failed. Total time: %.1fs. Output: %s",
        ok,
        failed,
        total_time,
        csv_path,
    )


if __name__ == "__main__":
    main()
