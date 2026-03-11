"""
Export generated stories to a folder and optionally trigger deploy hook.
Includes provenance metadata (source excerpt, factual anchors).
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OUTPUT_STORIES_DIR = Path(
    os.environ.get("OUTPUT_STORIES_DIR", str(Path(__file__).resolve().parent.parent / "generated" / "stories"))
)
DEPLOY_HOOK_URL = os.environ.get("VERCEL_DEPLOY_HOOK_URL") or os.environ.get("DEPLOY_HOOK_URL", "")


def extract_provenance(card: dict[str, Any], full_text: str) -> dict[str, Any]:
    """
    Extract provenance metadata from case card and full text.
    Returns dict with source_excerpt (<=120 words) and factual_anchors (6-12 facts).
    """
    provenance: dict[str, Any] = {
        "case_id": card.get("case_id"),
        "doc_id": card.get("doc_id"),
        "year": card.get("year"),
        "source_excerpt": "",
        "factual_anchors": [],
    }

    # Source excerpt: first 120 words of full_text
    words = (full_text or "").split()
    provenance["source_excerpt"] = " ".join(words[:120]) if words else ""

    # Factual anchors from card and full_text
    anchors: list[str] = []

    for o in (card.get("offences") or []):
        if isinstance(o, dict):
            for k in ("offence_text", "offenceCategory", "offenceSubcategory"):
                v = o.get(k)
                if v and str(v).strip():
                    anchors.append(str(v).strip())

    for persons_key in ("defendants", "victims"):
        for p in (card.get(persons_key) or []):
            if isinstance(p, dict) and p.get("display_name"):
                anchors.append(str(p["display_name"]).strip())

    for v in (card.get("verdicts") or []):
        if isinstance(v, dict):
            for k in ("verdict_text", "verdictCategory"):
                val = v.get(k)
                if val and str(val).strip():
                    anchors.append(str(val).strip())

    for p in (card.get("punishments") or []):
        if isinstance(p, dict):
            for k in ("punishment_text", "punishmentCategory"):
                val = p.get(k)
                if val and str(val).strip():
                    anchors.append(str(val).strip())

    for place in (card.get("places") or [])[:5]:
        if isinstance(place, str) and place.strip():
            anchors.append(place.strip())
        elif isinstance(place, dict) and place.get("display_name"):
            anchors.append(str(place["display_name"]).strip())

    # Add amounts and dates from full_text
    for m in re.finditer(r"£\s*\d+[\d.,\s]*(?:s\.|d\.)?", full_text or ""):
        anchors.append(m.group(0).strip())
    for m in re.finditer(r"\b(1[6-9]\d{2}|19[0-4]\d)\b", full_text or ""):
        anchors.append(m.group(0))

    # Dedupe and cap at 12
    seen: set[str] = set()
    unique: list[str] = []
    for a in anchors:
        if a and a not in seen and len(unique) < 12:
            seen.add(a)
            unique.append(a)

    provenance["factual_anchors"] = unique
    return provenance


def write_story_to_folder(
    story_id: int,
    case_id: str,
    created_at: str,
    model: str,
    mode: str,
    target_length: str,
    story_markdown: str,
    case_summary: dict[str, Any],
    provenance: dict[str, Any] | None = None,
) -> Path:
    """
    Write a story to OUTPUT_STORIES_DIR as {story_id}.json.
    Creates the directory if missing. Returns the path written.
    """
    OUTPUT_STORIES_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_STORIES_DIR / f"{story_id}.json"
    payload: dict[str, Any] = {
        "story_id": story_id,
        "case_id": case_id,
        "created_at": created_at,
        "model": model,
        "mode": mode,
        "target_length": target_length,
        "story_markdown": story_markdown,
        "case_summary": case_summary,
    }
    if provenance:
        payload["provenance"] = provenance
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def trigger_deploy_hook() -> None:
    """If DEPLOY_HOOK_URL is set, POST to it (e.g. Vercel). Log and ignore failures."""
    if not DEPLOY_HOOK_URL or not DEPLOY_HOOK_URL.strip():
        return
    try:
        import httpx
        r = httpx.post(DEPLOY_HOOK_URL.strip(), timeout=10.0)
        if r.is_success:
            logger.info("Deploy hook triggered successfully.")
        else:
            logger.warning("Deploy hook returned %s: %s", r.status_code, r.text[:200])
    except Exception as e:
        logger.warning("Deploy hook failed: %s", e)
