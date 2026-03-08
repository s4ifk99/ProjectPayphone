"""
Export generated stories to a folder and optionally trigger deploy hook.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OUTPUT_STORIES_DIR = Path(
    os.environ.get("OUTPUT_STORIES_DIR", str(Path(__file__).resolve().parent.parent / "generated" / "stories"))
)
DEPLOY_HOOK_URL = os.environ.get("VERCEL_DEPLOY_HOOK_URL") or os.environ.get("DEPLOY_HOOK_URL", "")


def write_story_to_folder(
    story_id: int,
    case_id: str,
    created_at: str,
    model: str,
    mode: str,
    target_length: str,
    story_markdown: str,
    case_summary: dict[str, Any],
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
