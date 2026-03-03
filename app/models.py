"""
Normalize case row to card dict; safe access for prompts and templates.
"""
from __future__ import annotations

from typing import Any


def normalize_case_row(row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """
    Build a case card dict from a DB row (with 'card' and 'full_text').
    Returns (card_for_prompt, card_json_valid).
    card_json_valid is False if card_json failed to parse.
    """
    card = row.get("card") if isinstance(row.get("card"), dict) else {}
    full_text = row.get("full_text") or ""
    card_json_valid = "card" in row and isinstance(row.get("card"), dict)

    # Merge so we have a single dict for prompt/template with all optional keys
    out: dict[str, Any] = {
        "case_id": card.get("case_id") or row.get("case_id"),
        "doc_id": card.get("doc_id") or row.get("doc_id"),
        "year": card.get("year"),
        "offences": card.get("offences") or [],
        "defendants": card.get("defendants") or [],
        "victims": card.get("victims") or [],
        "verdicts": card.get("verdicts") or [],
        "punishments": card.get("punishments") or [],
        "places": card.get("places") or [],
        "page_facsimiles": card.get("page_facsimiles") or [],
        "full_text": full_text,
    }
    return out, card_json_valid


def primary_offence_label(card: dict[str, Any]) -> str:
    """First offence text or category for display."""
    offences = card.get("offences") or []
    if not offences:
        return "—"
    o = offences[0] if isinstance(offences[0], dict) else {}
    return (
        (o.get("offence_text") or o.get("offenceCategory") or o.get("offenceSubcategory")) or "—"
    )


def verdict_category_label(card: dict[str, Any]) -> str:
    """First verdict category or text for display."""
    verdicts = card.get("verdicts") or []
    if not verdicts:
        return "—"
    v = verdicts[0] if isinstance(verdicts[0], dict) else {}
    return (v.get("verdictCategory") or v.get("verdict_text")) or "—"
