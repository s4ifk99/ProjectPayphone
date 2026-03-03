"""
Database layer for Old Bailey Case Blog.
Uses old_bailey.db (cases table); compatibility layer for schema detection.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

_DEFAULT_DB_PATH = Path("./old_bailey.db")


def get_db_path() -> Path:
    return Path(os.environ.get("OLD_BAILEY_DB_PATH", str(_DEFAULT_DB_PATH)))


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or get_db_path()
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _get_cases_columns(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute("PRAGMA table_info(cases)")
    return [row[1] for row in cur.fetchall()]


def _row_to_case_dict(row: sqlite3.Row, columns: list[str]) -> dict[str, Any]:
    d = dict(row)
    out: dict[str, Any] = {
        "case_id": d.get("case_id"),
        "doc_id": d.get("doc_id"),
        "sequence_in_doc": d.get("sequence_in_doc"),
        "full_text": d.get("full_text") or "",
        "card_json": d.get("card_json"),
    }
    if "page_facsimiles" in columns and d.get("page_facsimiles") is not None:
        out["page_facsimiles_raw"] = d.get("page_facsimiles")
    return out


def init_stories_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stories (
            story_id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            model TEXT NOT NULL,
            mode TEXT NOT NULL,
            target_length TEXT NOT NULL,
            prompt TEXT NOT NULL,
            story_markdown TEXT NOT NULL,
            FOREIGN KEY(case_id) REFERENCES cases(case_id)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_stories_case_created ON stories(case_id, created_at);"
    )
    conn.commit()


def _parse_card_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _card_matches_search(card: dict[str, Any], q: str, full_text: str) -> bool:
    q_lower = q.lower()
    if q_lower in (full_text or "").lower():
        return True
    case_id = (card.get("case_id") or "").lower()
    if q_lower in case_id:
        return True
    year = card.get("year")
    if year is not None and str(year) == q.strip():
        return True
    for o in card.get("offences") or []:
        if isinstance(o, dict):
            for v in (o.get("offence_text"), o.get("offenceCategory"), o.get("offenceSubcategory")):
                if v and q_lower in str(v).lower():
                    return True
    for key in ("defendants", "victims"):
        for p in card.get(key) or []:
            if isinstance(p, dict) and p.get("display_name") and q_lower in str(p.get("display_name")).lower():
                return True
    return False


def list_cases(
    conn: sqlite3.Connection,
    search: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    columns = _get_cases_columns(conn)
    init_stories_table(conn)

    select_cols = ["case_id", "doc_id", "sequence_in_doc", "full_text", "card_json"]
    if "page_facsimiles" in columns:
        select_cols.append("page_facsimiles")
    cols_str = ", ".join(select_cols)

    # Cap rows scanned for search performance
    fetch_limit = 3000 if (search and search.strip()) else limit

    sql = f"SELECT {cols_str} FROM cases ORDER BY case_id LIMIT ?"
    rows = conn.execute(sql, (fetch_limit,)).fetchall()

    if not search or not search.strip():
        return [_row_to_case_dict(row, columns) for row in rows]

    q = search.strip()
    out: list[dict[str, Any]] = []
    for row in rows:
        rec = _row_to_case_dict(row, columns)
        card = _parse_card_json(rec.get("card_json"))
        rec["card"] = card
        if _card_matches_search(card, q, rec.get("full_text") or ""):
            out.append(rec)
            if len(out) >= limit:
                break
    return out


def get_case(conn: sqlite3.Connection, case_id: str) -> dict[str, Any] | None:
    columns = _get_cases_columns(conn)
    select_cols = ["case_id", "doc_id", "sequence_in_doc", "full_text", "card_json"]
    if "page_facsimiles" in columns:
        select_cols.append("page_facsimiles")
    cols_str = ", ".join(select_cols)
    row = conn.execute(
        f"SELECT {cols_str} FROM cases WHERE case_id = ?",
        (case_id,),
    ).fetchone()
    if not row:
        return None
    rec = _row_to_case_dict(row, columns)
    card = _parse_card_json(rec.get("card_json"))
    rec["card"] = card
    if "page_facsimiles_raw" in rec:
        try:
            pf = json.loads(rec["page_facsimiles_raw"]) if isinstance(rec["page_facsimiles_raw"], str) else rec["page_facsimiles_raw"]
            card["page_facsimiles"] = pf if isinstance(pf, list) else card.get("page_facsimiles", [])
        except (json.JSONDecodeError, TypeError):
            pass
        del rec["page_facsimiles_raw"]
    return rec


def insert_story(
    conn: sqlite3.Connection,
    case_id: str,
    model: str,
    mode: str,
    target_length: str,
    prompt: str,
    story_markdown: str,
) -> int:
    from datetime import datetime, timezone
    init_stories_table(conn)
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cur = conn.execute(
        """INSERT INTO stories (case_id, created_at, model, mode, target_length, prompt, story_markdown)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (case_id, created_at, model, mode, target_length, prompt, story_markdown),
    )
    conn.commit()
    return cur.lastrowid or 0


def list_stories_for_case(
    conn: sqlite3.Connection,
    case_id: str,
) -> list[dict[str, Any]]:
    init_stories_table(conn)
    rows = conn.execute(
        """SELECT story_id, case_id, created_at, model, mode, target_length, prompt, story_markdown
           FROM stories WHERE case_id = ? ORDER BY created_at DESC""",
        (case_id,),
    ).fetchall()
    return [dict(row) for row in rows]
