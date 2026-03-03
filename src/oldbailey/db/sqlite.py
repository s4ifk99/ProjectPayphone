from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from oldbailey.model.schema import Case, Speech


def connect(db_path: str | Path) -> sqlite3.Connection:
    """
    Open a SQLite connection suitable for ingestion + search.

    Notes:
    - Enables foreign keys.
    - Uses WAL mode for better concurrent read performance.
    """

    path = str(db_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """
    Create core tables + indexes + FTS5 virtual table (with triggers).

    Raises RuntimeError if SQLite was compiled without FTS5 support.
    """

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cases (
          case_id TEXT PRIMARY KEY,
          session_date TEXT,
          date_iso TEXT,
          year INTEGER,
          court TEXT,
          offence_category TEXT,
          xml_path TEXT,
          source TEXT,
          metadata_json TEXT NOT NULL DEFAULT '{}',
          meta_json TEXT NOT NULL DEFAULT '{}'
        );
        """
    )
    # Migrate existing DBs: add new columns if missing
    for col, typ in [
        ("date_iso", "TEXT"),
        ("year", "INTEGER"),
        ("court", "TEXT"),
        ("offence_category", "TEXT"),
        ("xml_path", "TEXT"),
        ("meta_json", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE cases ADD COLUMN {col} {typ};")
        except sqlite3.OperationalError:
            pass  # column already exists

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS speeches (
          id INTEGER PRIMARY KEY,
          case_id TEXT NOT NULL,
          speech_no INTEGER NOT NULL,
          speaker_id TEXT,
          speaker_name TEXT,
          text TEXT NOT NULL,
          source TEXT,
          metadata_json TEXT NOT NULL DEFAULT '{}',
          FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE
        );
        """
    )

    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS speeches_case_order_uq
        ON speeches(case_id, speech_no);
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS speeches_case_idx
        ON speeches(case_id);
        """
    )

    # FTS5 setup (external content table).
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS speeches_fts
            USING fts5(
              text,
              content='speeches',
              content_rowid='id',
              tokenize='unicode61'
            );
            """
        )
    except sqlite3.OperationalError as e:  # pragma: no cover
        raise RuntimeError(
            "SQLite FTS5 is required but is not available in this Python/SQLite build."
        ) from e

    # Triggers to keep speeches_fts synchronized with speeches.
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS speeches_ai
        AFTER INSERT ON speeches BEGIN
          INSERT INTO speeches_fts(rowid, text) VALUES (new.id, new.text);
        END;
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS speeches_ad
        AFTER DELETE ON speeches BEGIN
          INSERT INTO speeches_fts(speeches_fts, rowid, text)
          VALUES ('delete', old.id, old.text);
        END;
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS speeches_au
        AFTER UPDATE ON speeches BEGIN
          INSERT INTO speeches_fts(speeches_fts, rowid, text)
          VALUES ('delete', old.id, old.text);
          INSERT INTO speeches_fts(rowid, text) VALUES (new.id, new.text);
        END;
        """
    )


def upsert_case(conn: sqlite3.Connection, case: Case) -> None:
    conn.execute(
        """
        INSERT INTO cases(
          case_id,
          session_date,
          date_iso,
          year,
          court,
          offence_category,
          xml_path,
          source,
          metadata_json,
          meta_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(case_id) DO UPDATE SET
          session_date=excluded.session_date,
          date_iso=excluded.date_iso,
          year=excluded.year,
          court=excluded.court,
          offence_category=excluded.offence_category,
          xml_path=excluded.xml_path,
          source=excluded.source,
          metadata_json=excluded.metadata_json,
          meta_json=excluded.meta_json;
        """,
        (
            case.case_id,
            case.date_iso,
            case.date_iso,
            case.year,
            case.court,
            case.offence_category,
            case.xml_path or "",
            case.source,
            json.dumps(case.metadata, ensure_ascii=False),
            case.meta_json,
        ),
    )


def insert_speech(conn: sqlite3.Connection, speech: Speech) -> int:
    cur = conn.execute(
        """
        INSERT INTO speeches(case_id, speech_no, speaker_id, speaker_name, text, source, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(case_id, speech_no) DO UPDATE SET
          speaker_id=excluded.speaker_id,
          speaker_name=excluded.speaker_name,
          text=excluded.text,
          source=excluded.source,
          metadata_json=excluded.metadata_json;
        """,
        (
            speech.case_id,
            speech.speech_no,
            speech.speaker_id,
            speech.speaker_name,
            speech.text,
            speech.source,
            json.dumps(speech.metadata, ensure_ascii=False),
        ),
    )
    return int(cur.lastrowid)


def bulk_insert_speeches(conn: sqlite3.Connection, speeches: Iterable[Speech]) -> int:
    rows: list[tuple[Any, ...]] = []
    for s in speeches:
        rows.append(
            (
                s.case_id,
                s.speech_no,
                s.speaker_id,
                s.speaker_name,
                s.text,
                s.source,
                json.dumps(s.metadata, ensure_ascii=False),
            )
        )

    conn.executemany(
        """
        INSERT INTO speeches(case_id, speech_no, speaker_id, speaker_name, text, source, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        rows,
    )
    return len(rows)


def stats(conn: sqlite3.Connection) -> dict[str, int]:
    cases = int(conn.execute("SELECT COUNT(*) FROM cases;").fetchone()[0])
    speeches = int(conn.execute("SELECT COUNT(*) FROM speeches;").fetchone()[0])
    return {"cases": cases, "speeches": speeches}


def offences_summary(conn: sqlite3.Connection, limit: int | None = None) -> list[sqlite3.Row]:
    """
    Return offence_category counts, ordered by count desc (NULLs last).
    """
    sql = """
        SELECT offence_category, COUNT(*) AS case_count
        FROM cases
        GROUP BY offence_category
        ORDER BY case_count DESC, offence_category IS NULL, offence_category
    """
    if limit is not None:
        sql += " LIMIT ?"
        rows = conn.execute(sql, (limit,)).fetchall()
    else:
        rows = conn.execute(sql).fetchall()
    return list(rows)


def cases_by_offence(
    conn: sqlite3.Connection, offence_category: str | None, limit: int | None = None
) -> list[sqlite3.Row]:
    """
    List cases, optionally filtered by offence_category.
    """
    params: list[Any] = []
    where = ""
    if offence_category is None:
        where = "WHERE offence_category IS NULL"
    elif offence_category != "*":
        where = "WHERE offence_category = ?"
        params.append(offence_category)

    sql = f"""
        SELECT case_id, date_iso, year, offence_category
        FROM cases
        {where}
        ORDER BY year, date_iso, case_id
    """
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(sql, tuple(params)).fetchall()
    return list(rows)


def search_speeches(
    conn: sqlite3.Connection, query: str, limit: int = 20
) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT
              s.case_id,
              s.speech_no,
              s.speaker_name,
              COALESCE(json_extract(s.metadata_json, '$.role'), '') AS speaker_role,
              snippet(speeches_fts, 0, '[', ']', '…', 12) AS snippet
            FROM speeches_fts
            JOIN speeches AS s ON speeches_fts.rowid = s.id
            WHERE speeches_fts MATCH ?
            ORDER BY bm25(speeches_fts)
            LIMIT ?;
            """,
            (query, limit),
        ).fetchall()
    )


def get_case(conn: sqlite3.Connection, case_id: str) -> sqlite3.Row | None:
    """Return a single case row by case_id, or None if not found."""
    row = conn.execute(
        """
        SELECT case_id, date_iso, year, court, offence_category, xml_path, source, meta_json
        FROM cases
        WHERE case_id = ?;
        """,
        (case_id,),
    ).fetchone()
    return row


def get_case_speeches(
    conn: sqlite3.Connection, case_id: str, limit: int = 200
) -> list[sqlite3.Row]:
    """Return speeches for a case, ordered by speech_no (sequence)."""
    return list(
        conn.execute(
            """
            SELECT speech_no, speaker_name, text
            FROM speeches
            WHERE case_id = ?
            ORDER BY speech_no
            LIMIT ?;
            """,
            (case_id, limit),
        ).fetchall()
    )

