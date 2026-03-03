from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

from oldbailey.db.sqlite import connect, init_db, insert_speech, stats, upsert_case
from oldbailey.io.obo_xml import iter_cases_and_speeches_from_xml_dir
from oldbailey.io.obv2_tsv import iter_speeches_from_obv2_zip
from oldbailey.model.schema import Case

logger = logging.getLogger(__name__)


def _ensure_case_exists(conn, case_id: str) -> None:
    """Insert a minimal case if it does not exist (for OBV2-only ingest)."""
    cur = conn.execute("SELECT 1 FROM cases WHERE case_id = ?", (case_id,))
    if cur.fetchone() is None:
        upsert_case(conn, Case(case_id=case_id, source="obv2"))


def ingest_cases(
    conn,
    xml_root: Path,
    *,
    progress: Callable[[int], None] | None = None,
) -> tuple[int, int]:
    """
    Ingest cases and speeches from a directory tree of Old Bailey XML files.

    Extracts case metadata and paragraph text (as Speech records) in a single
    pass. Returns (cases_ingested, speeches_ingested).
    """
    case_count = 0
    speech_count = 0
    for case, speeches in iter_cases_and_speeches_from_xml_dir(xml_root):
        upsert_case(conn, case)
        case_count += 1
        for speech in speeches:
            insert_speech(conn, speech)
            speech_count += 1
        if progress:
            progress(case_count)
        elif case_count % 500 == 0:
            logger.info("Ingested %d cases, %d speeches...", case_count, speech_count)
    return case_count, speech_count


def ingest_speeches(
    conn,
    obv2_zip_path: Path,
    *,
    progress: Callable[[int], None] | None = None,
) -> int:
    """
    Ingest speeches from an OBV2 ZIP containing a TSV.

    Ensures a case exists for each speech's case_id (creates minimal case if missing).
    Returns the number of speeches ingested.
    """
    count = 0
    for speech in iter_speeches_from_obv2_zip(obv2_zip_path):
        _ensure_case_exists(conn, speech.case_id)
        insert_speech(conn, speech)
        count += 1
        if progress:
            progress(count)
        elif count % 500 == 0:
            logger.info("Ingested %d speeches...", count)
    return count


def ingest(
    *,
    obo_xml_path: str | Path | None,
    obv2_zip_path: str | Path | None,
    db_path: str | Path,
    progress_cb: Callable[[str, int], None] | None = None,
    progress_file: Path | None = None,
) -> dict[str, int]:
    """
    Ingest Old Bailey sources into a SQLite database at `db_path`.

    - init_db if db doesn't exist (connect + init_db are idempotent)
    - ingest_cases from obo_xml_path if it exists
    - ingest_speeches from obv2_zip_path if it exists
    - Returns summary: cases_ingested, speeches_ingested, cases_total, speeches_total
    """
    conn = connect(db_path)
    init_db(conn)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

    def _progress(kind: str, n: int) -> None:
        if progress_cb:
            progress_cb(kind, n)
        if progress_file:
            try:
                progress_file.write_text(
                    json.dumps({
                        "phase": kind,
                        "count": n,
                        "started_at": started_at,
                    }),
                    encoding="utf-8",
                )
            except OSError:
                pass

    _started_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    cases_ingested = 0
    speeches_ingested = 0

    xml_root = Path(obo_xml_path) if obo_xml_path else None
    if xml_root and xml_root.exists():
        cases_ingested, obo_speeches = ingest_cases(
            conn,
            xml_root,
            progress=lambda n: _progress("cases", n),
        )
        speeches_ingested += obo_speeches

    obv2_path = Path(obv2_zip_path) if obv2_zip_path else None
    if obv2_path and obv2_path.exists():
        speeches_ingested = ingest_speeches(
            conn,
            obv2_path,
            progress=lambda n: _progress("speeches", n),
        )

    conn.commit()
    conn.close()

    if progress_file and progress_file.exists():
        try:
            progress_file.unlink()
        except OSError:
            pass

    conn2 = connect(db_path)
    try:
        s = stats(conn2)
    finally:
        conn2.close()

    return {
        "cases_ingested": cases_ingested,
        "speeches_ingested": speeches_ingested,
        "cases_total": s["cases"],
        "speeches_total": s["speeches"],
    }

