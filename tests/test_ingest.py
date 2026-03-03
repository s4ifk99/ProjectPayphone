"""Tests for ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

from oldbailey.db.sqlite import connect, init_db, stats
from oldbailey.pipeline.ingest import ingest, ingest_cases

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI.2>
  <text><body>
    <div1 type="trialAccount" id="t12345678-1">
      <interp inst="t12345678-1" type="date" value="17800628"/>
      <interp inst="t12345678-1" type="year" value="1780"/>
      <interp inst="t12345678-1" type="offenceCategory" value="theft"/>
    </div1>
  </body></text>
</TEI.2>"""

SAMPLE_XML_WITH_P = """<?xml version="1.0" encoding="UTF-8"?>
<TEI.2>
  <text><body>
    <div1 type="trialAccount" id="t12345678-2">
      <interp inst="t12345678-2" type="date" value="17800628"/>
      <interp inst="t12345678-2" type="offenceCategory" value="theft"/>
      <p>The prisoner was indicted for stealing a watch.</p>
      <p>The jury found him guilty.</p>
    </div1>
  </body></text>
</TEI.2>"""


def test_ingest_full(tmp_path):
    """Full ingest() with XML dir populates DB."""
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    (xml_dir / "17800628.xml").write_text(SAMPLE_XML, encoding="utf-8")
    db_path = tmp_path / "db.sqlite"

    summary = ingest(obo_xml_path=xml_dir, obv2_zip_path=None, db_path=db_path)

    assert summary["cases_ingested"] == 1
    assert summary["cases_total"] == 1
    conn = connect(db_path)
    try:
        s = stats(conn)
        assert s["cases"] == 1
    finally:
        conn.close()


def test_ingest_cases(tmp_path):
    """ingest_cases inserts cases from XML dir and returns count."""
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    (xml_dir / "17800628.xml").write_text(SAMPLE_XML, encoding="utf-8")

    db_path = tmp_path / "test.sqlite"
    conn = connect(db_path)
    try:
        from oldbailey.db.sqlite import init_db

        init_db(conn)
        cases_n, speeches_n = ingest_cases(conn, xml_dir)
        conn.commit()

        assert cases_n == 1
        assert speeches_n == 0  # SAMPLE_XML has no <p> in trialAccount
        s = stats(conn)
        assert s["cases"] == 1

        row = conn.execute(
            "SELECT case_id, date_iso, year, offence_category FROM cases"
        ).fetchone()
        assert row[0] == "t12345678-1"
        assert row[1] == "1780-06-28"
        assert row[2] == 1780
        assert row[3] == "theft"
    finally:
        conn.close()


def test_ingest_cases_with_speeches(tmp_path):
    """ingest_cases extracts paragraph text as Speech records."""
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    (xml_dir / "with_p.xml").write_text(SAMPLE_XML_WITH_P, encoding="utf-8")

    db_path = tmp_path / "test2.sqlite"
    conn = connect(db_path)
    try:
        init_db(conn)
        cases_n, speeches_n = ingest_cases(conn, xml_dir)
        conn.commit()

        assert cases_n == 1
        assert speeches_n == 2
        rows = conn.execute(
            "SELECT case_id, speech_no, text FROM speeches ORDER BY speech_no"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0][2] == "The prisoner was indicted for stealing a watch."
        assert rows[1][2] == "The jury found him guilty."
    finally:
        conn.close()
