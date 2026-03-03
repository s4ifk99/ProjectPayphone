"""Integration test: ingest from fixtures, then search."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from oldbailey.db.sqlite import connect, get_case_speeches, init_db, search_speeches, stats
from oldbailey.pipeline.ingest import ingest

# XML: case t17800628-1 with offenceCategory
XML_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<TEI.2>
  <text><body>
    <div1 type="trialAccount" id="t17800628-1">
      <interp inst="t17800628-1" type="date" value="17800628"/>
      <interp inst="t17800628-1" type="year" value="1780"/>
      <interp inst="t17800628-1" type="offenceCategory" value="violentTheft"/>
    </div1>
  </body></text>
</TEI.2>"""

# OBV2 Format A: speeches for t17800628-1 with searchable words "theft", "guilty"
OBV2_TSV = """case_id\tspeaker\ttext\trow_id
t17800628-1\tJudge\tYou are charged with theft.\t1
t17800628-1\tDefendant\tI plead not guilty.\t2"""


def _make_obv2_zip(tsv: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("obv2.tsv", tsv.encode("utf-8"))
    return buf.getvalue()


def test_ingest_then_search(tmp_path):
    """Ingest XML + OBV2, then search returns FTS5 results."""
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    (xml_dir / "17800628.xml").write_text(XML_FIXTURE, encoding="utf-8")

    obv2_path = tmp_path / "obv2.zip"
    obv2_path.write_bytes(_make_obv2_zip(OBV2_TSV))

    db_path = tmp_path / "test.sqlite"

    summary = ingest(
        obo_xml_path=xml_dir,
        obv2_zip_path=obv2_path,
        db_path=db_path,
    )

    assert summary["cases_ingested"] == 1
    assert summary["speeches_ingested"] == 2
    assert summary["cases_total"] == 1
    assert summary["speeches_total"] == 2

    conn = connect(db_path)
    try:
        init_db(conn)

        # FTS5 search for "theft"
        rows = search_speeches(conn, query="theft", limit=10)
        assert len(rows) >= 1
        assert rows[0]["case_id"] == "t17800628-1"
        assert "theft" in (rows[0]["snippet"] or "").lower()
        assert rows[0]["speaker_name"] == "Judge"

        # FTS5 search for "guilty"
        rows2 = search_speeches(conn, query="guilty", limit=10)
        assert len(rows2) >= 1
        assert rows2[0]["case_id"] == "t17800628-1"

        # case command data: speeches in order
        speeches = get_case_speeches(conn, case_id="t17800628-1", limit=10)
        assert len(speeches) == 2
        assert speeches[0]["speech_no"] == 1
        assert speeches[0]["speaker_name"] == "Judge"
        assert "theft" in speeches[0]["text"]
        assert speeches[1]["speech_no"] == 2
        assert speeches[1]["speaker_name"] == "Defendant"
        assert "guilty" in speeches[1]["text"]
    finally:
        conn.close()
