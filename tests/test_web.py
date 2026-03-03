"""Tests for web interface (browse cases by offence)."""

from __future__ import annotations

import json
from pathlib import Path

from oldbailey.db.sqlite import connect, init_db, insert_speech, upsert_case
from oldbailey.model.schema import Case, Speech
from oldbailey.web import create_app


def test_web_offences_index(tmp_path: Path) -> None:
    """GET / returns offence categories with case counts."""
    db_path = tmp_path / "web.sqlite"
    conn = connect(db_path)
    init_db(conn)
    upsert_case(conn, Case(case_id="t1", offence_category="theft", source="test"))
    conn.commit()
    conn.close()

    app = create_app(db_path)
    client = app.test_client()
    rv = client.get("/")
    assert rv.status_code == 200
    assert b"theft" in rv.data
    assert b"1" in rv.data


def test_web_cases_for_offence(tmp_path: Path) -> None:
    """GET /offences/<cat>/cases returns cases for that offence."""
    db_path = tmp_path / "web.sqlite"
    conn = connect(db_path)
    init_db(conn)
    upsert_case(conn, Case(case_id="t1780-1", offence_category="theft", date_iso="1780-06-28", source="test"))
    conn.commit()
    conn.close()

    app = create_app(db_path)
    client = app.test_client()
    rv = client.get("/offences/theft/cases")
    assert rv.status_code == 200
    assert b"t1780-1" in rv.data
    assert b"theft" in rv.data


def test_web_case_detail(tmp_path: Path) -> None:
    """GET /cases/<case_id> returns case metadata and speeches."""
    db_path = tmp_path / "web.sqlite"
    conn = connect(db_path)
    init_db(conn)
    upsert_case(conn, Case(case_id="t2", offence_category="burglary", source="test"))
    insert_speech(conn, Speech(case_id="t2", speech_no=1, text="First speech.", source="test"))
    conn.commit()
    conn.close()

    app = create_app(db_path)
    client = app.test_client()
    rv = client.get("/cases/t2")
    assert rv.status_code == 200
    assert b"t2" in rv.data
    assert b"burglary" in rv.data
    assert b"First speech" in rv.data


def test_web_case_detail_with_subtitles(tmp_path: Path) -> None:
    """Case detail page shows Case summary when meta_json has subtitles."""
    db_path = tmp_path / "web.sqlite"
    conn = connect(db_path)
    init_db(conn)
    meta = {
        "interp": {"offenceCategory": "theft", "offenceSubcategory": "burglary"},
        "subtitles": {
            "defendants": "others",
            "victims": "Mr. Bradbourn",
            "place": "St. Giles in the Fields",
            "verdict": "Guilty",
        },
    }
    upsert_case(
        conn,
        Case(
            case_id="t1674-1",
            offence_category="theft",
            meta_json=json.dumps(meta),
            source="test",
        ),
    )
    conn.commit()
    conn.close()

    app = create_app(db_path)
    client = app.test_client()
    rv = client.get("/cases/t1674-1")
    assert rv.status_code == 200
    assert b"Case details" in rv.data
    assert b"Mr. Bradbourn" in rv.data
    assert b"Guilty" in rv.data


def test_web_case_404(tmp_path: Path) -> None:
    """GET /cases/<unknown> returns 404."""
    db_path = tmp_path / "web.sqlite"
    conn = connect(db_path)
    init_db(conn)
    conn.close()

    app = create_app(db_path)
    client = app.test_client()
    rv = client.get("/cases/nonexistent-case-id")
    assert rv.status_code == 404


def test_api_offences(tmp_path: Path) -> None:
    """GET /api/offences returns JSON list of offence categories."""
    db_path = tmp_path / "web.sqlite"
    conn = connect(db_path)
    init_db(conn)
    upsert_case(conn, Case(case_id="t1", offence_category="theft", source="test"))
    conn.commit()
    conn.close()

    app = create_app(db_path)
    client = app.test_client()
    rv = client.get("/api/offences")
    assert rv.status_code == 200
    data = rv.get_json()
    assert isinstance(data, list)
    assert any(item.get("slug") == "theft" and item.get("case_count") == 1 for item in data)


def test_api_offences_slug_cases(tmp_path: Path) -> None:
    """GET /api/offences/<slug>/cases returns JSON list of cases."""
    db_path = tmp_path / "web.sqlite"
    conn = connect(db_path)
    init_db(conn)
    upsert_case(conn, Case(case_id="t1780-1", offence_category="theft", date_iso="1780-06-28", source="test"))
    conn.commit()
    conn.close()

    app = create_app(db_path)
    client = app.test_client()
    rv = client.get("/api/offences/theft/cases")
    assert rv.status_code == 200
    data = rv.get_json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["case_id"] == "t1780-1"
    assert data[0]["offence_category"] == "theft"


def test_api_case_detail(tmp_path: Path) -> None:
    """GET /api/cases/<case_id> returns JSON case with speeches."""
    db_path = tmp_path / "web.sqlite"
    conn = connect(db_path)
    init_db(conn)
    upsert_case(conn, Case(case_id="t2", offence_category="burglary", source="test"))
    insert_speech(conn, Speech(case_id="t2", speech_no=1, text="First speech.", source="test"))
    conn.commit()
    conn.close()

    app = create_app(db_path)
    client = app.test_client()
    rv = client.get("/api/cases/t2")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["case_id"] == "t2"
    assert data["offence_category"] == "burglary"
    assert "speeches" in data
    assert len(data["speeches"]) == 1
    assert data["speeches"][0]["text"] == "First speech."
    assert "prev_case_id" in data
    assert "next_case_id" in data
