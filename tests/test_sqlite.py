from __future__ import annotations

from oldbailey.db.sqlite import (
    connect,
    get_case_speeches,
    init_db,
    insert_speech,
    search_speeches,
    stats,
    upsert_case,
)
from oldbailey.model.schema import Case, Speech


def test_init_insert_search(tmp_path):
    db_path = tmp_path / "test.sqlite"
    conn = connect(db_path)
    try:
        init_db(conn)

        upsert_case(conn, Case(case_id="t00000000-1", metadata={"x": 1}))
        insert_speech(conn, Speech(case_id="t00000000-1", speech_no=1, text="Hello world"))
        insert_speech(conn, Speech(case_id="t00000000-1", speech_no=2, text="Another line"))
        conn.commit()

        s = stats(conn)
        assert s["cases"] == 1
        assert s["speeches"] == 2

        rows = search_speeches(conn, query="hello", limit=10)
        assert len(rows) >= 1
        assert rows[0]["case_id"] == "t00000000-1"

        speeches = get_case_speeches(conn, case_id="t00000000-1", limit=10)
        assert [r["speech_no"] for r in speeches] == [1, 2]
    finally:
        conn.close()

