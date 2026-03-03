"""
Flask app for browsing Old Bailey cases by criminal offence.

DB path is set via app.config["DB_PATH"] when creating the app (e.g. from CLI).
One SQLite connection per request.
"""

from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request

from oldbailey.db.sqlite import (
    cases_by_offence,
    connect,
    get_case,
    get_case_speeches,
    init_db,
    offences_summary,
    stats,
)

# Special path segment for NULL/empty offence_category in URLs
OFFENCE_NONE_SLUG = "_none"


def create_app(db_path: str | Path) -> Flask:
    """Create Flask app with db_path stored in config."""
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
        static_url_path="/static",
    )
    app.config["DB_PATH"] = str(Path(db_path).resolve())

    @app.after_request
    def _cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    @app.route("/api/offences", methods=["GET"])
    def api_offences():
        """List offence categories with case counts (JSON)."""
        limit = request.args.get("limit", type=int)
        conn = get_conn()
        try:
            rows = offences_summary(conn, limit=limit)
            out = []
            for r in rows:
                cat = r["offence_category"]
                slug = OFFENCE_NONE_SLUG if (cat is None or cat == "") else cat
                out.append({
                    "offence_category": cat,
                    "slug": slug,
                    "case_count": int(r["case_count"]),
                })
            return jsonify(out)
        finally:
            conn.close()

    @app.route("/api/offences/<path:offence_slug>/cases", methods=["GET"])
    def api_cases_for_offence(offence_slug: str):
        """List cases for one offence (JSON)."""
        offence_category = None if offence_slug == OFFENCE_NONE_SLUG else offence_slug
        limit = request.args.get("limit", type=int)
        conn = get_conn()
        try:
            rows = cases_by_offence(conn, offence_category=offence_category, limit=limit)
            out = []
            for r in rows:
                out.append({
                    "case_id": r["case_id"],
                    "date_iso": r["date_iso"],
                    "year": r["year"],
                    "offence_category": r["offence_category"],
                })
            return jsonify(out)
        finally:
            conn.close()

    @app.route("/api/cases/<path:case_id>", methods=["GET"])
    def api_case_detail(case_id: str):
        """Single case with metadata and speeches (JSON)."""
        conn = get_conn()
        try:
            case = get_case(conn, case_id)
            if case is None:
                abort(404)
            speeches = get_case_speeches(conn, case_id=case_id, limit=500)
            case_subtitles = {}
            offence_category = case["offence_category"] if "offence_category" in case.keys() else None
            meta_json = case["meta_json"] if "meta_json" in case.keys() else None
            if meta_json:
                try:
                    parsed = json.loads(meta_json)
                    if parsed:
                        interp = parsed.get("interp") or {}
                        subtitles = parsed.get("subtitles") or {}
                        cat = (
                            case["offence_category"]
                            if "offence_category" in case.keys()
                            else None
                        ) or interp.get("offenceCategory")
                        subcat = interp.get("offenceSubcategory")
                        if isinstance(cat, list):
                            cat = cat[0] if cat else None
                        if isinstance(subcat, list):
                            subcat = subcat[0] if subcat else None
                        if cat:
                            offence_str = f"{cat} ({subcat})" if subcat else str(cat)
                            case_subtitles["Offence"] = offence_str
                        if offence_category is None and cat is not None:
                            offence_category = cat
                        if subtitles.get("defendants"):
                            case_subtitles["Defendant(s)"] = subtitles["defendants"]
                        if subtitles.get("victims"):
                            case_subtitles["Victim"] = subtitles["victims"]
                        if subtitles.get("place"):
                            case_subtitles["Place"] = subtitles["place"]
                        if subtitles.get("offence_description"):
                            case_subtitles["Offence description"] = subtitles["offence_description"]
                        if subtitles.get("verdict"):
                            case_subtitles["Verdict"] = subtitles["verdict"]
                        if subtitles.get("punishment"):
                            case_subtitles["Punishment"] = subtitles["punishment"]
                except json.JSONDecodeError:
                    pass
            offence_slug = (
                OFFENCE_NONE_SLUG
                if (offence_category is None or offence_category == "")
                else offence_category
            )
            rows = cases_by_offence(conn, offence_category=offence_category, limit=None)
            case_ids = [r["case_id"] for r in rows]
            try:
                idx = case_ids.index(case_id)
            except ValueError:
                idx = -1
            prev_case_id = case_ids[idx - 1] if idx > 0 else None
            next_case_id = case_ids[idx + 1] if idx >= 0 and idx + 1 < len(case_ids) else None
            speeches_list = [
                {"speech_no": s["speech_no"], "speaker_name": s["speaker_name"], "text": s["text"]}
                for s in speeches
            ]
            return jsonify({
                "case_id": case["case_id"],
                "date_iso": case["date_iso"],
                "year": case["year"],
                "court": case["court"],
                "offence_category": case["offence_category"],
                "case_subtitles": case_subtitles,
                "speeches": speeches_list,
                "offence_slug": offence_slug,
                "prev_case_id": prev_case_id,
                "next_case_id": next_case_id,
            })
        finally:
            conn.close()

    def get_conn():
        conn = connect(app.config["DB_PATH"])
        try:
            init_db(conn)
            return conn
        except Exception:
            conn.close()
            raise

    @app.route("/")
    def index():
        conn = get_conn()
        try:
            rows = offences_summary(conn, limit=None)
            # Normalize for templates: use _none slug for NULL category in links
            categories = []
            for r in rows:
                cat = r["offence_category"]
                slug = OFFENCE_NONE_SLUG if (cat is None or cat == "") else cat
                categories.append({"offence_category": cat, "slug": slug, "case_count": r["case_count"]})
            return render_template("index.html", categories=categories)
        finally:
            conn.close()

    @app.route("/offences/<path:offence_slug>/cases")
    def cases_for_offence(offence_slug: str):
        # Map _none back to None for DB
        offence_category = None if offence_slug == OFFENCE_NONE_SLUG else offence_slug
        conn = get_conn()
        try:
            rows = cases_by_offence(conn, offence_category=offence_category, limit=None)
            display_name = "(unknown)" if offence_category is None or offence_category == "" else offence_category
            return render_template(
                "cases.html",
                offence_category=display_name,
                offence_slug=offence_slug,
                cases=rows,
            )
        finally:
            conn.close()

    @app.route("/api/status")
    def api_status():
        """Return ingest progress (if running) or DB stats. For status bar polling."""
        progress_path = Path(app.config["DB_PATH"] + ".ingest_progress.json")
        if progress_path.exists():
            try:
                data = json.loads(progress_path.read_text(encoding="utf-8"))
                return jsonify({
                    "ingesting": True,
                    "phase": data.get("phase", "unknown"),
                    "count": data.get("count", 0),
                    "started_at": data.get("started_at"),
                })
            except (json.JSONDecodeError, OSError):
                pass
        conn = get_conn()
        try:
            s = stats(conn)
            return jsonify({
                "ingesting": False,
                "cases": s["cases"],
                "speeches": s["speeches"],
            })
        finally:
            conn.close()

    @app.route("/cases/<path:case_id>")
    def case_detail(case_id: str):
        conn = get_conn()
        try:
            case = get_case(conn, case_id)
            if case is None:
                abort(404)
            speeches = get_case_speeches(conn, case_id=case_id, limit=500)
            case_subtitles = {}
            offence_display_name = "(unknown)"
            offence_category = case["offence_category"] if "offence_category" in case.keys() else None
            meta_json = case["meta_json"] if "meta_json" in case.keys() else None
            if meta_json:
                try:
                    parsed = json.loads(meta_json)
                    if parsed:
                        interp = parsed.get("interp") or {}
                        subtitles = parsed.get("subtitles") or {}
                        cat = (
                            case["offence_category"]
                            if "offence_category" in case.keys()
                            else None
                        ) or interp.get("offenceCategory")
                        subcat = interp.get("offenceSubcategory")
                        if isinstance(cat, list):
                            cat = cat[0] if cat else None
                        if isinstance(subcat, list):
                            subcat = subcat[0] if subcat else None
                        if cat:
                            offence_str = f"{cat} ({subcat})" if subcat else str(cat)
                            case_subtitles["Offence"] = offence_str
                            offence_display_name = offence_str
                        if offence_category is None and cat is not None:
                            offence_category = cat
                        if subtitles.get("defendants"):
                            case_subtitles["Defendant(s)"] = subtitles["defendants"]
                        if subtitles.get("victims"):
                            case_subtitles["Victim"] = subtitles["victims"]
                        if subtitles.get("place"):
                            case_subtitles["Place"] = subtitles["place"]
                        if subtitles.get("offence_description"):
                            case_subtitles["Offence description"] = subtitles["offence_description"]
                        if subtitles.get("verdict"):
                            case_subtitles["Verdict"] = subtitles["verdict"]
                        if subtitles.get("punishment"):
                            case_subtitles["Punishment"] = subtitles["punishment"]
                except json.JSONDecodeError:
                    pass
            offence_slug = (
                OFFENCE_NONE_SLUG
                if (offence_category is None or offence_category == "")
                else offence_category
            )
            rows = cases_by_offence(conn, offence_category=offence_category, limit=None)
            case_ids = [r["case_id"] for r in rows]
            try:
                idx = case_ids.index(case_id)
            except ValueError:
                idx = -1
            prev_case_id = case_ids[idx - 1] if idx > 0 else None
            next_case_id = case_ids[idx + 1] if idx >= 0 and idx + 1 < len(case_ids) else None
            return render_template(
                "case_detail.html",
                case=case,
                speeches=speeches,
                case_subtitles=case_subtitles,
                offence_slug=offence_slug,
                offence_display_name=offence_display_name,
                prev_case_id=prev_case_id,
                next_case_id=next_case_id,
            )
        finally:
            conn.close()

    return app
