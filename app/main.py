"""
Old Bailey Case Blog + Local Legal Fiction Generator.
FastAPI app: cases list, case detail, generate story via local LLM.
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import json
from pathlib import Path
import markdown
from pydantic import BaseModel

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import db, llm, models, prompts, story_export

app = FastAPI(title="Old Bailey Case Blog")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Bleach allow list for Markdown-rendered HTML
try:
    import bleach
    ALLOWED_TAGS = [
        "p", "br", "strong", "em", "b", "i", "u", "h1", "h2", "h3", "h4", "h5", "h6",
        "ul", "ol", "li", "blockquote", "code", "pre", "a", "hr",
    ]
    ALLOWED_ATTRS = {"a": ["href", "title"]}

    def sanitize_html(html: str) -> str:
        return bleach.clean(html, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)
except ImportError:
    def sanitize_html(html: str) -> str:
        return html
        # Fallback: could use html.escape on the whole string if no bleach
        # import html
        # return html.escape(html).replace('\n', '<br>\n')


def markdown_to_safe_html(md: str) -> str:
    if not md:
        return ""
    html = markdown.markdown(md, extensions=["nl2br", "sane_lists"])
    return sanitize_html(html)


def get_conn():
    conn = db.connect()
    try:
        yield conn
    finally:
        conn.close()


def _date_from_doc_id(doc_id: str) -> str | None:
    """Derive YYYY-MM-DD from doc_id like '16740429'."""
    if not doc_id or len(doc_id) < 8:
        return None
    s = str(doc_id)[:8]
    if s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return None


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Home: offence categories. Browse by offence or view all cases."""
    conn = db.connect()
    try:
        categories = db.offences_summary(conn)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "categories": categories},
        )
    finally:
        conn.close()


@app.get("/cases", response_class=HTMLResponse)
def cases_list(request: Request, q: str | None = None, limit: int = 500):
    """All cases with optional search."""
    conn = db.connect()
    try:
        rows = db.list_cases(conn, search=q, limit=limit)
        cases = []
        for row in rows:
            card = row.get("card") or {}
            case_id = row.get("case_id") or ""
            doc_id = card.get("doc_id") or row.get("doc_id") or ""
            date_iso = _date_from_doc_id(doc_id)
            cases.append({
                "case_id": case_id,
                "date_iso": date_iso,
                "year": card.get("year"),
                "primary_offence_label": models.primary_offence_label(card),
                "verdict_category": models.verdict_category_label(card),
            })
        return templates.TemplateResponse(
            "cases.html",
            {"request": request, "cases": cases, "search_query": q or ""},
        )
    finally:
        conn.close()


@app.get("/offences/{offence_slug:path}/cases", response_class=HTMLResponse)
def cases_by_offence(request: Request, offence_slug: str):
    """Cases for one offence category."""
    conn = db.connect()
    try:
        # Map slug back to display name; _unknown -> (unknown)
        offence_display = "(unknown)" if offence_slug in ("_unknown", "_none") else offence_slug.replace("-", " ")
        rows = db.list_cases_by_offence(conn, offence_slug, limit=None)
        cases = []
        for row in rows:
            card = row.get("card") or {}
            case_id = row.get("case_id") or ""
            doc_id = card.get("doc_id") or row.get("doc_id") or ""
            date_iso = _date_from_doc_id(doc_id)
            cases.append({
                "case_id": case_id,
                "date_iso": date_iso,
                "year": card.get("year"),
                "primary_offence_label": models.primary_offence_label(card),
                "verdict_category": models.verdict_category_label(card),
            })
        return templates.TemplateResponse(
            "cases.html",
            {
                "request": request,
                "cases": cases,
                "search_query": "",
                "offence_display": offence_display,
                "offence_slug": offence_slug,
            },
        )
    finally:
        conn.close()


@app.get("/case/{case_id:path}", response_class=HTMLResponse)
def case_detail(request: Request, case_id: str, message: str | None = None, error: str | None = None):
    conn = db.connect()
    try:
        row = db.get_case(conn, case_id)
        if not row:
            return templates.TemplateResponse(
                "case_detail.html",
                {"request": request, "case": None, "case_id": case_id},
                status_code=404,
            )
        card, card_valid = models.normalize_case_row(row)
        stories = db.list_stories_for_case(conn, case_id)
        latest_story_html = None
        if stories:
            latest = stories[0]
            latest_story_html = markdown_to_safe_html(latest.get("story_markdown") or "")
        older_stories = stories[1:] if len(stories) > 1 else []
        # Prev/next and similar cases (by offence)
        offence_slug = db.offence_slug_for_card(card)
        offence_display = db._offence_from_card(card)
        same_offence = db.list_cases_by_offence(conn, offence_slug, limit=None)
        case_ids = [r["case_id"] for r in same_offence]
        try:
            idx = case_ids.index(case_id)
        except ValueError:
            idx = -1
        prev_case_id = case_ids[idx - 1] if idx > 0 else None
        next_case_id = case_ids[idx + 1] if idx >= 0 and idx + 1 < len(case_ids) else None
        return templates.TemplateResponse(
            "case_detail.html",
            {
                "request": request,
                "case": row,
                "card": card,
                "card_valid": card_valid,
                "case_id": case_id,
                "stories": stories,
                "latest_story_html": latest_story_html,
                "older_stories": older_stories,
                "message": message,
                "error": error,
                "prev_case_id": prev_case_id,
                "next_case_id": next_case_id,
                "offence_slug": offence_slug,
                "offence_display": offence_display,
            },
        )
    finally:
        conn.close()


@app.post("/case/{case_id:path}/generate")
def generate(request: Request, case_id: str, mode: str = Form("courtroom_focused"), target_length: str = Form("800-1200"), model_override: str | None = Form(None)):
    conn = db.connect()
    try:
        row = db.get_case(conn, case_id)
        if not row:
            return RedirectResponse(url=f"/case/{case_id}?error=Case+not+found", status_code=302)
        card, _ = models.normalize_case_row(row)
        full_text = row.get("full_text") or ""
        prompt = prompts.build_story_prompt(card, full_text, mode, target_length)
        try:
            story_markdown = llm.generate_story(prompt, model_override)
        except llm.LLMError as e:
            return RedirectResponse(url=f"/case/{case_id}?error={_quote(str(e))}", status_code=302)
        compliant, _ = prompts.validate_story_has_twelve_stages(story_markdown)
        model_name = model_override or (llm.OLLAMA_MODEL if not llm.LLAMA_CPP_BASE_URL else "llama.cpp")
        story_id, created_at = db.insert_story(
            conn,
            case_id=case_id,
            model=model_name,
            mode=mode,
            target_length=target_length,
            prompt=prompt,
            story_markdown=story_markdown,
        )
        case_summary = {
            "year": card.get("year"),
            "primary_offence": models.primary_offence_label(card),
            "verdict": models.verdict_category_label(card),
        }
        try:
            story_export.write_story_to_folder(
                story_id=story_id,
                case_id=case_id,
                created_at=created_at,
                model=model_name,
                mode=mode,
                target_length=target_length,
                story_markdown=story_markdown,
                case_summary=case_summary,
            )
            story_export.trigger_deploy_hook()
        except Exception:
            pass  # do not fail the request if export/deploy fails
        return RedirectResponse(url=f"/case/{case_id}?message=Story+generated.", status_code=302)
    finally:
        conn.close()


class GenerateRequestBody(BaseModel):
    mode: str = "courtroom_focused"
    target_length: str = "800-1200"
    model_override: str | None = None


@app.post("/api/case/{case_id:path}/generate")
def api_generate(case_id: str, body: GenerateRequestBody | None = None):
    """
    JSON API for generating legal fiction. Use this from V0 / SPA frontends.
    Returns JSON and allows showing progress while the request is in flight.
    """
    b = body or GenerateRequestBody()
    mode = b.mode
    target_length = b.target_length
    model_override = b.model_override
    conn = db.connect()
    try:
        row = db.get_case(conn, case_id)
        if not row:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "Case not found"},
            )
        card, _ = models.normalize_case_row(row)
        full_text = row.get("full_text") or ""
        prompt = prompts.build_story_prompt(card, full_text, mode, target_length)
        try:
            story_markdown = llm.generate_story(prompt, model_override)
        except llm.LLMError as e:
            return JSONResponse(
                status_code=502,
                content={"success": False, "error": str(e)},
            )
        compliant, _ = prompts.validate_story_has_twelve_stages(story_markdown)
        model_name = model_override or (llm.OLLAMA_MODEL if not llm.LLAMA_CPP_BASE_URL else "llama.cpp")
        story_id, created_at = db.insert_story(
            conn,
            case_id=case_id,
            model=model_name,
            mode=mode,
            target_length=target_length,
            prompt=prompt,
            story_markdown=story_markdown,
        )
        case_summary = {
            "year": card.get("year"),
            "primary_offence": models.primary_offence_label(card),
            "verdict": models.verdict_category_label(card),
        }
        try:
            story_export.write_story_to_folder(
                story_id=story_id,
                case_id=case_id,
                created_at=created_at,
                model=model_name,
                mode=mode,
                target_length=target_length,
                story_markdown=story_markdown,
                case_summary=case_summary,
            )
            story_export.trigger_deploy_hook()
        except Exception:
            pass
        return JSONResponse(content={
            "success": True,
            "story_id": story_id,
            "case_id": case_id,
            "created_at": created_at,
            "compliant": compliant,
            "message": "Story generated." if compliant else "Story saved (some Hero's Journey headings missing).",
        })
    finally:
        conn.close()


def _quote(s: str) -> str:
    from urllib.parse import quote
    return quote(s, safe="")


# ---------- JSON API for V0 ----------

@app.get("/api/stories")
def api_list_stories(limit: int = 100, offset: int = 0):
    """List story summaries (story_id, case_id, created_at, model, mode, case_summary)."""
    conn = db.connect()
    try:
        rows = db.list_all_stories(conn, limit=limit, offset=offset)
        out = []
        for r in rows:
            case_summary = {
                "year": None,
                "primary_offence": None,
                "verdict": None,
            }
            # Try to load case_summary from exported file if present
            try:
                path = story_export.OUTPUT_STORIES_DIR / f"{r['story_id']}.json"
                if path.exists():
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                        case_summary = data.get("case_summary") or case_summary
            except Exception:
                pass
            out.append({
                "story_id": r["story_id"],
                "case_id": r["case_id"],
                "created_at": r["created_at"],
                "model": r["model"],
                "mode": r["mode"],
                "case_summary": case_summary,
            })
        return {"stories": out}
    finally:
        conn.close()


@app.get("/api/stories/{story_id:int}")
def api_get_story(story_id: int):
    """Full story; read from folder if present, else from DB."""
    path = story_export.OUTPUT_STORIES_DIR / f"{story_id}.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    conn = db.connect()
    try:
        row = db.get_story(conn, story_id)
        if not row:
            raise HTTPException(status_code=404, detail="Not found")
        return {
            "story_id": row["story_id"],
            "case_id": row["case_id"],
            "created_at": row["created_at"],
            "model": row["model"],
            "mode": row["mode"],
            "target_length": row["target_length"],
            "story_markdown": row["story_markdown"],
            "case_summary": {"year": None, "primary_offence": None, "verdict": None},
        }
    finally:
        conn.close()


@app.get("/api/cases/{case_id:path}/stories")
def api_list_stories_for_case(case_id: str):
    """List stories for one case."""
    conn = db.connect()
    try:
        rows = db.list_stories_for_case(conn, case_id)
        out = []
        for r in rows:
            case_summary = {"year": None, "primary_offence": None, "verdict": None}
            try:
                path = story_export.OUTPUT_STORIES_DIR / f"{r['story_id']}.json"
                if path.exists():
                    with open(path, encoding="utf-8") as f:
                        case_summary = json.load(f).get("case_summary") or case_summary
            except Exception:
                pass
            out.append({
                "story_id": r["story_id"],
                "case_id": r["case_id"],
                "created_at": r["created_at"],
                "model": r["model"],
                "mode": r["mode"],
                "case_summary": case_summary,
            })
        return {"stories": out}
    finally:
        conn.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
