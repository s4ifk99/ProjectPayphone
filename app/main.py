"""
Old Bailey Case Blog + Local Legal Fiction Generator.
FastAPI app: cases list, case detail, generate story via local LLM.
"""
from __future__ import annotations

from pathlib import Path

import markdown
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import db, llm, models, prompts

app = FastAPI(title="Old Bailey Case Blog")

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


@app.get("/", response_class=HTMLResponse)
def index(request: Request, q: str | None = None, limit: int = 500):
    conn = db.connect()
    try:
        rows = db.list_cases(conn, search=q, limit=limit)
        cases = []
        for row in rows:
            card = row.get("card") or {}
            case_id = row.get("case_id") or ""
            cases.append({
                "case_id": case_id,
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
        db.insert_story(
            conn,
            case_id=case_id,
            model=model_name,
            mode=mode,
            target_length=target_length,
            prompt=prompt,
            story_markdown=story_markdown,
        )
        if compliant:
            return RedirectResponse(url=f"/case/{case_id}?message=Story+generated.", status_code=302)
        return RedirectResponse(
            url=f"/case/{case_id}?message=Story+saved+but+missing+some+Hero's+Journey+headings.",
            status_code=302,
        )
    finally:
        conn.close()


def _quote(s: str) -> str:
    from urllib.parse import quote
    return quote(s, safe="")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
