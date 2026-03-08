from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from oldbailey.db.sqlite import (
    cases_by_offence,
    connect,
    get_case_speeches,
    init_db,
    offences_summary,
    search_speeches,
    stats,
)
from oldbailey.pipeline.ingest import ingest as ingest_pipeline
from oldbailey.web import create_app


app = typer.Typer(add_completion=False, help="Old Bailey → SQLite (FTS5) ingestion + search.")
console = Console()


@app.command()
def ingest(
    obo_xml: Path | None = typer.Option(
        None,
        "--obo-xml",
        help="Path to Old Bailey XML directory (ingests cases).",
    ),
    obv2_zip: Path | None = typer.Option(
        None,
        "--obv2-zip",
        help="Path to Old Bailey Voices OBV2 ZIP (ingests speeches).",
    ),
    db: Path = typer.Option(
        ...,
        "--db",
        help="Path to SQLite database file to create/update.",
    ),
) -> None:
    if not obo_xml and not obv2_zip:
        console.print("[red]Provide at least one of --obo-xml or --obv2-zip[/red]")
        raise typer.Exit(code=1)

    def progress(kind: str, n: int) -> None:
        console.print(f"  {kind}: {n}...", end="\r")

    progress_file = Path(str(db) + ".ingest_progress.json")

    try:
        summary = ingest_pipeline(
            obo_xml_path=obo_xml,
            obv2_zip_path=obv2_zip,
            db_path=db,
            progress_cb=progress,
            progress_file=progress_file,
        )
    except Exception as e:
        console.print(f"[red]Ingest failed:[/red] {e}")
        raise typer.Exit(code=1) from e

    console.print()
    console.print("[bold]Ingest summary[/bold]")
    console.print(f"  Cases ingested:  {summary['cases_ingested']}")
    console.print(f"  Speeches ingested: {summary['speeches_ingested']}")
    console.print(f"  Total cases in DB:  {summary['cases_total']}")
    console.print(f"  Total speeches in DB: {summary['speeches_total']}")


@app.command("serve")
def serve_cmd(
    db: Path = typer.Option(..., "--db", help="SQLite database path."),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(5000, "--port", help="Bind port."),
    backend_url: str | None = typer.Option(
        None, "--backend-url", help="FastAPI backend URL for Generate Legal Fiction (default: http://127.0.0.1:8000)."
    ),
) -> None:
    """Start web interface to browse cases by offence."""
    flask_app = create_app(db, backend_url=backend_url)
    init_db(connect(db))
    console.print(f"Starting server at http://{host}:{port}/")
    flask_app.run(host=host, port=port, debug=False)


@app.command("stats")
def stats_cmd(
    db: Path = typer.Option(..., "--db", help="SQLite DB path.")
) -> None:
    conn = connect(db)
    try:
        init_db(conn)
        s = stats(conn)
    finally:
        conn.close()

    table = Table(title="oldbailey stats")
    table.add_column("table")
    table.add_column("rows", justify="right")
    table.add_row("cases", str(s["cases"]))
    table.add_row("speeches", str(s["speeches"]))
    console.print(table)


@app.command("offences")
def offences_cmd(
    db: Path = typer.Option(..., "--db", help="SQLite DB path."),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Maximum number of offence categories to show (default: all).",
    ),
) -> None:
    """
    Show offence categories with case counts, ordered by frequency.
    """
    conn = connect(db)
    try:
        init_db(conn)
        rows = offences_summary(conn, limit=limit)
    finally:
        conn.close()

    table = Table(title="offences (by case count)")
    table.add_column("offence_category", no_wrap=True)
    table.add_column("cases", justify="right", no_wrap=True)

    for r in rows:
        table.add_row(str(r["offence_category"] or ""), str(r["case_count"]))

    console.print(table)


@app.command("cases")
def cases_cmd(
    db: Path = typer.Option(..., "--db", help="SQLite DB path."),
    offence: str | None = typer.Option(
        None,
        "--offence",
        help="Filter by offence_category (use '*' for all, omit for NULL only).",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Maximum number of cases to show (default: all).",
    ),
) -> None:
    """
    List cases, optionally filtered by offence_category.
    """
    conn = connect(db)
    try:
        init_db(conn)
        rows = cases_by_offence(conn, offence_category=offence, limit=limit)
    finally:
        conn.close()

    if not rows:
        console.print("[yellow]No cases found for the given criteria.[/yellow]")
        raise typer.Exit(code=0)

    table = Table(title="cases")
    table.add_column("case_id", no_wrap=True)
    table.add_column("date_iso", no_wrap=True)
    table.add_column("year", justify="right", no_wrap=True)
    table.add_column("offence_category", no_wrap=True)

    for r in rows:
        table.add_row(
            str(r["case_id"]),
            str(r["date_iso"] or ""),
            str(r["year"] or ""),
            str(r["offence_category"] or ""),
        )

    console.print(table)


@app.command()
def search(
    db: Path = typer.Option(..., "--db", help="SQLite DB path."),
    query: str = typer.Argument(..., help="FTS5 query (MATCH expression)."),
    limit: int = typer.Option(20, "--limit", min=1, max=5000, help="Max rows to return."),
) -> None:
    conn = connect(db)
    try:
        init_db(conn)
        rows = search_speeches(conn, query=query, limit=limit)
    finally:
        conn.close()

    table = Table(title=f"search: {query!r}")
    table.add_column("case_id", no_wrap=True)
    table.add_column("speaker_name", no_wrap=True)
    table.add_column("speaker_role", no_wrap=True)
    table.add_column("snippet")

    for r in rows:
        table.add_row(
            str(r["case_id"]),
            str(r["speaker_name"] or ""),
            str(r["speaker_role"] or ""),
            str(r["snippet"] or ""),
        )

    console.print(table)


@app.command("case")
def case_cmd(
    db: Path = typer.Option(..., "--db", help="SQLite DB path."),
    case_id: str = typer.Argument(..., help="Case identifier (cases.case_id)."),
    limit: int = typer.Option(200, "--limit", min=1, max=10000, help="Max speeches to show."),
) -> None:
    conn = connect(db)
    try:
        init_db(conn)
        rows = get_case_speeches(conn, case_id=case_id, limit=limit)
    finally:
        conn.close()

    if not rows:
        console.print(f"[red]No speeches found[/red] for case_id={case_id!r}.")
        raise typer.Exit(code=1)

    table = Table(title=f"case: {case_id}")
    table.add_column("sequence", justify="right", no_wrap=True)
    table.add_column("speaker", no_wrap=True)
    table.add_column("text")

    for r in rows:
        table.add_row(str(r["speech_no"]), str(r["speaker_name"] or ""), str(r["text"]))

    console.print(table)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

