# oldbailey

Python library + CLI to ingest:

- Old Bailey XML files (case / trial metadata)
- Old Bailey Voices OBV2 (TSV inside ZIP) (speech/utterance text or word-level rows)

…into a SQLite database with:

- `cases` table indexed by `case_id`
- `speeches` table (ordered per `case_id`)
- FTS5 full-text search over `speeches.text`

Parsers are **not implemented yet** (stubs with TODOs exist); the database schema and CLI are in place.

## Install (editable)

Using `uv` (recommended):

```bash
uv pip install -e .
```

Or with `pip`:

```bash
python3 -m pip install -e .
```

## CLI

Create/init a database (schema + FTS5). Ingestion will currently fail with a clear TODO error until parsers are implemented.

```bash
oldbailey ingest --obo-xml data/oldbailey_xml/sessionsPapers --obv2-zip path/to/obv2.zip --db oldbailey.sqlite
```

Show basic DB stats:

```bash
oldbailey stats --db oldbailey.sqlite
```

Full-text search (FTS5):

```bash
oldbailey search --db oldbailey.sqlite "knife NEAR/5 pocket" --limit 20
```

Show ordered speeches for a case:

```bash
oldbailey case --db oldbailey.sqlite t17800628-12 --limit 200
```

Browse cases by offence (CLI):

```bash
oldbailey offences --db oldbailey.sqlite
oldbailey cases --db oldbailey.sqlite --offence violentTheft
```

## Web interface

Start a local web server to browse cases by criminal offence in your browser:

```bash
oldbailey serve --db oldbailey.sqlite
```

Then open http://127.0.0.1:5000/

- **Home**: lists offence categories with case counts (sorted by frequency).
- Click **View cases** for an offence to see cases (sorted by year/date).
- Click **View** on a case to see its metadata and ordered speeches.

Optional: `--host` and `--port` to change bind address and port.

## Library usage

```python
from oldbailey.db.sqlite import connect, init_db, upsert_case, insert_speech
from oldbailey.model.schema import Case, Speech

conn = connect("oldbailey.sqlite")
init_db(conn)

upsert_case(conn, Case(case_id="t00000000-1", metadata={"note": "example"}))
insert_speech(conn, Speech(case_id="t00000000-1", speech_no=1, text="Hello world."))
conn.commit()
```

## Old Bailey Case Blog + Legal Fiction Generator

A separate FastAPI app in `app/` uses the **old_bailey.db** database (from `ingest_old_bailey.py`) to list cases, show case details, and generate legal fiction stories via a **local LLM** (Ollama or optional llama.cpp). No cloud APIs.

### Install

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Run Ollama

1. Install [Ollama](https://ollama.com) and start the server: `ollama serve`
2. Pull a model, e.g.: `ollama pull mistral:7b-instruct`
3. Optional: set `OLLAMA_MODEL=mistral:7b-instruct` (default) or another model name.

### Run the app

From the project root (where `old_bailey.db` and `app/` live):

```bash
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/

- **/** — Searchable cases list (case_id, year, offence, verdict).
- **/case/{case_id}** — Case card (offences, people, places, outcome, full text) and “Generate Legal Fiction” form. Stories are stored in SQLite and shown on the same page.

### Suggested models (Intel i7, 16GB RAM)

- Mistral 7B Instruct (GGUF Q4_K_M)
- Qwen2.5 7B Instruct (GGUF Q4_K_M)
- Llama 3.1 8B Instruct (GGUF Q4_K_M)
- Faster fallback: Phi-3.5 Mini or Qwen2.5 3B

### Optional: llama.cpp server

If `LLAMA_CPP_BASE_URL` is set (e.g. `http://localhost:8080`), the app uses that for generation instead of Ollama.

### Troubleshooting

- **Ollama not reachable**: Start the daemon with `ollama serve`.
- **Model not found**: Run `ollama pull <model>` (e.g. `ollama pull mistral:7b-instruct`).
- Test Ollama from the command line:
  ```bash
  curl http://localhost:11434/api/generate -d '{"model":"mistral:7b-instruct","prompt":"Hello","stream":false}'
  ```

Database path defaults to `./old_bailey.db`; override with `OLD_BAILEY_DB_PATH`.

---

## Notes

- SQLite must be compiled with **FTS5** enabled (most modern distros ship it).
- The Old Bailey XML sample corpus in this workspace was moved to `data/oldbailey_xml/` to avoid a name collision with the Python package `oldbailey`.

