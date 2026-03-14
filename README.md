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
- **Generate Legal Fiction**: on each case page (top and bottom). Requires the FastAPI backend on port 8000.

**If the Generate Legal Fiction button doesn't appear**, the app may be using cached templates from an installed package. Use the run script instead: `./run_flask.sh` (see below).

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

### Run everything (single machine)

Run the full stack (Flask case browser + FastAPI + LLM) on one computer:

1. **Data setup** — both databases must exist:
   - **Both at once** (stop servers first): `./scripts/ingest_all.sh`
   - Or individually:
     - `old_bailey.db` (FastAPI, Generate Legal Fiction): `./scripts/ingest_fastapi_data.sh`
     - `oldbailey.sqlite` (Flask browse): `./scripts/ingest_flask_data.sh` — **stop servers first**
   If migrating from a remote setup: copy `old_bailey.db` and `oldbailey.sqlite` from the other machine.

2. **Start Ollama** (in a separate terminal):
   ```bash
   ollama serve
   ollama pull smollm2:360m   # or your preferred model
   ```

3. **Start Flask + FastAPI**:
   ```bash
   ./run_local.sh
   # or: ./run_both.sh   (run_local.sh checks that Ollama is running first)
   ```

   - Flask (case browser): http://127.0.0.1:5000/
   - FastAPI (stories, generate): http://127.0.0.1:8000/

No `GENERATE_BACKEND_URL` needed — Flask uses `http://127.0.0.1:8000` by default.

**Split across two computers?** See [REMOTE_GENERATION.md](REMOTE_GENERATION.md).

### Run Ollama

1. Install [Ollama](https://ollama.com) and start the server: `ollama serve`
2. Pull a model. Default is `smollm2:360m` (optimised for 8 GB RAM); for 16 GB see below:
   ```bash
   ollama pull smollm2:360m
   ```
3. Optional env vars: `OLLAMA_MODEL`, `OLLAMA_NUM_CTX`, `PROMPT_FULL_TEXT_TRUNCATE` (see below).

### Run the app

From the project root (where `old_bailey.db` and `app/` live):

```bash
uvicorn app.main:app --reload
```

For 8 GB RAM (e.g. Surface Book, i5):

```bash
OLLAMA_MODEL=smollm2:360m OLLAMA_NUM_CTX=2048 uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/

- **/** — Searchable cases list (case_id, year, offence, verdict).
- **/case/{case_id}** — Case card (offences, people, places, outcome, full text) and “Generate Legal Fiction” form. Stories are stored in SQLite and shown on the same page. **V0 frontend:** If the button does nothing, see [V0_INTEGRATION.md](V0_INTEGRATION.md) to wire it up.

### Suggested models

**8 GB RAM (e.g. Surface Book, i5):**

- SmolLM2-360M (default)
- Qwen2.5-1.5B
- TinyLlama
- Phi-3.5 Mini

**16 GB RAM:**

- Mistral 7B Instruct
- Qwen2.5 7B Instruct
- Llama 3.1 8B Instruct

**Fine-tuned (after training):**

- `payphone-story` — custom QLoRA model for 400–600 word dark legal fiction. Set `OLLAMA_MODEL=payphone-story`. See [training/README.md](training/README.md) for export steps.

### Environment variables

| Variable | Default | Description |
|----------|---------|--------------|
| `OLLAMA_MODEL` | `smollm2:360m` | Model name for generation |
| `OLLAMA_NUM_CTX` | (unset) | Context window size; use `2048` on 8 GB to reduce RAM |
| `PROMPT_FULL_TEXT_TRUNCATE` | `5000` | Max case text chars in prompt; raise to `8000` on 16 GB |
| `OLD_BAILEY_DB_PATH` | `./old_bailey.db` | Database path |
| `VERCEL_DEPLOY_HOOK_URL` | (unset) | Vercel deploy hook; pinged after each story to rebuild your V0 site |

### Optional: Vercel deploy hook (for V0 frontend)

After each successful story generation, the app writes the story to `generated/stories/` and can trigger a Vercel rebuild so your V0 site picks up new content. See **[DEPLOY_HOOK_SETUP.md](DEPLOY_HOOK_SETUP.md)** for step-by-step instructions.

The app supports `VERCEL_DEPLOY_HOOK_URL` or `DEPLOY_HOOK_URL`; both work the same.

### Optional: llama.cpp server

If `LLAMA_CPP_BASE_URL` is set (e.g. `http://localhost:8080`), the app uses that for generation instead of Ollama.

### Troubleshooting

- **Ollama not reachable**: Start the daemon with `ollama serve`.
- **Model not found**: Run `ollama pull <model>` (e.g. `ollama pull smollm2:360m`).
- **8 GB RAM / slow or OOM**: Use `OLLAMA_MODEL=smollm2:360m`, `OLLAMA_NUM_CTX=2048`, close other apps.
- Test Ollama from the command line:
  ```bash
  curl http://localhost:11434/api/generate -d '{"model":"smollm2:360m","prompt":"Hello","stream":false}'
  ```

Database path defaults to `./old_bailey.db`; override with `OLD_BAILEY_DB_PATH`.

---

## Notes

- SQLite must be compiled with **FTS5** enabled (most modern distros ship it).
- The Old Bailey XML sample corpus in this workspace was moved to `data/oldbailey_xml/` to avoid a name collision with the Python package `oldbailey`.

