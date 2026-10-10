# Gilto

Upload PDFs and chat with them. Answers come only from your files, cite `[filename, p.N]`, and every citation is checked against the passages that were retrieved. When the files do not contain the answer, Gilto says so.

Stack: Python, Claude API, multilingual-e5-small embeddings, Chroma, Streamlit. See `SPEC.md` for the full design. This is slice 1 (single user, text-layer PDFs).

## Setup

Needs Python 3.11+ and an Anthropic API key.

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env
```

Put your key in `.env` as `ANTHROPIC_API_KEY=...`. The first run downloads the embedding model (about 470 MB).

## Run

```bash
streamlit run app.py
```

Upload PDFs in the sidebar, click **Index files**, then ask questions.

## Test

```bash
pytest
```

Claude calls are mocked in the tests, so they need no API key.

## Configuration

All settings are environment variables. See `.env.example` for the full list and defaults.

- `CLAUDE_MODEL`: answers. `CLAUDE_FAST_MODEL`: follow-up question rewriting.
- `SIMILARITY_THRESHOLD` (default 0.80): passages scoring below this are dropped. If nothing passes, Gilto answers "I couldn't find that in the uploaded documents." without calling Claude. Tuned on a small sample; raise it if unrelated questions get answers, lower it if good questions get refused.
- Upload limits: `MAX_FILE_MB`, `MAX_PAGES_PER_FILE`, `MAX_DOCS_PER_USER`, `MAX_TOTAL_PAGES`.

## Limits of slice 1

- Text-layer PDFs only. Pages without text are skipped with a warning. OCR comes in slice 2.
- One local user (`local`). Login, per-user isolation, and quotas come in slice 3.
- Document text is sent to Anthropic's API to answer questions.
