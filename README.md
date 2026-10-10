# Gilto

Upload PDFs and chat with them. Answers come only from your files, cite `[filename, p.N]`, and every citation is checked against the passages that were retrieved. When the files do not contain the answer, Gilto says so.

Stack: Python, Gemini API, multilingual-e5-small embeddings, Chroma, Streamlit. See `SPEC.md` for the full design. This is slice 1 (single user, text-layer PDFs).

## Setup

Needs Python 3.11 to 3.13 and a Gemini API key. Create a key at https://aistudio.google.com/apikey.

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv/bin/activate on macOS/Linux
python -m pip install -r requirements.txt
cp .env.example .env
```

Put your key in `.env` as `GEMINI_API_KEY=...`. Never put it in `.env.example` or any other tracked file. The first run downloads the embedding model (about 470 MB).

## Run

```bash
python -m streamlit run app.py
```

Upload PDFs in the sidebar, click **Add to library**, then ask questions.

## Test

```bash
python -m pytest
```

Gemini calls are mocked in the tests, so they need no API key.

## Configuration

All settings are environment variables. See `.env.example` for the full list and defaults.

- `GEMINI_MODEL`: answers. `GEMINI_FAST_MODEL`: follow-up question rewriting. `GEMINI_FALLBACK_MODEL`: used when the main model keeps returning transient errors (500/503/504). Gemini is sometimes overloaded, so answers retry with backoff before falling back.
- `SIMILARITY_THRESHOLD` (default 0.80): passages scoring below this are dropped. If nothing passes, Gilto answers "I couldn't find that in the uploaded documents." without calling Gemini. Tuned on a small sample; raise it if unrelated questions get answers, lower it if good questions get refused.
- Upload limits: `MAX_FILE_MB`, `MAX_PAGES_PER_FILE`, `MAX_DOCS_PER_USER`, `MAX_TOTAL_PAGES`.

## Limits of slice 1

- Text-layer PDFs only. Pages without text are skipped with a warning. OCR comes in slice 2.
- One local user (`local`). Login, per-user isolation, and quotas come in slice 3.
- Document text is sent to Google's Gemini API to answer questions. On Google's free tier, submitted content may be used to improve Google products (check Google's current terms), so use a paid-tier key for confidential documents.
- A free-tier key has low rate limits. If you see "rate limit or quota was reached", wait a minute and try again.
