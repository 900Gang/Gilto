# Gilto

Upload PDFs and chat with them. Answers come only from your files, cite `[filename, p.N]`, and every citation is checked against the passages that were retrieved. When the files do not contain the answer, Gilto says so.

Stack: Python, Gemini API (answers and embeddings), Chroma, Streamlit. See `SPEC.md` for the full design. It reads text and scanned PDFs and images, supports Google sign-in with private libraries and saved chats, applies usage limits, combines vector and keyword search, summarizes whole documents, and ships with a Docker setup for Render.

## Setup

Needs Python 3.11 to 3.13 and a Gemini API key. Create a key at https://aistudio.google.com/apikey.

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv/bin/activate on macOS/Linux
python -m pip install -r requirements.txt
cp .env.example .env
```

Put your key in `.env` as `GEMINI_API_KEY=...`. Never put it in `.env.example` or any other tracked file. Embeddings also come from the Gemini API, so no model is downloaded and almost no memory is needed. To run the embedding model on your own machine instead, set `EMBEDDING_PROVIDER=local`, install `requirements-local.txt` and keep about 1.2 GB of RAM free.

## Run

```bash
python -m streamlit run app.py
```

Tick the consent box (first time only), upload PDFs or images (PNG, JPG) in the sidebar, click **Add to library**, then ask questions. Chats are saved and listed in the sidebar.

## Searching and summarizing

- **Hybrid search.** Each question runs a vector search and a keyword search, and the two ranked lists are merged. Keyword search finds exact terms such as invoice numbers, names and codes that vector search can miss. A passage found only by keyword shows "keyword match" in Sources instead of a relevance score. A question about something that is not in your files still gets "I couldn't find that in the uploaded documents." Set `HYBRID_SEARCH=false` for vector search only.
- **Summaries.** Ask to "summarize this document", "give me an overview", "what are the main points" or "compare ..." and Gilto reads the whole document in parts, then answers with citations. With two or more documents, a **Search in** picker appears in the sidebar. Leave it empty to use all documents. A request is limited to 4 documents and about 100 pages, and above that Gilto asks you to pick fewer. A summary makes several Gemini calls, so it uses more of your daily token allowance than a normal question and can take a while. Summaries have no Sources list because they draw on every page.

## Users and sign-in

Gilto uses Google sign-in. Each user has a private library and private chats, and only emails listed in `ALLOWED_EMAILS` can get in. An empty list lets nobody in.

1. Create a Google OAuth client (type Web application) in Google Cloud Console. Add `http://localhost:8501/oauth2callback` as an authorized redirect URI.
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in `client_id`, `client_secret` and a random `cookie_secret`. This file is ignored by git. Never commit it.
3. Set `ALLOWED_EMAILS=you@gmail.com,friend@gmail.com` in `.env`.

For local testing without Google, set `AUTH_DISABLED=true` in `.env`. That skips sign-in and uses one shared user called `local`. Never use it on a server.

**Limits** (set to 0 to turn one off): `HOURLY_QUESTION_LIMIT` (30), `DAILY_TOKEN_LIMIT` (200000) and an app-wide `MONTHLY_TOKEN_CAP` (5000000). Tokens are estimated at 3 characters each, and vision OCR calls are not counted.

**Data:** users can delete their documents, chats and usage history from **Account** in the sidebar. Accounts not seen for `INACTIVE_DAYS` (90) are deleted when the server starts. Chats and accounts live in `gilto.db`.

## Scanned PDFs and images

Pages without a text layer, and every image, go through OCR. Tesseract runs first (local and free). When Tesseract is missing, unsure (confidence below `OCR_CONFIDENCE_THRESHOLD`), finds under 50 characters, or reads Hindi (Devanagari) text, the page goes to Gemini vision instead. Hindi always goes to vision because Tesseract reports high confidence on Hindi even when its words are wrong. Without a Gemini key, Hindi pages stay with Tesseract and are flagged as low confidence. Each source in the answer shows how its page was read: `text`, `ocr`, or `vision`.

- **Gemini vision only (no install).** Works with just your API key. Each scanned page is one Gemini request, so a long scan on a free-tier key can hit rate limits. Results are cached by file hash and page in `ocr_cache/`, so a retry never pays twice.
- **With Tesseract (recommended for long scans).** Install it, plus the `eng` and `hin` language data. On Windows use the installer from https://github.com/UB-Mannheim/tesseract/wiki and tick Hindi under additional language data. Gilto finds `C:\Program Files\Tesseract-OCR\tesseract.exe` by itself. Set `TESSERACT_CMD` if it is installed elsewhere.
- Handwriting is not supported. Page images read by Gemini vision are sent to Google's Gemini API.

## Test

```bash
python -m pytest
```

Gemini calls are mocked in the tests, so they need no API key.

### Answer-quality evaluation

This runs real questions against the real Gemini API, so it is skipped in the normal run. It needs your `GEMINI_API_KEY`. Each run asks Gemini about 10 questions and embeds a few pages, so it uses a little of your quota.

```bash
python -m pytest -m eval -s
```

It indexes a document, asks each question, and checks that answers contain the expected words with a citation to the right page, and that out-of-scope questions are refused. The bar is 90% of answerable questions correct and 100% of out-of-scope questions refused. It also checks that an instruction hidden inside a document is not followed. With no `eval/cases.json` it uses a small built-in sample.

To test your own documents, put PDFs in `eval/docs/`, copy `eval/cases.example.json` to `eval/cases.json`, and list your documents and questions. Both paths are ignored by git. Set `EVAL_DELAY_SECONDS` (default 4) higher if you hit rate limits.

## Deploy to Render

The repo includes a `Dockerfile`, `start.sh` and `render.yaml`. These have not been built or deployed yet, so expect to fix small problems on the first run.

**Cost and size.** Embeddings run on the Gemini API, so the image has no PyTorch and the service needs little memory. `render.yaml` uses the `starter` plan (about 512 MB). If it runs out of memory, change the plan to `standard`. Persistent disks are not available on the free plan, and without one your documents, chats and accounts vanish on every restart. Check Render's current pricing before you create the service.

1. Commit and push to GitHub. Check that `git status` does not list `.env` or `.streamlit/secrets.toml`.
2. In the Render dashboard choose **New > Blueprint**, select the repo, and Render reads `render.yaml`. Enter `GEMINI_API_KEY` and `ALLOWED_EMAILS` when asked. Data goes to a 5 GB disk mounted at `/data`.
3. Open the new service, go to **Environment > Secret Files**, and add a file named `secrets.toml` with the `[auth]` block from `.streamlit/secrets.toml.example`. Set `redirect_uri` to `https://YOUR-SERVICE.onrender.com/oauth2callback` and generate a new `cookie_secret`. `start.sh` copies this file into place at startup.
4. In Google Cloud Console, add the same redirect URI to your OAuth client.
5. Redeploy, open the service URL, and sign in with an allowlisted email.

To try the image on your own machine first, install Docker, then:

```bash
docker build -t gilto .
docker run -p 8501:8501 -e AUTH_DISABLED=true -e GEMINI_API_KEY=your-key gilto
```

Use `AUTH_DISABLED=true` only for this local test.

## Configuration

All settings are environment variables. See `.env.example` for the full list and defaults.

- `GEMINI_MODEL`: answers. `GEMINI_FAST_MODEL`: follow-up question rewriting. `GEMINI_FALLBACK_MODEL`: used when the main model keeps returning transient errors (500/503/504) or is over its quota (429). Gemini is sometimes overloaded, so answers retry with backoff, then try the fallback model, then `GEMINI_FAST_MODEL` as a last resort. The last resort is cheaper and can be lower quality, and the app does not show which model answered.
- `EMBEDDING_PROVIDER` (`gemini` or `local`), `GEMINI_EMBEDDING_MODEL` (gemini-embedding-001), `EMBEDDING_DIMENSIONS` (768). Each embedding model gets its own collections, so after changing any of these you need to upload your documents again. Scanned pages are not re-read, because OCR results are cached.
- `SIMILARITY_THRESHOLD` (default 0.60 for Gemini embeddings, 0.80 for the local model): passages scoring below this are dropped. If nothing passes, Gilto answers "I couldn't find that in the uploaded documents." without calling Gemini. Tuned on a small sample (relevant questions scored 0.65 to 0.82, unrelated ones up to 0.54); raise it if unrelated questions get answers, lower it if good questions get refused. Leave it unset in `.env` unless you have measured your own documents, because a value left over from the local model (0.80) would reject nearly everything.
- Upload limits: `MAX_FILE_MB`, `MAX_PAGES_PER_FILE`, `MAX_DOCS_PER_USER`, `MAX_TOTAL_PAGES`.
- OCR: `OCR_LANGS`, `OCR_CONFIDENCE_THRESHOLD`, `OCR_DPI`, `OCR_CACHE_PATH`, `TESSERACT_CMD`.
- Search: `HYBRID_SEARCH` (default true).
- Evaluation: `EVAL_DELAY_SECONDS` (default 4), `EVAL_CASES` (path to a cases file, default `eval/cases.json`).

## Limits of this version

- Handwriting is not read. Pages with no readable text are skipped with a warning.
- When the main Gemini model hits its quota (429), answers move to the next model straight away, because quotas are per model. If every model is busy or over quota you see a plain message and can try again.
- Original uploaded files are not stored, only extracted text chunks and the OCR cache.
- Google sign-in is the only login method. It was not tested end to end with real Google credentials.
- Summaries are limited to 4 documents and about 100 pages per request.
- The Docker image, the Render deployment, and Hindi OCR on real scans have not been verified yet.
- The evaluation suite passed on the built-in sample (9 questions) using Gemini embeddings, but the answers came from `gemini-3.5-flash-lite` because the main models were rate limited at the time.
- Document text is sent to Google's Gemini API to answer questions. On Google's free tier, submitted content may be used to improve Google products (check Google's current terms), so use a paid-tier key for confidential documents.
- A free-tier key has low rate limits. If you see "rate limit or quota was reached", wait a minute and try again.
