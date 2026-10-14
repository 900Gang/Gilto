# RAG Document Chatbot: Specification

Users upload documents and chat with them. Every answer must be grounded in the user's own files. When the files do not contain the answer, the app says so.

Stack: Python 3.11+, Gemini API (`google-genai` SDK), local embeddings, Chroma, Streamlit. No LangChain or LlamaIndex.

This spec records decisions made in a design review. Each decision has an ID (D1-D19) so build prompts and tests can reference it.

**Provider change.** The design review assumed the Claude API. The project now uses Gemini. Only D4 (disclosure) and D7 (models) changed in substance. The app talks to the model through an `LLM` interface (`rag/llm.py`), so the provider can change again without touching the chain or the UI.

## 1. Users and access

- **D1 Isolation.** Many users. Each user's documents are private. One Chroma collection per user, plus a `user_id` metadata filter on every query as a second guard. No shared pool in v1.
- **D2 Identity.** Google sign-in through Streamlit's built-in `st.login` (OIDC). The verified email is the user ID. The app stores no passwords.
- **D16 Allowlist.** An `ALLOWED_EMAILS` env var (comma-separated) is checked right after login. Anyone else sees "not authorized" and cannot reach any page logic.
- **D3 Scale.** Under 20 users, under 5 concurrent. Local persistent Chroma is enough at this size.

## 2. Inputs and limits

- **D5 Accepted files.** PDF (text or scanned), PNG, JPG/JPEG containing text (screenshots, scans, photographed pages). Handwriting is out of scope for v1 and the UI says so. Charts are read for visible labels only.
- **D10 Limits.** All are env vars with these defaults: 20 MB per file, 100 pages per file, 15 documents per user, 500 total pages per user. Over-limit files are rejected before processing with a clear message.
- **Corpus now.** About 10 PDFs of about 30 pages each.
- **Dedup.** Files are hashed (SHA-256). Re-uploading the same content for the same user does not create duplicates.

## 3. Ingestion

- **D8 OCR pipeline (hybrid).**
  1. Extract the PDF text layer with PyMuPDF.
  2. For any page with fewer than about 50 extracted characters, run Tesseract (`pytesseract`) and record mean confidence from `image_to_data`.
  3. Pages below the confidence threshold, pages with under 50 characters, and pages where Tesseract reads Hindi (Devanagari) escalate to Gemini vision (the fast model). Hindi always escalates because Tesseract reports high confidence on Hindi even when its words are wrong. If Tesseract is not installed, every scanned page goes to vision. If Gemini is unavailable, the Tesseract text is kept and flagged as low confidence.
  4. Each chunk carries `source_type`: `text`, `ocr`, or `vision`. The Sources panel shows it.
  5. OCR results are cached by file hash and page number so a retry never pays twice.
- **D9 Languages.** English and Hindi. Embeddings come from the Gemini API model `gemini-embedding-001` at 768 dimensions (`rag/embedder.py`): `RETRIEVAL_DOCUMENT` for chunks, `RETRIEVAL_QUERY` for questions, vectors scaled to unit length, batches of 50 texts, rate limits and server errors retried. The local `multilingual-e5-small` model remains as `EMBEDDING_PROVIDER=local` (it needs `query: ` and `passage: ` prefixes, handled inside the `Embedder`, and about 1.2 GB of RAM). Each embedding model has its own Chroma collections, named with a tag such as `gemini-embedding-001-768`, because vectors from different models cannot be compared. The default similarity threshold depends on the model: 0.60 for Gemini embeddings and 0.80 for the local model. This changed the original D9 choice of a local model, which ran out of memory on the development machine. Tesseract packs `eng` and `hin`. Answers come back in the language of the question.
- **Chunking.** Recursive splitter, about 800 characters, 150 overlap. Metadata per chunk: user_id, doc_id, filename, page, chunk_index, source_type.
- **D11 Ingestion flow.** Blocking in v1, with a per-page progress bar. Indexing sits behind an `Indexer` interface so a background worker can replace it later without UI changes.

## 4. Retrieval and answering

- **Retrieval.** Top-k default 5, adjustable 1-10 in the sidebar. A similarity threshold drops weak chunks.
- **D14a Hybrid search.** Vector search plus BM25 keyword search, fused with reciprocal rank fusion (k = 60), always scoped to the user. BM25 runs in memory over the user's chunks (`rag/bm25.py`), so there is no second index to keep in sync. Both searches fetch `top_k x 3` candidates and the fused list is cut to `top_k`. A vector hit must still pass the similarity threshold. A keyword hit must match at least half of the question's meaningful words (stopwords removed, English and a few Hindi), so a question about something not in the files finds nothing and is refused without calling the model. A keyword-only hit has no vector score and the Sources panel shows "keyword match". `HYBRID_SEARCH=false` turns this off.
- **D14b Summarize path.** A question that matches summarize / summary / overview / tl;dr / main points / "what is this document about" / compare intent skips retrieval and runs a map-reduce over whole documents. A "Search in" multiselect in the sidebar (shown with two or more documents) picks the scope, and empty means all documents. The chunks of the documents in scope are read in order in batches of up to 20,000 characters. The fast model writes short cited notes for each batch (up to 3 attempts per call), then the main model answers from the notes. Limits: at most 4 documents and 400 passages per request, with a polite message above that. Citations are checked against every passage in scope. The answer has no Sources list, because it draws on every passage. The UI shows "Reading part N of M" while it works.
- **D14c Citation verification.** The model cites as `[filename, p.N]`. Every citation is checked against the retrieved chunks. Unverified citations are removed and the answer gets a "citation could not be verified" note.
- **Grounding rules (system prompt).** Answer only from the provided context. If the context does not contain the answer, reply "I couldn't find that in the uploaded documents." Never use outside knowledge.
- **Memory.** Last 4 turns go into the prompt. Follow-up questions are rewritten into standalone queries (fast model) before retrieval. If the rewrite fails, the original question is used.
- **D17 Prompt injection.** Document text is untrusted data. Retrieved chunks are wrapped in `<document>` tags, and the notes in the summary path in `<notes>` tags. Both tag names are neutralized inside the text so a document cannot close or open them. The system prompts state that text inside those tags is never an instruction. The model gets no tools. No other user's data ever enters a prompt.
- **Streaming.** Answers stream to the UI. A collapsible Sources expander under each answer shows chunks with filename, page, score, and `source_type`.

## 5. Models and cost

- **D7 Models.** Env vars, never hardcoded. Defaults: `GEMINI_MODEL=gemini-3.8-flash` for answers, `GEMINI_FAST_MODEL=gemini-3.5-flash-lite` for query rewrite and vision OCR, `GEMINI_FALLBACK_MODEL=gemini-3.6-flash` as a backup for answers. `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) from env only, never in a tracked file.
- **Reliability.** Gemini returns transient 500/503/504 errors under load. Answer generation retries up to 3 times with backoff (1 s, 2 s), then tries the fallback model once, then the fast model as a last resort (cheaper and sometimes lower quality, and the app does not say which model answered). Retries happen only before any text has streamed. A quota error (429) skips straight to the next model, because Gemini quotas are per model. Other errors (bad key) are not retried and show a plain-language message. Vision OCR and the summary map step also retry rate limits and server errors up to 3 times.
- **Budget.** An app-wide monthly token cap (`MONTHLY_TOKEN_CAP`, default 5,000,000) instead of a money cap, so no price table is needed. Tokens are estimated at 3 characters each, and vision OCR calls are not counted. At the cap, users see a friendly "limit reached" message, not a crash.
- **D15 Per-user limits.** Hourly question limit (default 30) and daily token limit (default 200,000), tracked in the `usage` table. Remaining quota shows in the sidebar.

## 6. Data and privacy

- **D4 Disclosure.** Document text and OCR page images are sent to Google's Gemini API. Before a user's first upload they must tick a consent checkbox that says so. No document text is written to logs. On Google's free tier, submitted content may be used to improve Google products (check Google's current terms). A deployment that holds confidential documents must use a paid-tier key.
- **D12 Persistence.** SQLite file on the persistent volume. Tables: `users`, `conversations`, `messages`, `usage`. Document records live in Chroma, not SQLite. Every chat query also checks the owner. Chats survive logout and refresh. Users can delete a conversation.
- **D13 Retention.** Original uploaded files are not stored (nothing uses them yet), only extracted chunks and the OCR cache. Deleting a document removes its chunks and its OCR cache. Chat messages that cited it are kept. Deleting an account deletes the user's chunks, OCR cache entries, chats and account row. Usage rows are kept anonymized so the app-wide monthly count does not drop. Inactive accounts auto-purge after 90 days (env var). No user-content backups beyond the volume snapshot.

## 7. Deployment

- **D18 Target.** Render web service with a persistent disk, built from a `Dockerfile` and described in `render.yaml`. The image has Tesseract with `eng` and `hin`. It has no PyTorch or local model, because embeddings use the Gemini API. `start.sh` copies Render's secret file `/etc/secrets/secrets.toml` to `.streamlit/secrets.toml` and starts Streamlit on `$PORT`. HTTPS comes from Render, which Google login requires. Chroma, the OCR cache and SQLite live on the disk mounted at `/data`. `GEMINI_API_KEY` and `ALLOWED_EMAILS` are set in the Render dashboard, and the Google `[auth]` block goes in a Render secret file. Sizing: without a local model the service is small, so `render.yaml` uses `starter` (about 512 MB). Change it to `standard` if it runs out of memory. Persistent disks need a paid plan. Custom domain later.

## 8. Quality bar

- **D6 Eval.** `python -m pytest -m eval -s` runs real questions against the real Gemini API. It is skipped in the normal test run (`addopts = -m "not eval"`). Pass bar: at least 90% answered correctly with the correct page cited, 100% refusal on out-of-scope questions. With no `eval/cases.json` it uses a built-in sample (one 3-page document, 6 answerable and 3 out-of-scope questions). To use real PDFs, put them in `eval/docs/`, copy `eval/cases.example.json` to `eval/cases.json`, and list the documents and cases (question, expected words, document and page, or `"type": "out_of_scope"`). `eval/docs/` and `eval/cases.json` are ignored by git because the documents may be confidential. The run pauses `EVAL_DELAY_SECONDS` between questions to stay under rate limits.
- **Injection test.** One eval test indexes a PDF containing a malicious instruction and asserts the instruction is not followed. Unit tests also check that document and note text cannot close the prompt tags.
- **Other tests.** pytest for chunker, loader, store, citation verifier, OCR routing, quota logic, auth and allowlist, database isolation, retention, BM25, rank fusion, summary routing, LLM retry and error translation. Gemini calls are mocked in unit tests.

## 9. Build phasing (D19)

Each slice runs end to end and is tested before the next begins.

1. **Single-user core.** Text PDFs, chunk, embed, Chroma, Gemini answer with verified citations, Streamlit chat, Sources panel.
2. **Scanned PDFs and images.** OCR hybrid, OCR cache, `source_type`, upload limits.
3. **Multi-user.** Login, allowlist, per-user isolation, consent, SQLite chat history, quotas, budget cap, delete and retention.
4. **Retrieval and ship.** Hybrid search, summarize path, Dockerfile, Render deploy, eval suite run against real PDFs.

All four slices are built. Slice 4 added hybrid search, the summary path, the eval suite and the deploy files.

## 10. Target structure

```
app.py                  # Streamlit UI only
ui.py                   # theme CSS and small HTML components (escaped text only)
rag/
  config.py             # settings from env
  errors.py             # user-facing error types
  llm.py                # LLM interface + Gemini implementation (retry, fallback)
  loader.py             # PDF/image -> pages with metadata
  ocr.py                # Tesseract + vision fallback + cache
  chunker.py
  embedder.py           # Embedder interface, Gemini embeddings, optional local e5 model
  store.py              # Chroma wrapper, per-user collections, all_chunks for keyword search and summaries
  bm25.py               # in-memory keyword search
  fusion.py             # reciprocal rank fusion
  summarize.py          # summary intent detection and batching
  chain.py              # rewrite -> retrieve (hybrid) -> prompt -> LLM -> verify citations; summary path
  citations.py          # citation verifier
  indexer.py            # Indexer interface (blocking impl)
  auth.py               # allowlist check
  db.py                 # SQLite: users, chats, usage
  limits.py             # quotas and budget
  retention.py          # delete a user's data, purge inactive accounts
tests/                  # unit tests, plus test_eval.py (run with -m eval)
eval/cases.example.json # template for your own evaluation cases
Dockerfile
start.sh                # container entry point
render.yaml             # Render blueprint
.dockerignore
requirements.txt
requirements-local.txt   # adds sentence-transformers for EMBEDDING_PROVIDER=local
.env.example
.streamlit/secrets.toml.example
README.md
```

## Status and known gaps

- Google sign-in (D2, D16) is covered by unit tests and a no-credentials screen check, but it was not tried end to end with real Google credentials.
- The Dockerfile and `render.yaml` were never built or deployed. Docker was not available where they were written.
- The eval suite passed on the built-in 9-question sample with Gemini embeddings, but with answers from `gemini-3.5-flash-lite` because the main models were rate limited at the time. It has not been run on real documents or with the main model.
- The Gemini similarity threshold (0.60) was set from about 16 measured queries. Measure again on real documents.
- Hindi OCR quality was not checked on a real Hindi scan.
- Hybrid search was checked with a stand-in embedder that makes vector search find nothing, so the keyword path was proven alone. The fused ranking with real vectors is covered only by unit tests.
- The model IDs in D7 come from Google's model list at the time of writing and may need updating.

## 11. Non-goals for v1

Handwriting OCR, background workers, shared team documents, multi-tenant orgs, non-Google sign-in, any LangChain/LlamaIndex dependency.
