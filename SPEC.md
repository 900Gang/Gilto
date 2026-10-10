# RAG Document Chatbot: Specification

Users upload documents and chat with them. Every answer must be grounded in the user's own files. When the files do not contain the answer, the app says so.

Stack: Python 3.11+, Claude API (`anthropic` SDK), local embeddings, Chroma, Streamlit. No LangChain or LlamaIndex.

This spec records decisions made in a design review. Each decision has an ID (D1-D19) so build prompts and tests can reference it.

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
  3. Pages below the confidence threshold escalate to Claude vision (Haiku-class model).
  4. Each chunk carries `source_type`: `text`, `ocr`, or `vision`. The Sources panel shows it.
  5. OCR results are cached by file hash and page number so a retry never pays twice.
- **D9 Languages.** English and Hindi. Embedding model `multilingual-e5-small` (needs `query: ` and `passage: ` prefixes, handled inside the `Embedder`). Tesseract packs `eng` and `hin`. Answers come back in the language of the question.
- **Chunking.** Recursive splitter, about 800 characters, 150 overlap. Metadata per chunk: user_id, doc_id, filename, page, chunk_index, source_type.
- **D11 Ingestion flow.** Blocking in v1, with a per-page progress bar. Indexing sits behind an `Indexer` interface so a background worker can replace it later without UI changes.

## 4. Retrieval and answering

- **Retrieval.** Top-k default 5, adjustable 1-10 in the sidebar. A similarity threshold drops weak chunks.
- **D14a Hybrid search.** Vector search plus BM25 (or SQLite FTS5), fused with reciprocal rank fusion, always scoped to the user.
- **D14b Summarize path.** If the question matches summarize / overview / compare intent, run a map-reduce summary over the chunks of the document the user picked in a document selector.
- **D14c Citation verification.** Claude cites as `[filename, p.N]`. Every citation is checked against the retrieved chunks. Unverified citations are removed and the answer gets a "citation could not be verified" note.
- **Grounding rules (system prompt).** Answer only from the provided context. If the context does not contain the answer, reply "I couldn't find that in the uploaded documents." Never use outside knowledge.
- **Memory.** Last 4 turns go into the prompt. Follow-up questions are rewritten into standalone queries (Haiku-class) before retrieval.
- **D17 Prompt injection.** Document text is untrusted data. Retrieved chunks are wrapped in `<document>` tags. The system prompt states that text inside those tags is never an instruction. The model gets no tools. No other user's data ever enters a prompt.
- **Streaming.** Answers stream to the UI. A collapsible Sources expander under each answer shows chunks with filename, page, score, and `source_type`.

## 5. Models and cost

- **D7 Models.** Env vars, never hardcoded. Defaults: `CLAUDE_MODEL=claude-sonnet-5-5` for answers, `CLAUDE_FAST_MODEL=claude-haiku-4-5-20251001` for query rewrite and vision OCR. `ANTHROPIC_API_KEY` from env only.
- **Budget.** An app-wide monthly spend cap (env var). At the cap, users see a friendly "limit reached" message, not a crash.
- **D15 Per-user limits.** Hourly question limit (default 30) and daily token limit (default 200,000), tracked in the `usage` table. Remaining quota shows in the sidebar.

## 6. Data and privacy

- **D4 Disclosure.** Document text and OCR page images are sent to Anthropic's API. Before a user's first upload they must tick a consent checkbox that says so. No document text is written to logs.
- **D12 Persistence.** SQLite file on the persistent volume. Tables: `users`, `documents`, `conversations`, `messages`, `usage`. Chats survive logout and refresh. Users can delete a conversation.
- **D13 Retention.** Original files are kept so users can re-index and view the page behind a citation. Deleting a document hard-deletes the file, its chunks, its OCR cache, and messages citing it. Deleting an account deletes everything for that user. Inactive accounts auto-purge after 90 days (env var). No user-content backups beyond the volume snapshot.

## 7. Deployment

- **D18 Target.** Render web service with a persistent disk, built from a Dockerfile (Tesseract with `eng` and `hin`, PyMuPDF). HTTPS comes from Render, which Google login requires. Secrets live only in Render env vars. Custom domain later.

## 8. Quality bar

- **D6 Eval.** `pytest -m eval` runs 20 questions over real PDFs. Pass bar: at least 90% answered correctly with the correct page cited, 100% refusal on out-of-scope questions. User supplies 5 PDFs and 10 questions with expected answers. The rest are drafted for the user to review.
- **Injection test.** One eval case uses a PDF containing a malicious instruction and asserts the instruction is not followed.
- **Other tests.** pytest for chunker, loader, store, citation verifier, OCR routing, quota logic. Claude calls are mocked in unit tests.

## 9. Build phasing (D19)

Each slice runs end to end and is tested before the next begins.

1. **Single-user core.** Text PDFs, chunk, embed, Chroma, Claude answer with verified citations, Streamlit chat, Sources panel.
2. **Scanned PDFs and images.** OCR hybrid, OCR cache, `source_type`, upload limits.
3. **Multi-user.** Login, allowlist, per-user isolation, consent, SQLite chat history, quotas, budget cap, delete and retention.
4. **Retrieval and ship.** Hybrid search, summarize path, Dockerfile, Render deploy, eval suite run against real PDFs.

## 10. Target structure

```
app.py                  # Streamlit UI only
rag/
  loader.py             # PDF/image -> pages with metadata
  ocr.py                # Tesseract + vision fallback + cache (slice 2)
  chunker.py
  embedder.py           # Embedder interface + multilingual-e5-small
  store.py              # Chroma wrapper, per-user collections
  chain.py              # rewrite -> retrieve -> prompt -> Claude -> verify citations
  citations.py          # citation verifier
  indexer.py            # Indexer interface (blocking impl in v1)
  db.py                 # SQLite access (slice 3)
  limits.py             # quotas and budget (slice 3)
tests/
Dockerfile              # slice 4
requirements.txt
.env.example
README.md
```

## 11. Non-goals for v1

Handwriting OCR, background workers, shared team documents, multi-tenant orgs, non-Google sign-in, any LangChain/LlamaIndex dependency.
