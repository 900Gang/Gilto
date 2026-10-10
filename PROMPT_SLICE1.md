# Build prompt: Slice 1 (single-user core)

Read `SPEC.md` in this folder first. It is the source of truth. Build only Slice 1 from section 9. Do not start slices 2-4.

Provider: Gemini (`google-genai` SDK), not Claude. Put all model calls behind an `LLM` interface in `rag/llm.py` so `chain.py` never imports the SDK.

## Goal

A working Streamlit app where one user uploads text-layer PDFs and chats with them. Answers are grounded in the PDFs, cite `[filename, p.N]`, and every citation is verified against retrieved chunks.

## Scope (Slice 1 only)

- Text-layer PDFs only (PyMuPDF). Pages with no text are skipped with a visible warning ("scanned pages need OCR, coming in a later version").
- Chunker, `Embedder` interface with `multilingual-e5-small` implementation (query:/passage: prefixes), Chroma wrapper with persistent storage in `./chroma_db`.
- Per-user collection naming and a `user_id` metadata filter from the start. Slice 1 uses a fixed placeholder user ID `local`, so slice 3 only has to swap in the real ID.
- `chain.py`: query rewrite from the last 4 turns (fast model), retrieval with top-k and similarity threshold, prompt with `<document>`-wrapped chunks, streaming answer, citation verification.
- Grounding system prompt exactly as in SPEC section 4, including the not-found sentence and the untrusted-document rule.
- Streamlit UI: sidebar with multi-file PDF upload, progress bar per page, list of indexed documents with delete, top-k slider. Chat with streaming and a Sources expander (filename, page, score, `source_type`). Chat input disabled with an upload prompt when no documents exist.
- Dedup by SHA-256 file hash.
- Visible, non-technical errors for: missing `GEMINI_API_KEY`, unreadable PDF, empty file, rate limit, Gemini overload. No stack traces in the UI.
- Retry transient Gemini errors (500/503/504) with backoff before any text streams, then try `GEMINI_FALLBACK_MODEL` once (SPEC D7).
- Env vars: `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_FAST_MODEL`, `GEMINI_FALLBACK_MODEL`, limits from SPEC D10. Provide `.env.example` with an empty key value. Never put a real key in any tracked file. Add `.env` and `chroma_db/` to `.gitignore`.
- A question that fails or is interrupted must not leave an unanswered message in the chat history.

## Out of scope for this slice

OCR and images, login, SQLite, quotas, hybrid search, summarize path, Docker, deployment.

## Rules

- Plain `google-genai`, `chromadb`, `sentence-transformers`, `pymupdf`, `streamlit`. No LangChain or LlamaIndex.
- Test first for each `rag/` module (pytest). Mock the `LLM` in chain tests and the Gemini client in `llm` tests.
- Required tests: chunker overlap and metadata; loader on a text PDF and on an empty-text PDF; store add/query/delete-by-doc with user filter; citation verifier (valid, invalid, mixed); the not-found path when nothing passes the threshold; prompt-injection text inside a chunk stays inside `<document>` tags.
- No placeholder or stub code.

## Delivery order

1. Show the file tree and `requirements.txt`.
2. Implement `rag/` modules one at a time, each with passing tests.
3. Build `app.py` last.
4. Run `python -m pytest` and `python -m streamlit run app.py`. Report real results. Do not claim it works until you ran it.
5. Write `README.md`: setup, env vars, run command.

Stop after Slice 1 and report. Ask before deviating from SPEC.md.
