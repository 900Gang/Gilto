"""Gilto: chat with your own documents. UI only; the logic lives in rag/."""
from __future__ import annotations

import streamlit as st

import ui
from rag.chain import Chain
from rag.config import Settings
from rag.embedder import E5Embedder
from rag.errors import ConfigError, GiltoError
from rag.indexer import BlockingIndexer
from rag.llm import make_llm
from rag.store import VectorStore

USER_ID = "local"  # slice 3 replaces this with the signed-in user

st.set_page_config(page_title="Gilto", layout="wide")


@st.cache_resource
def get_settings() -> Settings:
    return Settings.from_env()


@st.cache_resource(show_spinner="Loading the embedding model (first run downloads it)...")
def get_embedder() -> E5Embedder:
    embedder = E5Embedder(get_settings().embedding_model)
    embedder.embed_query("warm up")
    return embedder


@st.cache_resource
def get_store() -> VectorStore:
    return VectorStore(path=get_settings().chroma_path)


@st.cache_resource
def get_llm():
    return make_llm(get_settings())


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("notices", [])


def render_sources(sources) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        cards = "".join(
            ui.source_card(
                s.metadata["filename"], s.metadata["page"], s.score, s.metadata["source_type"], s.text
            )
            for s in sources
        )
        st.markdown(cards, unsafe_allow_html=True)


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(ui.pretty_citations(message["content"]))
            render_sources(message.get("sources"))
        else:
            st.markdown(message["content"])


def index_uploads(files, indexer: BlockingIndexer) -> None:
    for file in files:
        bar = st.sidebar.progress(0.0, text=f"Reading {file.name}...")

        def on_progress(done: int, total: int, message: str, bar=bar, name=file.name) -> None:
            bar.progress(done / total, text=f"{name}: {message}")

        try:
            result = indexer.index(USER_ID, file.name, file.getvalue(), progress=on_progress)
        except GiltoError as exc:
            st.session_state.notices.append(("error", f"{file.name}: {exc}"))
        except Exception:
            st.session_state.notices.append(("error", f"{file.name}: indexing failed. Try the file again."))
        else:
            if result.duplicate:
                st.session_state.notices.append(("info", f"{file.name}: already in your library."))
            else:
                st.session_state.notices.append(
                    ("success", f"{file.name}: added ({result.pages_indexed} pages, {result.chunks} chunks).")
                )
                if result.skipped_pages:
                    pages = ", ".join(str(p) for p in result.skipped_pages)
                    st.session_state.notices.append(
                        ("warning", f"{file.name}: skipped pages {pages} (no text layer). Scanned pages need OCR, coming later.")
                    )
        bar.empty()


def sidebar(store: VectorStore, indexer: BlockingIndexer) -> int:
    with st.sidebar:
        ui.brand()

        ui.label("Add documents")
        files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
            label_visibility="collapsed",
        )
        if st.button("Add to library", disabled=not files, type="primary"):
            index_uploads(files, indexer)
            st.session_state.uploader_key += 1
            st.rerun()

        for level, message in st.session_state.notices:
            getattr(st, level)(message)
        st.session_state.notices = []

        ui.label("Library")
        docs = store.list_docs(USER_ID)
        if not docs:
            st.caption("No documents yet.")
        for doc in docs:
            left, right = st.columns([5, 1], vertical_alignment="center")
            left.markdown(ui.doc_card(doc.filename, doc.page_count, doc.chunks), unsafe_allow_html=True)
            if right.button("✕", key=f"del_{doc.doc_id}", help=f"Remove {doc.filename}"):
                store.delete_doc(USER_ID, doc.doc_id)
                st.rerun()

        ui.label("Chat")
        if st.button("New chat", disabled=not st.session_state.messages):
            st.session_state.messages = []
            st.rerun()
        with st.expander("Settings"):
            top_k = st.slider("Passages to retrieve", 1, 10, 5)

        ui.note("Document text is sent to Google's Gemini API to generate answers.")
    return top_k


def main() -> None:
    init_state()
    ui.inject_css()
    settings = get_settings()
    store = get_store()
    embedder = get_embedder()
    indexer = BlockingIndexer(store, embedder, settings)
    top_k = sidebar(store, indexer)

    st.title("Ask your documents")

    chain = None
    try:
        chain = Chain(get_llm(), store, embedder, settings)
    except ConfigError as exc:
        st.error(str(exc))

    docs = store.list_docs(USER_ID)
    if docs:
        pages = sum(d.page_count for d in docs)
        ui.subtitle(f"{len(docs)} document{'s' if len(docs) != 1 else ''} · {pages} pages. Answers cite the page they came from.")
    else:
        ui.empty_state()

    for message in st.session_state.messages:
        render_message(message)

    question = st.chat_input("Ask about your documents", disabled=not (docs and chain))
    if not question:
        return

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed = ""
        result = None
        try:
            for kind, payload in chain.stream(USER_ID, question, history, top_k=top_k):
                if kind == "delta":
                    streamed += payload
                    placeholder.markdown(ui.pretty_citations(streamed) + "▌")
                else:
                    result = payload
        except GiltoError as exc:
            placeholder.empty()
            st.error(str(exc))
            return  # nothing is saved, so a failed or interrupted turn leaves no orphan question
        placeholder.markdown(ui.pretty_citations(result.answer))
        render_sources(result.sources)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )


main()