"""Gilto: chat with your own documents. UI only; the logic lives in rag/."""
from __future__ import annotations

import streamlit as st

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
        for s in sources:
            meta = s.metadata
            st.markdown(
                f"**{meta['filename']}**, page {meta['page']} · score {s.score:.2f} · {meta['source_type']}"
            )
            st.text(s.text[:600] + ("..." if len(s.text) > 600 else ""))


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
                st.session_state.notices.append(("info", f"{file.name}: already indexed."))
            else:
                st.session_state.notices.append(
                    ("success", f"{file.name}: indexed {result.pages_indexed} pages, {result.chunks} chunks.")
                )
                if result.skipped_pages:
                    pages = ", ".join(str(p) for p in result.skipped_pages)
                    st.session_state.notices.append(
                        ("warning", f"{file.name}: skipped pages {pages} (no text layer). Scanned pages need OCR, coming later.")
                    )
        bar.empty()


def sidebar(store: VectorStore, indexer: BlockingIndexer) -> int:
    st.sidebar.title("Gilto")
    st.sidebar.caption("Answers come only from the files you upload.")

    files = st.sidebar.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if st.sidebar.button("Index files", disabled=not files, type="primary"):
        index_uploads(files, indexer)
        st.session_state.uploader_key += 1
        st.rerun()

    for level, message in st.session_state.notices:
        getattr(st.sidebar, level)(message)
    st.session_state.notices = []

    st.sidebar.divider()
    st.sidebar.subheader("Your documents")
    docs = store.list_docs(USER_ID)
    if not docs:
        st.sidebar.caption("Nothing indexed yet.")
    for doc in docs:
        left, right = st.sidebar.columns([4, 1])
        left.markdown(f"**{doc.filename}**  \n{doc.page_count} pages · {doc.chunks} chunks")
        if right.button("✕", key=f"del_{doc.doc_id}", help=f"Delete {doc.filename}"):
            store.delete_doc(USER_ID, doc.doc_id)
            st.rerun()

    st.sidebar.divider()
    top_k = st.sidebar.slider("Passages to retrieve", 1, 10, 5)
    if st.sidebar.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    return top_k


def main() -> None:
    init_state()
    settings = get_settings()
    store = get_store()
    embedder = get_embedder()
    indexer = BlockingIndexer(store, embedder, settings)
    top_k = sidebar(store, indexer)

    st.title("Chat with your documents")

    chain = None
    try:
        chain = Chain(get_llm(), store, embedder, settings)
    except ConfigError as exc:
        st.error(str(exc))

    has_docs = bool(store.list_docs(USER_ID))
    if not has_docs:
        st.info("Upload a PDF in the sidebar and click **Index files** to start chatting.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("sources"))

    question = st.chat_input("Ask about your documents", disabled=not (has_docs and chain))
    if not question:
        return

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed = ""
        try:
            for kind, payload in chain.stream(USER_ID, question, history, top_k=top_k):
                if kind == "delta":
                    streamed += payload
                    placeholder.markdown(streamed + "▌")
                else:
                    result = payload
        except GiltoError as exc:
            placeholder.empty()
            st.error(str(exc))
            st.session_state.messages.pop()  # drop the unanswered question
            return
        placeholder.markdown(result.answer)
        render_sources(result.sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )


main()
