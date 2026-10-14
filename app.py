"""Gilto: chat with your own documents. UI only; the logic lives in rag/."""
from __future__ import annotations

import os
import sys

try:  # Community Cloud can ship an old SQLite that Chroma rejects. Use the bundled one when it is installed.
    import pysqlite3

    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import streamlit as st

import ui
from rag.auth import AccessError, check_access, parse_allowed
from rag.chain import Chain
from rag.config import Settings
from rag.db import Database
from rag.embedder import Embedder, embedding_tag, make_embedder
from rag.errors import ConfigError, GiltoError, LimitError
from rag.indexer import BlockingIndexer
from rag.limits import QuotaGuard
from rag.llm import make_llm
from rag.ocr import OCRCache, build_ocr
from rag.retention import delete_user_data, purge_inactive
from rag.store import Retrieved, VectorStore

st.set_page_config(
    page_title="Gilto",
    page_icon=":material/description:",
    layout="wide",
    initial_sidebar_state="expanded",
)

USER_AVATAR = ":material/person:"
BOT_AVATAR = ":material/auto_awesome:"
SUGGESTIONS = ["Summarize my documents", "What are the key points?", "What is this document about?"]


@st.cache_resource
def get_settings() -> Settings:
    return Settings.from_env()


@st.cache_resource
def get_embedder() -> Embedder:
    return make_embedder(get_settings())


@st.cache_resource
def get_store() -> VectorStore:
    settings = get_settings()
    return VectorStore(path=settings.chroma_path, tag=embedding_tag(settings))


@st.cache_resource
def get_llm():
    return make_llm(get_settings())


@st.cache_resource
def get_ocr():
    """OCR pipeline, or None when there is no Tesseract and no Gemini key."""
    try:
        llm = get_llm()
    except ConfigError:
        llm = None
    return build_ocr(get_settings(), llm)


@st.cache_resource
def get_db() -> Database:
    return Database(get_settings().sqlite_path)


@st.cache_resource
def get_cache() -> OCRCache:
    return OCRCache(get_settings().ocr_cache_path)


@st.cache_resource
def run_retention() -> int:
    """Runs once per server start: remove accounts that have been inactive too long."""
    s = get_settings()
    return len(purge_inactive(s.inactive_days, get_store(), get_cache(), get_db()))


def auth_configured() -> bool:
    try:
        return "auth" in st.secrets
    except Exception:
        return False


def require_user(settings: Settings) -> str:
    """Return the signed-in user's email. Otherwise show the right screen and stop."""
    if settings.auth_disabled:
        return "local"  # local testing only, set with AUTH_DISABLED=true
    if not auth_configured():
        with ui.auth_card(
            "Sign-in is not set up",
            "Add an [auth] section to .streamlit/secrets.toml, or set AUTH_DISABLED=true in .env for local testing only.",
            tone="danger",
        ):
            pass
        st.stop()
    if not st.user.is_logged_in:
        with ui.auth_card(
            "Restricted access",
            "Sign in with an approved Google account to chat with your documents. "
            "Every answer cites the page it came from.",
            footnote="Access is limited to approved accounts. Your documents stay private to you.",
        ):
            st.button("Continue with Google ↗", type="primary", on_click=st.login)
        st.stop()
    claims = st.user.to_dict()
    try:
        return check_access(claims.get("email"), claims.get("email_verified"), parse_allowed(settings.allowed_emails))
    except AccessError as exc:
        with ui.auth_card("Access denied", str(exc), tone="danger"):
            st.button("Sign out", on_click=st.logout)
        st.stop()


def init_state() -> None:
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("notices", [])
    st.session_state.setdefault("scope", [])


def render_sources(sources: list[Retrieved]) -> None:
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
    if message["role"] == "assistant":
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown(ui.pretty_citations(message["content"]))
            render_sources([Retrieved.from_dict(d) for d in message.get("sources") or []])
    else:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(message["content"])


def render_suggestions() -> None:
    with st.container(key="suggestions"):
        for index, text in enumerate(SUGGESTIONS):
            if st.button(f"› {text}", key=f"suggest_{index}"):
                st.session_state.pending_question = text
                st.rerun()


def index_uploads(files, indexer: BlockingIndexer, user: str) -> None:
    for file in files:
        bar = st.sidebar.progress(0.0, text=f"Reading {file.name}...")

        def on_progress(done: int, total: int, message: str, bar=bar, name=file.name) -> None:
            bar.progress(done / total, text=f"{name}: {message}")

        try:
            result = indexer.index(user, file.name, file.getvalue(), progress=on_progress)
        except GiltoError as exc:
            st.session_state.notices.append(("error", f"{file.name}: {exc}"))
        except Exception:
            st.session_state.notices.append(("error", f"{file.name}: indexing failed. Try the file again."))
        else:
            if result.duplicate:
                st.session_state.notices.append(("info", f"{file.name}: already in your library."))
            else:
                st.session_state.notices.append(
                    (
                        "success",
                        f"{file.name}: added ({ui.plural(result.pages_indexed, 'page')}, "
                        f"{ui.plural(result.chunks, 'passage')}).",
                    )
                )
                scanned = result.ocr_pages + result.vision_pages
                if scanned:
                    st.session_state.notices.append(
                        ("info", f"{file.name}: {scanned} scanned page(s) read with OCR ({result.vision_pages} with Gemini vision).")
                    )
                if result.low_confidence_pages:
                    pages = ", ".join(str(p) for p in result.low_confidence_pages)
                    st.session_state.notices.append(
                        ("warning", f"{file.name}: OCR was not confident on page(s) {pages}. Check answers from those pages.")
                    )
                if result.skipped_pages:
                    pages = ", ".join(str(p) for p in result.skipped_pages)
                    st.session_state.notices.append(
                        ("warning", f"{file.name}: no readable text on page(s) {pages}, so they were skipped.")
                    )
        bar.empty()


def usage_rows(guard: QuotaGuard, settings: Settings, user: str) -> tuple[list[tuple[str, float, str]], bool]:
    """Bars for the System status panel, and whether any quota is nearly used up."""
    status = guard.status(user)
    rows: list[tuple[str, float, str]] = []
    if status.questions_left is not None and settings.hourly_question_limit > 0:
        used = 1 - status.questions_left / settings.hourly_question_limit
        rows.append(("Questions", used, f"{status.questions_left} left"))
    if status.tokens_left is not None and settings.daily_token_limit > 0:
        used = 1 - status.tokens_left / settings.daily_token_limit
        rows.append(("Tokens", used, f"{status.tokens_left:,} left"))
    return rows, any(used >= 0.9 for _, used, _ in rows)


def sidebar(
    user: str,
    settings: Settings,
    db: Database,
    guard: QuotaGuard,
    indexer: BlockingIndexer,
    store: VectorStore,
    cache: OCRCache,
    ocr_ready: bool,
) -> int:
    with st.sidebar:
        ui.brand()
        notice = os.getenv("DEMO_NOTICE")
        if notice:
            st.warning(notice)
        rows, low = usage_rows(guard, settings, user)
        if rows:
            ui.usage_panel(rows, low)

        ui.label("Documents", 1)
        consented = db.has_consent(user)
        if not consented:
            agreed = st.checkbox(
                "I understand that my document text, and scanned pages read by Gemini vision, "
                "are sent to Google's Gemini API."
            )
            if agreed:
                db.give_consent(user)
                st.rerun()
        files = st.file_uploader(
            "Upload PDFs or images",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
            label_visibility="collapsed",
            disabled=not consented,
        )
        if not ocr_ready:
            st.caption("Scanned files need OCR: install Tesseract or set GEMINI_API_KEY.")
        if st.button("Add to library ↗", disabled=not (files and consented), type="primary"):
            index_uploads(files, indexer, user)
            st.session_state.uploader_key += 1
            st.rerun()

        for level, message in st.session_state.notices:
            getattr(st, level)(message)
        st.session_state.notices = []

        docs = store.list_docs(user)
        with st.container(key="library"):
            if not docs:
                st.caption("No documents yet.")
            for doc in docs:
                left, right = st.columns([5, 1], vertical_alignment="center")
                left.markdown(ui.doc_card(doc.filename, doc.page_count, doc.chunks), unsafe_allow_html=True)
                if right.button("✕", key=f"del_{doc.doc_id}", help=f"Remove {doc.filename}"):
                    indexer.forget(user, doc.doc_id)
                    st.rerun()

        ids = [d.doc_id for d in docs]
        names = {d.doc_id: d.filename for d in docs}
        st.session_state.scope = [i for i in st.session_state.scope if i in ids]
        if len(docs) > 1:
            ui.label("Search in")
            st.multiselect(
                "Search in",
                options=ids,
                format_func=names.get,
                key="scope",
                placeholder="All documents",
                label_visibility="collapsed",
            )

        ui.label("Chats", 2)
        with st.container(key="chats"):
            if st.button("+ New chat", disabled=st.session_state.conversation_id is None):
                st.session_state.conversation_id = None
                st.rerun()
            for conv in db.list_conversations(user):
                left, right = st.columns([5, 1], vertical_alignment="center")
                active = conv.id == st.session_state.conversation_id
                if left.button(
                    ui.chat_title(conv.title),
                    key=f"conv_{conv.id}",
                    type="primary" if active else "secondary",
                    help=conv.title,
                ):
                    st.session_state.conversation_id = conv.id
                    st.rerun()
                if right.button("✕", key=f"delc_{conv.id}", help="Delete this chat"):
                    db.delete_conversation(user, conv.id)
                    if active:
                        st.session_state.conversation_id = None
                    st.rerun()

        ui.label("System", 3)
        with st.expander("Settings"):
            top_k = st.slider("Passages to retrieve", 1, 10, 5)

        with st.expander("Account"):
            st.caption(user)
            if not settings.auth_disabled:
                st.button("Sign out", on_click=st.logout)
            confirm = st.checkbox("Delete all my documents, chats and usage history")
            if st.button("Delete my data", disabled=not confirm):
                delete_user_data(user, store, cache, db)
                st.session_state.conversation_id = None
                st.session_state.scope = []
                st.session_state.notices = []
                if settings.auth_disabled:
                    st.rerun()
                else:
                    st.logout()

        ui.note("Document text, and page images read by Gemini vision, are sent to Google's Gemini API.")
        ui.hazard()
    return top_k


def main() -> None:
    init_state()
    ui.inject_css()
    settings = get_settings()
    run_retention()
    user = require_user(settings)

    db = get_db()
    db.upsert_user(user)
    store = get_store()
    try:
        embedder = get_embedder()
    except ConfigError as exc:
        st.error(str(exc))
        st.stop()
    ocr = get_ocr()
    cache = get_cache()
    indexer = BlockingIndexer(store, embedder, settings, ocr=ocr, cache=cache)
    guard = QuotaGuard(db, settings)
    top_k = sidebar(user, settings, db, guard, indexer, store, cache, ocr is not None)

    docs = store.list_docs(user)
    ui.status_strip(
        [("Sys", "online"), ("Search", "hybrid" if settings.hybrid_search else "vector"), ("User", user)]
    )
    chips: list[str] = []
    if docs:
        chips = [ui.plural(len(docs), "document"), ui.plural(sum(d.page_count for d in docs), "page")]
        scope = st.session_state.scope
        if scope and len(docs) > 1:
            chips.append(f"Searching {len(scope)} of {len(docs)}")
    ui.header("Ask your documents", chips, eyebrow="Workspace", tagline="Answers come only from your files.")

    chain = None
    try:
        chain = Chain(get_llm(), store, embedder, settings)
    except ConfigError as exc:
        st.error(str(exc))

    if not docs:
        ui.empty_state()

    conv_id = st.session_state.conversation_id
    if conv_id is not None and not db.conversation_exists(user, conv_id):
        conv_id = st.session_state.conversation_id = None
    messages = db.get_messages(user, conv_id) if conv_id is not None else []
    pending = st.session_state.pop("pending_question", None)  # a starter question that was just clicked
    for message in messages:
        render_message(message)

    if docs and chain and not messages and not pending:
        ui.welcome()
        render_suggestions()

    typed = st.chat_input("Ask about your documents", disabled=not (docs and chain))
    question = typed or pending
    if not question:
        return

    try:
        guard.check(user)
    except LimitError as exc:
        st.error(str(exc))
        return

    doc_ids = st.session_state.scope or None
    history = [{"role": m["role"], "content": m["content"]} for m in messages]
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        placeholder = st.empty()
        placeholder.markdown("_Searching your documents…_")
        streamed = ""
        result = None
        try:
            for kind, payload in chain.stream(user, question, history, top_k=top_k, doc_ids=doc_ids):
                if kind == "status":
                    placeholder.markdown(f"_{payload}_")
                elif kind == "delta":
                    streamed += payload
                    placeholder.markdown(ui.pretty_citations(streamed) + "▌")
                else:
                    result = payload
        except GiltoError as exc:
            placeholder.empty()
            st.error(str(exc))
            return  # nothing is saved, so a failed or interrupted turn leaves no orphan question

    if conv_id is None:
        conv_id = db.create_conversation(user, question.strip()[:40] or "New chat")
        st.session_state.conversation_id = conv_id
    db.add_message(user, conv_id, "user", question)
    db.add_message(user, conv_id, "assistant", result.answer, [s.to_dict() for s in result.sources])
    guard.record(user, result.tokens_used)
    st.rerun()


main()