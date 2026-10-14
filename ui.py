"""Look and feel: a dark, lime, monospace theme and small HTML components.

Only static markup and escaped user text go through unsafe_allow_html.
Model output never does.
"""
from __future__ import annotations

import html
import re
from collections.abc import Iterator
from contextlib import contextmanager

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

:root {
  --lime: #C8F31D; --lime-dim: rgba(200, 243, 29, .28); --ink: #0A0B09; --panel: #11130E;
  --line: #2A2E22; --text: #E6E8DF; --muted: #8A9080; --danger: #FF5A47;
  --display: 'Space Mono', ui-monospace, 'Courier New', monospace;
}

/* ---- page frame ---- */
.stApp {
  background-color: var(--ink);
  background-image: linear-gradient(rgba(200, 243, 29, .035) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(200, 243, 29, .035) 1px, transparent 1px);
  background-size: 44px 44px;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 840px !important; padding-top: 1.6rem; padding-bottom: 7rem; }
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"] { background: var(--ink) !important; }
[data-testid="stBottomBlockContainer"] { max-width: 840px !important; margin: 0 auto; }
[data-testid="stSidebar"] { background: #0D0F0B; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .55rem; }
[data-testid="stSidebar"] .stButton > button { width: 100%; }
*::-webkit-scrollbar { width: 8px; height: 8px; }
*::-webkit-scrollbar-track { background: transparent; }
*::-webkit-scrollbar-thumb { background: var(--line); }
*::-webkit-scrollbar-thumb:hover { background: var(--lime); }

/* ---- square everything ---- */
.stButton > button, [data-testid="stExpander"], [data-testid="stExpander"] details,
[data-testid="stChatInput"], [data-testid="stChatInput"] textarea, [data-testid="stFileUploaderDropzone"],
[data-testid="stAlert"], [data-baseweb="select"] > div, [data-baseweb="tag"],
[data-testid="stChatInputSubmitButton"] { border-radius: 0 !important; }

/* ---- buttons ---- */
.stButton > button {
  text-transform: uppercase; letter-spacing: .07em; font-size: .74rem; font-weight: 600;
  background: transparent; color: var(--text); border: 1px solid var(--lime-dim);
  transition: border-color .12s, color .12s, background .12s;
}
.stButton > button:hover { border-color: var(--lime); color: var(--lime); }
.stButton > button:focus-visible { outline: 2px solid var(--lime); outline-offset: 2px; }
.stButton > button:disabled { opacity: .4; }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
  background: var(--lime); border-color: var(--lime); color: var(--ink) !important;
}
.stButton > button[kind="primary"] p, .stButton > button[data-testid="stBaseButton-primary"] p { color: var(--ink) !important; }
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
  background: #DDFF4A; border-color: #DDFF4A; color: var(--ink) !important;
}
[data-testid="stExpander"] { background: var(--panel); border: 1px solid var(--line); }
[data-testid="stExpander"] summary { text-transform: uppercase; letter-spacing: .08em; font-size: .74rem; }
[data-testid="stFileUploaderDropzone"] { background: var(--panel); border: 1px dashed var(--lime-dim); padding: .8rem; }

/* ---- corner-bracket panels ---- */
.g-panel { position: relative; border: 1px solid var(--line); background: var(--panel); }
.g-panel::before, .g-panel::after,
[data-testid="stVerticalBlockBorderWrapper"]:has(.g-auth-title)::before,
[data-testid="stVerticalBlockBorderWrapper"]:has(.g-auth-title)::after {
  content: ""; position: absolute; width: 9px; height: 9px; pointer-events: none;
}
.g-panel::before, [data-testid="stVerticalBlockBorderWrapper"]:has(.g-auth-title)::before {
  top: -1px; left: -1px; border-top: 2px solid var(--lime); border-left: 2px solid var(--lime);
}
.g-panel::after, [data-testid="stVerticalBlockBorderWrapper"]:has(.g-auth-title)::after {
  bottom: -1px; right: -1px; border-bottom: 2px solid var(--lime); border-right: 2px solid var(--lime);
}

/* ---- brand, labels, strip, hazard ---- */
.g-logo-box { display: inline-block; background: var(--lime); color: var(--ink); font-family: var(--display);
  font-weight: 700; letter-spacing: .06em; padding: .3rem .8rem; font-size: 1.05rem; }
.g-tagline { color: var(--muted); font-size: .76rem; letter-spacing: .04em; margin: .35rem 0 .5rem; }
.g-label { text-transform: uppercase; font-size: .68rem; letter-spacing: .16em; color: var(--muted); margin: 1.1rem 0 .15rem; }
.g-label b { color: var(--lime); font-weight: 700; margin-right: .55rem; }
.g-note { color: #6C7263; font-size: .7rem; line-height: 1.45; margin-top: .8rem; }
.g-hazard { height: 10px; margin-top: 1.2rem;
  background: repeating-linear-gradient(135deg, var(--lime) 0 12px, var(--ink) 12px 24px); }
.g-strip { display: flex; flex-wrap: wrap; border: 1px solid var(--line); background: var(--panel); margin-bottom: 1.3rem; }
.g-cell { padding: .35rem .85rem; border-right: 1px solid var(--line); font-size: .66rem; letter-spacing: .12em;
  text-transform: uppercase; color: var(--muted); }
.g-cell:last-child { border-right: none; }
.g-cell b { color: var(--lime); font-weight: 600; margin-left: .45rem; }

/* ---- page header ---- */
.g-eyebrow { color: var(--lime); font-size: .72rem; letter-spacing: .16em; text-transform: uppercase; }
.g-header-title { font-family: var(--display); font-weight: 700; font-size: 2.1rem; line-height: 1.1;
  letter-spacing: -.01em; text-transform: uppercase; color: #fff; margin: .25rem 0 .5rem; }
.g-tag-lime { color: var(--lime); font-size: .8rem; letter-spacing: .06em; text-transform: uppercase; }
.g-chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: .8rem 0 1.4rem; }
.g-chip { border: 1px solid var(--line); color: var(--muted); padding: .1rem .55rem; font-size: .68rem;
  letter-spacing: .1em; text-transform: uppercase; }
.g-cursor::after { content: "_"; color: var(--lime); animation: g-blink 1.1s steps(2, start) infinite; }
@keyframes g-blink { to { visibility: hidden; } }
@media (prefers-reduced-motion: reduce) { .g-cursor::after { animation: none; } }

/* ---- sidebar cards ---- */
.g-doc { padding: .5rem .75rem; }
.g-doc-name { font-weight: 600; font-size: .82rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.g-doc-meta { color: var(--muted); font-size: .7rem; margin-top: .1rem; letter-spacing: .04em; }
.st-key-chats .stButton > button { justify-content: flex-start; text-align: left; text-transform: none;
  letter-spacing: 0; font-weight: 400; font-size: .8rem; }
.st-key-chats .stButton > button[kind="primary"], .st-key-chats .stButton > button[data-testid="stBaseButton-primary"] { font-weight: 700; }
.g-usage { padding: .7rem .8rem; }
.g-usage-title { font-size: .68rem; letter-spacing: .16em; color: var(--lime); text-transform: uppercase; margin-bottom: .45rem; }
.g-bar-row { display: grid; grid-template-columns: 5.4rem 1fr auto; gap: .5rem; align-items: center;
  font-size: .64rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; margin: .3rem 0; }
.g-bar { height: 6px; background: #1C2016; border: 1px solid var(--line); }
.g-bar i { display: block; height: 100%; background: var(--lime); }
.g-bar.hot i { background: var(--danger); }
.g-ok, .g-warn { margin-top: .55rem; padding: .3rem .5rem; font-size: .62rem; letter-spacing: .12em; text-transform: uppercase;
  border: 1px solid var(--lime-dim); color: var(--lime); background: rgba(200, 243, 29, .06); }
.g-warn { border-color: rgba(255, 90, 71, .5); color: var(--danger); background: rgba(255, 90, 71, .07); }

/* ---- welcome, empty state, suggestions ---- */
.g-welcome-title { font-family: var(--display); font-weight: 700; font-size: 1rem; text-transform: uppercase;
  margin-top: 1.1rem; letter-spacing: .02em; }
.g-welcome-sub { color: var(--muted); font-size: .82rem; margin: .25rem 0 .9rem; }
.st-key-suggestions .stButton > button { width: 100%; justify-content: flex-start; text-align: left;
  background: var(--panel); border: 1px solid var(--line); padding: .7rem .95rem; color: var(--text); }
.st-key-suggestions .stButton > button:hover { border-color: var(--lime); color: var(--lime); }
.g-empty { padding: 1.5rem 1.5rem 1.3rem; margin-top: .4rem; }
.g-empty-title { font-family: var(--display); font-weight: 700; font-size: 1.15rem; text-transform: uppercase; }
.g-empty-sub { color: var(--muted); margin: .35rem 0 1.3rem; font-size: .85rem; }
.g-steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; }
.g-step { border-top: 1px solid var(--line); padding-top: .7rem; }
.g-step-num { color: var(--lime); font-size: .7rem; letter-spacing: .14em; }
.g-step-title { font-weight: 700; font-size: .85rem; text-transform: uppercase; margin: .2rem 0; }
.g-step-body { color: var(--muted); font-size: .76rem; line-height: 1.5; }

/* ---- chat ---- */
[data-testid^="stChatMessageAvatar"] { display: none !important; }
[data-testid="stChatMessage"] { background: transparent; border: none; padding: .55rem 0; gap: 1rem; }
[data-testid="stChatMessage"]::before { flex: none; width: 3.6rem; padding-top: .3rem; font-size: .66rem; letter-spacing: .14em; color: var(--muted); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])::before { content: "YOU"; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])::before { content: "GILTO"; color: var(--lime); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: var(--panel); border: 1px solid var(--line); padding: .7rem 1rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) { border-left: 2px solid var(--lime); padding-left: 1rem; }
[data-testid="stChatMessage"] code { background: rgba(200, 243, 29, .1); color: var(--lime); border: 1px solid var(--lime-dim);
  border-radius: 0; padding: .05rem .4rem; font-size: .72rem; }
[data-testid="stChatInput"] { background: var(--panel); border: 1px solid var(--lime-dim); }
[data-testid="stChatInput"]:focus-within { border-color: var(--lime); box-shadow: 0 0 0 1px var(--lime); }
[data-testid="stChatInputSubmitButton"] { background: var(--lime) !important; color: var(--ink) !important; }

.g-source { padding: .6rem .8rem; margin-bottom: .6rem; }
.g-source-head { display: flex; flex-wrap: wrap; align-items: center; gap: .45rem; }
.g-source-name { font-weight: 700; font-size: .8rem; }
.g-pill { border: 1px solid var(--lime-dim); color: var(--lime); padding: .02rem .45rem; font-size: .66rem; letter-spacing: .06em; }
.g-pill-muted { border-color: var(--line); color: var(--muted); }
.g-score { margin-left: auto; color: var(--muted); font-size: .68rem; text-transform: uppercase; letter-spacing: .08em; }
.g-source-text { color: #B9BDAF; font-size: .78rem; margin-top: .45rem; white-space: pre-wrap; line-height: 1.5; }

/* ---- sign-in card ---- */
[data-testid="stVerticalBlockBorderWrapper"]:has(.g-auth-title) { position: relative; border: 1px solid var(--line);
  background: var(--panel); border-radius: 0; padding: 1.4rem 1.2rem 1.2rem; }
[data-testid="stVerticalBlockBorderWrapper"]:has(.g-auth-title) .stButton > button { width: 100%; padding: .65rem 1rem; }
.g-auth-spacer { height: 4vh; }
.g-auth-title { font-family: var(--display); font-weight: 700; font-size: 1.5rem; letter-spacing: .02em;
  text-transform: uppercase; margin: 1.2rem 0 .4rem; }
.g-auth-title.danger { color: var(--danger); }
.g-auth-body { color: var(--muted); font-size: .85rem; line-height: 1.55; margin-bottom: 1.1rem; }
.g-foot { color: #6C7263; font-size: .7rem; letter-spacing: .08em; text-align: center; margin-top: .9rem; text-transform: uppercase; }
</style>
"""

_CITATION = re.compile(
    r"\[([^\[\]]+?),\s*pp?\.\s*(\d+(?:\s*(?:,|-|–|&|and)\s*(?:pp?\.\s*)?\d+)*)\]",
    re.IGNORECASE,
)

_BRAND = '<div class="g-logo-box">GILTO_</div>'


def plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def chat_title(title: str, limit: int = 34) -> str:
    """A saved chat title tidied for the sidebar: one line, capitalized, shortened with an ellipsis."""
    text = " ".join(title.split())
    text = text[:1].upper() + text[1:]
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def brand() -> None:
    st.markdown(
        f'{_BRAND}<div class="g-tagline">Answers come only from your files.</div>',
        unsafe_allow_html=True,
    )


def label(text: str, num: int | None = None) -> None:
    prefix = f"<b>/{num:02d}</b>" if num is not None else ""
    st.markdown(f'<div class="g-label">{prefix}{html.escape(text)}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="g-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def hazard() -> None:
    st.markdown('<div class="g-hazard"></div>', unsafe_allow_html=True)


def status_strip(cells: list[tuple[str, str]]) -> None:
    items = "".join(
        f'<div class="g-cell">{html.escape(name)}<b>{html.escape(value)}</b></div>' for name, value in cells
    )
    st.markdown(f'<div class="g-strip">{items}</div>', unsafe_allow_html=True)


def header(title: str, chips: list[str] | None = None, eyebrow: str = "", tagline: str = "") -> None:
    chip_html = "".join(f'<span class="g-chip">{html.escape(c)}</span>' for c in chips or [])
    st.markdown(
        (f'<div class="g-eyebrow">/ {html.escape(eyebrow)}</div>' if eyebrow else "")
        + f'<div class="g-header-title g-cursor">{html.escape(title)}</div>'
        + (f'<div class="g-tag-lime">{html.escape(tagline)}</div>' if tagline else "")
        + f'<div class="g-chips">{chip_html}</div>',
        unsafe_allow_html=True,
    )


def usage_panel(rows: list[tuple[str, float, str]], low: bool) -> None:
    """rows are (name, fraction used from 0 to 1, text on the right)."""
    bars = ""
    for name, used, text in rows:
        pct = max(0.0, min(1.0, used)) * 100
        hot = " hot" if used >= 0.9 else ""
        bars += (
            f'<div class="g-bar-row"><span>{html.escape(name)}</span>'
            f'<div class="g-bar{hot}"><i style="width:{pct:.0f}%"></i></div>'
            f"<span>{html.escape(text)}</span></div>"
        )
    status = '<div class="g-warn">Quota low</div>' if low else '<div class="g-ok">All systems operational</div>'
    st.markdown(
        f'<div class="g-panel g-usage"><div class="g-usage-title">System status</div>{bars}{status}</div>',
        unsafe_allow_html=True,
    )


def welcome() -> None:
    st.markdown(
        '<div class="g-welcome-title">What do you want to know?</div>'
        '<div class="g-welcome-sub">Ask about your documents, or start with one of these.</div>',
        unsafe_allow_html=True,
    )


def doc_card(filename: str, pages: int, chunks: int) -> str:
    name = html.escape(filename)
    return (
        f'<div class="g-panel g-doc"><div class="g-doc-name" title="{name}">{name}</div>'
        f'<div class="g-doc-meta">{plural(int(pages), "page")} · {plural(int(chunks), "passage")}</div></div>'
    )


def empty_state() -> None:
    steps = [
        ("01", "Add documents", "Upload PDFs or images in the sidebar and choose Add to library."),
        ("02", "Ask a question", "Ask about anything in your files, or ask for a summary."),
        ("03", "Check the sources", "Every answer cites the page it came from."),
    ]
    items = "".join(
        f'<div class="g-step"><div class="g-step-num">/{n}</div>'
        f'<div class="g-step-title">{t}</div><div class="g-step-body">{b}</div></div>'
        for n, t, b in steps
    )
    st.markdown(
        '<div class="g-panel g-empty"><div class="g-empty-title g-cursor">No documents loaded</div>'
        '<div class="g-empty-sub">If the answer is not in your files, Gilto says so instead of guessing.</div>'
        f'<div class="g-steps">{items}</div></div>',
        unsafe_allow_html=True,
    )


def source_card(filename: str, page: int, score: float | None, source_type: str, text: str) -> str:
    snippet = html.escape(text[:500]) + ("…" if len(text) > 500 else "")
    relevance = "keyword match" if score is None else f"relevance {score:.2f}"
    return (
        '<div class="g-panel g-source"><div class="g-source-head">'
        f'<span class="g-source-name">{html.escape(filename)}</span>'
        f'<span class="g-pill">p.{int(page)}</span>'
        f'<span class="g-pill g-pill-muted">{html.escape(source_type)}</span>'
        f'<span class="g-score">{relevance}</span></div>'
        f'<div class="g-source-text">{snippet}</div></div>'
    )


def pretty_citations(text: str) -> str:
    """Turn [file, p.3] into an inline-code chip. Plain markdown, so no raw HTML from the model."""

    def chip(match: re.Match) -> str:
        name = match.group(1).strip().replace("`", "")
        return f"`{name} · p.{match.group(2).strip()}`"

    return _CITATION.sub(chip, text)


@contextmanager
def auth_card(title: str, body: str = "", footnote: str = "", tone: str = "neutral") -> Iterator[None]:
    """A centered card for sign-in and access messages. Put buttons inside the `with` block.
    tone="danger" colors the title red."""
    _, middle, _ = st.columns([1, 1.5, 1])
    with middle:
        st.markdown('<div class="g-auth-spacer"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            danger = " danger" if tone == "danger" else ""
            st.markdown(
                f'{_BRAND}<div class="g-auth-title{danger}">{html.escape(title)}</div>'
                + (f'<div class="g-auth-body">{html.escape(body)}</div>' if body else ""),
                unsafe_allow_html=True,
            )
            yield
        if footnote:
            st.markdown(f'<div class="g-foot">{html.escape(footnote)}</div>', unsafe_allow_html=True)
        hazard()