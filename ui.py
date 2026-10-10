"""Look and feel: theme CSS and small HTML components.

Only static markup and escaped user text go through unsafe_allow_html.
Model output never does.
"""
from __future__ import annotations

import html
import re

import streamlit as st

CSS = """
<style>
.block-container { max-width: 860px; padding-top: 2.2rem; padding-bottom: 6rem; }
[data-testid="stSidebar"] { border-right: 1px solid #E5E7EB; }
[data-testid="stSidebar"] .stButton > button { width: 100%; }
.stButton > button { border-radius: 8px; font-weight: 500; }
h1 { font-weight: 650; letter-spacing: -0.02em; font-size: 1.8rem; }

.g-brand { display: flex; align-items: center; gap: .65rem; margin-bottom: .2rem; }
.g-logo { width: 34px; height: 34px; border-radius: 9px; background: #4F46E5; color: #fff;
  display: grid; place-items: center; font-weight: 700; font-size: 1.05rem; }
.g-brand-name { font-weight: 650; font-size: 1.2rem; letter-spacing: -0.01em; }
.g-tagline { color: #6B7280; font-size: .84rem; margin-bottom: .9rem; }
.g-label { text-transform: uppercase; font-size: .7rem; letter-spacing: .09em; color: #6B7280;
  font-weight: 600; margin: 1.1rem 0 .45rem; }
.g-note { color: #6B7280; font-size: .76rem; line-height: 1.4; margin-top: 1rem; }

.g-doc { border: 1px solid #E5E7EB; border-radius: 10px; padding: .5rem .75rem; background: #fff; }
.g-doc-name { font-weight: 600; font-size: .88rem; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
.g-doc-meta { color: #6B7280; font-size: .76rem; margin-top: .1rem; }

.g-sub { color: #6B7280; font-size: .92rem; margin: -.4rem 0 1.4rem; }
.g-empty { border: 1px solid #E5E7EB; border-radius: 14px; padding: 1.8rem 1.8rem 1.4rem;
  background: #F7F8FA; margin-top: .5rem; }
.g-empty-title { font-size: 1.2rem; font-weight: 650; letter-spacing: -0.01em; }
.g-empty-sub { color: #6B7280; margin: .3rem 0 1.3rem; font-size: .95rem; }
.g-steps { display: grid; gap: .9rem; }
.g-step { display: flex; gap: .85rem; align-items: flex-start; }
.g-step-num { flex: none; width: 26px; height: 26px; border-radius: 50%; background: #4F46E5;
  color: #fff; display: grid; place-items: center; font-size: .8rem; font-weight: 600; }
.g-step-title { font-weight: 600; font-size: .95rem; }
.g-step-body { color: #6B7280; font-size: .86rem; }

[data-testid="stChatMessage"] { border: 1px solid #E5E7EB; border-radius: 12px;
  padding: .9rem 1.1rem; background: #fff; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: #F7F8FA; }
[data-testid="stChatMessage"] code { background: #EEF2FF; color: #3730A3; border-radius: 6px;
  padding: .08rem .4rem; font-size: .78rem; }
[data-testid="stChatInput"] { border-radius: 14px; }

.g-source { border: 1px solid #E5E7EB; border-radius: 10px; padding: .6rem .8rem;
  margin-bottom: .6rem; background: #fff; }
.g-source-head { display: flex; flex-wrap: wrap; align-items: center; gap: .45rem; }
.g-source-name { font-weight: 600; font-size: .86rem; }
.g-pill { background: #EEF2FF; color: #3730A3; border-radius: 999px; padding: .05rem .5rem;
  font-size: .72rem; font-weight: 600; }
.g-pill-muted { background: #F3F4F6; color: #4B5563; }
.g-score { margin-left: auto; color: #6B7280; font-size: .74rem; }
.g-source-text { color: #374151; font-size: .84rem; margin-top: .4rem; white-space: pre-wrap; }
</style>
"""

_CITATION = re.compile(
    r"\[([^\[\]]+?),\s*pp?\.\s*(\d+(?:\s*(?:,|-|–|&|and)\s*(?:pp?\.\s*)?\d+)*)\]",
    re.IGNORECASE,
)


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def brand() -> None:
    st.markdown(
        '<div class="g-brand"><div class="g-logo">G</div><div class="g-brand-name">Gilto</div></div>'
        '<div class="g-tagline">Answers come only from your files.</div>',
        unsafe_allow_html=True,
    )


def label(text: str) -> None:
    st.markdown(f'<div class="g-label">{html.escape(text)}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="g-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def subtitle(text: str) -> None:
    st.markdown(f'<div class="g-sub">{html.escape(text)}</div>', unsafe_allow_html=True)


def doc_card(filename: str, pages: int, chunks: int) -> str:
    name = html.escape(filename)
    return (
        f'<div class="g-doc"><div class="g-doc-name" title="{name}">{name}</div>'
        f'<div class="g-doc-meta">{int(pages)} pages · {int(chunks)} chunks</div></div>'
    )


def empty_state() -> None:
    steps = [
        ("1", "Add documents", "Upload PDFs in the sidebar and choose Add to library."),
        ("2", "Ask a question", "Ask about anything in your files."),
        ("3", "Check the sources", "Every answer cites the page it came from."),
    ]
    items = "".join(
        f'<div class="g-step"><div class="g-step-num">{n}</div>'
        f'<div><div class="g-step-title">{t}</div><div class="g-step-body">{b}</div></div></div>'
        for n, t, b in steps
    )
    st.markdown(
        '<div class="g-empty"><div class="g-empty-title">Chat with your own documents</div>'
        '<div class="g-empty-sub">If the answer is not in your files, Gilto says so instead of guessing.</div>'
        f'<div class="g-steps">{items}</div></div>',
        unsafe_allow_html=True,
    )


def source_card(filename: str, page: int, score: float, source_type: str, text: str) -> str:
    snippet = html.escape(text[:500]) + ("…" if len(text) > 500 else "")
    return (
        '<div class="g-source"><div class="g-source-head">'
        f'<span class="g-source-name">{html.escape(filename)}</span>'
        f'<span class="g-pill">p.{int(page)}</span>'
        f'<span class="g-pill g-pill-muted">{html.escape(source_type)}</span>'
        f'<span class="g-score">relevance {score:.2f}</span></div>'
        f'<div class="g-source-text">{snippet}</div></div>'
    )


def pretty_citations(text: str) -> str:
    """Turn [file, p.3] into an inline-code chip. Plain markdown, so no raw HTML from the model."""

    def chip(match: re.Match) -> str:
        name = match.group(1).strip().replace("`", "")
        return f"`{name} · p.{match.group(2).strip()}`"

    return _CITATION.sub(chip, text)