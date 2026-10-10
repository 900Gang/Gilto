"""Check `[filename, p.N]` citations in an answer against the chunks that were retrieved."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .store import Retrieved

_CITATION = re.compile(
    r"\[([^\[\]]+?),\s*pp?\.\s*(\d+(?:\s*(?:,|-|–|&|and)\s*(?:pp?\.\s*)?\d+)*)\]",
    re.IGNORECASE,
)

UNVERIFIED_NOTE = "_Note: some citations could not be verified against the documents and were removed._"


@dataclass
class VerifiedAnswer:
    text: str
    removed: list[str]

    @property
    def all_valid(self) -> bool:
        return not self.removed


def verify_citations(answer: str, retrieved: list[Retrieved]) -> VerifiedAnswer:
    allowed = {(r.metadata["filename"], int(r.metadata["page"])) for r in retrieved}
    removed: list[str] = []

    def check(match: re.Match) -> str:
        filename = match.group(1).strip()
        pages = [int(n) for n in re.findall(r"\d+", match.group(2))]
        if all((filename, page) in allowed for page in pages):
            return match.group(0)
        removed.append(match.group(0))
        return ""

    text = _CITATION.sub(check, answer)
    if removed:
        text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)
        text = re.sub(r"[ \t]{2,}", " ", text).rstrip()
        text = f"{text}\n\n{UNVERIFIED_NOTE}"
    return VerifiedAnswer(text=text, removed=removed)
