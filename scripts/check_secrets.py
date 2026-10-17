"""Block secrets before they reach git.

    python scripts/check_secrets.py          scan what is staged for commit
    python scripts/check_secrets.py --all    scan every tracked file (used in CI)

Exit code 1 means something suspicious was found. The secret itself is never printed.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

FORBIDDEN_FILES = (
    re.compile(r"(^|/)\.env$"),
    re.compile(r"(^|/)secrets\.toml$"),
    re.compile(r"\.db(-wal|-shm)?$"),
)

PATTERNS = {
    "Google API key (AIza)": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "Google API key (AQ)": re.compile(r"\bAQ\.[0-9A-Za-z_\-]{30,}"),
    "Google OAuth client secret": re.compile(r"GOCSPX-[0-9A-Za-z_\-]{20,}"),
    "GitHub token": re.compile(r"\bgh[pousr]_[0-9A-Za-z]{30,}"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "secret assigned a long literal": re.compile(
        r"""(?ix)(api[_-]?key|client[_-]?secret|cookie[_-]?secret|secret[_-]?key|access[_-]?token)
        \s*[:=]\s*["']?[A-Za-z0-9_\-./+]{24,}"""
    ),
}

# Lines that show how to set a value, not a real value.
PLACEHOLDER = re.compile(r"your|replace|example|placeholder|xxx|\.\.\.|changeme|random|token_hex|getenv|environ", re.I)

SKIP_SUFFIXES = (".png", ".jpg", ".jpeg", ".pdf", ".ico", ".woff", ".woff2", ".db")


def is_forbidden(path: str) -> bool:
    return any(pattern.search(path.replace("\\", "/")) for pattern in FORBIDDEN_FILES)


def find_secrets(text: str) -> list[tuple[str, int]]:
    """Return (kind, line number) for each suspicious line. Never returns the secret."""
    found: list[tuple[str, int]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if PLACEHOLDER.search(line):
            continue
        for kind, pattern in PATTERNS.items():
            if pattern.search(line):
                found.append((kind, number))
                break
    return found


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", check=True
    )
    return result.stdout


def _staged_added_text(path: str) -> str:
    diff = _git("diff", "--cached", "-U0", "--", path)
    return "\n".join(line[1:] for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))


def main(argv: list[str]) -> int:
    scan_all = "--all" in argv
    files = _git("ls-files").splitlines() if scan_all else _git("diff", "--cached", "--name-only", "--diff-filter=ACM").splitlines()

    problems: list[str] = []
    for path in files:
        if is_forbidden(path):
            problems.append(f"{path}: this file must never be committed")
            continue
        if path.lower().endswith(SKIP_SUFFIXES):
            continue
        if scan_all:
            try:
                text = Path(path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
        else:
            text = _staged_added_text(path)
        for kind, number in find_secrets(text):
            problems.append(f"{path}:{number}: looks like a {kind}")

    if problems:
        print("Possible secrets found. Nothing was committed.\n", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print("\nRemove the secret and put it in .env or .streamlit/secrets.toml (both are ignored by git).", file=sys.stderr)
        return 1
    print(f"Secret check passed ({len(files)} files).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))