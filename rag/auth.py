"""Who may use Gilto. Pure functions, so they are easy to test without Streamlit."""
from __future__ import annotations

import re

from .errors import GiltoError


class AccessError(GiltoError):
    pass


def parse_allowed(raw: str) -> frozenset[str]:
    """ALLOWED_EMAILS is a list separated by commas, semicolons or spaces."""
    return frozenset(part.strip().lower() for part in re.split(r"[,;\s]+", raw or "") if part.strip())


def check_access(email: str | None, email_verified, allowed: frozenset[str]) -> str:
    """Return the normalized email, or raise AccessError. Fails closed:
    an empty allowlist lets nobody in."""
    clean = (email or "").strip().lower()
    if not clean or "@" not in clean:
        raise AccessError("Your account has no email address, so Gilto cannot identify you.")
    if str(email_verified).lower() == "false":
        raise AccessError("Your Google email address is not verified.")
    if not allowed:
        raise AccessError("No one is allowed to sign in yet. The owner must set ALLOWED_EMAILS.")
    if clean not in allowed:
        raise AccessError(f"{clean} is not authorized to use Gilto. Ask the owner to add you.")
    return clean