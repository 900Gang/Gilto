import pytest

from rag.auth import AccessError, check_access, parse_allowed


def test_parse_allowed_handles_separators_and_case():
    assert parse_allowed("A@x.com, b@x.com;C@X.com  d@x.com") == {"a@x.com", "b@x.com", "c@x.com", "d@x.com"}
    assert parse_allowed("") == frozenset()
    assert parse_allowed(None) == frozenset()


def test_allowed_user_is_normalized():
    assert check_access(" Alice@X.com ", True, parse_allowed("alice@x.com")) == "alice@x.com"


def test_unlisted_user_is_denied():
    with pytest.raises(AccessError, match="not authorized"):
        check_access("eve@x.com", True, parse_allowed("alice@x.com"))


def test_empty_allowlist_denies_everyone():
    with pytest.raises(AccessError, match="ALLOWED_EMAILS"):
        check_access("alice@x.com", True, frozenset())


@pytest.mark.parametrize("email", [None, "", "   ", "not-an-email"])
def test_missing_or_invalid_email_is_denied(email):
    with pytest.raises(AccessError, match="no email"):
        check_access(email, True, parse_allowed("alice@x.com"))


@pytest.mark.parametrize("verified", [False, "false", "False"])
def test_unverified_email_is_denied(verified):
    with pytest.raises(AccessError, match="not verified"):
        check_access("alice@x.com", verified, parse_allowed("alice@x.com"))


def test_missing_verified_claim_is_accepted():
    assert check_access("alice@x.com", None, parse_allowed("alice@x.com")) == "alice@x.com"