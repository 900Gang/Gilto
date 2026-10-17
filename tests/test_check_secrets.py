import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "check_secrets", Path(__file__).resolve().parent.parent / "scripts" / "check_secrets.py"
)
check_secrets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_secrets)

find_secrets = check_secrets.find_secrets
is_forbidden = check_secrets.is_forbidden


def test_google_api_keys_are_flagged_without_printing_them():
    text = 'key = "' + "AIza" + "B" * 35 + '"'
    assert find_secrets(text) == [("Google API key (AIza)", 1)]
    newer = "token: " + "AQ." + "x1" * 20
    assert find_secrets(newer) == [("Google API key (AQ)", 1)]


def test_oauth_client_secret_and_private_key_are_flagged():
    assert find_secrets("s = " + "GOCSPX-" + "a" * 24)[0][0] == "Google OAuth client secret"
    assert find_secrets("-----BEGIN " + "RSA PRIVATE KEY-----")[0][0] == "private key"


def test_a_long_literal_assigned_to_a_secret_name_is_flagged():
    line = 'cookie_secret = "' + "a1B2" * 8 + '"'
    assert find_secrets(line) == [("secret assigned a long literal", 1)]


def test_line_numbers_point_at_the_right_line():
    text = "fine\nstill fine\n" + 'api_key = "' + "Q9z" * 10 + '"\n'
    assert find_secrets(text) == [("secret assigned a long literal", 3)]


@pytest.mark.parametrize(
    "line",
    [
        'client_secret = "your-client-secret"',
        'cookie_secret = "a-new-random-string"',
        "GEMINI_API_KEY=",
        'api_key=os.getenv("GEMINI_API_KEY")',
        'GEMINI_API_KEY = "your-gemini-key"',
        'client_id = "your-id.apps.googleusercontent.com"',
        "ALLOWED_EMAILS=you@gmail.com",
    ],
)
def test_placeholders_and_environment_reads_are_not_flagged(line):
    assert find_secrets(line) == []


@pytest.mark.parametrize(
    "path",
    [".env", "sub/.env", ".streamlit/secrets.toml", "gilto.db", "gilto.db-wal", "data\\gilto.db-shm"],
)
def test_secret_and_data_files_are_forbidden(path):
    assert is_forbidden(path)


@pytest.mark.parametrize("path", [".env.example", ".streamlit/secrets.toml.example", "app.py", "README.md"])
def test_templates_and_code_are_allowed(path):
    assert not is_forbidden(path)