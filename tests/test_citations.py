from rag.citations import UNVERIFIED_NOTE, verify_citations
from rag.store import Retrieved


def hit(filename, page):
    return Retrieved(text="x", metadata={"filename": filename, "page": page}, score=0.9)


RETRIEVED = [hit("policy.pdf", 3), hit("policy.pdf", 4), hit("guide.pdf", 1)]


def test_valid_citation_is_kept():
    answer = "Refunds take 14 days [policy.pdf, p.3]."
    result = verify_citations(answer, RETRIEVED)
    assert result.text == answer
    assert result.all_valid


def test_invalid_page_is_removed():
    result = verify_citations("Refunds take 14 days [policy.pdf, p.9].", RETRIEVED)
    assert "[policy.pdf, p.9]" not in result.text
    assert result.removed == ["[policy.pdf, p.9]"]
    assert result.text.startswith("Refunds take 14 days.")
    assert result.text.endswith(UNVERIFIED_NOTE)


def test_invented_filename_is_removed():
    result = verify_citations("It is free [secret.pdf, p.1].", RETRIEVED)
    assert result.removed == ["[secret.pdf, p.1]"]
    assert not result.all_valid


def test_mixed_keeps_valid_and_removes_invalid():
    answer = "A [policy.pdf, p.3]. B [fake.pdf, p.2]. C [guide.pdf, p.1]."
    result = verify_citations(answer, RETRIEVED)
    assert "[policy.pdf, p.3]" in result.text
    assert "[guide.pdf, p.1]" in result.text
    assert "[fake.pdf, p.2]" not in result.text
    assert result.removed == ["[fake.pdf, p.2]"]


def test_multi_page_citation_checks_every_page():
    ok = verify_citations("See [policy.pdf, p.3, p.4].", RETRIEVED)
    assert ok.all_valid
    bad = verify_citations("See [policy.pdf, p.3, p.8].", RETRIEVED)
    assert not bad.all_valid


def test_page_range_checks_endpoints():
    assert verify_citations("See [policy.pdf, p.3-4].", RETRIEVED).all_valid
    assert not verify_citations("See [policy.pdf, p.3-7].", RETRIEVED).all_valid


def test_answer_without_citations_is_unchanged():
    result = verify_citations("I couldn't find that in the uploaded documents.", RETRIEVED)
    assert result.all_valid
    assert result.text == "I couldn't find that in the uploaded documents."


def test_filename_with_comma_and_spaces():
    retrieved = [hit("Q3 report, final.pdf", 2)]
    assert verify_citations("x [Q3 report, final.pdf, p.2]", retrieved).all_valid
