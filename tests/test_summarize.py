import pytest

from rag.store import Retrieved
from rag.summarize import batch_chunks, is_summary_request


def chunk(text):
    return Retrieved(text=text, metadata={"filename": "a.pdf", "page": 1}, score=None)


@pytest.mark.parametrize(
    "question",
    [
        "Summarize this document",
        "Please summarise the report",
        "give me an overview",
        "tl;dr please",
        "What are the main points?",
        "what is this pdf about",
        "Compare the two contracts",
        "Write a summary of the policy",
    ],
)
def test_summary_requests_are_detected(question):
    assert is_summary_request(question)


@pytest.mark.parametrize(
    "question",
    [
        "How long do I have to return an item?",
        "What is the warranty period?",
        "Is the summerhouse covered?",
        "Who won the 2018 football world cup?",
    ],
)
def test_normal_questions_are_not_summary_requests(question):
    assert not is_summary_request(question)


def test_batches_respect_the_size_limit_and_order():
    chunks = [chunk("a" * 40), chunk("b" * 40), chunk("c" * 40), chunk("d" * 10)]
    batches = batch_chunks(chunks, 100)
    assert [[c.text[0] for c in b] for b in batches] == [["a", "b"], ["c", "d"]]


def test_oversized_chunk_gets_its_own_batch():
    batches = batch_chunks([chunk("a" * 10), chunk("b" * 500), chunk("c" * 10)], 100)
    assert [len(b) for b in batches] == [1, 1, 1]


def test_no_chunks_gives_no_batches():
    assert batch_chunks([], 100) == []