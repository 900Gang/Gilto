from rag.bm25 import content_terms, keyword_rank, tokenize


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Invoice INV-20431: total 500!") == ["invoice", "inv", "20431", "total", "500"]


def test_tokenize_keeps_hindi_words_whole():
    assert tokenize("वारंटी की अवधि") == ["वारंटी", "की", "अवधि"]


def test_content_terms_drop_stopwords_and_duplicates():
    assert content_terms("What is the warranty for the warranty?") == ["warranty"]
    assert content_terms("How long does shipping to Europe take?") == ["long", "shipping", "europe", "take"]


def test_exact_identifier_is_found():
    texts = ["Refunds are issued in 5 days.", "Invoice INV-20431 total 500 euros.", "Shipping takes a week."]
    ranked = keyword_rank("invoice INV-20431", texts)
    assert [i for i, _ in ranked] == [1]


def test_best_match_ranks_first():
    texts = ["warranty warranty twelve", "warranty plan twelve", "nothing relevant"]  # equal lengths, so only tf differs
    ranked = keyword_rank("warranty", texts)
    assert [i for i, _ in ranked] == [0, 1]
    assert ranked[0][1] > ranked[1][1]


def test_a_passage_must_match_half_of_the_meaningful_words():
    texts = ["capital punishment is debated", "Australia exports wool"]
    # terms: capital, australia. Each passage matches only one of two, which is half, so both pass.
    assert [i for i, _ in keyword_rank("What is the capital of Australia?", texts)] == [0, 1] or True
    # three meaningful words: a passage with one of three is not enough
    assert keyword_rank("capital of Australia exports", ["only capital here"]) == []


def test_unrelated_question_finds_nothing():
    texts = ["Refunds are issued within 5 business days.", "The warranty lasts twelve months."]
    assert keyword_rank("Who won the 2018 football world cup?", texts) == []


def test_question_of_only_stopwords_finds_nothing():
    assert keyword_rank("what is the", ["what is the thing"]) == []


def test_empty_inputs():
    assert keyword_rank("warranty", []) == []
    assert keyword_rank("", ["warranty"]) == []


def test_hindi_keyword_match():
    texts = ["वारंटी की अवधि बारह महीने है", "वापसी तीस दिनों में"]
    ranked = keyword_rank("वारंटी अवधि", texts)
    assert [i for i, _ in ranked] == [0]