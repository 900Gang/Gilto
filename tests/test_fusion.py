from rag.fusion import rrf


def test_item_in_both_lists_beats_item_in_one():
    assert rrf([["a", "b"], ["b", "c"]]) == ["b", "a", "c"]


def test_single_list_keeps_its_order():
    assert rrf([["x", "y", "z"]]) == ["x", "y", "z"]


def test_ties_are_broken_by_key():
    assert rrf([["b"], ["a"]]) == ["a", "b"]


def test_empty_rankings():
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_tuple_keys_work():
    assert rrf([[("d1", 0), ("d1", 1)], [("d1", 1)]]) == [("d1", 1), ("d1", 0)]