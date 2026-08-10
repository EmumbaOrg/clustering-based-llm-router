from router.common.embedding import normalize_line_endings, prepare_embedding_input


def test_normalizes_crlf_and_bare_cr_to_lf():
    assert normalize_line_endings("a\r\nb\rc\n") == "a\nb\nc\n"


def test_trims_surrounding_whitespace():
    assert prepare_embedding_input("  hello world  \n", max_chars=100) == "hello world"


def test_head_truncates_at_max_chars():
    text = "a" * 100
    assert prepare_embedding_input(text, max_chars=10) == "a" * 10


def test_short_text_only_normalized_not_truncated():
    text = "def foo():\n    return 1\n"
    assert prepare_embedding_input(text, max_chars=1000) == "def foo():\n    return 1"


def test_does_not_collapse_internal_whitespace_or_change_case():
    text = "  Def Foo():\n\tPASS  "
    assert prepare_embedding_input(text, max_chars=1000) == "Def Foo():\n\tPASS"
