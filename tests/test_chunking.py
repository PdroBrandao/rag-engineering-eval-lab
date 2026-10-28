import pytest

from retrieval_lab.chunking import chunk_pages, split_tokens, validate_chunk_params


def test_split_tokens_respects_size_and_overlap():
    tokens = list(range(11))
    chunks = split_tokens(tokens, chunk_size=4, overlap=1)
    assert chunks == [
        [0, 1, 2, 3],
        [3, 4, 5, 6],
        [6, 7, 8, 9],
        [9, 10],
    ]
    assert chunks[0][-1:] == chunks[1][:1]


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        validate_chunk_params(4, 4)
    with pytest.raises(ValueError):
        validate_chunk_params(0, 0)
    with pytest.raises(ValueError):
        validate_chunk_params(4, -1)


def test_chunk_pages_keep_page_numbers_and_skip_blank_pages():
    page_one = "alpha " * 12
    page_two = "beta " * 12
    chunks = chunk_pages(
        [(1, page_one), (2, " \n"), (3, page_two)],
        chunk_size=4,
        overlap=1,
    )
    assert 2 not in {chunk.page for chunk in chunks}
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert sum(chunk.page == 1 for chunk in chunks) > 1
    assert sum(chunk.page == 3 for chunk in chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 4
        if chunk.page == 1:
            assert "alpha" in chunk.content
            assert "beta" not in chunk.content
        if chunk.page == 3:
            assert "beta" in chunk.content
            assert "alpha" not in chunk.content


def test_short_page_is_one_chunk():
    chunks = chunk_pages([(1, "alpha beta")], chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].page == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].content == "alpha beta"
