import pytest

from retrieval_lab.rrf import fuse_rankings


def test_rrf_uses_constant_60_and_deduplicates():
    fused = fuse_rankings([{1: 1, 2: 2}, {2: 1, 3: 2}], rrf_k=60)
    scores = dict(fused)
    assert [chunk_id for chunk_id, _score in fused] == [2, 1, 3]
    assert scores[1] == pytest.approx(1 / 61)
    assert scores[2] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[3] == pytest.approx(1 / 62)
    assert len(fused) == 3


def test_chunk_present_in_only_one_list_still_scores():
    fused = fuse_rankings([{7: 1}, {}], rrf_k=60)
    assert fused == [(7, pytest.approx(1 / 61))]


def test_rrf_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        fuse_rankings([{1: 1}], rrf_k=0)
    with pytest.raises(ValueError):
        fuse_rankings([{1: 0}], rrf_k=60)
