"""Tests for kv.chunk — pure logic, no I/O, no tensors."""

from __future__ import annotations

import pytest

from plastic_infer.kv.chunk import (
    ChunkKey,
    compute_prefix_hashes,
    compute_prefix_hashes_and_final,
    longest_prefix_hits,
)


CHUNK = 16  # small chunk size for fast tests


def _tokens(n: int, start: int = 0) -> list[int]:
    return list(range(start, start + n))


class TestChunkKey:
    def test_full_key_is_deterministic(self) -> None:
        k = ChunkKey(prefix_key=b"\x00" * 32, cumulative_key=b"\xaa" * 32)
        assert k.full_key == k.cumulative_key
        assert len(k.full_key) == 32

    def test_different_prefix_different_key(self) -> None:
        k1 = ChunkKey(prefix_key=b"\x00" * 32, cumulative_key=b"\xaa" * 32)
        k2 = ChunkKey(prefix_key=b"\x01" * 32, cumulative_key=b"\xaa" * 32)
        # full_key is cumulative_key; prefix_key distinguishes provenance
        assert k1.prefix_key != k2.prefix_key
        assert k1.full_key == k2.full_key  # same cumulative → same key

    def test_different_cumulative_different_key(self) -> None:
        k1 = ChunkKey(prefix_key=b"\x00" * 32, cumulative_key=b"\xaa" * 32)
        k2 = ChunkKey(prefix_key=b"\x00" * 32, cumulative_key=b"\xbb" * 32)
        assert k1.full_key != k2.full_key


class TestComputePrefixHashes:
    def test_single_chunk(self) -> None:
        keys = compute_prefix_hashes(_tokens(CHUNK), chunk_size=CHUNK)
        assert len(keys) == 1

    def test_multiple_chunks(self) -> None:
        keys = compute_prefix_hashes(_tokens(CHUNK * 4), chunk_size=CHUNK)
        assert len(keys) == 4
        # Chaining: each chunk's prefix = previous chunk's cumulative
        for i in range(1, 4):
            assert keys[i].prefix_key == keys[i - 1].cumulative_key

    def test_deterministic(self) -> None:
        tokens = _tokens(CHUNK * 3, start=42)
        k1 = compute_prefix_hashes(tokens, chunk_size=CHUNK)
        k2 = compute_prefix_hashes(tokens, chunk_size=CHUNK)
        assert [k.full_key for k in k1] == [k.full_key for k in k2]

    def test_prefix_property(self) -> None:
        """If A is a prefix of B, A's keys are a prefix of B's keys."""
        short = _tokens(CHUNK * 2)
        long = short + _tokens(CHUNK * 2, start=100)
        k_short = compute_prefix_hashes(short, chunk_size=CHUNK)
        k_long = compute_prefix_hashes(long, chunk_size=CHUNK)
        assert len(k_short) == 2
        assert len(k_long) == 4
        assert [k.full_key for k in k_short] == [k.full_key for k in k_long[:2]]

    def test_unaligned_raises(self) -> None:
        with pytest.raises(ValueError, match="not aligned"):
            compute_prefix_hashes(_tokens(CHUNK + 1), chunk_size=CHUNK)

    def test_empty(self) -> None:
        keys = compute_prefix_hashes([], chunk_size=CHUNK)
        assert keys == []

    def test_different_content_different_keys(self) -> None:
        k1 = compute_prefix_hashes(_tokens(CHUNK), chunk_size=CHUNK)
        k2 = compute_prefix_hashes(_tokens(CHUNK, start=1),
                                   chunk_size=CHUNK)
        assert k1[0].full_key != k2[0].full_key


class TestFinalHash:
    def test_consistency(self) -> None:
        tokens = _tokens(CHUNK * 3)
        keys, final = compute_prefix_hashes_and_final(tokens, chunk_size=CHUNK)
        # Adding one more chunk: its prefix_key should equal final.
        more = tokens + _tokens(CHUNK, start=1000)
        keys2, final2 = compute_prefix_hashes_and_final(more,
                                                        chunk_size=CHUNK)
        assert keys2[3].prefix_key == final
        assert final != final2  # different after adding chunk

    def test_empty_final_is_salt_hash(self) -> None:
        _, final = compute_prefix_hashes_and_final([], chunk_size=CHUNK)
        assert len(final) == 32


class TestLongestPrefixHits:
    def test_full_hit(self) -> None:
        tokens = _tokens(CHUNK * 3)
        keys, _ = compute_prefix_hashes_and_final(tokens, chunk_size=CHUNK)
        cache = {k.full_key: k for k in keys}
        matched, tail = longest_prefix_hits(cache, tokens, chunk_size=CHUNK)
        assert len(matched) == 3
        assert tail == 3 * CHUNK

    def test_partial_hit(self) -> None:
        tokens = _tokens(CHUNK * 4)
        keys, _ = compute_prefix_hashes_and_final(
            tokens[:2 * CHUNK], chunk_size=CHUNK,
        )
        cache = {k.full_key: k for k in keys}
        matched, tail = longest_prefix_hits(cache, tokens, chunk_size=CHUNK)
        assert len(matched) == 2
        assert tail == 2 * CHUNK

    def test_no_hit(self) -> None:
        tokens = _tokens(CHUNK * 2)
        other_tokens = _tokens(CHUNK * 2, start=9999)
        keys, _ = compute_prefix_hashes_and_final(other_tokens,
                                                  chunk_size=CHUNK)
        cache = {k.full_key: k for k in keys}
        matched, tail = longest_prefix_hits(cache, tokens, chunk_size=CHUNK)
        assert matched == []
        assert tail == 0

    def test_unmatched_second_chunk_stops_early(self) -> None:
        """If chunk 1 misses, don't return chunk 2 even if it's somehow
        in cache (prefix mismatch means it can't be the same prefix)."""
        tokens = _tokens(CHUNK * 3)
        # Cache chunks 0 and 2 but not 1
        keys0, _ = compute_prefix_hashes_and_final(
            tokens[:1 * CHUNK], chunk_size=CHUNK,
        )
        keys2 = compute_prefix_hashes(tokens, chunk_size=CHUNK)
        cache = {keys0[0].full_key: keys0[0],
                 keys2[2].full_key: keys2[2]}
        matched, tail = longest_prefix_hits(cache, tokens, chunk_size=CHUNK)
        # Only chunk 0 should match (chunk 1 misses, so we stop)
        assert len(matched) == 1
        assert tail == CHUNK

    def test_unaligned_tokens_match_full_chunks(self) -> None:
        """3.5 chunks → match up to 3 full chunks."""
        tokens = _tokens(CHUNK * 3 + CHUNK // 2)
        keys, _ = compute_prefix_hashes_and_final(
            tokens[:3 * CHUNK], chunk_size=CHUNK,
        )
        cache = {k.full_key: k for k in keys}
        matched, tail = longest_prefix_hits(cache, tokens, chunk_size=CHUNK)
        assert len(matched) == 3
        assert tail == 3 * CHUNK
