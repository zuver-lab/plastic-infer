"""KV chunking and deterministic prefix hashing.

Design rules (from §5.5 / D9 / D10 of DESIGN.md):
  - Chunk = CHUNK_SIZE tokens (default 256), aligned to page boundaries.
  - ChunkKey is a deterministic chained hash over token content.
  - Hash is computed incrementally over a token stream, so identical
    prefixes produce identical keys regardless of process (D9).
  - We never use builtin hash or process-local IDs.

This module is pure logic — no tensors, no I/O.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


# Defaults — kept small so tests run fast
DEFAULT_CHUNK_SIZE: int = 256
# Salt for the very first chunk, so empty prefix has a well-defined key.
_INITIAL_SALT = b"plastic-infer-kv-v1"


@dataclass(frozen=True)
class ChunkKey:
    """A deterministic content-address for one KV chunk.

    `prefix_key` = cumulative hash of all chunks before this one.
    `cumulative_key` = cumulative hash including this chunk's content.

    Both are derived from token content only (D9): same tokens → same
    keys, regardless of process.
    """

    prefix_key: bytes       # sha256 digest up to (not including) this chunk
    cumulative_key: bytes   # sha256 digest up to and including this chunk

    def __repr__(self) -> str:
        return f"ChunkKey({self.cumulative_key.hex()[:16]})"

    @property
    def full_key(self) -> bytes:
        """Alias for the key used in cache lookups."""
        return self.cumulative_key


def compute_prefix_hashes(
    tokens: list[int],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    initial_salt: bytes = _INITIAL_SALT,
) -> list[ChunkKey]:
    """Compute the chained chunk key for each chunk of `tokens`.

    Hash function:
      h_0 = sha256(salt)                     # prefix of chunk 0
      c_i = sha256(tokens[i*chunk_size : (i+1)*chunk_size])
      h_{i+1} = sha256(h_i + c_i)           # cumulative after chunk i

    ChunkKey i has prefix_key = h_i, cumulative_key = h_{i+1}.
    Token list must be aligned to chunk_size; partial chunks raise.
    """
    keys, _ = _compute_hashes(tokens, chunk_size, initial_salt)
    return keys


def compute_prefix_hashes_and_final(
    tokens: list[int],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    initial_salt: bytes = _INITIAL_SALT,
) -> tuple[list[ChunkKey], bytes]:
    """Same as compute_prefix_hashes, plus the final cumulative hash.

    The final hash is the cumulative_key of the last chunk, i.e. the
    prefix_key of the next chunk yet to be written.
    """
    return _compute_hashes(tokens, chunk_size, initial_salt)


def _compute_hashes(
    tokens: list[int],
    chunk_size: int,
    initial_salt: bytes,
) -> tuple[list[ChunkKey], bytes]:
    if len(tokens) % chunk_size != 0:
        raise ValueError(
            f"token count {len(tokens)} not aligned to chunk_size {chunk_size}"
        )

    n_chunks = len(tokens) // chunk_size
    keys: list[ChunkKey] = []
    running = hashlib.sha256(initial_salt).digest()

    for _ in range(n_chunks):
        chunk_tokens = tokens[len(keys) * chunk_size:
                              (len(keys) + 1) * chunk_size]

        # Hash this chunk's content
        chunk_hash = hashlib.sha256()
        for tok in chunk_tokens:
            chunk_hash.update(tok.to_bytes(4, "little", signed=False))

        # Cumulative key = sha256(prefix_key + chunk_content_hash)
        combined = hashlib.sha256(running)
        combined.update(chunk_hash.digest())
        cumulative = combined.digest()

        keys.append(ChunkKey(prefix_key=running, cumulative_key=cumulative))
        running = cumulative

    return keys, running


def longest_prefix_hits(
    cache_index: dict[bytes, ChunkKey],
    tokens: list[int],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[list[ChunkKey], int]:
    """Find the longest prefix of `tokens` that is already in cache.

    Args:
        cache_index: maps full_key (bytes) -> ChunkKey
        tokens: token stream to match against
        chunk_size: tokens per chunk

    Returns:
        (matched_keys, tail_start_token):
        - matched_keys: list of cached ChunkKeys for the prefix, in order
        - tail_start_token: index where the unmatched tail begins
          (== len(matched_keys) * chunk_size)

    Only matches whole chunks. If token count isn't chunk-aligned,
    we still match as many full chunks as we can; the remainder is
    part of the tail.
    """
    full_chunks = len(tokens) // chunk_size
    keys, final_h = compute_prefix_hashes_and_final(
        tokens[:full_chunks * chunk_size], chunk_size=chunk_size,
    )

    matched: list[ChunkKey] = []
    for key in keys:
        if key.full_key in cache_index:
            matched.append(key)
        else:
            break

    tail_start = len(matched) * chunk_size
    return matched, tail_start
