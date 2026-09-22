"""HostExpertLru tests: host LRU over a disk expert source (HOST_FIRST).

The middle tier of the three-tier expert story: a byte-budgeted host LRU
whose backing store is DiskExpertSource. Verifies:
  - miss -> load from disk -> hit on re-access (counters move right)
  - budget eviction: oldest expert evicted, re-fetched byte-identical
    on the next miss (D5: placement never changes values)
  - budget smaller than one expert is refused (no silent overflow)
  - duck-types ExpertSlotPool's bank interface so the slot pool can
    sit on top of it (three tiers wired by composition)
"""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3MoeForCausalLM

from plastic_infer.store.experts import ExpertSlotPool, HostExpertLru
from plastic_infer.weights.convert import convert_from_dict
from plastic_infer.weights.disk import DiskExpertSource
from plastic_infer.weights.layout import LayoutIndex

from test_qwen3moe_equiv import (
    build_weights_and_bank,
    tiny_qwen3moe_config,
    tiny_qwen3moe_config_dict,
)


@pytest.fixture(scope="module")
def disk_model(tmp_path_factory) -> tuple:
    """A converted tiny Qwen3Moe on disk + its layout + reference bank."""
    torch.manual_seed(11)
    model = Qwen3MoeForCausalLM(tiny_qwen3moe_config())
    model.eval()
    out = tmp_path_factory.mktemp("model") / "converted"
    convert_from_dict(model.state_dict(), tiny_qwen3moe_config_dict(),
                      out, dtype=torch.float32)
    layout = LayoutIndex.load(out / "layout.index.json")
    ref_dense, ref_bank = build_weights_and_bank(model)
    return out, layout, ref_bank


@pytest.fixture(scope="module")
def layout(disk_model) -> LayoutIndex:
    return disk_model[1]


@pytest.fixture(scope="module")
def ref_bank(disk_model):
    return disk_model[2]


def _per_expert(layout: LayoutIndex) -> int:
    return layout.expert_total_bytes(0)


def _make_lru(disk_model, budget_bytes: int | None = None) -> HostExpertLru:
    out, layout, _bank = disk_model
    per_expert = _per_expert(layout)
    budget = budget_bytes if budget_bytes is not None else per_expert * 4
    return HostExpertLru(DiskExpertSource(out, layout),
                         budget_bytes=budget, per_expert_bytes=per_expert)


class TestHostLruBasics:
    def test_miss_then_hit(self, disk_model) -> None:
        lru = _make_lru(disk_model)
        key = (0, 0)
        first = lru[key]          # miss: loaded from disk
        assert lru.misses == 1
        assert lru.hits == 0
        assert key in lru
        second = lru[key]         # hit: served from cache
        assert lru.hits == 1
        assert lru.misses == 1
        assert lru.hit_rate == 0.5
        assert first is second    # same object, no re-fetch

    def test_loaded_matches_disk_and_bank(self, disk_model, ref_bank) -> None:
        lru = _make_lru(disk_model)
        got = lru[(0, 1)]
        want = ref_bank[(0, 1)]
        for tag in ("w1", "w2", "w3"):
            assert torch.equal(getattr(got, tag), getattr(want, tag)), tag

    def test_budget_smaller_than_one_expert_raises(self, disk_model,
                                                   layout) -> None:
        per_expert = _per_expert(layout)
        lru = _make_lru(disk_model, budget_bytes=per_expert - 1)
        with pytest.raises(AssertionError, match="too small"):
            lru[(0, 0)]

    def test_non_uniform_expert_asserts(self, disk_model, layout) -> None:
        """per_expert_bytes must match the actual source size."""
        out, _l, _b = disk_model
        lru = HostExpertLru(DiskExpertSource(out, layout),
                            budget_bytes=_per_expert(layout) * 4,
                            per_expert_bytes=_per_expert(layout) + 1)
        with pytest.raises(AssertionError, match="non-uniform"):
            lru[(0, 0)]


class TestHostLruEviction:
    def test_eviction_refetches_byte_identical(self, disk_model,
                                               layout) -> None:
        """D5 at the host tier: evicted expert is re-fetched from disk
        and is byte-identical to what the bank serves."""
        lru = _make_lru(disk_model, budget_bytes=_per_expert(layout) * 2)

        a = lru[(0, 0)]
        b = lru[(0, 1)]           # both fit (2-expert budget)
        c = lru[(0, 2)]           # evicts (0,0) — LRU
        assert (0, 0) not in lru
        assert (0, 1) in lru and (0, 2) in lru
        assert lru.misses == 3
        assert lru._used <= lru.budget_bytes

        a2 = lru[(0, 0)]          # miss again, evicts (0,1)
        assert (0, 1) not in lru
        for tag in ("w1", "w2", "w3"):
            assert torch.equal(getattr(a2, tag), getattr(a, tag)), tag

    def test_reaccess_promotes_lru_position(self, disk_model, layout) -> None:
        lru = _make_lru(disk_model, budget_bytes=_per_expert(layout) * 2)
        lru[(0, 0)]
        lru[(0, 1)]
        lru[(0, 0)]               # (0,0) is now MRU
        lru[(0, 2)]               # evicts (0,1), keeps (0,0)
        assert (0, 0) in lru
        assert (0, 1) not in lru

    def test_evicted_reload_counts_as_miss(self, disk_model, layout) -> None:
        lru = _make_lru(disk_model, budget_bytes=_per_expert(layout) * 1)
        lru[(0, 0)]
        lru[(0, 1)]               # evicts (0,0)
        assert lru.misses == 2
        lru[(0, 0)]               # evicts (0,1)
        assert lru.misses == 3
        assert lru.hits == 0


class TestSlotPoolOverHostLru:
    def test_slot_pool_served_by_host_lru(self, disk_model, layout) -> None:
        """Three tiers composed: slot pool (GPU) over host LRU over disk.
        The slot pool's copy matches the disk bytes."""
        out, _l, _bank = disk_model
        per_expert = _per_expert(layout)
        host = HostExpertLru(DiskExpertSource(out, layout),
                             budget_bytes=per_expert * 2,
                             per_expert_bytes=per_expert)
        pool = ExpertSlotPool(host, budget_bytes=per_expert * 2,
                              device=torch.device("cpu"))

        got = pool.ensure(0, [0, 1])
        assert set(got) == {0, 1}
        pool.release(0, [0, 1])
        assert pool.misses == 2
        assert pool.hit_rate == 0.0

        # Re-ensure from the (still host-resident) set: host hits now.
        got2 = pool.ensure(0, [0, 1])
        pool.release(0, [0, 1])
        assert pool.misses == 2
        assert pool.hits >= 1
        for eid in (0, 1):
            for tag in ("w1", "w2", "w3"):
                assert torch.equal(getattr(got2[eid], tag),
                                   getattr(got[eid], tag)), (eid, tag)

    def test_slot_pool_over_host_lru_evicts_across(self, disk_model,
                                                   layout) -> None:
        """GPU pool with 2 slots over a host LRU with room for all:
        loading a 3rd expert evicts the GPU slot and re-fetches from the
        host LRU (which is itself a hit — no disk I/O)."""
        out, _l, _bank = disk_model
        per_expert = _per_expert(layout)
        host = HostExpertLru(DiskExpertSource(out, layout),
                             budget_bytes=per_expert * 4,
                             per_expert_bytes=per_expert)
        pool = ExpertSlotPool(host, budget_bytes=per_expert * 2,
                              device=torch.device("cpu"))

        pool.ensure(0, [0, 1])
        pool.release(0, [0, 1])
        pool.ensure(0, [2, 3])            # evicts GPU slots 0,1
        pool.release(0, [2, 3])
        assert pool.pool.used_bytes <= pool.pool.budget_bytes

        # 0,1 must come back from the host LRU (host hits, no disk miss).
        host_misses_before = host.misses
        pool.ensure(0, [0, 1])
        pool.release(0, [0, 1])
        assert host.misses == host_misses_before
        assert host.hits == 2
