"""Storage tier definitions and bandwidth/cost constants.

All tier constants live here so every module agrees on the same
vocabulary. Device-specific numbers come from DeviceProfile (probed
at runtime), not from this file.
"""

from __future__ import annotations

from enum import Enum, auto


class Tier(Enum):
    """Where a piece of data currently lives."""

    GPU = auto()     # on-device HBM — bandwidth highest, capacity smallest
    HOST = auto()    # host RAM (pinned or pageable)
    DISK = auto()    # NVMe / SSD — bandwidth lowest, capacity largest


# A fixed ordering we can reuse for comparison / promotion checks.
TIER_ORDER: dict[Tier, int] = {
    Tier.GPU: 0,
    Tier.HOST: 1,
    Tier.DISK: 2,
}


def tier_colder(a: Tier, b: Tier) -> bool:
    """Return True if tier a is colder (slower, cheaper) than tier b."""
    return TIER_ORDER[a] > TIER_ORDER[b]
