"""Device profile — measured capacity and bandwidth numbers.

In production these come from actual probes. For planning and tests
they are plain dataclasses you construct with any values.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceProfile:
    """Hardware budget and per-tier bandwidths (bytes/sec).

    All values are measured, not assumed — the planner trusts them.
    """

    # Capacity
    hbm_bytes: int              # total usable GPU VRAM
    host_ram_bytes: int         # total usable host RAM
    pin_limit_bytes: int        # max bytes we may pin for H2D

    # Bandwidth (bytes/sec)
    b_h2d: float                # PCIe host -> device effective
    b_host: float               # host RAM read bandwidth
    b_disk: float               # NVMe sequential read bandwidth

    @property
    def hbm_gib(self) -> float:
        return self.hbm_bytes / (1024 ** 3)

    @property
    def ram_gib(self) -> float:
        return self.host_ram_bytes / (1024 ** 3)
