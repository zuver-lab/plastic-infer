"""Small shared helpers for the offload stack (mirror of FreeToken's
freetoken.utils subset that the ported modules import).

Kept intentionally tiny: div/align arithmetic plus a logger that exposes
the ``*_rank0`` methods the ported code calls (single-process here, so
rank0 logging is just normal logging).
"""

from __future__ import annotations

import logging
import os
import sys

_LOG_LEVEL = None


def div_even(a: int, b: int, allow_replicate: bool = False) -> int:
    """Divides two integers. If allow_replicate=True, allows b > a when b % a == 0, returning 1."""
    if allow_replicate and b > a:
        assert b % a == 0, f"{b = } must be divisible by {a = } for KV head replication"
        return 1
    assert a % b == 0, f"{a = } must be divisible by {b = }"
    return a // b


def div_ceil(a: int, b: int) -> int:
    """Divides two integers, rounding up"""
    return (a + b - 1) // b


def align_ceil(a: int, b: int) -> int:
    """Aligns a to the next multiple of b"""
    return div_ceil(a, b) * b


def align_down(a: int, b: int) -> int:
    """Aligns a to the previous multiple of b"""
    return (a // b) * b


def init_logger(
    name: str,
    suffix: str = "",
    *,
    strip_file: bool = True,
    level: str | None = None,
    use_pid: bool | None = None,
    use_tp_rank: bool | None = None,
):
    """Initialize a named logger with the rank0-style methods the ported code uses."""
    global _LOG_LEVEL
    if _LOG_LEVEL is None:
        LEVEL_MAP = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        level = level or os.getenv("LOG_LEVEL", "").upper()
        _LOG_LEVEL = LEVEL_MAP.get(level, logging.INFO)

    if strip_file and suffix:
        suffix = os.path.basename(suffix)
    if use_pid is None:
        use_pid = os.getenv("LOG_PID", "0").lower() in ("1", "true", "yes")
    if use_pid:
        suffix = f"{suffix}|pid={os.getpid()}"

    logger = logging.getLogger(name)
    logger.setLevel(_LOG_LEVEL)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(f"%(asctime)s|{suffix} %(levelname)-8s %(message)s",
                          "[%Y-%m-%d|%H:%M:%S]"))
    logger.addHandler(handler)
    logger.propagate = False

    def _rank0(msg, *args, _which, **kwargs):
        getattr(logger, _which)(msg, *args, **kwargs)

    logger.info_rank0 = lambda *a, **k: _rank0(*a, _which="info", **k)
    logger.debug_rank0 = lambda *a, **k: _rank0(*a, _which="debug", **k)
    logger.critical_rank0 = lambda *a, **k: _rank0(*a, _which="critical", **k)
    logger.warning_rank0 = lambda *a, **k: _rank0(*a, _which="warning", **k)
    return logger
