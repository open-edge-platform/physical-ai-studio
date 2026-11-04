"""Keep a worker's loop at its own frequency, and say when it falls behind."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

_REPORT_INTERVAL_S = 1.0
"""Overruns are summarised at most this often; at 100Hz one line each would bury them."""


class Rate:
    """Sleep until the next tick of a fixed-rate loop.

    ::

        rate = Rate(30.0, name=self.name)
        while not self.should_stop():
            do_one_pass()
            await rate.sleep()

    Ticks are scheduled from when the Rate was created, not from when each pass ended, so
    the loop does not drift. A pass that overruns its period starts the next
    one immediately and resets the schedule from there, rather than running a
    burst of catch-up ticks.
    """

    def __init__(self, hz: float, *, name: str, clock: Callable[[], float] = time.monotonic) -> None:
        if hz <= 0:
            raise ValueError(f"Rate must be positive, got {hz}")
        self._period = 1.0 / hz
        self._hz = hz
        self._name = name
        self._clock = clock
        # The schedule starts now, so build the Rate right before the loop.
        self._next = clock()
        self._overruns = 0
        self._worst = 0.0
        self._next_report = self._next + _REPORT_INTERVAL_S

    @property
    def period(self) -> float:
        return self._period

    async def sleep(self) -> None:
        """Wait out the rest of this tick's period."""
        now = self._clock()
        self._next += self._period
        remaining = self._next - now
        if remaining > 0:
            await asyncio.sleep(remaining)
        else:
            self._overruns += 1
            self._worst = max(self._worst, self._period - remaining)
            self._next = now
            # Still yield, so a loop that always overruns cannot starve the event loop.
            await asyncio.sleep(0)
        self._report(now)

    def _report(self, now: float) -> None:
        if self._overruns and now >= self._next_report:
            logger.warning(
                "{} missed {} of its {:.0f}Hz deadlines, worst {:.1f}ms",
                self._name,
                self._overruns,
                self._hz,
                self._worst * 1000,
            )
            self._overruns, self._worst = 0, 0.0
            self._next_report = now + _REPORT_INTERVAL_S
