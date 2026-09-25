"""Suspend-inclusive elapsed deadlines, without civil-clock dependence.

macOS uses mach_continuous_time (macOS 10.12+) and mach_timebase_info;
Linux uses CLOCK_BOOTTIME. Unavailable or broken clocks fail closed, never
falling back to mach_absolute_time, CLOCK_MONOTONIC, or time.time.

Primary references:
https://developer.apple.com/documentation/kernel/1646199-mach_continuous_time
https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/mach/mach_time.h
https://man7.org/linux/man-pages/man2/clock_gettime.2.html
https://docs.python.org/3/library/time.html#time.CLOCK_BOOTTIME

This helper does not wake a sleeping machine or interrupt a blocked operation.
An external supervisor must poll/check the deadline, including on wake; using
one remaining-time value in a suspend-excluding wait is insufficient. Injected
clocks are intended for tests: their suspend behavior is the caller's contract.
Readings are process-local elapsed values, not portable civil timestamps.
"""

from __future__ import annotations

import ctypes
import math
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

NANOSECONDS = 1_000_000_000


class ClockError(RuntimeError):
    """The selected suspend-inclusive timer cannot be trusted."""


class _Timebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


def _macos_reader(loader: Callable | None = None) -> Callable[[], int]:
    """Bind the published ABI; convert ticks with integer arithmetic."""
    try:
        library = (loader or ctypes.CDLL)("/usr/lib/libSystem.B.dylib")
        continuous = library.mach_continuous_time
        continuous.argtypes = []
        continuous.restype = ctypes.c_uint64
        timebase = library.mach_timebase_info
        timebase.argtypes = [ctypes.POINTER(_Timebase)]
        timebase.restype = ctypes.c_int
        info = _Timebase()
        status = timebase(ctypes.byref(info))
        if status != 0 or info.numer == 0 or info.denom == 0:
            raise ClockError("mach_timebase_info failed or returned an invalid ratio")
        numerator, denominator = int(info.numer), int(info.denom)
    except Exception as error:
        raise ClockError("macOS continuous clock initialization failed") from error

    def read() -> int:
        # Keep the library alive together with its bound function.
        _ = library
        ticks = continuous()
        if type(ticks) is not int or ticks < 0:
            raise ClockError("invalid mach_continuous_time reading")
        return ticks * numerator // denominator

    return read


def _native_reader() -> tuple[Callable[[], int], str]:
    if sys.platform == "darwin":
        return _macos_reader(), "mach_continuous_time"
    if sys.platform.startswith("linux"):
        try:
            clock_id = time.CLOCK_BOOTTIME
            read = time.clock_gettime_ns
        except AttributeError as error:
            raise ClockError("Linux CLOCK_BOOTTIME is unavailable") from error
        return lambda: read(clock_id), "CLOCK_BOOTTIME"
    raise ClockError(f"no suspend-inclusive clock supported on {sys.platform!r}")


def _duration_ns(seconds: float) -> int:
    if type(seconds) is int:
        if seconds < 0:
            raise ValueError("duration must be nonnegative")
        return seconds * NANOSECONDS
    if type(seconds) is not float or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("duration must be a finite nonnegative int or float")
    numerator, denominator = seconds.as_integer_ratio()
    # Round upward, so a fractional nanosecond never expires early.
    return (numerator * NANOSECONDS + denominator - 1) // denominator


class SuspendClock:
    """Validated clock with permanent failure on bad or regressing readings.

    Construction reads the clock once to reject unusable backends immediately.
    A lock orders concurrent reads; it does not provide asynchronous timeouts.
    """

    def __init__(self, clock_ns: Callable[[], int] | None = None) -> None:
        if clock_ns is None:
            self._read, self.backend = _native_reader()
        elif callable(clock_ns):
            self._read, self.backend = clock_ns, "injected"
        else:
            raise TypeError("clock_ns must be callable")
        self._lock = threading.Lock()
        self._last: int | None = None
        self._failure: Exception | None = None
        self.now_ns()

    def now_ns(self) -> int:
        with self._lock:
            if self._failure is not None:
                raise ClockError("clock remains failed") from self._failure
            try:
                value = self._read()
                if type(value) is not int or value < 0:
                    raise ClockError("clock must return nonnegative integer nanoseconds")
                if self._last is not None and value < self._last:
                    raise ClockError("clock regressed")
            except Exception as error:
                self._failure = error
                raise ClockError("suspend-inclusive clock read failed") from error
            self._last = value
            return value

    def deadline_after(self, seconds: float) -> Deadline:
        duration = _duration_ns(seconds)
        start = self.now_ns()
        return Deadline(self, start, start + duration)


@dataclass(frozen=True)
class Deadline:
    """Use SuspendClock.deadline_after(); equality with the cap is expired."""

    clock: SuspendClock
    started_ns: int
    expires_ns: int

    def elapsed_ns(self) -> int:
        return self.clock.now_ns() - self.started_ns

    def remaining_ns(self) -> int:
        return max(0, self.expires_ns - self.clock.now_ns())

    def expired(self) -> bool:
        return self.clock.now_ns() >= self.expires_ns

    def check(self) -> None:
        if self.expired():
            raise TimeoutError("suspend-inclusive deadline expired")
