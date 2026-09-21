"""Artificial clocks only: no real sleep, native timer, or model invocation."""

from types import SimpleNamespace

import pytest

from openjev.research import suspend_clock as clocks


class FakeTime:
    def __init__(self):
        self.continuous = 10_000_000_000
        self.awake = self.continuous
        self.civil = 1_700_000_000_000_000_000

    def read(self):
        return self.continuous


def test_suspend_expires_even_when_awake_clock_does_not_advance():
    fake = FakeTime()
    deadline = clocks.SuspendClock(fake.read).deadline_after(8)
    fake.continuous += 9_000_000_000
    fake.civil += 9_000_000_000
    assert fake.awake == 10_000_000_000
    assert deadline.elapsed_ns() == 9_000_000_000
    assert deadline.remaining_ns() == 0
    with pytest.raises(TimeoutError):
        deadline.check()


def test_civil_jumps_do_not_change_deadline():
    fake = FakeTime()
    deadline = clocks.SuspendClock(fake.read).deadline_after(8)
    for jump in (10**20, -(10**21)):
        fake.civil += jump
        assert deadline.remaining_ns() == 8_000_000_000
        deadline.check()
    fake.continuous += 8_000_000_000
    assert deadline.expired()


def test_exact_boundary_zero_and_fractional_nanosecond():
    fake = FakeTime()
    clock = clocks.SuspendClock(fake.read)
    assert clock.deadline_after(0).expired()
    deadline = clock.deadline_after(0.0000000001)
    assert deadline.remaining_ns() == 1
    deadline.check()
    fake.continuous += 1
    assert deadline.expired()
    with pytest.raises(TimeoutError):
        deadline.check()


@pytest.mark.parametrize("duration", [-1, -0.1, float("inf"), float("nan"), True, "1"])
def test_invalid_duration_is_rejected_without_new_read(duration):
    reads = []
    clock = clocks.SuspendClock(lambda: reads.append(1) or 0)
    with pytest.raises(ValueError):
        clock.deadline_after(duration)
    assert len(reads) == 1


@pytest.mark.parametrize("bad", [9, -1, 11.0, True, None, OSError("timer failed")])
def test_read_failure_is_latched_and_cannot_resume(bad):
    readings = iter([10, 10, bad, 1_000_000_000])
    calls = []

    def read():
        calls.append(1)
        value = next(readings)
        if isinstance(value, Exception):
            raise value
        return value

    clock = clocks.SuspendClock(read)
    deadline = clock.deadline_after(1)
    with pytest.raises(clocks.ClockError):
        deadline.check()
    with pytest.raises(clocks.ClockError, match="remains failed"):
        deadline.remaining_ns()
    assert len(calls) == 3


def test_initial_read_failure_rejects_construction():
    with pytest.raises(clocks.ClockError):
        clocks.SuspendClock(lambda: -1)


class FakeFunction:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


def fake_mach_library(*, numerator=125, denominator=3, status=0, ticks=2**63 + 17):
    def timebase(pointer):
        pointer._obj.numer = numerator
        pointer._obj.denom = denominator
        return status

    return SimpleNamespace(
        mach_timebase_info=FakeFunction(timebase),
        mach_continuous_time=FakeFunction(lambda: ticks),
    )


def test_mach_abi_and_large_integer_timebase_conversion():
    library = fake_mach_library()
    paths = []
    read = clocks._macos_reader(lambda path: paths.append(path) or library)
    assert paths == ["/usr/lib/libSystem.B.dylib"]
    assert read() == ((2**63 + 17) * 125) // 3
    assert library.mach_continuous_time.argtypes == []
    assert library.mach_continuous_time.restype is clocks.ctypes.c_uint64
    assert library.mach_timebase_info.restype is clocks.ctypes.c_int


@pytest.mark.parametrize("kwargs", [{"status": 5}, {"numerator": 0}, {"denominator": 0}])
def test_mach_bad_timebase_fails_closed(kwargs):
    with pytest.raises(clocks.ClockError):
        clocks._macos_reader(lambda _: fake_mach_library(**kwargs))


def test_missing_continuous_symbol_does_not_use_absolute_clock(monkeypatch):
    library = SimpleNamespace(mach_absolute_time=lambda: pytest.fail("fallback called"))
    monkeypatch.setattr(clocks.sys, "platform", "darwin")
    monkeypatch.setattr(clocks.ctypes, "CDLL", lambda _: library)
    with pytest.raises(clocks.ClockError):
        clocks.SuspendClock()


def test_linux_selects_boottime_only(monkeypatch):
    monkeypatch.setattr(clocks.sys, "platform", "linux")
    monkeypatch.setattr(clocks.time, "CLOCK_BOOTTIME", 123, raising=False)
    seen = []
    monkeypatch.setattr(clocks.time, "clock_gettime_ns", lambda k: seen.append(k) or 7)
    clock = clocks.SuspendClock()
    assert clock.backend == "CLOCK_BOOTTIME"
    assert clock.now_ns() == 7
    assert seen == [123, 123]


def test_linux_missing_boottime_has_no_fallback(monkeypatch):
    monkeypatch.setattr(clocks.sys, "platform", "linux")
    monkeypatch.delattr(clocks.time, "CLOCK_BOOTTIME", raising=False)
    monkeypatch.setattr(clocks.time, "monotonic_ns", lambda: pytest.fail("fallback called"))
    with pytest.raises(clocks.ClockError, match="unavailable"):
        clocks.SuspendClock()


def test_unsupported_platform_rejects_native_clock(monkeypatch):
    monkeypatch.setattr(clocks.sys, "platform", "unsupported")
    with pytest.raises(clocks.ClockError, match="no suspend-inclusive"):
        clocks.SuspendClock()
