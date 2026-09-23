"""Fabricated durable bytes, bounded accounting and failure-injection fixtures."""
from __future__ import annotations

import json
import os

import pytest

from openjev.research import durable_jsonl as module
from openjev.research.durable_jsonl import DurableJSONL


def writer(directory, **kwargs):
    defaults = {"total_limit": 4096, "non_sampler_limit": 2048, "failure_reserve": 128}
    return DurableJSONL(directory, **(defaults | kwargs))


def test_exact_old_bytes_single_encoding_and_persistent_handle(tmp_path, monkeypatch):
    (tmp_path / "started.json").write_bytes(b"{}\n")
    (tmp_path / "work.jsonl").touch()
    values = [{"z": [1, 2.5, None], "a": "caf\u00e9\n"}, {"boolean": True, "tuple": (0, -1)}, {}]
    expected = b"".join((json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
                        for v in values)
    original_dump, original_sync = module.json.dumps, module.os.fsync
    encodes, syncs, budgets = [], [], []

    def encode(*args, **kwargs):
        encodes.append(1)
        return original_dump(*args, **kwargs)

    def sync(fd):
        syncs.append(fd)
        return original_sync(fd)

    monkeypatch.setattr(module.json, "dumps", encode)
    monkeypatch.setattr(module.os, "fsync", sync)
    journal = writer(tmp_path, initial_inventory={"started.json": 3, "work.jsonl": 0}, check=lambda: budgets.append(1))
    sizes, handles = [], []
    for value in values:
        sizes.append(journal.append("work.jsonl", value))
        handles.append(journal._streams["work.jsonl"])
    assert len(encodes) == len(syncs) == len(budgets) == 3
    assert handles[0] is handles[1] is handles[2]
    assert sum(sizes) == len(expected)
    assert (tmp_path / "work.jsonl").read_bytes() == expected
    assert journal.total_bytes == journal.non_sampler_bytes == 3 + len(expected)
    assert journal.reconcile() == {"started.json": 3, "work.jsonl": len(expected)}
    assert journal.close() is True and journal.pending == []


def test_external_reservation_reconciliation_and_failure_space(tmp_path):
    journal = writer(tmp_path, total_limit=100, non_sampler_limit=60, failure_reserve=10)
    journal.reserve("setup.json", 20)
    journal.reserve("sampler-events.jsonl", 30)
    assert journal.allocated_total_bytes == 60
    assert journal.allocated_non_sampler_bytes == 30
    with pytest.raises(ValueError, match="non-sampler"):
        journal.reserve("too-large.json", 31)
    (tmp_path / "setup.json").write_bytes(b"x" * 12)
    with pytest.raises(ValueError, match="unpublished"):
        journal.cancel_reservation("setup.json")
    journal.reconcile()
    assert journal.inventory == {"setup.json": 12}
    assert journal.reservations == {"sampler-events.jsonl": 30}
    assert journal.allocated_total_bytes == 52 and journal.allocated_non_sampler_bytes == 22
    journal.cancel_reservation("sampler-events.jsonl")
    n = journal.append("sampler-events.jsonl", {"x": 1})
    assert journal.total_bytes == 12 + n and journal.non_sampler_bytes == 12
    assert journal.close()


@pytest.mark.parametrize("limit", ["record", "total", "ordinary"])
def test_admission_refuses_before_write_and_never_spends_failure_reserve(tmp_path, limit):
    payload = {"value": "x" * 40}
    size = len((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode())
    opts = {"failure_reserve": 10}
    if limit == "record":
        opts["per_file_record_limits"] = {"work.jsonl": size - 1}
    elif limit == "total":
        opts["total_limit"] = size + 9
    else:
        opts["non_sampler_limit"] = size + 9
    journal = writer(tmp_path, **opts)
    with pytest.raises(ValueError, match="exceeded"):
        journal.append("work.jsonl", payload, context={"step": 2})
    assert journal.total_bytes == 0 and not (tmp_path / "work.jsonl").exists()
    assert journal.pending[0]["uncertain"] is False and journal.pending[0]["payload_bytes"] == size
    assert journal.poisoned
    with pytest.raises(ValueError, match="poisoned"):
        journal.append("work.jsonl", {})
    journal.close(suppress=True)


class FaultStream:
    def __init__(self, stream, fault, closed=None):
        self.stream, self.fault, self.closed_log = stream, fault, closed

    def write(self, data):
        if self.fault == "write":
            raise OSError("injected write")
        if self.fault == "short":
            self.stream.write(data[:2])
            return 2
        return self.stream.write(data)

    def flush(self):
        self.stream.flush()
        if self.fault == "flush":
            raise OSError("injected flush")

    def fileno(self):
        return self.stream.fileno()

    def close(self):
        self.stream.close()
        if self.closed_log is not None:
            self.closed_log.append(self.fault)
        if self.fault == "close":
            raise OSError("injected close")


@pytest.mark.parametrize("fault", ["write", "short", "flush", "fsync"])
def test_uncertain_write_never_acknowledged_or_retried(tmp_path, monkeypatch, fault):
    journal = writer(tmp_path)
    initial = journal.append("work.jsonl", {"initial": True})
    journal._streams["work.jsonl"] = FaultStream(journal._streams["work.jsonl"], fault)
    original_sync = module.os.fsync
    if fault == "fsync":
        def fail_sync(_fd):
            raise OSError("injected fsync")
        monkeypatch.setattr(module.os, "fsync", fail_sync)
    context = {"anchor": {"id": 7}}
    with pytest.raises((OSError, ValueError), match="injected|short"):
        journal.append("work.jsonl", {"later": [1, 2]}, context=context)
    context["anchor"]["id"] = 999
    assert journal.total_bytes == initial and journal.inventory["work.jsonl"] == initial
    pending = journal.pending
    assert pending[0]["uncertain"] and pending[0]["context"] == {"anchor": {"id": 7}}
    assert pending[0]["stage"] == ("write" if fault == "short" else fault)
    assert journal.allocated_total_bytes == initial + pending[0]["payload_bytes"] + journal.failure_reserve
    pending[0]["context"]["anchor"]["id"] = -1
    assert journal.pending[0]["context"]["anchor"]["id"] == 7
    with pytest.raises(ValueError, match="poisoned"):
        journal.append("work.jsonl", {})
    monkeypatch.setattr(module.os, "fsync", original_sync)
    journal.close(suppress=True)
    assert journal.poisoned and journal.pending[0]["stage"] == ("write" if fault == "short" else fault)


def test_close_attempts_every_handle_and_retains_cleanup_error(tmp_path):
    journal, closed = writer(tmp_path), []
    for name, fault in (("first.jsonl", "close"), ("second.jsonl", "okay")):
        journal.append(name, {})
        journal._streams[name] = FaultStream(journal._streams[name], fault, closed)
    with pytest.raises(OSError, match="injected close"):
        journal.close()
    assert closed == ["close", "okay"]
    assert journal.closed and journal.poisoned and journal.pending[0]["stage"] == "close"
    assert journal.close(suppress=True) is False and closed == ["close", "okay"]


@pytest.mark.parametrize("drift", ["truncate", "replace", "unreserved", "external_change", "oversize"])
def test_boundary_detects_file_drift_and_unowned_publication(tmp_path, drift):
    (tmp_path / "started.json").write_bytes(b"{}\n")
    journal = writer(tmp_path)
    journal.append("work.jsonl", {"x": 1})
    path = tmp_path / "work.jsonl"
    if drift == "truncate":
        path.write_bytes(b"")
    elif drift == "replace":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    elif drift == "unreserved":
        (tmp_path / "new.json").write_bytes(b"{}")
    elif drift == "external_change":
        (tmp_path / "started.json").write_bytes(b"bad")
    else:
        journal.reserve("large.json", 1)
        (tmp_path / "large.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="drift|unreserved|changed|reservation"):
        journal.reconcile()
    assert journal.poisoned and journal.pending[-1]["operation"] == "reconcile"
    journal.close(suppress=True)


def test_initial_ownership_nonempty_resume_and_missing_publication(tmp_path):
    (tmp_path / "old.jsonl").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="inventory"):
        writer(tmp_path, initial_inventory={})
    journal = writer(tmp_path)
    with pytest.raises(ValueError, match="resumed"):
        journal.append("old.jsonl", {})
    assert (tmp_path / "old.jsonl").read_bytes() == b"{}\n"
    journal.close(suppress=True)
    fresh = writer(tmp_path)
    fresh.reserve("never-published.json", 10)
    with pytest.raises(ValueError, match="unpublished"):
        fresh.close()
    assert fresh.poisoned and fresh.reservations == {"never-published.json": 10}


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_rejects_shared_or_symlink_files(tmp_path, kind):
    source = tmp_path / "source"
    source.write_bytes(b"x")
    if kind == "symlink":
        (tmp_path / "alias").symlink_to(source)
    else:
        os.link(source, tmp_path / "alias")
    with pytest.raises(ValueError, match="singly owned"):
        writer(tmp_path)


def test_budget_failure_is_pending_before_any_bytes(tmp_path):
    def expired():
        raise TimeoutError("injected deadline")

    journal = writer(tmp_path, check=expired)
    with pytest.raises(TimeoutError, match="deadline"):
        journal.append("work.jsonl", {})
    assert journal.pending[0]["stage"] == "check" and journal.pending[0]["uncertain"] is False
    assert journal.total_bytes == 0 and not list(tmp_path.iterdir())
    journal.close(suppress=True)
