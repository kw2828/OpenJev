"""Single-owner durable JSONL streams with explicit artifact boundaries.

Each append encodes once, writes once, flushes and fsyncs before acknowledgement.
Counters replace per-record directory scans. The caller must reserve external
JSON/NPZ files before creating them and reconcile immediately after publication.
No other writer may touch owned streams. Boundary checks verify file identity
and size, not cryptographic content; final artifact hashes remain caller-owned.
An uncertain operation permanently poisons this instance. Close preserves its
pending evidence and attempts cleanup of every opened handle without resuming it.
"""
from __future__ import annotations

import copy
import json
import os
import stat
from pathlib import Path


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _size(value, name):
    _require(type(value) is int and value >= 0, f"{name} must be a nonnegative integer")
    return value


class DurableJSONL:
    """Persistent flat-file writer; not thread safe and never a resume facility.

    ``initial_inventory`` optionally asserts exact existing name -> byte counts.
    Existing nonempty artifacts are immutable. Initially empty JSONL files may
    become owned streams. ``reserve(name, bytes)`` accounts an external file's
    maximum final size; ``reconcile()`` consumes its reservation only when that
    file exists within its allocation. Unpublished reservations remain active.

    ``failure_reserve`` is never spendable by normal appends or publications.
    The caller owns bounded failure receipts and may use that space after close.
    ``check`` is a lightweight caller budget check before each operation. It
    must not mutate this writer. ``append`` returns encoded bytes only after
    successful flush and fsync. Metadata access returns detached copies.
    """

    def __init__(self, directory, *, total_limit, non_sampler_limit, failure_reserve,
                 sampler_names=("sampler-events.jsonl",), initial_inventory=None,
                 per_file_record_limits=None, check=None):
        self.directory = Path(directory)
        _require(self.directory.is_absolute() and ".." not in self.directory.parts and self.directory.is_dir()
                 and not any(p.is_symlink() for p in (self.directory, *self.directory.parents)),
                 "existing absolute directory without symlink ancestors required")
        self.total_limit = _size(total_limit, "total_limit")
        self.non_sampler_limit = _size(non_sampler_limit, "non_sampler_limit")
        self.failure_reserve = _size(failure_reserve, "failure_reserve")
        self.sampler_names = frozenset(self._name(n) for n in sampler_names)
        self.record_limits = {self._name(n): _size(v, "record limit")
                              for n, v in (per_file_record_limits or {}).items()}
        _require(check is None or callable(check), "check must be callable or None")
        self._check = check
        self._streams, self._owned, self._reservations = {}, set(), {}
        self._pending, self._errors, self._sequence = [], [], 0
        self._poisoned = self._closed = self._active = False
        self._metadata = self._scan()
        self._inventory = {name: value[2] for name, value in self._metadata.items()}
        self._observed = dict(self._inventory)
        if initial_inventory is not None:
            expected = {self._name(n): _size(v, "initial byte count") for n, v in initial_inventory.items()}
            _require(expected == self._inventory, "initial inventory mismatch")
        self._total = sum(self._inventory.values())
        self._non_sampler = sum(v for k, v in self._inventory.items() if k not in self.sampler_names)
        self._capacity()

    @staticmethod
    def _name(value):
        _require(type(value) is str and value not in ("", ".", "..")
                 and Path(value).name == value and "/" not in value and "\\" not in value,
                 "flat artifact filename required")
        return value

    def _scan(self):
        result = {}
        for path in self.directory.iterdir():
            info = path.lstat()
            _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "only singly owned regular files")
            result[self._name(path.name)] = (info.st_dev, info.st_ino, info.st_size,
                                            info.st_mtime_ns, info.st_ctime_ns)
        return result

    def _capacity(self, extra=0, *, sampler=False):
        reserved = sum(self._reservations.values())
        ordinary = sum(v for k, v in self._reservations.items() if k not in self.sampler_names)
        _require(self._total + reserved + extra + self.failure_reserve <= self.total_limit,
                 "total byte allocation exceeded")
        _require(self._non_sampler + ordinary + (0 if sampler else extra) + self.failure_reserve
                 <= self.non_sampler_limit, "non-sampler byte allocation exceeded")

    def _ready(self):
        _require(not self._closed and not self._poisoned and not self._active,
                 "writer is closed, poisoned or reentered")

    def _budget(self):
        if self._check is not None:
            self._check()

    def _failure(self, error, intent):
        self._poisoned = True
        self._errors.append({"error": repr(error), "intent": copy.deepcopy(intent)})

    def _open(self, name):
        path = self.directory / name
        flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        if name in self._inventory:
            _require(self._inventory[name] == 0 and name not in self._owned,
                     "nonempty existing JSONL cannot be resumed")
        else:
            flags |= os.O_CREAT | os.O_EXCL
        fd = os.open(path, flags, 0o600)
        try:
            info = os.fstat(fd)
            _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size == 0,
                     "new stream is not an empty singly owned regular file")
            if name in self._metadata:
                _require((info.st_dev, info.st_ino, info.st_size) == self._metadata[name][:3],
                         "initial stream identity changed")
            handle = os.fdopen(fd, "ab")
        except BaseException:
            os.close(fd)
            raise
        self._metadata[name] = (info.st_dev, info.st_ino, 0, info.st_mtime_ns, info.st_ctime_ns)
        self._inventory[name] = 0
        self._owned.add(name)
        self._streams[name] = handle
        return handle

    def append(self, name, value, *, context=None):
        self._ready()
        name = self._name(name)
        _require(name.endswith(".jsonl") and name not in self._reservations, "unreserved JSONL stream required")
        detached_context = copy.deepcopy(context)
        self._active = True
        self._sequence += 1
        intent = {"id": self._sequence, "operation": "append", "file": name,
                  "context": detached_context, "stage": "check", "uncertain": False}
        self._pending.append(intent)
        try:
            self._budget()
            intent["stage"] = "encode"
            payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
            intent.update(payload_bytes=len(payload), offset=self._inventory.get(name, 0), stage="admit")
            _require(name not in self.record_limits or len(payload) <= self.record_limits[name],
                     "per-file encoded record limit exceeded")
            self._capacity(len(payload), sampler=name in self.sampler_names)
            stream = self._streams.get(name)
            if stream is None:
                intent["stage"] = "open"
                stream = self._open(name)
            intent.update(stage="write", uncertain=True)
            written = stream.write(payload)
            intent["write_returned_bytes"] = written
            _require(type(written) is int and written == len(payload), "short or invalid binary write")
            intent["stage"] = "flush"
            stream.flush()
            intent["stage"] = "fsync"
            os.fsync(stream.fileno())
            self._inventory[name] += len(payload)
            self._total += len(payload)
            if name not in self.sampler_names:
                self._non_sampler += len(payload)
            self._pending.pop()
            return len(payload)
        except BaseException as error:
            self._failure(error, intent)
            raise
        finally:
            self._active = False

    def reserve(self, name, maximum_bytes):
        self._ready()
        self._budget()
        name, maximum_bytes = self._name(name), _size(maximum_bytes, "reservation")
        _require(name not in self._inventory and name not in self._reservations,
                 "external reservation requires a new artifact name")
        self._capacity(maximum_bytes, sampler=name in self.sampler_names)
        self._reservations[name] = maximum_bytes

    def cancel_reservation(self, name):
        self._ready()
        name = self._name(name)
        _require(name in self._reservations and not (self.directory / name).exists(),
                 "only unpublished reservations may be cancelled")
        del self._reservations[name]

    def reconcile(self):
        """Explicit boundary scan; cannot clear poisoning or acknowledge failed IO."""
        _require(not self._active, "cannot reconcile inside an append")
        intent = {"operation": "reconcile", "stage": "inventory", "uncertain": False}
        try:
            if not self._poisoned:
                self._budget()
            observed = self._scan()
            self._observed = {n: v[2] for n, v in observed.items()}
            _require(set(self._inventory) <= set(observed), "tracked artifact disappeared")
            for name, size in self._inventory.items():
                intent["file"] = name
                current, previous = observed[name], self._metadata[name]
                if name in self._owned:
                    _require(current[:3] == (*previous[:2], size), "owned stream identity or byte count drift")
                else:
                    _require(current == previous, "existing external artifact changed")
            added = set(observed) - set(self._inventory)
            _require(added <= set(self._reservations), "unreserved external artifact appeared")
            for name in added:
                intent["file"] = name
                _require(observed[name][2] <= self._reservations[name], "external artifact exceeded reservation")
            self._capacity()
            for name in added:
                size = observed[name][2]
                self._inventory[name], self._metadata[name] = size, observed[name]
                self._total += size
                if name not in self.sampler_names:
                    self._non_sampler += size
                del self._reservations[name]
            self._capacity()
            return self.inventory
        except BaseException as error:
            self._pending.append(intent)
            self._failure(error, intent)
            raise

    def close(self, *, suppress=False):
        """Attempt every close once; preserve earlier failures and cleanup evidence."""
        _require(type(suppress) is bool and not self._active, "invalid close or active append")
        if self._closed:
            return not self._poisoned
        first = None
        for name, stream in self._streams.items():
            intent = {"operation": "close", "file": name, "stage": "flush", "uncertain": True}
            try:
                stream.flush()
                intent["stage"] = "fsync"
                os.fsync(stream.fileno())
            except BaseException as error:  # noqa: BLE001 - clean every remaining handle after interruption
                self._pending.append(copy.deepcopy(intent))
                self._failure(error, intent)
                first = error if first is None else first
            try:
                intent["stage"] = "close"
                stream.close()
            except BaseException as error:  # noqa: BLE001 - preserve cleanup errors without skipping other handles
                self._pending.append(copy.deepcopy(intent))
                self._failure(error, intent)
                first = error if first is None else first
        try:
            self.reconcile()
            _require(not self._reservations, "unpublished external reservations remain at close")
        except BaseException as error:  # noqa: BLE001 - retain boundary failure alongside cleanup failures
            if not self._poisoned:
                intent = {"operation": "close", "stage": "reservations", "uncertain": False}
                self._pending.append(intent)
                self._failure(error, intent)
            first = error if first is None else first
        self._closed = True
        if first is not None and not suppress:
            raise first
        return not self._poisoned

    @property
    def inventory(self):
        return dict(self._inventory)

    @property
    def observed_inventory(self):
        return dict(self._observed)

    @property
    def reservations(self):
        return dict(self._reservations)

    @property
    def total_bytes(self):
        return self._total

    @property
    def non_sampler_bytes(self):
        return self._non_sampler

    @property
    def allocated_total_bytes(self):
        uncertain = sum(p.get("payload_bytes", 0) for p in self._pending
                        if p.get("operation") == "append" and p.get("uncertain"))
        return self._total + sum(self._reservations.values()) + uncertain + self.failure_reserve

    @property
    def allocated_non_sampler_bytes(self):
        uncertain = sum(p.get("payload_bytes", 0) for p in self._pending
                        if p.get("operation") == "append" and p.get("uncertain")
                        and p["file"] not in self.sampler_names)
        return (self._non_sampler + sum(v for k, v in self._reservations.items() if k not in self.sampler_names)
                + uncertain + self.failure_reserve)

    @property
    def pending(self):
        return copy.deepcopy(self._pending)

    @property
    def errors(self):
        return copy.deepcopy(self._errors)

    @property
    def poisoned(self):
        return self._poisoned

    @property
    def closed(self):
        return self._closed
