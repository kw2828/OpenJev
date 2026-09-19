"""Artifact authentication for the bounded public-card experiment."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def deadline_check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Card experiment wall cap exceeded')


def runtime():
    import gymnasium
    import numpy
    import torch
    return {'python': platform.python_version(), 'executable': sys.executable,
            'machine': platform.machine(), 'platform': platform.platform(),
            'numpy': str(numpy.__version__), 'torch': str(torch.__version__),
            'gymnasium': str(gymnasium.__version__)}


def authenticate(root, protocol_path, inputs_path, bindings_path):
    root = Path(root).resolve()
    bindings = json.loads(Path(bindings_path).read_text())
    check(bindings['version'] == 'card-memory-pilot-v1', 'Wrong source binding version')
    for relative, expected in bindings['files'].items():
        path = (root / relative).resolve()
        check(path.is_relative_to(root), 'Source path outside repository')
        check(sha(path) == expected, f'Source changed: {relative}')
    for path in (protocol_path, inputs_path):
        relative = str(Path(path).resolve().relative_to(root))
        check(relative in bindings['files'], f'Unbound input: {relative}')
    actual = runtime()
    for field, expected in bindings['runtime'].items():
        check(actual[field] == expected, f'Runtime changed: {field}')
    return json.loads(Path(protocol_path).read_text()), json.loads(Path(inputs_path).read_text()), bindings


def preserve_failure(error, actions):
    """Attempt every preservation action without replacing the original error."""
    for action in actions:
        try:
            action()
        except BaseException as secondary:  # noqa: BLE001 - preserve the original failure
            note = getattr(error, 'add_note', None)
            if callable(note):
                note(f'Failure preservation also failed: {secondary!r}')
