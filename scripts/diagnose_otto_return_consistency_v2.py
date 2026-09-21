"""Prospective terminal-packet repair of the preserved failed v1 diagnostic.

Only public packet eligibility changes: a found terminal packet has no available
actions. Its preceding step's allowed_actions remains a nonempty movement mask.
All inherited authentication, selection, joins and arithmetic are unchanged.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = 'scripts/diagnose_otto_return_consistency.py'
ORIGINAL_PIN = '5fe0e5d242862cb3edacfb9c12b63d4c2c5da515129be91e3151979dad8f12cc'
VERSION = 'otto-return-consistency-v2'
ADDITIONS = {'scripts/diagnose_otto_return_consistency_v2.py',
             'tests/test_otto_return_consistency_v2.py', 'research/otto-return-consistency-repair.md'}


def load_original():
    path = ROOT / ORIGINAL
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('original diagnostic must be a regular nonsymlink source')
    if hashlib.sha256(path.read_bytes()).hexdigest() != ORIGINAL_PIN:
        raise ValueError('original diagnostic source pin changed')
    spec = importlib.util.spec_from_file_location('_return_consistency_v2_inherited', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_original()
INHERITED_SOURCES = frozenset(BASE.SOURCES)
SOURCES = set(INHERITED_SOURCES) | ADDITIONS


def packet(value):
    BASE.require(type(value) is dict and set(value) == {'position', 'hit', 'done', 'step', 'valid_actions'},
                 'public packet whitelist')
    pos = value['position']
    BASE.require(type(pos) is list and len(pos) == 2 and all(type(x) is int and 0 <= x < 53 for x in pos),
                 'grid position')
    BASE.integer(value['step'])
    BASE.require(type(value['done']) is bool and type(value['hit']) is int
                 and (value['hit'] == -2 if value['done'] else 0 <= value['hit'] < 4), 'public hit/done')
    if value['done']:
        BASE.require(type(value['valid_actions']) is list and value['valid_actions'] == [],
                     'terminal packet has no available actions')
    else:
        actions = [a for a in range(4) if 0 <= pos[a // 2] + 2 * (a % 2) - 1 < 53]
        BASE.require(BASE.allowed(value['valid_actions']) == actions, 'public movement eligibility')
    return value


# The imported functions retain their original implementation and refer to these
# explicitly replaced globals. No fallback, skipped row or numerical repair.
BASE.packet = packet
BASE.VERSION = VERSION
BASE.SOURCES = SOURCES
Diagnostic = BASE.Diagnostic
main = BASE.main


if __name__ == '__main__':
    raise SystemExit(main())
