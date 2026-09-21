"""Run the immutable pretrained qualifier in the separately pinned NumPy 2.5.3 runtime.

Only the engine's receipt version, required package versions, and required source
membership change. Fixture generation, all 82 calls per framework, inference,
numeric tolerances, exact-action gate, and failure lifecycle remain the original
source. The failed v1 artifacts and environment are never modified or retried.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'otto-pretrained-qualification-runtime-v2'
ENGINE = 'scripts/qualify_otto_pretrained.py'
ENGINE_SHA256 = '7c6514831ea204938905bf36575d50bc9abb5365be19bcc4c396fdad672c035a'
PORT = 'src/openjev/research/otto_pretrained_value.py'
PORT_SHA256 = '20e2f83389bc6bfbc890636904ca4aaba028937673446845866e6156a41493c1'
WRAPPER = 'scripts/qualify_otto_pretrained_runtime_v2.py'
ADDED_SOURCES = {WRAPPER, 'tests/test_qualify_otto_pretrained_runtime_v2.py',
                 'research/otto-pretrained-reference-runtime-v2-protocol.md'}
ORIGINAL_VERSIONS = {'tensorflow': '2.20.0', 'tf-keras': '2.20.1', 'numpy': '2.2.6', 'h5py': '3.14.0'}
RUNTIME_VERSIONS = {**ORIGINAL_VERSIONS, 'numpy': '2.5.3'}
INTERPRETER = '.venv-otto-reference-v2/bin/python'
ADAPTATION = {'version': {'from': 'otto-pretrained-qualification-v1', 'to': VERSION},
              'package_version': {'numpy': {'from': '2.2.6', 'to': '2.5.3'}},
              'required_source_additions': sorted(ADDED_SOURCES),
              'inference_changes': [], 'fixture_changes': [], 'tolerance_changes': []}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def source_path(name):
    relative = Path(name)
    require(not relative.is_absolute() and relative.parts and '..' not in relative.parts, 'contained source path')
    path = ROOT / relative
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
            and path.resolve().is_relative_to(ROOT), 'regular contained source')
    return path


def authenticate_wrapper(args):
    """Bind wrapper, unchanged engine/port, and new interpreter before importing engine."""
    require(args.plan.is_absolute() and args.plan.is_file()
            and not any(p.is_symlink() for p in (args.plan, *args.plan.parents)), 'regular absolute plan')
    require(sha(args.plan) == args.plan_sha256, 'external v2 plan pin')
    plan = json.loads(args.plan.read_text())
    require(plan['version'] == VERSION and plan['status'] == 'frozen_before_run', 'separate frozen v2 plan')
    sources = plan['sources']
    require(ADDED_SOURCES | {ENGINE, PORT} <= sources.keys(), 'wrapper, test, protocol, engine, and port membership')
    require(sources[ENGINE] == ENGINE_SHA256 and sources[PORT] == PORT_SHA256, 'unchanged engine and port pins')
    for name, pin in sources.items():
        require(sha(source_path(name)) == pin, f'v2 source bytes: {name}')
    runtime = plan['runtime']
    require(runtime['versions'] == RUNTIME_VERSIONS, 'only the declared NumPy runtime version changes')
    require(runtime['python_executable'] == str(ROOT / INTERPRETER)
            == str(Path(sys.executable).absolute()), 'separate v2 interpreter identity')
    return plan


def load_engine():
    path = source_path(ENGINE)
    require(sha(path) == ENGINE_SHA256, 'immutable engine hash before import')
    spec = importlib.util.spec_from_file_location('_otto_pretrained_runtime_v2_engine', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_engine(engine):
    """Exactly three module globals change; no inference callable is replaced."""
    require(Path(engine.__file__).resolve() == (ROOT / ENGINE).resolve()
            and sha(source_path(ENGINE)) == ENGINE_SHA256, 'immutable engine configuration identity')
    require(engine.VERSION == 'otto-pretrained-qualification-v1' and engine.VERSIONS == ORIGINAL_VERSIONS,
            'fresh original engine metadata')
    require(engine.ROOT == ROOT and engine.PORT == PORT, 'original root and NumPy port route')
    engine.VERSION = VERSION
    engine.VERSIONS = dict(RUNTIME_VERSIONS)
    engine.REQUIRED_SOURCES = engine.REQUIRED_SOURCES | ADDED_SOURCES
    return engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'supervision', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    authenticate_wrapper(args)
    # Original main parses the unchanged argv and owns all output/receipt lifecycle.
    return configure_engine(load_engine()).main()


if __name__ == '__main__':
    main()
