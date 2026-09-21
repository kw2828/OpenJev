"""Pure runtime-wrapper qualification; no numeric imports, weights, or model calls."""
from __future__ import annotations

import builtins
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'scripts/qualify_otto_pretrained_runtime_v2.py'
SPEC = importlib.util.spec_from_file_location('otto_runtime_v2_wrapper_fixture', PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_wrapper_and_old_engine_import_no_numeric_or_model_packages(monkeypatch):
    original = builtins.__import__
    calls = []

    def guarded(name, *args, **kwargs):
        calls.append(name)
        assert name.split('.')[0] not in {'numpy', 'tensorflow', 'tf_keras', 'h5py', 'torch'}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    engine = module.load_engine()
    assert engine.VERSION == 'otto-pretrained-qualification-v1'
    assert calls


def test_only_three_metadata_globals_change_and_all_inference_settings_remain():
    engine = M.load_engine()
    before = dict(vars(engine))
    configuration = copy.deepcopy(engine.CONFIGURATION)
    original_sources = set(engine.REQUIRED_SOURCES)
    assert M.configure_engine(engine) is engine
    changed = {key for key, value in before.items() if vars(engine)[key] is not value}
    assert changed == {'VERSION', 'VERSIONS', 'REQUIRED_SOURCES'}
    assert engine.CONFIGURATION == configuration
    assert engine.CONFIGURATION['atol'] == 1e-4 and engine.CONFIGURATION['rtol'] == 1e-5
    assert engine.CONFIGURATION['exact_action_agreement'] is True
    assert engine.VERSION == M.VERSION
    assert engine.VERSIONS == {**M.ORIGINAL_VERSIONS, 'numpy': '2.5.3'}
    assert engine.REQUIRED_SOURCES == original_sources | M.ADDED_SOURCES
    assert engine.CONFIGURATION['raw_value_inputs'] == 16
    assert engine.CONFIGURATION['physical_policy_fixtures'] == 32
    # Six raw routes produce 46 forwards; 32 physical +4 floor fixtures add 36.
    raw_calls = sum(len(range(0, 16, size)) for _sym in (False, True) for size in (1, 3, 16))
    assert raw_calls + 32 + len(engine.FLOOR_MASSES) == 82
    assert engine.SCOPE == before['SCOPE']
    assert engine.LIMITS == {'native_seconds': 600, 'rss_bytes': 4 * 1024**3,
                             'output_bytes': 256 * 1024**2, 'simulator_calls': 0}
    with pytest.raises(ValueError, match='fresh original'):
        M.configure_engine(engine)


def test_loading_new_engine_does_not_mutate_other_imports_or_old_file():
    pin = M.sha(ROOT / M.ENGINE)
    original = M.load_engine()
    adapted = M.configure_engine(M.load_engine())
    assert original is not adapted
    assert original.VERSION == 'otto-pretrained-qualification-v1'
    assert original.VERSIONS['numpy'] == '2.2.6'
    assert original.REQUIRED_SOURCES.isdisjoint(M.ADDED_SOURCES)
    assert M.sha(ROOT / M.ENGINE) == pin == M.ENGINE_SHA256


def write_plan(path, plan):
    path.write_text(json.dumps(plan, sort_keys=True))
    return SimpleNamespace(plan=path, plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def pinned_plan(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    names = M.ADDED_SOURCES | {M.ENGINE, M.PORT}
    sources = {}
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name in {M.ENGINE, M.PORT, M.WRAPPER}:
            path.write_bytes((ROOT / name).read_bytes())
        else:
            path.write_text('prospective synthetic source fixture\n')
        sources[name] = M.sha(path)
    monkeypatch.setattr(M, 'ROOT', root)
    monkeypatch.setattr(M.sys, 'executable', str(root / M.INTERPRETER))
    plan = {'version': M.VERSION, 'status': 'frozen_before_run', 'sources': sources,
            'runtime': {'versions': dict(M.RUNTIME_VERSIONS), 'python_executable': str(root / M.INTERPRETER)}}
    return root, plan, write_plan(root / 'plan.json', plan)


def test_external_plan_and_all_source_bytes_authenticate_before_engine_import(pinned_plan, monkeypatch):
    _, plan, args = pinned_plan
    monkeypatch.setattr(M, 'load_engine', lambda: pytest.fail('authentication imported engine'))
    assert M.authenticate_wrapper(args) == plan
    args.plan_sha256 = '0' * 64
    with pytest.raises(ValueError, match='external v2 plan pin'):
        M.authenticate_wrapper(args)


@pytest.mark.parametrize('name', sorted(M.ADDED_SOURCES | {M.ENGINE, M.PORT}))
def test_missing_required_source_rejected(pinned_plan, name):
    root, plan, _ = pinned_plan
    del plan['sources'][name]
    args = write_plan(root / 'plan.json', plan)
    with pytest.raises(ValueError, match='membership'):
        M.authenticate_wrapper(args)


@pytest.mark.parametrize('name', [M.ENGINE, M.PORT])
def test_newly_repinned_engine_or_port_is_not_accepted(pinned_plan, name):
    root, plan, _ = pinned_plan
    path = root / name
    path.write_bytes(path.read_bytes() + b'\n# changed bytes\n')
    plan['sources'][name] = M.sha(path)
    args = write_plan(root / 'plan.json', plan)
    with pytest.raises(ValueError, match='unchanged engine and port'):
        M.authenticate_wrapper(args)


def test_changed_wrapper_bytes_rejected(pinned_plan):
    root, _, args = pinned_plan
    (root / M.WRAPPER).write_text('changed')
    with pytest.raises(ValueError, match='source bytes'):
        M.authenticate_wrapper(args)


def test_engine_hash_checked_before_dynamic_import(pinned_plan, monkeypatch):
    root, _, _ = pinned_plan
    (root / M.ENGINE).write_text('raise AssertionError("untrusted source executed")')
    monkeypatch.setattr(M.importlib.util, 'spec_from_file_location', lambda *_: pytest.fail('untrusted import attempted'))
    with pytest.raises(ValueError, match='before import'):
        M.load_engine()


@pytest.mark.parametrize('mutation', ['old_version', 'old_numpy', 'other_package', 'old_interpreter', 'actual_interpreter'])
def test_wrong_runtime_or_version_cannot_silently_fall_back(pinned_plan, mutation, monkeypatch):
    root, plan, _ = pinned_plan
    if mutation == 'old_version':
        plan['version'] = 'otto-pretrained-qualification-v1'
    elif mutation == 'old_numpy':
        plan['runtime']['versions']['numpy'] = '2.2.6'
    elif mutation == 'other_package':
        plan['runtime']['versions']['tensorflow'] = '2.21.0'
    elif mutation == 'old_interpreter':
        plan['runtime']['python_executable'] = str(root / '.venv-otto-reference/bin/python')
    else:
        monkeypatch.setattr(M.sys, 'executable', str(root / '.venv/bin/python'))
    args = write_plan(root / 'plan.json', plan)
    with pytest.raises(ValueError, match='plan|version|identity'):
        M.authenticate_wrapper(args)


def test_main_preserves_original_argv_and_delegates_only_after_authentication(monkeypatch):
    argv = [str(PATH), '--plan', '/tmp/plan.json', '--plan-sha256', 'abc',
            '--supervision', '/tmp/launch.json', '--output', '/tmp/run-01']
    events = []
    engine = object()
    monkeypatch.setattr(M.sys, 'argv', argv)
    monkeypatch.setattr(M, 'authenticate_wrapper', lambda args: events.append(('auth', args.plan_sha256)))
    monkeypatch.setattr(M, 'load_engine', lambda: events.append('load') or engine)

    def configure(actual):
        assert actual is engine
        events.append('configure')

        def main():
            assert M.sys.argv == argv
            events.append('original_main')
            return 7
        return SimpleNamespace(main=main)

    monkeypatch.setattr(M, 'configure_engine', configure)
    assert M.main() == 7
    assert events == [('auth', 'abc'), 'load', 'configure', 'original_main']
