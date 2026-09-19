"""Pure packing and fake-environment failure tests; no native or RNG calls."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def collector(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('test_only_card_data_collector', scripts / 'prepare_card_memory_data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The actual public tracker/teacher are used; choice is a fixed, legal,
    # handwritten mismatch sequence, with no stochastic behavior construction.
    monkeypatch.setattr(module.PublicCardTracker, 'choose', lambda self, *_args, **_kwargs: self.step % 2)
    monkeypatch.setattr(module.np.random, 'PCG64', lambda _seed: None)
    monkeypatch.setattr(module.np.random, 'Generator', lambda _bitgen: None)
    return module


class FakeEnv:
    """104 scripted public reveals of two unequal cards, not a simulator."""

    def __init__(self, *, fail_reset=False, fail_step=None, bad_reward=None, bad_frame=None,
                 fail_close=False, on_step=None, on_close=None):
        self.fail_reset, self.fail_step, self.bad_reward = fail_reset, fail_step, bad_reward
        self.bad_frame, self.fail_close = bad_frame, fail_close
        self.on_step, self.on_close = on_step, on_close
        self.steps, self.closed = 0, False

    def reset(self, *, seed):
        self.seed = seed
        if self.fail_reset:
            raise RuntimeError('fake reset failed')
        return np.full(52, 13, dtype=np.int64), {}

    def step(self, action):
        if self.steps == self.fail_step:
            raise RuntimeError('fake step failed')
        assert action == self.steps % 2
        after = np.full(52, 13, dtype=np.int64)
        after[0] = 0
        if action == 1:
            after[1] = 1
        reward = 0.0 if action == 0 else -2 / 104
        if self.steps == self.bad_reward:
            reward = 99.0
        if self.steps == self.bad_frame:
            after = np.empty(0, dtype=np.int64)
        self.steps += 1
        if self.on_step:
            self.on_step()
        return after, reward, False, self.steps == 104, {}

    def close(self):
        self.closed = True
        if self.on_close:
            self.on_close()
        if self.fail_close:
            raise OSError('fake close failed')


def work():
    return {f'{phase}_{kind}_{state}': 0 for phase in ('native', 'replay')
            for kind in ('resets', 'steps') for state in ('attempted', 'returned')}


def collect(module, env=None, counts=None):
    env = FakeEnv() if env is None else env
    return module.collect_episode(410, 410, deadline=float('inf'), work=work() if counts is None else counts,
                                  env_factory=lambda: env)


def test_collection_keeps_public_before_write_targets_and_full_prefix(collector):
    env, counts = FakeEnv(), work()
    arrays, receipt = collect(collector, env, counts)
    assert env.closed and receipt['steps'] == 104
    assert arrays['observations'].shape == (105, 52)
    assert not arrays['target_mask'][:3].any()  # Two-card mismatch frame still visible at boundary 2.
    assert arrays['target_mask'][3, 1] and arrays['targets'][3, 1] == 1 and arrays['ages'][3, 1] == 1
    assert arrays['ranks'][:4].tolist() == [0, 1, 0, 1]
    assert not arrays['terminated'].any() and arrays['truncated'].sum() == 1 and arrays['truncated'][-1]
    assert counts['native_resets_attempted'] == counts['native_resets_returned'] == 1
    assert counts['native_steps_attempted'] == counts['native_steps_returned'] == 104
    assert receipt['return'] == pytest.approx(-1.0)


def test_pack_exact_prefix_padding_dtypes_and_no_alias(collector):
    whole, _ = collect(collector)
    short = {name: value[:3].copy() for name, value in whole.items() if name != 'observations'}
    short['observations'] = whole['observations'][:4].copy()
    packed = collector.pack([short, whole])
    assert set(packed) == {'positions', 'ranks', 'valid', 'targets', 'target_mask', 'ages'}
    assert packed['valid'].sum(axis=1).tolist() == [3, 104]
    assert np.all(packed['targets'][0, 3:] == -1) and np.all(packed['ages'][0, 3:] == -1)
    assert not packed['target_mask'][0, 3:].any()
    assert packed['ages'].dtype == np.int32 and packed['positions'].dtype == np.int64
    packed['positions'][0, 0] = 50
    assert short['positions'][0] == 0


@pytest.mark.parametrize('episodes', ([], [{'positions': []}], [{'positions': np.zeros(105)}]))
def test_pack_rejects_empty_or_oversized_episode(collector, episodes):
    with pytest.raises(ValueError):
        collector.pack(episodes)


def test_failed_reset_count_and_close_preserve_original_exception(collector):
    env, counts = FakeEnv(fail_reset=True, fail_close=True), work()
    with pytest.raises(RuntimeError, match='fake reset failed') as error:
        collect(collector, env, counts)
    assert env.closed and counts['native_resets_attempted'] == 1 and counts['native_resets_returned'] == 0
    assert error.value.partial_card_episode['observations'] == []
    assert any('fake close failed' in note for note in error.value.__notes__)


def test_failed_native_call_and_successful_return_validation_are_distinct(collector):
    counts = work()
    with pytest.raises(RuntimeError, match='fake step failed') as error:
        collect(collector, FakeEnv(fail_step=2), counts)
    assert counts['native_steps_attempted'] == 3 and counts['native_steps_returned'] == 2
    assert len(error.value.partial_card_episode['positions']) == 2
    counts = work()
    with pytest.raises(ValueError, match='reward') as error:
        collect(collector, FakeEnv(bad_reward=2), counts)
    assert counts['native_steps_attempted'] == counts['native_steps_returned'] == 3
    assert error.value.partial_card_episode['rewards'][-1] == 99.0
    assert len(error.value.partial_card_episode['observations']) == 4


def test_malformed_return_keeps_raw_event_before_derived_rank_indexing(collector):
    counts = work()
    with pytest.raises(IndexError) as error:
        collect(collector, FakeEnv(bad_frame=0), counts)
    partial = error.value.partial_card_episode
    assert partial['positions'] == [0] and partial['rewards'] == [0.0]
    assert partial['terminated'] == [False] and partial['truncated'] == [False]
    assert len(partial['observations']) == 2 and partial['ranks'] == []
    assert counts['native_steps_returned'] == 1


def test_successful_episode_close_failure_preserves_all_returned_data(collector):
    counts = work()
    with pytest.raises(OSError, match='fake close failed') as error:
        collect(collector, FakeEnv(fail_close=True), counts)
    assert len(error.value.partial_card_episode['positions']) == 104
    assert len(error.value.partial_card_episode['observations']) == 105
    assert counts['native_steps_returned'] == 104


def test_deadline_before_creation_and_after_return_preserves_paid_work(collector, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(collector.time, 'monotonic', lambda: clock[0])
    calls = []
    with pytest.raises(TimeoutError):
        collector.collect_episode(410, 410, deadline=0., work=work(), env_factory=lambda: calls.append(True))
    assert calls == []
    counts = work()
    env = FakeEnv(on_step=lambda: clock.__setitem__(0, 2.))
    with pytest.raises(TimeoutError) as error:
        collector.collect_episode(410, 410, deadline=1., work=counts, env_factory=lambda: env)
    assert env.closed and counts['native_steps_returned'] == 1
    assert len(error.value.partial_card_episode['positions']) == 1


def test_cleanup_time_included_and_post_cleanup_deadline_enforced(collector, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(collector.time, 'monotonic', lambda: clock[0])
    env = FakeEnv(on_close=lambda: clock.__setitem__(0, 5.))
    _, receipt = collect(collector, env)
    assert receipt['elapsed_seconds'] == 5.
    clock[0] = 0.
    env = FakeEnv(on_close=lambda: clock.__setitem__(0, 5.))
    with pytest.raises(TimeoutError) as error:
        collector.collect_episode(410, 410, deadline=3., work=work(), env_factory=lambda: env)
    assert len(error.value.partial_card_episode['positions']) == 104


def test_replay_authenticates_labels_behavior_and_counts_before_failure(collector):
    arrays, _ = collect(collector)
    counts, env = work(), FakeEnv()
    collector.replay_episode(410, 410, arrays, deadline=float('inf'), work=counts, env_factory=lambda: env)
    assert env.closed and counts['replay_steps_returned'] == 104
    corrupted = {k: v.copy() for k, v in arrays.items()}
    corrupted['ages'][3, 1] = 99
    counts, env = work(), FakeEnv()
    with pytest.raises(ValueError, match='ages'):
        collector.replay_episode(410, 410, corrupted, deadline=float('inf'), work=counts, env_factory=lambda: env)
    assert env.closed and counts['replay_steps_returned'] == 3
    counts, env = work(), FakeEnv(bad_reward=2, fail_close=True)
    with pytest.raises(ValueError, match='reward') as error:
        collector.replay_episode(410, 410, arrays, deadline=float('inf'), work=counts, env_factory=lambda: env)
    assert counts['replay_steps_attempted'] == counts['replay_steps_returned'] == 3
    assert any('fake close failed' in note for note in error.value.__notes__)


def setup_prepare(module, monkeypatch, tmp_path, *, first_env=None):
    files = []
    for name in ('inputs', 'protocol', 'bindings'):
        path = tmp_path / f'{name}.json'
        path.write_text('{}')
        files.append(path)
    cases = [{'seed': 410 + i, 'behavior_seed': 410} for i in range(4)]
    monkeypatch.setattr(module, 'authenticate', lambda *_args: ({'data': {'wall_cap_seconds': 300}},
                        {'preflight': {'engineering': cases}}, {}))
    original_collect, original_replay = module.collect_episode, module.replay_episode
    environments = [first_env] if first_env is not None else []

    def collect_fake(seed, behavior_seed, **kwargs):
        env = environments.pop() if environments else FakeEnv()
        return original_collect(seed, behavior_seed, env_factory=lambda: env, **kwargs)

    monkeypatch.setattr(module, 'collect_episode', collect_fake)
    monkeypatch.setattr(module, 'replay_episode', lambda seed, bs, arrays, **kwargs:
                        original_replay(seed, bs, arrays, env_factory=FakeEnv, **kwargs))
    return tmp_path / 'attempt', *files, tmp_path


def test_complete_fake_preflight_exclusive_hashes_counts_and_no_retry(collector, monkeypatch, tmp_path):
    args = setup_prepare(collector, monkeypatch, tmp_path)
    collector.prepare(*args, preflight=True)
    out = args[0]
    done = json.loads((out / 'completed.json').read_text())
    assert done['episodes'] == 4 and done['status'] == 'complete'
    assert done['work']['native_steps_returned'] == done['work']['replay_steps_returned'] == 416
    assert collector.sha(out / 'engineering.npz') == done['parts']['engineering']
    assert collector.sha(out / 'episodes.json') == done['episodes_sha256']
    assert collector.sha(out / 'started.json') == done['started_sha256']
    assert len((out / 'ledger.jsonl').read_text().splitlines()) == 4
    with pytest.raises(FileExistsError):
        collector.prepare(*args, preflight=True)


def test_prepare_failure_keeps_partial_and_attempted_returned_counts(collector, monkeypatch, tmp_path):
    args = setup_prepare(collector, monkeypatch, tmp_path, first_env=FakeEnv(fail_step=2))
    with pytest.raises(RuntimeError, match='fake step failed'):
        collector.prepare(*args, preflight=True)
    out = args[0]
    done = json.loads((out / 'failed.json').read_text())
    assert done['work']['native_steps_attempted'] == 3 and done['work']['native_steps_returned'] == 2
    assert done['current']['index'] == 0 and done['episodes'] == 0 and not done['automatic_retry']
    assert not (out / 'completed.json').exists()
    with np.load(out / 'partial-episode.npz', allow_pickle=False) as partial:
        assert len(partial['positions']) == 2 and len(partial['observations']) == 3


def test_prepare_preservation_failure_cannot_replace_original(collector, monkeypatch, tmp_path):
    args = setup_prepare(collector, monkeypatch, tmp_path, first_env=FakeEnv(fail_step=0))
    original = collector.write_json

    def fail(path, value):
        if Path(path).name == 'failed.json':
            raise OSError('fake failure receipt unavailable')
        return original(path, value)

    monkeypatch.setattr(collector, 'write_json', fail)
    with pytest.raises(RuntimeError, match='fake step failed') as error:
        collector.prepare(*args, preflight=True)
    assert any('failure receipt unavailable' in note for note in error.value.__notes__)


def test_late_completion_demoted_without_repeating_work(collector, monkeypatch, tmp_path):
    args = setup_prepare(collector, monkeypatch, tmp_path)
    original = collector.deadline_check

    def deadline(value):
        if (args[0] / 'completed.json').exists():
            raise TimeoutError('fake late terminal write')
        original(value)

    monkeypatch.setattr(collector, 'deadline_check', deadline)
    with pytest.raises(TimeoutError, match='fake late'):
        collector.prepare(*args, preflight=True)
    assert not (args[0] / 'completed.json').exists()
    assert (args[0] / 'invalid-completion.json').exists()
    failure = json.loads((args[0] / 'failed.json').read_text())
    assert failure['episodes'] == 4 and failure['work']['native_steps_returned'] == 416
