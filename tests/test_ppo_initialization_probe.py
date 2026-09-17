import importlib.util
from pathlib import Path

import pytest

pytest.importorskip('minigrid')
torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('initialization_probe_test', ROOT/'scripts/ppo_initialization_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize('arm', probe.PROTOCOL['arms'])
def test_zero_head_changes_only_value_parameters_without_rng_draws(arm):
    models = probe.paired_models(arm, 7)
    after_pair = torch.get_rng_state().clone()
    torch.manual_seed(7)
    independent = probe.AssociativePolicy(arm, probe.OBS_SIZE, probe.PROTOCOL['hidden'])
    assert torch.equal(after_pair, torch.get_rng_state())
    for name, value in independent.state_dict().items():
        assert torch.equal(value, models['original'].state_dict()[name])
        if name.startswith('value.'):
            assert not torch.count_nonzero(models['zero'].state_dict()[name])
        else:
            assert torch.equal(value, models['zero'].state_dict()[name])
    assert torch.count_nonzero(models['original'].value.weight)


@torch.no_grad()
def reference_rollout(arm, seed, count, length):
    """Independent first-rollout path, preserving the frozen trainer's call order."""
    p = probe.reference.PROTOCOL
    torch.manual_seed(seed)
    probe.np.random.seed(seed)
    model = probe.AssociativePolicy(arm, probe.OBS_SIZE, p['hidden'])
    torch.manual_seed(p['policy_rng_offset']+seed)
    rows = []
    with probe.CueMemoryBatch(count, p['train_size'], p['training_seed_base']+seed*100000,
                             p['max_steps']) as envs:
        obs = torch.from_numpy(envs.reset())
        previous = torch.zeros(count, dtype=torch.long)
        state = model.initial_state(count)
        reset = torch.ones(count, dtype=torch.bool)
        for _ in range(length):
            logits, value, state = model.observe(obs, previous, state, reset, False)
            distribution = probe.Categorical(logits=logits)
            actions = distribution.sample()
            following, rewards, terminated, truncated, infos = envs.step(actions.numpy())
            ended = torch.from_numpy(terminated | truncated)
            true_next = following.copy()
            for index, info in enumerate(infos):
                if 'episode' in info:
                    true_next[index] = info['terminal_observation']
            _, next_value, _ = model.observe(torch.from_numpy(true_next), actions, state,
                                             torch.zeros_like(reset), False)
            rows.append({'obs': obs, 'previous': previous, 'reset': reset, 'actions': actions,
                         'logprob': distribution.log_prob(actions), 'values_original': value,
                         'next_values_original': next_value, 'rewards': torch.from_numpy(rewards),
                         'terminated': torch.from_numpy(terminated), 'ended': ended})
            obs, previous, reset = torch.from_numpy(following), actions, ended
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}, torch.get_rng_state()


@pytest.mark.parametrize('arm', probe.PROTOCOL['arms'])
def test_paired_collection_matches_frozen_recipe_and_sampling_rng(arm):
    # Full rollout length exercises time order; two environments keep this test small.
    models = probe.paired_models(arm, 7)
    data, receipt = probe.collect(models, 7, count=2, length=64)
    rng = torch.get_rng_state().clone()
    expected, expected_rng = reference_rollout(arm, 7, 2, 64)
    assert torch.equal(rng, expected_rng)
    for key, value in expected.items():
        assert torch.equal(data[key], value), key
    assert not torch.count_nonzero(data['values_zero'])
    assert not torch.count_nonzero(data['next_values_zero'])
    assert receipt['interactions'] == 128
    assert sum(receipt['action_counts']) == 128
    assert receipt['paired_logits_states_and_actions_identical']
    assert receipt['positive_reward_events'] == int((data['rewards'] > 0).sum())
    assert receipt['data_sha256'] == probe.tensor_digest(data)


def test_auto_reset_and_terminal_bootstrap_match_reference(monkeypatch):
    # This test-only short cap guarantees two internal episode boundaries.
    monkeypatch.setitem(probe.PROTOCOL, 'max_steps', 2)
    monkeypatch.setitem(probe.reference.PROTOCOL, 'max_steps', 2)
    models = probe.paired_models('fast_global', 7)
    data, receipt = probe.collect(models, 7, count=2, length=5)
    expected, _ = reference_rollout('fast_global', 7, 2, 5)
    for key, value in expected.items():
        assert torch.equal(data[key], value), key
    assert data['reset'][:, 0].tolist() == [True, False, True, False, True]
    assert receipt['truncation_only_events'] == 4
    assert len(receipt['completed_episodes']) == 4
    probe.analyze(models['original'], data, 'original', 'actual')


@pytest.mark.parametrize('arm', probe.PROTOCOL['arms'])
def test_zero_critic_zero_rewards_has_no_actor_or_critic_gradient(arm):
    models = probe.paired_models(arm, 7)
    data, _ = probe.collect(models, 7, count=2, length=8)
    model = models['zero']
    before = probe.tensor_digest(model.state_dict())
    result = probe.analyze(model, data, 'zero', 'synthetic_all_zero')
    assert result['synthetic_rewards']
    assert result['raw_advantage']['nonzero_count'] == 0
    assert result['normalized_advantage']['nonzero_count'] == 0
    assert result['return_targets']['nonzero_count'] == 0
    for group in result['gradient_components'].values():
        norms = group['l2_norms']
        for component in ('actor_raw', 'actor_centered_raw', 'actor_normalized', 'critic_weighted'):
            assert norms[component] == 0
        assert all(value is None for value in group['cosines'].values())
    assert result['gradient_components']['all']['l2_norms']['entropy_weighted'] > 0
    assert result['gradient_components']['value_head']['l2_norms']['entropy_weighted'] == 0
    assert before == probe.tensor_digest(model.state_dict())
    assert all(value.grad is None for value in model.parameters())


def test_actual_rewards_preserved_and_counterfactual_explicit():
    models = probe.paired_models('gru', 7)
    data, _ = probe.collect(models, 7, count=2, length=8)
    data['rewards'] = torch.zeros_like(data['rewards'])
    data['rewards'][-1, 0] = .75  # A test-only positive event, distinct from environment receipts.
    before = data['rewards'].clone()
    actual = probe.analyze(models['zero'], data, 'zero', 'actual')
    synthetic = probe.analyze(models['zero'], data, 'zero', 'synthetic_all_zero')
    assert not actual['synthetic_rewards'] and synthetic['synthetic_rewards']
    assert actual['raw_advantage']['nonzero_count'] > 0
    assert actual['gradient_components']['all']['l2_norms']['actor_normalized'] > 0
    assert actual['gradient_components']['value_head']['l2_norms']['critic_weighted'] > 0
    # With a zero value head its critic gradient cannot yet reach the shared trunk.
    assert actual['gradient_components']['shared_trunk']['l2_norms']['critic_weighted'] == 0
    assert synthetic['raw_advantage']['nonzero_count'] == 0
    assert torch.equal(data['rewards'], before)


def test_advantage_rescaling_and_gradient_groups_are_accounted_for():
    models = probe.paired_models('gru', 7)
    data, _ = probe.collect(models, 7, count=2, length=8)
    model = models['original']
    before = probe.tensor_digest(model.state_dict())
    result = probe.analyze(model, data, 'original', 'synthetic_all_zero')
    assert result['raw_advantage']['std_population'] > 0
    assert result['normalized_advantage']['std_population'] == pytest.approx(1, abs=1e-5)
    for group in ('all', 'shared_trunk', 'actor_head'):
        components = result['gradient_components'][group]
        norms = components['l2_norms']
        assert norms['actor_normalized'] == pytest.approx(
            norms['actor_centered_raw']/result['normalization_divisor'], rel=2e-5)
        assert components['cosines']['actor_normalized__actor_centered_raw'] == pytest.approx(1, abs=1e-7)
    assert result['gradient_components']['shared_trunk']['l2_norms']['critic_weighted'] > 0
    assert result['gradient_components']['actor_head']['l2_norms']['critic_weighted'] == 0
    assert result['gradient_components']['value_head']['l2_norms']['actor_normalized'] == 0
    assert before == probe.tensor_digest(model.state_dict())
    assert all(value.grad is None for value in model.parameters())


def test_replay_rejects_changed_policy_and_unknown_conditions():
    models = probe.paired_models('gru', 7)
    data, _ = probe.collect(models, 7, count=2, length=4)
    with pytest.raises(ValueError, match='Unknown probe condition'):
        probe.analyze(models['original'], data, 'original', 'quietly_remove_rewards')
    with torch.no_grad():
        models['original'].actor.bias[0] += 1
    with pytest.raises(ValueError, match='replay differs'):
        probe.analyze(models['original'], data, 'original', 'actual')


def test_probe_rejects_undeclared_identities_before_collection(tmp_path):
    for arm, seed, smoke in [('gru', 7, False), ('gru', 101, True), ('unknown', 7, True)]:
        with pytest.raises(ValueError, match='declared identities'):
            probe.probe(arm, seed, tmp_path/'unused', smoke=smoke)
    assert not (tmp_path/'unused').exists()


def test_full_panel_requires_a_plan_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match='frozen plan is required'):
        probe.run(None, tmp_path/'unused')
    assert not (tmp_path/'unused').exists()


def test_freeze_signature_tracks_reference_and_rejects_drift(monkeypatch):
    signature = probe.signature()
    assert signature['protocol']['optimizer_steps'] == 0
    assert signature['source_sha256']['scripts/associative_ppo_study.py'] == probe.sha(
        ROOT/'scripts/associative_ppo_study.py')
    changed = dict(probe.PROTOCOL, rollout=1)
    monkeypatch.setattr(probe, 'PROTOCOL', changed)
    with pytest.raises(ValueError, match='differ from the reference'):
        probe.signature()
