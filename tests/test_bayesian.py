import numpy as np
import pytest

from openjev.research.bayesian import BayesianLogistic, brier_reward


def test_brier_truthful_probability_maximizes_expected_score():
    q = .7
    expected = lambda p: q*brier_reward(p, 1)+(1-q)*brier_reward(p, 0)
    assert expected(q) > expected(.5)
    assert expected(q) > expected(.9)
    with pytest.raises(ValueError):
        brier_reward(float('nan'), 1)


def test_laplace_fit_probabilities_and_variance_decomposition():
    model = BayesianLogistic(1).fit(np.ones((100, 1)), [0]*40+[1]*60, ['train']*100)
    result = model.predict([1], unit_id='test')
    assert result.mean_probability == pytest.approx(.6, abs=.02)
    assert result.lower_probability < result.mean_probability < result.upper_probability
    assert result.epistemic_variance + result.aleatoric_variance == pytest.approx(
        result.mean_probability*(1-result.mean_probability), abs=1e-10)
    assert np.linalg.eigvalsh(model.covariance).min() > 0
    more = BayesianLogistic(1).fit(np.ones((1000, 1)), [0]*400+[1]*600, ['train']*1000)
    assert more.predict([1], unit_id='test').epistemic_variance < result.epistemic_variance


def test_no_train_test_overlap_or_invalid_observations():
    model = BayesianLogistic(2).fit([[1, 0], [1, 1]], [0, 1], ['a', 'b'])
    with pytest.raises(ValueError, match='held out'):
        model.predict([1, 0], unit_id='a')
    for x, y in [([[1, float('inf')]], [1]), ([[1, 0]], [2]), ([], [])]:
        with pytest.raises(ValueError):
            model.fit(x, y, ['x'])
    with pytest.raises(ValueError):
        model.predict([1], unit_id='x')
    rng = np.random.default_rng(7)
    assert 0 <= model.sample_probability([1, 0], rng) <= 1


def test_real_line_scenario_labels_do_not_depend_on_net_ammo():
    # Regression for v1: this scenario replenishes ammo while shots still hit.
    import importlib.util
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('bayesian_doom_runner', root/'research/bayesian_doom.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    protocol = json.loads((root/'research/protocols/bayesian-doom-v2.json').read_text())
    protocol['max_steps_per_episode'] = 50
    summary, rows = runner.run_episode('defend_the_line', 998876, 'collect', protocol,
                                       rng=np.random.default_rng(123))
    assert summary['shots'] == 0  # net decrease is zero, not actual shot count
    assert summary['hits'] > 0 and rows
    assert all(row['issued_fire'] and row['outcome'] in (0, 1) for row in rows)
    assert summary['utility'] == sum(r['outcome'] for r in rows)-.25*len(rows)
    with pytest.raises(ValueError, match='No common-audit'):
        runner.common_audit(None, [])
