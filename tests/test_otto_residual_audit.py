"""Fabricated-only independent scalar checks and producer compatibility tests."""
from copy import deepcopy
from itertools import pairwise

import numpy as np
import pytest

from openjev.research import otto_query_memory_metrics as metrics
from openjev.research import otto_residual_audit as audit
from openjev.research import otto_residual_contract as contract
from openjev.research import otto_residual_gate as gate
from openjev.research import otto_residual_replay as replay


def cohort(stage="dev", length=9):
    cases = 3 if stage == "dev" else 6
    identities = []
    for setting, regime in enumerate(("lambda3", "lambda4")):
        for case in range(cases):
            for arm in ("analytic", "neural", "period4_hold"):
                identities.append({"stage": "dev" if stage == "dev" else "test",
                    "episode_id": f"synthetic:{regime}:{case}:{arm}", "episode_index": len(identities),
                    "seed": 100 + setting * 10 + case, "case": case, "regime": regime, "arm": arm})
    lengths = [length(row) if callable(length) else length for row in identities]
    offsets = np.asarray([0, *np.cumsum(lengths)], dtype=np.int64)
    raw = np.tile(np.asarray([0., 2., 4., 6.], np.float32), (int(offsets[-1]), 1))
    legal = np.ones_like(raw, dtype=np.bool_)
    return identities, raw, legal, offsets


def producer_view(identities, raw, legal, actions, offsets, method, tau, seed, prior=None):
    episodes = []
    for index, identity in enumerate(identities):
        low, high = offsets[index:index + 2]
        episodes.append(metrics.episode_metrics(identity, 4, raw[low:high], legal[low:high], actions[low:high],
            prequery_forecast=None if prior is None else prior[low:high]))
    return metrics.aggregate_episodes(episodes, family=contract.view_name(method, tau), fit_seed=seed,
                                      expected_identities=identities)


def known_reports(stage="dev", tau=None, support=None):
    cases = 3 if stage == "dev" else 6
    support = cases if support is None else support
    identities, _, _, _ = cohort(stage)
    manifest = [{**row, "length": 9 if row["case"] < support else 1, "target_sha256": "0" * 64} for row in identities]
    reports = []
    for method, ratio, seed in contract.view_specs(stage, tau):
        value = (1. if method == "rls_full" else 10.) if support else 0.
        leaf = {"episodes": 3 * cases, "declared_case_count": cases,
                "supported_case_count": support, "case_weighted_raw_gap": value}
        reports.append({"version": metrics.VERSION, "family": contract.view_name(method, ratio), "seed": seed,
            "query_period": 4, "stage": identities[0]["stage"], "episodes": len(identities),
            "identity_manifest": deepcopy(manifest), "scopes": {
                name: {"by_regime": {regime: deepcopy(leaf) for regime in ("lambda3", "lambda4")}}
                for name in ("full", "later")}})
    return reports, identities


def set_gap(reports, family, value, *, regime=None, seed=None, scope="later"):
    for report in reports:
        if report["family"] == family and (seed is None or report["seed"] == seed):
            for setting in ((regime,) if regime else ("lambda3", "lambda4")):
                report["scopes"][scope]["by_regime"][setting]["case_weighted_raw_gap"] = value


def test_scalar_choice_near_minimum_and_centering_have_known_values():
    raw = np.array([0., 2., 4., 6.], np.float32)
    chosen = np.array([3., -1., 4., 6.], np.float32)
    assert audit.scalar_row(raw, [True] * 4, chosen) == (0., 2., 0., 4.5)
    assert audit.centered_mse([3., 4., 5., 6.], [0., 1., 2., 3.]) == 0.
    tiny = np.array([5e-11, 0., 7., 8.], np.float32)
    assert audit.scalar_row(tiny, [True] * 4, tiny)[:3] == (1., float(tiny[0]), 1.)
    # Illegal low scores never influence the selected action or legal MSE.
    illegal = np.array([-100., 2., 4., 6.], np.float32)
    assert audit.scalar_row(raw, [False, True, False, False], illegal) == (1., 0., 1., 0.)


def test_complete_denominators_and_known_scope_sums():
    ids, raw, legal, offsets = cohort(length=lambda row: 1 if row["case"] == 0 else 6)
    actions = raw.copy()
    for low, high in pairwise(offsets):
        if high - low > 1:
            actions[low + 1] = [3., -1., 4., 6.]
    actual = audit.report_view(ids, raw, legal, actions, offsets, "pretrained", None, audit.FIT_SEEDS[0])
    for regime in ("lambda3", "lambda4"):
        full = actual["scopes"]["full"]["by_regime"][regime]
        assert full["rows"] == 24 and full["episodes"] == 9 and full["supported_episodes"] == 6
        assert full["declared_case_count"] == 3 and full["supported_case_count"] == 2
        assert full["case_weighted_raw_gap"] == 1 / 3
        assert full["supported_episode_raw_gap"] == .5
        assert full["by_age"]["1"]["case_weighted_raw_gap"] == 2 / 3
    assert actual == producer_view(ids, raw, legal, actions, offsets, "pretrained", None, audit.FIT_SEEDS[0])


@pytest.mark.parametrize("stage,selected_tau", [("dev", None), ("confirm", .1)])
def test_actual_fabricated_replays_match_all_canonical_reports_and_gates(stage, selected_tau):
    ids, raw, legal, offsets = cohort(stage, length=lambda row: (1, 6, 11)[row["case"] % 3])
    rng = np.random.default_rng(84)
    raw[:] = rng.normal(size=raw.shape).astype(np.float32) * np.float32(16)
    legal[:, 2] = False
    masks = contract.masks_for_offsets(offsets, len(raw))
    query = masks["query_mask"]
    scores = np.full_like(raw, np.nan)
    scores[query] = raw[query]
    baseline = raw + rng.normal(size=raw.shape).astype(np.float32)
    baseline[query] = raw[query]
    joint = baseline * np.float32(.7)
    joint[query] = raw[query]
    shadow = np.full_like(raw, np.nan)
    shadow[masks["prior_mask"]] = 0.
    cues = rng.normal(size=(len(raw), 8)).astype(np.float32)
    cues /= np.linalg.norm(cues, axis=1, keepdims=True)
    cues[~masks["key_mask"]] = np.nan
    cache = {"version": contract.CACHE_VERSION, "query_period": 4, "base_action": baseline,
             "joint_action": joint, "shadow_prior": shadow, "cues": cues, "query_scores": scores,
             "episode_offsets": offsets, "work_counts": {}, **masks}
    independent, producer = [], []
    for method, tau, seed in contract.view_specs(stage, selected_tau):
        actions = replay.replay(cache, method, tau=tau)["action_scores"]
        actual = audit.report_view(ids, raw, legal, actions, offsets, method, tau, seed)
        expected = producer_view(ids, raw, legal, actions, offsets, method, tau, seed)
        assert actual == expected
        independent.append(actual)
        producer.append(expected)
    kwargs = {"stage": stage, "selected_tau": selected_tau, "technical_complete": True}
    assert audit.evaluate_reports(independent, ids, **kwargs) == gate.evaluate(producer, ids, **kwargs)
    assert len(independent) == (72 if stage == "dev" else 27)


def test_optional_prior_uses_all_four_scores_excludes_first_query_and_accepts_masked_poison():
    ids, raw, legal, offsets = cohort(length=6)
    legal[:, 1:] = False
    prior = np.full_like(raw, np.nan)
    for low in offsets[:-1]:
        prior[low + 4] = raw[low + 4] + np.array([0., 2., 0., -2.], np.float32)
    actual = audit.report_view(ids, raw, legal, raw.copy(), offsets, "joint_aux", None, audit.FIT_SEEDS[1], prequery_forecast=prior)
    assert actual["prequery"]["overall"]["case_weighted_centered_mse"] == 2.
    assert actual == producer_view(ids, raw, legal, raw.copy(), offsets, "joint_aux", None, audit.FIT_SEEDS[1], prior)
    assert audit.report_view(ids, raw, legal, raw.copy(), offsets, "joint_aux", None, audit.FIT_SEEDS[1])["prequery"] is None


def test_reverse_episode_storage_preserves_canonical_report_and_input_bytes():
    ids, raw, legal, offsets = cohort(length=6)
    original = audit.report_view(ids, raw, legal, raw.copy(), offsets, "pretrained", None, audit.FIT_SEEDS[0])
    order = list(reversed(range(len(ids))))
    reverse_raw = np.concatenate([raw[offsets[i]:offsets[i + 1]] for i in order])
    reverse_legal = np.concatenate([legal[offsets[i]:offsets[i + 1]] for i in order])
    actions = reverse_raw.copy()
    values = (reverse_raw, reverse_legal, actions, offsets)
    before = [value.tobytes() for value in values]
    for value in values:
        value.flags.writeable = False
    assert audit.report_view([ids[i] for i in order], reverse_raw, reverse_legal, actions, offsets,
                             "pretrained", None, audit.FIT_SEEDS[0]) == original
    assert [value.tobytes() for value in values] == before


@pytest.mark.parametrize("defect", ["query_signed_zero", "query_change", "nan", "infinity", "dtype", "shape", "offsets",
                                    "no_legal", "duplicate_identity", "bad_prior", "bad_method", "bad_seed", "bad_tau"])
def test_view_malformed_data_fail_closed(defect):
    ids, raw, legal, offsets = cohort(length=6)
    actions, prior = raw.copy(), None
    method, tau, seed = "pretrained", None, audit.FIT_SEEDS[0]
    if defect == "query_signed_zero":
        actions[0, 0] = -0.
    elif defect == "query_change":
        actions[4, 3] += 1
    elif defect in ("nan", "infinity"):
        actions[1, 3] = np.nan if defect == "nan" else np.inf
    elif defect == "dtype":
        raw = raw.astype(np.float64)
    elif defect == "shape":
        legal = legal[:, :3]
    elif defect == "offsets":
        offsets[1] = 0
    elif defect == "no_legal":
        legal[1] = False
    elif defect == "duplicate_identity":
        ids[1] = dict(ids[0])
    elif defect == "bad_prior":
        prior = np.full_like(raw, np.nan)
    elif defect == "bad_method":
        method = "promoted_control"
    elif defect == "bad_seed":
        seed = True
    else:
        tau = .1
    with pytest.raises(ValueError):
        audit.report_view(ids, raw, legal, actions, offsets, method, tau, seed, prequery_forecast=prior)


def test_known_minimax_selection_tie_and_no_control_promotion():
    reports, identities = known_reports()
    values = {0.01: (1., 9.), .1: (4., 4.), 1.: (3., 5.), 10.: (2., 8.)}
    for tau, pair in values.items():
        for regime, value in zip(audit.REGIMES, pair, strict=True):
            set_gap(reports, f"rls_full@tau={tau:g}", value, regime=regime)
    set_gap(reports, "rls_shrink_025@tau=10", 0.)
    result = audit.evaluate_reports(reports, identities, stage="dev", technical_complete=True)
    assert result["selected_tau"] == .1 and result["passed_conditions"] == 13
    assert [row["objective"] for row in result["selection"]["grid"]] == [9., 4., 5., 8.]
    assert result == gate.evaluate(reports, identities, stage="dev", technical_complete=True)
    for tau in audit.TAUS:
        set_gap(reports, f"rls_full@tau={tau:g}", 1.)
    tied = audit.evaluate_reports(list(reversed(reports)), identities, stage="dev", technical_complete=True)
    assert tied["selected_tau"] == .01 and tied["candidate"] == "rls_full"
    assert not any(row.get("confirmed_with_usefulness", False) for row in tied["mechanistic_contrasts"])


@pytest.mark.parametrize("stage,tau,support,required", [("dev", None, 1, 2), ("confirm", .1, 3, 4)])
def test_support_veto_is_case_based_and_never_rescued(stage, tau, support, required):
    reports, ids = known_reports(stage, tau, support)
    result = audit.evaluate_reports(reports, ids, stage=stage, selected_tau=tau, technical_complete=True)
    failed = [row for row in result["conditions"] if not row["passed"]]
    assert len(failed) == 2 and all(row["actual"] == support and row["required"] == required for row in failed)
    assert result == gate.evaluate(reports, ids, stage=stage, selected_tau=tau, technical_complete=True)


def test_zero_support_and_zero_best_control_fail():
    reports, ids = known_reports(support=0)
    result = audit.evaluate_reports(reports, ids, stage="dev", technical_complete=True)
    assert result["passed_conditions"] == 9 and not result["passed"]
    assert all(not row["passed"] for row in result["conditions"] if row["name"].endswith("later_gap_10pct"))


def test_paired_seed_veto_and_full_scope_guard_survive_mean_improvement():
    reports, ids = known_reports()
    for tau in audit.TAUS:
        set_gap(reports, f"rls_full@tau={tau:g}", 11., seed=audit.FIT_SEEDS[-1])
    set_gap(reports, "rls_full@tau=0.01", 12., regime="lambda4", scope="full")
    result = audit.evaluate_reports(reports, ids, stage="dev", technical_complete=True)
    failed = [row["name"] for row in result["conditions"] if not row["passed"]]
    assert failed == [f"lambda3:P4:seed_{audit.FIT_SEEDS[-1]}_nonregression", "lambda4:P4:full_gap_nonregression",
                      f"lambda4:P4:seed_{audit.FIT_SEEDS[-1]}_nonregression"]
    assert result == gate.evaluate(reports, ids, stage="dev", technical_complete=True)


@pytest.mark.parametrize("technical,diagonal_gap,contrast,confirmed", [(True, 10., True, True), (False, 10., True, False),
                                                                     (True, 1., False, False)])
def test_covariance_ten_conditions_require_confirmation_and_usefulness(technical, diagonal_gap, contrast, confirmed):
    reports, ids = known_reports("confirm", 1.)
    set_gap(reports, "rls_diagonal@tau=1", diagonal_gap)
    result = audit.evaluate_reports(reports, ids, stage="confirm", selected_tau=1., technical_complete=technical)
    diagonal = result["mechanistic_contrasts"][0]
    assert len(diagonal["conditions"]) == 10 and diagonal["contrast_passed"] is contrast
    assert diagonal["confirmed_with_usefulness"] is confirmed
    assert result == gate.evaluate(reports, ids, stage="confirm", selected_tau=1., technical_complete=technical)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "ratio", "seed", "stage", "manifest", "support", "nan", "boolean", "cohort"])
def test_bad_saved_rosters_and_unused_view_values_are_rejected(defect):
    reports, ids = known_reports()
    if defect == "missing":
        reports.pop()
    elif defect == "duplicate":
        reports[-1] = deepcopy(reports[-2])
    elif defect == "ratio":
        reports[-1]["family"] = "rls_shrink_075@tau=2"
    elif defect == "seed":
        reports[-1]["seed"] = 1
    elif defect == "stage":
        reports[-1]["stage"] = "test"
    elif defect == "manifest":
        reports[-1]["identity_manifest"][0]["target_sha256"] = "f" * 64
    elif defect == "support":
        reports[-1]["scopes"]["later"]["by_regime"]["lambda3"]["supported_case_count"] = 1
    elif defect in ("nan", "boolean"):
        set_gap(reports, reports[-1]["family"], np.nan if defect == "nan" else True)
    else:
        ids[0]["case"] = 9
    with pytest.raises(ValueError):
        audit.evaluate_reports(reports, ids, stage="dev", technical_complete=True)


def test_deadline_callback_interrupts_rows_and_views_without_modification():
    class Deadline(Exception):
        pass

    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise Deadline

    ids, raw, legal, offsets = cohort()
    original = raw.tobytes(), legal.tobytes(), offsets.tobytes()
    with pytest.raises(Deadline):
        audit.report_view(ids, raw, legal, raw.copy(), offsets, "pretrained", None, audit.FIT_SEEDS[0], check=check)
    assert calls == 3 and original == (raw.tobytes(), legal.tobytes(), offsets.tobytes())
    reports, ids = known_reports()
    snapshot = deepcopy(reports)
    calls = 0
    with pytest.raises(Deadline):
        audit.evaluate_reports(reports, ids, stage="dev", check=check)
    assert calls == 3 and reports == snapshot
