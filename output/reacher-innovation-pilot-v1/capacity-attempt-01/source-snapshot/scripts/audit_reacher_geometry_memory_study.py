"""Independent saved-output audit for the geometry-scored memory comparison.

No runner, learned inference, trainer or optimizer is invoked. NumPy rederives
score arithmetic; CEM replays saved scores. MuJoCo replays executed controls,
public observers and every paid nominal physics candidate/selected advance.
These native scopes are separate in all counts and cost reports.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from contextvars import ContextVar
from itertools import pairwise
from pathlib import Path

import audit_reacher_cache_study as inherited
import audit_reacher_geometry_study as previous
import numpy as np
import torch

from openjev.research import reacher_geometry_memory_protocol as protocol

base, search_audit = inherited.base, inherited.search_audit
require, read, write, sha = base.require, base.read, base.write, base.sha
array = search_audit.array
ROOT = Path(__file__).resolve().parents[1]
VERSION = "reacher-geometry-memory-v1"
FAMILIES = ("residual_gru", "encoded_current_gru", "cached_gru", "cached_mlp")
PANELS = ("full", "ordinary", "shift")
PAIRS = ("pair0", "pair1", "pair2")
PHYSICS_REFERENCES = ("known_state", "particle", "public_kinematic")
REFERENCES = (*PHYSICS_REFERENCES, "zero", "uniform")
GEOMETRY_FIELDS = ("joint_angles", "pair_norms", "fingertip", "distance", "action_cost")
finite, integer = inherited.finite, inherited.exact_integer
tensor_hash, state_hash = inherited.canonical_tensor_hash, inherited.canonical_state_hash
geometry_components, close_geometry = previous.geometry_components, previous.close_geometry
audit_bank_arrays, audit_bank_metadata = previous.audit_bank_arrays, previous.audit_bank_metadata
audit_selected, np_state_hash = previous.audit_selected, previous.np_state_hash
_DEADLINE = ContextVar("geometry_memory_audit_deadline", default=float("inf"))


def check_budget():
    if time.monotonic() > _DEADLINE.get():
        raise TimeoutError("Frozen geometry-memory audit cap exceeded")


def checked(path, digest):
    check_budget()
    return search_audit.checked(Path(path), digest)


def scoring_configuration(plan, mode="geometry"):
    require(mode == "geometry", "Geometry-only comparison")
    return {**previous.scoring_configuration(plan, mode),
            "version": "reacher-geometry-memory-control-v1",
            "scoring_kernel_version": "reacher-geometry-control-v1",
            "permitted_model_kinds": list(FAMILIES)}


def audit_scoring_trace(plan, stem, mode, root, fit, trace, step, *, carried=None, executed_angles=None, executed_rewards=None):
    values, meta = base.load_npz(stem.with_suffix(".npz")), read(stem.with_suffix(".json"))
    result, raw = trace["result"], trace["raw_rewards"]
    commands = result.sequences
    # SearchResult carries compressed chunks in older versions; expanded
    # sequences are an explicit property in the frozen planner.
    n, k, h, _ = commands.shape
    require(k == 256, "All paid CEM candidates retained")
    common = {"commands", "root_target", "predicted_angles", "learned_rewards", "selected_rewards"}
    bank_keys = common | ({"geometry_" + key for key in GEOMETRY_FIELDS} if mode == "geometry" else set())
    selected_keys = {"selected_angles", "selected_learned_reward", "selected_reward"}
    if mode == "geometry":
        selected_keys |= {"selected_geometry_" + key for key in GEOMETRY_FIELDS}
    wanted = bank_keys | {"selected_ids", "selected_actions"} | (selected_keys if carried is not None else set())
    require(set(values) == wanted, "Exact search scoring arrays")
    counts = audit_bank_arrays(plan, mode, {key: values[key] for key in bank_keys}, commands, root["packet"][:, 4:6])
    require(np.array_equal(values["selected_rewards"], raw)
            and np.array_equal(result.scores, search_audit.clipped_returns(raw)), "Exact clipped sequential score-to-CEM binding")
    require(np.array_equal(array(values["selected_ids"], (n,), np.int64, "Saved global-best IDs"), result.selected_ids)
            and np.array_equal(array(values["selected_actions"], (n, 2), np.float32, "Saved first actions"), result.selected_actions),
            "Selected candidate identity")
    expected_keys = {"version", "model", "step", "score_mode", "configuration", "root_sha256", "callbacks", "work", "search_seconds"}
    if carried is not None:
        expected_keys.add("selected_advance")
    require(set(meta) == expected_keys and meta["version"] == "reacher-geometry-memory-control-v1"
            and meta["model"] == inherited.model_identity(plan, fit["arm"], fit)
            and meta["step"] == step and meta["score_mode"] == mode
            and meta["configuration"] == scoring_configuration(plan, mode)
            and meta["root_sha256"] == np_state_hash(root) and len(meta["callbacks"]) == 4
            and meta["search_seconds"] == trace["search_seconds"], "Search scoring metadata identity")
    callback_costs = []
    for index, callback in enumerate(meta["callbacks"]):
        require(callback["candidate_start"] == index * 64 and callback["candidate_stop"] == (index + 1) * 64,
                "Complete paid callback intervals")
        part = {key: value if key == "root_target" else value[:, index * 64:(index + 1) * 64] for key, value in values.items() if key in bank_keys}
        callback_costs.append(audit_bank_metadata(plan, mode,
            {key: value for key, value in callback.items() if key not in ("candidate_start", "candidate_stop")}, part, root))
    work = meta["work"]
    require(set(work) == {"candidate_evaluations", "imagined_transitions", "root_tensor_bytes", "max_single_candidate_state_tensor_bytes",
        "tensor_bytes_are_not_peak_process_memory", "model_work", "geometry_samples", "geometry_seconds", "model_advance_seconds", "diagnostic_array_bytes"},
        "Complete search work accounting schema")
    integer(work["candidate_evaluations"], n * 256, "Paid candidate count")
    integer(work["imagined_transitions"], n * 256 * h, "Paid imagined transition count")
    integer(work["root_tensor_bytes"], sum(value.nbytes for value in root.values()), "Root state payload")
    integer(work["max_single_candidate_state_tensor_bytes"], work["root_tensor_bytes"] * 64, "Maximum paid branch payload")
    integer(work["geometry_samples"], counts["geometry_samples"], "Total search geometry work")
    require(work["tensor_bytes_are_not_peak_process_memory"] is True, "Payload versus resident memory distinction")
    inherited.audit_work(plan, fit["arm"], work["model_work"], 0, n * 256 * h)
    for key in ("geometry_seconds", "model_advance_seconds"):
        require(math.isclose(work[key], sum(row[key] for row in callback_costs), rel_tol=1e-12, abs_tol=1e-12), "Summed search time " + key)
    integer(work["diagnostic_array_bytes"], sum(value.nbytes for key, value in values.items() if key not in selected_keys), "Search array payload")
    require(sum(callback["bank_wall_seconds"] for callback in meta["callbacks"]) <= meta["search_seconds"] + 1e-6, "All callbacks charged")
    if carried is not None:
        audit_selected(plan, mode, {key: values[key] for key in selected_keys}, meta["selected_advance"], root, carried,
                       trace, executed_angles, executed_rewards)
        chosen = (np.arange(n), result.selected_ids, np.zeros(n, dtype=np.int64))
        for selected_key, bank_key in (("selected_angles", "predicted_angles"),
                                       ("selected_learned_reward", "learned_rewards"),
                                       ("selected_reward", "selected_rewards")):
            # Re-advance uses N roots rather than the N*64 search batch. CPU
            # affine reductions can differ slightly while sharing the action.
            require(np.allclose(values[selected_key], values[bank_key][chosen], rtol=2e-5, atol=2e-6),
                    "Selected re-advance agrees with chosen candidate first transition")
    return values, meta, counts


def audit_cohort(plan, records, panel):
    require(len(records) == plan["control_episodes"], "All fresh control cases")
    total, maximum = 0, 0.
    for index, record in enumerate(records):
        check_budget()
        meta = record["metadata"]
        require(meta["seed"] == protocol.seed(plan, f"control/reset/{index}")
                and meta["noise_seed"] == protocol.seed(plan, f"control/actuator_noise/{index}")
                and meta["noise_std"] == plan["noise_std"] and meta["policy_keys"] == ["packets", "commands"]
                and meta["sensor_schedule"] == protocol.schedule(plan, index, panel).tolist(), "Fresh control reset/noise/public sensor binding")
        replay = search_audit.native.native_replay(record)
        require(replay["transitions"] == plan["steps"] and replay["new_policy_calls"] == 0 and replay["saved_output_only"] is True,
                "Native replay scope and terminal boundary")
        total += replay["transitions"]
        maximum = max(maximum, replay["max_abs_error"])
    return {"episodes": len(records), "transitions": total, "max_abs_error": maximum}


def audit_learned_control(plan, folder, row, records, inputs, fit):
    n, steps, mode = plan["control_episodes"], plan["steps"], row["score_mode"]
    times = search_audit.timing_row(plan, folder / "timings.json", True)
    state_work = inherited.audit_states(plan, folder, records, fit["arm"], fit)
    states = base.load_npz(folder / "states.npz")
    predictions = base.load_npz(folder / "executed_predictions.npz", {"angles", "rewards"})
    angle = array(predictions["angles"], (n, steps, 4), np.float32, "Executed native-head angles")
    reward = array(predictions["rewards"], (n, steps), np.float32, "Executed original learned reward")
    commands, native_rewards = base.stack(records, "policy", "commands"), base.stack(records, "audit", "rewards")
    costs = -native_rewards.sum(1)
    work = {"candidate_evaluations": 0, "imagined_transitions": 0, "geometry_samples": 0,
            "geometry_seconds": 0., "model_advance_seconds": 0., "selected_geometry_samples": 0,
            "selected_geometry_seconds": 0., "selected_model_advance_seconds": 0.}
    searches, clipped, count = [], 0, 0
    selected_rewards = np.zeros((n, steps), np.float32)
    schema = inherited.state_shapes(plan, fit["arm"])
    for step in range(steps):
        check_budget()
        trace = search_audit.audit_trace({**plan, "planners": ["cem256"]}, inputs[step], folder / "decisions" / f"{step:03d}", step=step)
        require(np.array_equal(commands[:, step], trace["result"].selected_actions), "Executed action is recorded CEM global best")
        root = {name: states["root__" + name][:, step] for name in schema}
        carried = {name: states["carried__" + name][:, step] for name in schema}
        values, meta, counts = audit_scoring_trace(plan, folder / "scoring" / f"{step:03d}", mode, root, fit, trace, step,
            carried=carried, executed_angles=angle[:, step], executed_rewards=reward[:, step])
        require(trace["search_seconds"] + meta["selected_advance"]["model_advance_seconds"]
                + meta["selected_advance"]["geometry_seconds"] <= times["decision_seconds"][step] + 1e-6,
                "Search plus selected action computation charged to decision")
        searches.append(trace["search_seconds"])
        for key in ("candidate_evaluations", "imagined_transitions", "geometry_samples"):
            work[key] += counts[key]
        for key in ("geometry_seconds", "model_advance_seconds"):
            work[key] += meta["work"][key]
        for key in ("geometry_samples", "geometry_seconds", "model_advance_seconds"):
            work["selected_" + key] += meta["selected_advance"][key]
        selected_rewards[:, step] = values["selected_reward"]
        clipped += trace["clipped_predictions"]
        count += trace["reward_predictions"]
    decisions = np.asarray(times["decision_seconds"], dtype=float)
    valid = base.stack(records, "policy", "packets")[:, 1:, 6].astype(bool)
    truth = base.stack(records, "audit", "raw_obs")[:, 1:, :4]
    return {"episode_costs": costs.tolist(), "mean_cost": float(costs.mean()),
        "setup_seconds": times["setup_seconds"], "decision_wall_seconds": float(decisions.sum()),
        "native_step_seconds": float(sum(times["native_step_seconds"])), "row_wall_seconds": times["row_wall_seconds"],
        "decision_seconds": times["decision_seconds"], "search_seconds": searches,
        "batch_latency_seconds": {"mean": float(decisions.mean()), "p50": float(np.quantile(decisions, .5)),
            "p95": float(np.quantile(decisions, .95)), "max": float(decisions.max())},
        "per_case_amortized_seconds": float(decisions.mean() / n), "candidate_evaluations": work["candidate_evaluations"],
        "imagined_transitions": work["imagined_transitions"], "clipping_fraction": clipped / count, "planner_used": True,
        "on_policy_angle_mse": base.masked_mse(angle, truth, np.ones_like(valid)),
        "on_policy_valid_angle_mse": base.masked_mse(angle, truth, valid),
        "on_policy_blackout_angle_mse": base.masked_mse(angle, truth, ~valid) if (~valid).any() else None,
        "on_policy_original_learned_reward_mse": float(np.mean((reward.astype(float) - native_rewards) ** 2)),
        "on_policy_selected_reward_mse": float(np.mean((selected_rewards.astype(float) - native_rewards) ** 2)),
        "state_and_work": state_work, "scoring_work": work,
        "reward_error_limit": "Different closed-loop states/actions across scorers; not a common-input causal reward-quality comparison."}


def validate_runtime_sources(plan):
    check_budget()
    inherited.validate_runtime_sources(plan)
    check_budget()


def prior_streams():
    # Independent frozen auditor reconstruction, never a runner import.
    return previous.prior_streams()


def validate_streams(plan, execution):
    expected = protocol.stream_contract(plan, *prior_streams())
    require(plan['random_stream_contract'] == expected and read(execution / 'random-streams.json') == expected,
            'Complete pinned historical/new/literal random-stream contract')
    require(expected['torch_registry'] == expected['torch_generators'] == {}
            and expected['new_torch_scored_streams'] == 0
            and expected['training_and_stochastic_inference_rng_draws'] == 0,
            'No training or stochastic-inference streams')
    return {'numpy_generators': len(expected['generators']), 'torch_generators': 0,
            'draws_for_manifest': 0, 'full_prior_namespaces_excluded': True,
            'literal_engineering_exclusions': len(expected['engineering_literal_exclusions'])}


def inherited_members():
    names = {'train.npz', 'train.json'}
    for pair in PAIRS:
        names |= {f'{folder}/{pair}.{suffix}' for folder in ('initializations', 'orders') for suffix in ('pt', 'json')}
        for family in FAMILIES:
            names |= {f'fits/{family}-{pair}/{member}' for member in
                      ('initial-weights.pt', 'training.jsonl', 'weights.pt', 'checkpoint.pt', 'completed.json')}
    return names


def validate_lineage(plan, execution):
    parents = {}
    fieldset = {'plan_path', 'plan_sha256', 'audit_path', 'audit_receipt_sha256', 'execution_path',
        'completed_sha256', 'summary_sha256', 'members', 'prior_costs', 'engineering',
        'previous_scientific_gate_passed', 'training_weights_reused', 'new_fits'}
    for kind in ('cache', 'geometry'):
        source = plan[kind + '_source']
        require(set(source) == fieldset, 'Exact completed lineage descriptor')
        engineering = plan['engineering']
        require(source['engineering'] is engineering and source['new_fits'] == 0
                and source['training_weights_reused'] is (kind == 'cache'), 'Explicit lineage scope')
        binding = protocol.lineage_contract()[kind]
        if not engineering:
            require(source['plan_sha256'] == binding['plan_sha256']
                    and source['audit_receipt_sha256'] == binding['audit_receipt_sha256']
                    and source['previous_scientific_gate_passed'] is (kind == 'geometry'), 'Pinned completed study lineage')
        else:
            require(source['previous_scientific_gate_passed'] is None, 'Engineering has no scientific gate')
        parent = read(checked(ROOT / source['plan_path'], source['plan_sha256']))
        audit = read(checked(ROOT / source['audit_path'], source['audit_receipt_sha256']))
        completed = read(checked(ROOT / source['execution_path'] / 'completed.json', source['completed_sha256']))
        summary = read(checked((ROOT / source['audit_path']).parent / 'summary.json', source['summary_sha256']))
        require(audit['status'] == completed['status'] == summary['status'] == 'completed'
                and audit['saved_output_only'] is True
                and audit['engineering'] is parent['engineering'] is summary['engineering'] is engineering
                and audit['plan_sha256'] == completed['plan_sha256'] == summary['plan_sha256'] == source['plan_sha256']
                and audit['execution_completed_sha256'] == source['completed_sha256']
                and audit['execution_members'] == completed['files']
                and audit['source_sha256'] == parent['sources']
                and audit['runtime'] == parent['runtime'] == plan['runtime']
                and audit['files']['summary.json'] == source['summary_sha256']
                and source['prior_costs'] == audit['costs'], 'Authenticated completed parent evidence')
        require(finite(completed['wall_seconds'], 'Parent execution seconds', positive=True) <= parent['cap_seconds'],
                'Completed parent within original cap')
        for name, digest in parent['sources'].items():
            require(plan['sources'].get(name) == digest, 'Inherited source unchanged in new scope')
            checked(ROOT / name, digest)
        if kind == 'cache':
            require(len(parent['sources']) == 70, 'Original cache seventy-source cohort')
            require(all(plan[key] == parent[key] for key in ('hidden_size', 'mlp_width', 'dt', 'noise_std', 'residual_reward')),
                    'Unchanged inherited model dimensions and assumptions')
        else:
            protocol.validate_source_hashes(plan['sources'], parent['sources'])
        if not engineering:
            gate = summary['continuation_gate']
            require(gate['passed'] is (kind == 'geometry') and len(gate['checks']) == (28 if kind == 'cache' else 25)
                    and sum(row['passed'] for row in gate['checks']) == (15 if kind == 'cache' else 25),
                    'Historical failed cache/successful geometry gates preserved')
        for name, digest in audit['files'].items():
            checked((ROOT / source['audit_path']).parent / name, digest)
        wanted = inherited_members() if kind == 'cache' else set()
        require(set(source['members']) == wanted, 'All twelve fits; no reused old control histories')
        for name, digest in source['members'].items():
            require(completed['files'].get(name) == digest, 'Inherited member was audited')
            checked(ROOT / source['execution_path'] / name, digest)
            checked(execution / 'inherited' / name, digest)
        for suffix, digest in (('plan', source['plan_sha256']), ('audit', source['audit_receipt_sha256']),
                               ('completed', source['completed_sha256']), ('summary', source['summary_sha256'])):
            checked(execution / 'inherited' / f'{kind}-{suffix}.json', digest)
        parents[kind] = parent
    require(read(execution / 'inheritance.json') == {kind: plan[kind + '_source'] for kind in ('cache', 'geometry')},
            'Exact copied dual lineage receipt')
    geometry_parent = parents['geometry']['parent_source']
    require(geometry_parent['plan_sha256'] == plan['cache_source']['plan_sha256']
            and geometry_parent['audit_receipt_sha256'] == plan['cache_source']['audit_receipt_sha256'],
            'Geometry context and inherited checkpoints share the same cache parent')
    return parents['cache'], parents['geometry']


def expected_members(plan):
    names = {'started.json', 'random-streams.json', 'inheritance.json', 'all-models-restored.json',
             'evaluation-started.json', 'control-completed.json', 'final-models.json', 'costs.json'}
    names |= {'inherited/' + name for name in inherited_members()}
    names |= {f'inherited/{kind}-{suffix}.json' for kind in ('cache', 'geometry')
              for suffix in ('plan', 'audit', 'completed', 'summary')}
    for fit in protocol.fit_manifest(plan):
        names |= {f"model-states/{fit['name']}-{phase}.pt" for phase in ('before', 'after')}
    names |= {f'innovations/control/{step:03d}.{suffix}' for step in range(plan['steps']) for suffix in ('npz', 'json')}
    for row in protocol.execution_order(plan):
        path = row['path']
        names |= {f'{path}/{name}' for name in ('episodes.npz', 'episodes.json', 'timings.json')}
        if 'fit' in row:
            names |= {f'{path}/{name}' for name in ('executed_predictions.npz', 'states.npz', 'state-work.json')}
            folders = ('decisions', 'scoring')
        else:
            names.add(f'{path}/planning.npz')
            folders = ()
            if row['reference'] in PHYSICS_REFERENCES:
                names.add(f'{path}/physics-final.json')
                folders = ('decisions', 'physics')
            if row['reference'] in ('particle', 'public_kinematic'):
                names.add(f'{path}/observer-final.json')
        names |= {f'{path}/{folder}/{step:03d}.{suffix}' for folder in folders
                  for step in range(plan['steps']) for suffix in ('npz', 'json')}
    return names


def validate_members(plan, expected, execution):
    require(previous.valid_digest(expected), 'External exact plan SHA256')
    done = read(execution / 'completed.json')
    fields = {'status', 'study', 'plan_sha256', 'new_fits', 'restored_models', 'control_rows', 'diagnostic_roots',
              'astra_calls', 'wall_seconds', 'evaluation_started_elapsed_seconds', 'cumulative_attempt_wall_seconds', 'files'}
    require(set(done) == fields and done['status'] == 'completed' and done['study'] == VERSION
            and done['plan_sha256'] == expected, 'Complete memory execution identity')
    for name, count in (('new_fits', 0), ('restored_models', 12), ('control_rows', 51), ('diagnostic_roots', 0), ('astra_calls', 0)):
        integer(done[name], count, 'Completed ' + name)
    require(finite(done['wall_seconds'], 'Whole execution seconds', positive=True) <= plan['cap_seconds'], 'Original execution cap')
    paths = list(execution.rglob('*'))
    require(not execution.is_symlink() and not any(path.is_symlink() for path in paths), 'No artifact symlinks')
    members = expected_members(plan)
    require(set(done['files']) == members and {path.relative_to(execution).as_posix() for path in paths if path.is_file()}
            == members | {'completed.json'}, 'Exact completed artifact membership; no failed or partial files')
    for name, digest in done['files'].items():
        checked(execution / name, digest)
    return done


def audit_restoration(plan, parent, execution):
    restored, final = read(execution / 'all-models-restored.json'), read(execution / 'final-models.json')
    require(set(restored) == {'models', 'optimizer_constructed', 'new_updates', 'constructor_seed',
        'constructor_rng_isolated_and_weights_overwritten', 'wall_seconds', 'unix_time'}
        and restored['optimizer_constructed'] is False and restored['constructor_rng_isolated_and_weights_overwritten'] is True,
        'Deployment-only restoration boundary')
    integer(restored['new_updates'], 0, 'No restoration updates')
    integer(restored['constructor_seed'], 0, 'Isolated overwritten constructor seed')
    finite(restored['wall_seconds'], 'Restoration wall seconds', positive=True)
    require(set(final) == {'student_tensor_sha256', 'files', 'new_updates'}, 'Final tensor receipt')
    integer(final['new_updates'], 0, 'No new updates')
    names = {row['name'] for row in protocol.fit_manifest(plan)}
    require(set(restored['models']) == set(final['student_tensor_sha256']) == names
            and set(final['files']) == {f'model-states/{name}-after.pt' for name in names}, 'All twelve actual-class snapshots')
    training = base.load_records(execution / 'inherited' / 'train', parent['train_episodes'])
    data = inherited.public_learning_tensors(training)
    data_hash = tensor_hash(data)
    initializations, orders = {}, {}
    for pair in PAIRS:
        for folder, target in (('initializations', initializations), ('orders', orders)):
            path = execution / 'inherited' / folder / f'{pair}.pt'
            value = inherited.load_tensors(path)
            inherited.unseal(value)
            require(read(path.with_suffix('.json')) == {'pair': pair, 'file_sha256': sha(path),
                'integrity_sha256': value['integrity_sha256']}, 'Original initialization/order sidecar')
            target[pair] = value
        initial, order = initializations[pair], orders[pair]
        require(initial['version'] == inherited.TRAINING_VERSION and initial['settings'] == inherited.training_configuration(parent)
                and initial['roles'] == {family: f'fit/{family}/{pair}' for family in ('gru', 'mlp')}
                and initial['seeds'] == {family: inherited.protocol.seed(parent, f'fit/{family}/{pair}') for family in ('gru', 'mlp')},
                'Paired initialization original names/settings/seeds')
        for family, arm in (('gru', 'residual_gru'), ('mlp', 'cached_mlp')):
            inherited.tensors(initial['states'][family], inherited.shapes(parent, arm), 'Original family tensor schema')
            require(initial['tensor_hashes'][family] == tensor_hash(initial['states'][family]), 'Paired initial tensor identities')
        require(order['version'] == inherited.TRAINING_VERSION and order['role'] == f'fit/minibatch/{pair}'
                and order['seed'] == inherited.protocol.seed(parent, order['role']), 'Original order role and seed')
        value = order['orders']
        require(value.dtype == torch.int64 and tuple(value.shape) == (parent['epochs'], parent['train_episodes'])
                and torch.equal(value.sort(1).values, torch.arange(parent['train_episodes']).repeat(parent['epochs'], 1)),
                'Every inherited epoch is a complete paired permutation')
    fits = {}
    for row in protocol.fit_manifest(plan):
        check_budget()
        name, arm, pair = row['name'], row['arm'], row['pair']
        folder = execution / 'inherited' / 'fits' / name
        done = read(folder / 'completed.json')
        checkpoint = inherited.load_tensors(folder / 'checkpoint.pt')
        # This validates saved Adam tensors/counters numerically, without
        # constructing or restoring an optimizer or replaying optimization.
        inherited.audit_checkpoint(parent, checkpoint, initializations[pair], orders[pair], arm, data_hash)
        weights, initial = inherited.load_tensors(folder / 'weights.pt'), inherited.load_tensors(folder / 'initial-weights.pt')
        schema, config = inherited.shapes(parent, arm), inherited.model_configuration(parent, arm)
        inherited.tensors(weights, schema, 'Inherited deployment tensor schema')
        inherited.tensors(initial, schema, 'Inherited initial tensor schema')
        digest, family = tensor_hash(weights), 'mlp' if arm == 'cached_mlp' else 'gru'
        require(tensor_hash(initial) == initializations[pair]['tensor_hashes'][family], 'Paired initial weights')
        updates = parent['epochs'] * (parent['train_episodes'] // parent['batch_size'])
        require(digest == done['student_tensor_sha256'] == tensor_hash(checkpoint['student_state'])
                and done['name'] == name and done['arm'] == arm and done['pair'] == pair
                and done['model_configuration'] == config and done['settings'] == inherited.training_configuration(parent)
                and done['updates'] == done['optimizer_steps'] == updates
                and done['data_sha256'] == data_hash
                and done['initialization_sha256'] == initializations[pair]['integrity_sha256']
                and done['orders_sha256'] == orders[pair]['integrity_sha256']
                and done['checkpoint_integrity_sha256'] == checkpoint['integrity_sha256'], 'Completed inherited fit identities')
        require(set(done['files']) == {'initial-weights.pt', 'weights.pt', 'checkpoint.pt', 'training.jsonl'}, 'All original fit members')
        for member, digest_value in done['files'].items():
            checked(folder / member, digest_value)
        for phase in ('before', 'after'):
            path = execution / 'model-states' / f'{name}-{phase}.pt'
            actual = inherited.load_tensors(path)
            inherited.tensors(actual, schema, 'Actual-class snapshot tensors')
            require(tensor_hash(actual) == digest and all(torch.equal(actual[key], weights[key]) for key in weights),
                    'Observed before/after deployment tensors unchanged')
            if phase == 'after':
                checked(path, final['files'][path.relative_to(execution).as_posix()])
        require(final['student_tensor_sha256'][name] == digest, 'Final student identity')
        expected = {'kind': arm, 'model_class': protocol.MODEL_CLASSES[arm], 'checkpoint_sha256': sha(folder / 'checkpoint.pt'),
            'weights_sha256': sha(folder / 'weights.pt'), 'student_tensor_sha256': digest, 'successful_updates': updates,
            'configuration': config, 'snapshot_path': f'model-states/{name}-before.pt',
            'snapshot_sha256': sha(execution / 'model-states' / f'{name}-before.pt')}
        require(restored['models'][name] == expected, 'Exact restored class receipt')
        fits[name] = {'arm': arm, 'pair': pair, 'parameters': sum(value.numel() for value in weights.values()),
            'student_tensor_sha256': digest, 'inherited_optimizer_updates': updates, 'new_optimizer_updates': 0,
            'observed_before_after_equal': True, 'model_configuration': config}
    return fits, restored
PHYSICS_COUNTS = (
    'candidate_sequences_requested', 'candidate_sequences_initialized',
    'candidate_sequences_native_completed', 'candidate_sequences_scored',
    'native_transitions_attempted', 'native_transitions_completed',
    'native_substeps_attempted', 'native_substeps_completed',
    'reset_calls_attempted', 'reset_calls_completed', 'forward_calls_attempted', 'forward_calls_completed',
    'postconstraint_calls_attempted', 'postconstraint_calls_completed',
    'geometry_calls_attempted', 'geometry_calls_completed', 'geometry_samples_attempted', 'geometry_samples_completed',
)


def physics_array_hash(value):
    header = json.dumps({'dtype': value.dtype.str, 'shape': list(value.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + b'\n' + value.tobytes(order='C')).hexdigest()


def model_binary_hash(model):
    native = search_audit.native.mujoco
    value = np.empty(native.mj_sizeModel(model), np.uint8)
    native.mj_saveModel(model, buffer=value)
    return hashlib.sha256(value.tobytes()).hexdigest()


def physics_configuration(plan, model):
    return {'version': 'reacher-nominal-geometry-cem-v1', 'classification': 'supplied-physics nominal competence reference',
        'model_binary_sha256': model_binary_hash(model), 'model_copied': True,
        'native_timestep': float(model.opt.timestep), 'frame_skip': 2, 'decision_dt': plan['dt'],
        'steps': plan['steps'], 'planning_horizon': plan['planning_horizon'], 'action_block': plan['action_block'],
        'planner': 'cem256', 'callback_sizes': [64, 64, 64, 64], 'warm_start': False,
        'reward_clip': [-2.5, 0.], 'score_sum': 'sequential float32, per-step clipping',
        'geometry': plan['score_contract']['geometry'], 'rng_draws': 0,
        'root_semantics': 'Copied qpos/qvel; static explicit public target; mj_resetData then mj_forward; no full integration-state continuation.',
        'transition_semantics': 'Nominal issued command held across frame_skip substeps, zero realized noise, followed by mj_rnePostConstraint.',
        'angle_precision': 'cos/sin of nominal float64 qpos, quantized to float32 before the shared geometry scorer',
        'information_boundary': 'Caller declares known-state, filtered or public-kinematic root provenance; this adapter cannot infer it.',
        'selected_advance': 'Separate one-action prediction from the original root; not permission to replace an observer with privileged state.',
        'limits': 'Equal CEM proposal budgets do not imply matched wall time or stochastic optimal MPC; FK scoring is not exact native reward.',
        'run_status_authority': 'enclosing protocol and execution receipts'}


def physics_counts(n, k, h):
    sequence, transition = n * k, n * k * h
    values = {}
    for key in PHYSICS_COUNTS:
        if key.startswith(('candidate_sequences_', 'reset_calls_', 'forward_calls_')):
            values[key] = sequence
        elif key.startswith('native_substeps_'):
            values[key] = 2 * transition
        elif key.startswith('geometry_calls_'):
            values[key] = 1
        else:
            values[key] = transition
    return values


def replay_nominal_bank(plan, model, roots, values, *, step):
    """Independently reset/advance each supplied sequence, including selection.

    This does not call a physics controller, geometry scorer or learned model.
    Full native integration state is intentionally reset as in the declared
    nominal reference, not continued from the disturbed environment trajectory.
    """
    mj = search_audit.native.mujoco
    commands = values['commands']
    n, k, h, _ = commands.shape
    data, maximum = mj.MjData(model), 0.
    require(float(model.opt.timestep) * 2 == plan['dt'], 'Pinned nominal frame skip and dt')
    for case in range(n):
        for candidate in range(k):
            check_budget()
            mj.mj_resetData(model, data)
            data.qpos[:], data.qvel[:], data.time = roots['qpos'][case], roots['qvel'][case], step * plan['dt']
            mj.mj_forward(model, data)
            for offset in range(h + 1):
                if offset:
                    check_budget()
                    data.ctrl[:] = commands[case, candidate, offset - 1]
                    mj.mj_step(model, data)
                    mj.mj_step(model, data)
                    mj.mj_rnePostConstraint(model, data)
                for key, actual in (('qpos', data.qpos), ('qvel', data.qvel)):
                    difference = float(np.max(np.abs(actual - values[key][case, candidate, offset])))
                    require(math.isfinite(difference) and difference <= 1e-10, 'Nominal native ' + key + ' replay')
                    maximum = max(maximum, difference)
                if offset:
                    angles = np.concatenate((np.cos(data.qpos[:2]), np.sin(data.qpos[:2]))).astype(np.float32)
                    require(np.array_equal(angles, values['predicted_angles'][case, candidate, offset - 1]),
                            'Nominal angle quantization from independently replayed qpos')
    return {'transitions': n * k * h, 'substeps': 2 * n * k * h, 'max_abs_error': maximum,
            'new_model_calls': 0, 'new_planner_calls': 0}


def audit_physics_bank(plan, model, roots, values, metadata, commands, *, step, candidate_start=None):
    n, k, h, _ = commands.shape
    require(set(roots) == {'qpos', 'qvel', 'public_target'}, 'Exact nominal root schema')
    array(roots['qpos'], (n, 4), np.float64, 'Nominal qpos roots')
    array(roots['qvel'], (n, 4), np.float64, 'Nominal qvel roots')
    array(roots['public_target'], (n, 2), np.float32, 'Actual public target')
    require(np.array_equal(roots['qpos'][:, 2:].astype(np.float32), roots['public_target'])
            and np.all(roots['qvel'][:, 2:] == 0), 'Static public target agrees with nominal roots')
    require(type(step) is int and 0 <= step < plan['steps'] and h <= min(plan['planning_horizon'], plan['steps'] - step)
            and h > 0 and 0 < k <= 64, 'Bounded nonterminal paid nominal bank')
    names = {'commands', 'predicted_angles', 'qpos', 'qvel', 'initialized', 'native_completed',
             'scored', 'scores', 'native_substeps_completed'} | {'geometry_' + key for key in (*GEOMETRY_FIELDS, 'reward')}
    require(set(values) == names, 'Exact native bank arrays')
    array(values['commands'], (n, k, h, 2), np.float32, 'Nominal command bank')
    require(np.array_equal(values['commands'], commands) and np.isfinite(commands).all()
            and np.all(np.abs(commands) <= 1), 'Exact already-bounded nominal commands')
    angle = array(values['predicted_angles'], (n, k, h, 4), np.float32, 'Nominal predicted angle features')
    for key in ('qpos', 'qvel'):
        actual = array(values[key], (n, k, h + 1, 4), np.float64, 'Nominal trajectory ' + key)
        require(np.array_equal(actual[:, :, 0], np.broadcast_to(roots[key][:, None], (n, k, 4))), 'Every candidate restored from root')
    for key, shape in (('initialized', (n, k)), ('native_completed', (n, k, h)), ('scored', (n, k, h))):
        require(array(values[key], shape, np.bool_, 'Complete ' + key).all(), 'No partial nominal bank accepted')
    require(np.all(array(values['native_substeps_completed'], (n, k, h), np.int64, 'Native substep masks') == 2),
            'Exactly two returned substeps per completed nominal transition')
    expanded = np.broadcast_to(roots['public_target'][:, None, None], (n, k, h, 2))
    derived = geometry_components(angle, expanded, commands, plan['noise_std'])
    for key in (*GEOMETRY_FIELDS, 'reward'):
        close_geometry(values['geometry_' + key], derived[key], 'Nominal geometry ' + key)
    require(np.array_equal(values['geometry_reward'], -values['geometry_distance'] - values['geometry_action_cost']),
            'Nominal expected actuator penalty charged exactly once')
    array(values['scores'], (n, k), np.float32, 'Nominal sequential float32 scores')
    require(np.array_equal(values['scores'], search_audit.clipped_returns(values['geometry_reward'])),
            'Nominal clipping before sequential sum')
    expected_counts = physics_counts(n, k, h)
    fields = set(PHYSICS_COUNTS) | {'bank_sha256', 'root_sha256', 'purpose', 'candidate_start', 'bank_shape',
        'cursor', 'native_seconds', 'geometry_seconds', 'wall_seconds', 'sum_offsets_completed', 'status'}
    require(set(metadata) == fields and metadata['status'] == 'completed'
            and metadata['bank_sha256'] == physics_array_hash(commands)
            and metadata['root_sha256'] == {key: physics_array_hash(value) for key, value in roots.items()}
            and metadata['bank_shape'] == [n, k, h, 2]
            and metadata['candidate_start'] == candidate_start
            and metadata['purpose'] == ('selected_root_advance' if candidate_start is None else 'candidate_scoring')
            and metadata['cursor'] == {'case': n - 1, 'candidate': k - 1, 'offset': h - 1}, 'Native bank provenance/cursor')
    if candidate_start is None:
        require(k == h == 1, 'Selected native advance is one step per original root')
    for key, wanted in expected_counts.items():
        integer(metadata[key], wanted, 'Nominal ' + key)
    integer(metadata['sum_offsets_completed'], h, 'Every score offset counted')
    for key in ('native_seconds', 'geometry_seconds', 'wall_seconds'):
        finite(metadata[key], 'Nominal ' + key, positive=True)
    require(metadata['native_seconds'] + metadata['geometry_seconds'] <= metadata['wall_seconds'] + 1e-6,
            'Nominal bank wall includes native and geometry work')
    return {**replay_nominal_bank(plan, model, roots, values, step=step), 'counts': expected_counts,
            'native_seconds': metadata['native_seconds'], 'geometry_seconds': metadata['geometry_seconds'],
            'wall_seconds': metadata['wall_seconds']}


def audit_physics_trace(plan, model, stem, trace, root_estimates, public_target, *, step):
    values, meta = base.load_npz(stem.with_suffix('.npz')), read(stem.with_suffix('.json'))
    result, n = trace['result'], plan['control_episodes']
    root_names = ('qpos', 'qvel', 'public_target')
    roots = {name: values['root__' + name] for name in root_names}
    require(np.array_equal(roots['qpos'], root_estimates[:, :4])
            and np.array_equal(roots['qvel'], root_estimates[:, 4:])
            and np.array_equal(roots['public_target'], public_target), 'Physics root information boundary')
    fields = {'operation', 'status', 'step', 'configuration', 'root_sha256', 'banks', 'selected',
              'search', 'started', 'wall_seconds'}
    require(set(meta) == fields and meta['operation'] == 'cem256' and meta['status'] == 'completed'
            and meta['step'] == step and meta['configuration'] == physics_configuration(plan, model)
            and meta['root_sha256'] == {key: physics_array_hash(value) for key, value in roots.items()}
            and len(meta['banks']) == 4 and meta['selected'] is not None, 'Complete paid nominal CEM journal')
    selected = meta['search']
    require(set(selected) == {'status', 'candidate_evaluations', 'imagined_transitions', 'input_identities',
            'selected_ids', 'selected_actions', 'selected_sequences', 'candidate_ids'}
            and selected['status'] == 'completed' and selected['candidate_evaluations'] == result.candidate_evaluations
            and selected['imagined_transitions'] == result.imagined_transitions
            and selected['input_identities'] == [list(item) for item in result.input_identities]
            and selected['candidate_ids'] == list(result.candidate_ids), 'Exact nominal CEM identity/work')
    for key in ('selected_ids', 'selected_actions', 'selected_sequences'):
        require(np.array_equal(np.asarray(selected[key]), getattr(result, key)), 'Nominal selected sequence identity')
    banks, costs, expected_keys = [], [], {'root__' + key for key in root_names}
    for index in range(4):
        prefix = f'bank{index}__'
        bank = {key.removeprefix(prefix): value for key, value in values.items() if key.startswith(prefix)}
        require(set(meta['banks'][index]) == {'metadata'}, 'No failed scratch state in completed nominal bank')
        costs.append(audit_physics_bank(plan, model, roots, bank, meta['banks'][index]['metadata'],
            result.sequences[:, index * 64:(index + 1) * 64], step=step, candidate_start=index * 64))
        banks.append(bank)
        expected_keys |= {prefix + key for key in bank}
    selected_arrays = {key.removeprefix('selected__'): value for key, value in values.items() if key.startswith('selected__')}
    require(set(meta['selected']) == {'metadata'}, 'Completed selected-root advance')
    selected_cost = audit_physics_bank(plan, model, roots, selected_arrays, meta['selected']['metadata'],
                                     result.selected_actions[:, None, None], step=step)
    expected_keys |= {'selected__' + key for key in selected_arrays}
    require(set(values) == expected_keys, 'No unbound native journal arrays')
    require(np.array_equal(np.concatenate([bank['geometry_reward'] for bank in banks], 1), trace['raw_rewards']),
            'Nominal score arrays bind exact paid CEM trace')
    # All nominal branches use identical native state initialization and kernels;
    # unlike neural batch reductions this deterministic native equality is exact.
    chosen = np.arange(n), result.selected_ids
    for key in ('qpos', 'qvel'):
        full = np.concatenate([bank[key] for bank in banks], 1)
        require(np.array_equal(selected_arrays[key][:, 0], full[chosen][:, :2]), 'Selected native advance restarted at root')
    full_angles = np.concatenate([bank['predicted_angles'] for bank in banks], 1)
    require(np.array_equal(selected_arrays['predicted_angles'][:, 0, 0], full_angles[chosen][:, 0]),
            'Selected native angle features match chosen first step exactly')
    for key in ('geometry_' + name for name in (*GEOMETRY_FIELDS, 'reward')):
        full = np.concatenate([bank[key] for bank in banks], 1)
        # The selected and candidate geometry calls have different batch shapes;
        # CPU vector/reduction paths need not share exact final floating bits.
        close_geometry(selected_arrays[key][:, 0, 0], full[chosen][:, 0], 'Selected native ' + key)
    finite(meta['started'], 'Nominal monotonic start', positive=True)
    require(meta['wall_seconds'] == trace['search_seconds']
            and sum(row['wall_seconds'] for row in (*costs, selected_cost)) <= meta['wall_seconds'] + 1e-6,
            'Nominal CEM wall charges candidate and selected work')
    return {'candidate_transitions': sum(row['transitions'] for row in costs),
            'selected_transitions': selected_cost['transitions'],
            'max_abs_error': max(row['max_abs_error'] for row in (*costs, selected_cost)),
            'counts': {key: sum(row['counts'][key] for row in (*costs, selected_cost)) for key in PHYSICS_COUNTS},
            'native_seconds': sum(row['native_seconds'] for row in (*costs, selected_cost)),
            'geometry_seconds': sum(row['geometry_seconds'] for row in (*costs, selected_cost)),
            'operation_wall_seconds': meta['wall_seconds']}


def audit_kinematic_observer(plan, records, estimates, snapshots):
    """Reconstruct only public-derived state, never rerun MPC or a learned model.

    The final real packet is not assimilated: the last planning root is step49.
    Nominal observer propagation is additional audit work, counted separately
    from replay of the saved disturbed control episodes.
    """
    native = search_audit.native
    require(
        isinstance(snapshots, list) and len(snapshots) == len(records), "Complete public observer snapshots"
    )
    env = native.make_env()
    transitions, maximum = 0, 0.0
    try:
        model = copy.copy(env.unwrapped.model)
        require(float(model.opt.timestep) * 2 == plan["dt"], "Nominal observer decision time")
        steps, dt = plan["steps"], plan["dt"]
        for index, (record, saved) in enumerate(zip(records, snapshots, strict=True)):
            check_budget()
            packets = np.asarray(record["policy"]["packets"], dtype=np.float64)
            commands = record["policy"]["commands"]
            data = native.mujoco.MjData(model)
            target = packets[0, 4:6].copy()
            last_angles = np.arctan2(packets[0, 2:4], packets[0, :2])
            data.qpos[:] = np.concatenate((last_angles, target))
            data.qvel[:] = 0.0
            native.mujoco.mj_forward(model, data)
            last_valid, interval, visible = 0, None, [0]
            for step in range(steps):
                check_budget()
                packet = packets[step]
                require(np.array_equal(packet[4:6], target), "Observer uses a static public target")
                if step:
                    data.ctrl[:] = commands[step - 1].astype(np.float64)
                    native.mujoco.mj_step(model, data, nstep=2)
                    native.mujoco.mj_rnePostConstraint(model, data)
                    transitions += 1
                    elapsed = (step - last_valid) * dt
                    require(
                        np.isclose(packet[7], 0.0 if packet[6] else elapsed, atol=1e-6, rtol=1e-6),
                        "Public elapsed age",
                    )
                    if packet[6]:
                        angles = np.arctan2(packet[2:4], packet[:2])
                        data.qvel[:2] = ((angles - last_angles + np.pi) % (2 * np.pi) - np.pi) / elapsed
                        data.qvel[2:] = 0.0
                        data.qpos[:2] += (angles - data.qpos[:2] + np.pi) % (2 * np.pi) - np.pi
                        data.qpos[2:] = target
                        native.mujoco.mj_forward(model, data)
                        last_angles, last_valid, interval = angles.copy(), step, elapsed
                        visible.append(step)
                expected = np.concatenate((data.qpos, data.qvel))
                error = float(np.max(np.abs(estimates[index, step] - expected)))
                require(math.isfinite(error) and error <= 1e-10, "Public-only nominal state reconstruction")
                maximum = max(maximum, error)
            observer = saved
            require(
                set(observer)
                == {
                    "configuration",
                    "step_index",
                    "elapsed_seconds",
                    "last_valid_step",
                    "last_velocity_interval_seconds",
                    "valid_measurements",
                    "last_packet",
                    "last_issued_command",
                    "qpos_estimate",
                    "qvel_estimate",
                    "failed",
                    "costs",
                },
                "Observer snapshot schema",
            )
            require(
                observer["configuration"] == inherited.kinematic_configuration(model, plan)
                and observer["step_index"] == steps - 1
                and observer["elapsed_seconds"] == (steps - 1) * dt
                and observer["last_valid_step"] == last_valid
                and observer["last_velocity_interval_seconds"] == interval
                and observer["failed"] is False,
                "Final observer time, visibility and semantics",
            )
            require(
                observer["valid_measurements"]
                == [{"step": t, "time": t * dt, "packet": packets[t].tolist()} for t in visible[-2:]],
                "Actual last two public measurements",
            )
            for name, wanted in (
                ("last_packet", packets[steps - 1]),
                ("last_issued_command", commands[steps - 2]),
                ("qpos_estimate", data.qpos),
                ("qvel_estimate", data.qvel),
            ):
                actual = np.asarray(observer[name], dtype=np.float64)
                require(
                    actual.shape == wanted.shape
                    and np.isfinite(actual).all()
                    and np.allclose(actual, wanted, atol=1e-10, rtol=0),
                    "Final observer field: " + name,
                )
            oc = observer["costs"]
            expected_counts = {
                "native_transition_attempts": steps - 1,
                "native_substeps_requested": 2 * (steps - 1),
                "native_transitions_completed": steps - 1,
                "native_substeps_completed": 2 * (steps - 1),
                "setup_forward_calls": 1,
                "measurement_reanchor_forward_calls": len(visible) - 1,
                "postconstraint_refresh_calls_completed": steps - 1,
            }
            require(
                set(oc)
                == set(expected_counts)
                | {
                    "setup_wall_seconds",
                    "update_wall_seconds",
                    "counts_exclude_forward_and_postconstraint_from_native_substeps",
                }
                and oc["counts_exclude_forward_and_postconstraint_from_native_substeps"] is True,
                "Observer work schema",
            )
            for name, value in expected_counts.items():
                integer(oc[name], value, "Observer " + name)
            for name in ("setup_wall_seconds", "update_wall_seconds"):
                finite(oc[name], "Observer " + name, positive=True)
    finally:
        env.close()
    return {
        "observer_native_transitions_replayed": transitions,
        "max_abs_error": maximum,
        "final_observer_root": plan["steps"] - 1,
        "new_planner_calls": 0,
        "scope": "Public packets and issued commands only; nominal observer replay, no realized noise or true state.",
    }

def audit_particle_observer(plan, records, estimates, saved):
    """Rebuild the public filter from named child streams, not private state.

    The native simulator is explicitly supplied prior knowledge. Resampling
    and disturbance draws replay old recorded estimator streams; no new policy
    or environment evaluation cohort is chosen by the auditor.
    """
    native, count = search_audit.native, plan['particles']
    mj, env = native.mujoco, native.make_env()
    transitions, maximum = 0, 0.
    try:
        model = copy.copy(env.unwrapped.model)
        require(float(model.opt.timestep) * 2 == plan['dt'], 'Particle nominal timestep')
        for case, record in enumerate(records):
            packets = np.asarray(record['policy']['packets'], np.float64)
            commands = record['policy']['commands']
            streams = np.random.SeedSequence(protocol.seed(plan, f'planner/particle_filter/{case}')).spawn(3)
            initial_rng, process_rng, resample_rng = [np.random.default_rng(value) for value in streams]
            target, age = packets[0, 4:6].copy(), 0.
            particles = [mj.MjData(model) for _ in range(count)]
            for data in particles:
                data.qpos[:] = np.r_[np.arctan2(packets[0, 2:4], packets[0, :2]), target]
                data.qvel[:2] = initial_rng.uniform(-.005, .005, 2)
                data.qvel[2:] = 0.
                mj.mj_forward(model, data)
            for step in range(plan['steps']):
                check_budget()
                packet = packets[step]
                require(np.array_equal(packet[4:6], target), 'Static public particle target')
                if step:
                    require(np.isclose(packet[7], 0. if packet[6] else age + plan['dt'], atol=1e-6, rtol=1e-6), 'Public particle age')
                    disturbances = process_rng.normal(0., plan['noise_std'], (count, 2))
                    for data, noise in zip(particles, disturbances, strict=True):
                        check_budget()
                        data.ctrl[:] = np.clip(commands[step - 1].astype(float) + noise, -1., 1.)
                        mj.mj_step(model, data, nstep=2)
                        mj.mj_rnePostConstraint(model, data)
                        transitions += 1
                    if packet[6]:
                        measured = np.arctan2(packet[2:4], packet[:2])
                        residual = (np.stack([data.qpos[:2] for data in particles]) - measured + np.pi) % (2 * np.pi) - np.pi
                        squared = np.sum(residual ** 2, axis=1)
                        with np.errstate(over='ignore', under='ignore'):
                            weights = np.exp(-.5 * ((squared - squared.min()) / plan['filter_bandwidth']) / plan['filter_bandwidth'])
                        weights /= weights.sum()
                        points = (resample_rng.random() + np.arange(count)) / count
                        cumulative = np.cumsum(weights)
                        cumulative[-1] = 1.
                        indices = np.searchsorted(cumulative, points, side='right')
                        spec = mj.mjtState.mjSTATE_INTEGRATION
                        states = np.empty((count, mj.mj_stateSize(model, spec)))
                        for index, data in enumerate(particles):
                            mj.mj_getState(model, data, states[index], spec)
                        for data, index in zip(particles, indices, strict=True):
                            mj.mj_setState(model, data, states[index], spec)
                            data.qpos[:2] += (measured - data.qpos[:2] + np.pi) % (2 * np.pi) - np.pi
                            data.qpos[2:], data.qvel[2:] = target, 0.
                            mj.mj_forward(model, data)
                    age = float(packet[7])
                angles = np.stack([data.qpos[:2] for data in particles])
                position = angles.mean(axis=0)
                position[0] = np.arctan2(np.sin(angles[:, 0]).mean(), np.cos(angles[:, 0]).mean())
                velocity = np.mean([data.qvel[:2] for data in particles], axis=0)
                expected = np.r_[position, target, velocity, [0., 0.]]
                error = float(np.max(np.abs(estimates[case, step] - expected)))
                require(math.isfinite(error) and error <= 1e-10, 'Public-only particle state reconstruction')
                maximum = max(maximum, error)
        updates = len(records) * (plan['steps'] - 1)
        expected_counts = {'setup_forward_calls': len(records) * count,
            'nominal_transition_calls': updates * count, 'native_substeps': updates * count * 2,
            'measurement_reanchor_forward_calls': sum(int(np.asarray(record['metadata']['sensor_schedule'])[1:50].sum())
                                                      for record in records) * count}
        require(set(saved) == {'reference', 'public_inputs_only', 'updates', 'particles', 'estimates', 'counts', 'scope'}
                and saved['reference'] == 'particle' and saved['public_inputs_only'] is True
                and saved['updates'] == updates and saved['particles'] == count and saved['counts'] == expected_counts
                and np.array_equal(np.asarray(saved['estimates']), estimates[:, -1])
                and saved['scope'] == 'Detached estimate after final decision assimilation, before terminal packet; not a resumable particle/RNG state.',
                'Complete public particle observer receipt')
    finally:
        env.close()
    return {'observer_native_transitions_replayed': transitions, 'max_abs_error': maximum,
            'final_observer_root': plan['steps'] - 1, 'new_planner_calls': 0,
            'scope': 'Named estimator children and actual public packets/actions only; no realized environment disturbances.'}


def reference_timing(plan, path, arm):
    value = read(path)
    fields = {'setup_seconds', 'decision_seconds', 'native_step_seconds', 'row_wall_seconds',
        'candidate_evaluations_per_decision', 'selected_nominal_advances', 'observer_updates', 'privileged_state_reads',
        'information', 'score_mode', 'planner', 'floors_zero_scores_are_predictions'}
    require(set(value) == fields, 'Exact matched reference cost fields')
    n, steps, planned = plan['control_episodes'], plan['steps'], arm in PHYSICS_REFERENCES
    for key in ('setup_seconds', 'row_wall_seconds'):
        finite(value[key], 'Reference ' + key, positive=True)
    for key in ('decision_seconds', 'native_step_seconds'):
        require(isinstance(value[key], list) and len(value[key]) == steps, 'Every reference decision/native cost')
        for seconds in value[key]:
            finite(seconds, key, positive=True)
    require(value['setup_seconds'] + sum(value['decision_seconds']) + sum(value['native_step_seconds'])
            <= value['row_wall_seconds'] + 1e-6, 'Reference row includes all nested phases')
    for key, expected in (('candidate_evaluations_per_decision', 256 if planned else 0),
        ('selected_nominal_advances', n * steps if planned else 0),
        ('observer_updates', n * (steps - 1) if arm in ('particle', 'public_kinematic') else 0),
        ('privileged_state_reads', n * steps if arm == 'known_state' else 0)):
        integer(value[key], expected, 'Reference cost ' + key)
    require(value['score_mode'] == ('geometry' if planned else None)
            and value['planner'] == ('cem256' if planned else None)
            and value['floors_zero_scores_are_predictions'] is False
            and value['information'] == ('Current native qpos/qvel, public goal; no future noise' if arm == 'known_state'
                else 'Public packets and issued commands only, supplied nominal model for planned references'),
            'Reference scoring and information contract')
    return value


def audit_reference_control(plan, folder, row, records, inputs):
    arm, n, steps = row['reference'], plan['control_episodes'], plan['steps']
    require(arm in REFERENCES, 'Declared matched reference')
    planned = arm in PHYSICS_REFERENCES
    times = reference_timing(plan, folder / 'timings.json', arm)
    values = base.load_npz(folder / 'planning.npz')
    require(set(values) == {'candidate_scores', 'planner_used'}
            | ({'root_estimates', 'public_packets', 'previous_commands'} if planned else set()), 'Reference planning artifact schema')
    scores = array(values['candidate_scores'], (n, steps, 256), np.float64, 'Reference CEM scores')
    used = array(values['planner_used'], (), np.bool_, 'Reference used flag')
    require(bool(used) is planned, 'Reference planned flag')
    commands, packets = base.stack(records, 'policy', 'commands'), base.stack(records, 'policy', 'packets')
    costs = -base.stack(records, 'audit', 'rewards').sum(1)
    observer, work, clipped, predictions, searches = None, [], 0, 0, []
    if planned:
        roots = array(values['root_estimates'], (n, steps, 8), np.float64, 'Reference original supplied roots')
        require(np.array_equal(array(values['public_packets'], (n, steps, 8), np.float32, 'Reference public packets'), packets[:, :steps]),
                'Only actual public packets enter observers/planner')
        prior_actions = np.concatenate((np.zeros((n, 1, 2), np.float32), commands[:, :-1]), 1)
        require(np.array_equal(array(values['previous_commands'], (n, steps, 2), np.float32, 'Previous issued commands'), prior_actions),
                'Public observer causal action alignment')
        if arm == 'known_state':
            native = np.concatenate((base.stack(records, 'audit', 'qpos')[:, :steps], base.stack(records, 'audit', 'qvel')[:, :steps]), -1)
            require(np.array_equal(roots, native), 'Privileged current root only')
        elif arm == 'public_kinematic':
            observer = audit_kinematic_observer(plan, records, roots, read(folder / 'observer-final.json'))
        else:
            observer = audit_particle_observer(plan, records, roots, read(folder / 'observer-final.json'))
        env = search_audit.native.make_env()
        try:
            model = copy.copy(env.unwrapped.model)
            final = read(folder / 'physics-final.json')
            require(set(final) == {'configuration', 'setup_seconds', 'failed', 'lifetime'}
                    and final['configuration'] == physics_configuration(plan, model) and final['failed'] is False,
                    'Complete successful nominal physics lifetime')
            require(finite(final['setup_seconds'], 'Physics setup', positive=True) <= times['setup_seconds'], 'Physics setup charged')
            for step in range(steps):
                check_budget()
                trace = search_audit.audit_trace({**plan, 'planners': ['cem256']}, inputs[step], folder / 'decisions' / f'{step:03d}', step=step)
                require(np.array_equal(trace['result'].selected_actions, commands[:, step])
                        and np.array_equal(trace['result'].scores, scores[:, step]), 'Reference actual action and all paid scores')
                work.append(audit_physics_trace(plan, model, folder / 'physics' / f'{step:03d}', trace,
                                               roots[:, step], packets[:, step, 4:6], step=step))
                require(trace['search_seconds'] <= times['decision_seconds'][step] + 1e-6, 'Nominal work charged to decision')
                searches.append(trace['search_seconds'])
                clipped += trace['clipped_predictions']
                predictions += trace['reward_predictions']
            lifetime = final['lifetime']
            require(set(lifetime) == set(PHYSICS_COUNTS) | {'operations_completed', 'operations_failed', 'operation_wall_seconds'},
                    'Complete nominal lifetime counters')
            for key in PHYSICS_COUNTS:
                integer(lifetime[key], sum(item['counts'][key] for item in work), 'Total nominal ' + key)
            integer(lifetime['operations_completed'], steps, 'All nominal operations')
            integer(lifetime['operations_failed'], 0, 'No failed operation hidden')
            require(math.isclose(lifetime['operation_wall_seconds'], sum(searches), rel_tol=1e-12, abs_tol=1e-9),
                    'All nominal operation time accounted')
        finally:
            env.close()
    else:
        require(np.all(scores == 0), 'Floor scores are placeholders, not predictions')
        uniform = np.random.default_rng(protocol.seed(plan, 'floor/uniform/0')) if arm == 'uniform' else None
        for step in range(steps):
            wanted = np.zeros((n, 2), np.float32) if arm == 'zero' else uniform.uniform(-1, 1, (n, 2)).astype(np.float32)
            require(np.array_equal(commands[:, step], wanted), 'Exact floor command/no hidden search')
    decisions = np.asarray(times['decision_seconds'], float)
    return {'episode_costs': costs.tolist(), 'mean_cost': float(costs.mean()),
        'setup_seconds': times['setup_seconds'], 'decision_wall_seconds': float(decisions.sum()),
        'native_step_seconds': float(sum(times['native_step_seconds'])), 'row_wall_seconds': times['row_wall_seconds'],
        'decision_seconds': times['decision_seconds'], 'search_seconds': searches,
        'batch_latency_seconds': {'mean': float(decisions.mean()), 'p50': float(np.quantile(decisions, .5)),
            'p95': float(np.quantile(decisions, .95)), 'max': float(decisions.max())},
        'per_case_amortized_seconds': float(decisions.mean() / n), 'planner_used': planned,
        'candidate_evaluations': n * steps * 256 if planned else 0,
        'imagined_transitions': sum(item['candidate_transitions'] for item in work),
        'clipping_fraction': clipped / predictions if predictions else None,
        'public_observer': observer, 'physics_work': {
            'candidate_native_transitions_replayed': sum(item['candidate_transitions'] for item in work),
            'selected_native_transitions_replayed': sum(item['selected_transitions'] for item in work),
            'max_abs_error': max((item['max_abs_error'] for item in work), default=0.),
            'native_seconds': sum(item['native_seconds'] for item in work),
            'geometry_seconds': sum(item['geometry_seconds'] for item in work),
            'operation_wall_seconds': sum(item['operation_wall_seconds'] for item in work),
            'counts': {key: sum(item['counts'][key] for item in work) for key in PHYSICS_COUNTS}},
        'reference_limit': 'Native nominal supplied dynamics; candidate-budget matched, not wall-time matched or stochastic-optimal.'}
def qualification(plan, controls):
    """Independently reconstruct all25 inequalities, never summary booleans."""
    expected = {f'{family}-{pair}' for family in FAMILIES for pair in PAIRS} | set(REFERENCES)
    require(set(controls) == set(PANELS), 'All three sensing panels')
    for panel in PANELS:
        require(set(controls[panel]) == expected, 'All12 learned and five reference rows')
        for row in controls[panel].values():
            cost = np.asarray(row['episode_costs'], np.float64)
            require(cost.shape == (plan['control_episodes'],) and np.isfinite(cost).all() and (cost >= 0).all()
                    and math.isclose(row['mean_cost'], float(cost.mean()), rel_tol=1e-12, abs_tol=1e-12),
                    'Complete paired native cost means')
    means = {panel: {family: float(np.mean([controls[panel][f'{family}-{pair}']['episode_costs'] for pair in PAIRS]))
                    for family in FAMILIES} for panel in PANELS}
    checks = []
    def add(name, left, right):
        checks.append({'name': name, 'left': float(left), 'right': float(right), 'comparison': 'le', 'passed': bool(left <= right)})
    for comparator in ('encoded_current_gru', 'cached_gru'):
        for panel in ('ordinary', 'shift'):
            add(f'gap_mean/{panel}/{comparator}', means[panel]['residual_gru'], .97 * means[panel][comparator])
            for pair in PAIRS:
                add(f'gap_pair/{panel}/{comparator}/{pair}', controls[panel][f'residual_gru-{pair}']['mean_cost'],
                    controls[panel][f'{comparator}-{pair}']['mean_cost'])
        add(f'full_mean/{comparator}', means['full']['residual_gru'], 1.02 * means['full'][comparator])
    for panel in ('ordinary', 'shift'):
        for pair in PAIRS:
            add(f'competence/{panel}/residual_gru-{pair}', controls[panel][f'residual_gru-{pair}']['mean_cost'],
                .9 * controls[panel]['zero']['mean_cost'])
    add('competence/ordinary/known_state', controls['ordinary']['known_state']['mean_cost'], .9 * controls['ordinary']['zero']['mean_cost'])
    require(len(checks) == len({item['name'] for item in checks}) == 25
            and [item['name'] for item in checks] == [item['name'] for item in protocol.criterion_manifest(plan)],
            'Exact independently derived prospective gate')
    return {'passed': all(item['passed'] for item in checks), 'checks': checks,
            'secondary_cannot_rescue_primary': True, 'family_mean_costs': means,
            'scope': 'Trained persistent update policy versus trained reset GRUs, all saved fit pairs retained.'}


def paired_comparisons(plan, controls):
    contrasts = {'persistent_minus_encoded_current_gru': ('residual_gru', 'encoded_current_gru'),
        'persistent_minus_cached_gru': ('residual_gru', 'cached_gru'),
        'persistent_minus_cached_mlp': ('residual_gru', 'cached_mlp'),
        'cached_gru_minus_encoded_current_gru': ('cached_gru', 'encoded_current_gru'),
        'cached_gru_minus_cached_mlp': ('cached_gru', 'cached_mlp')}
    output = {}
    for panel in PANELS:
        families = {arm: np.asarray([controls[panel][f'{arm}-{pair}']['episode_costs'] for pair in PAIRS], np.float64) for arm in FAMILIES}
        output[panel] = {}
        for name, (a, b) in contrasts.items():
            differences = families[a] - families[b]
            cases = differences.mean(0)
            rng = np.random.default_rng(protocol.seed(plan, 'analysis/bootstrap/0'))
            indices = rng.integers(0, len(cases), (plan['bootstrap_samples'], len(cases)))
            denominator, pair_denominator = float(families[b].mean()), families[b].mean(1)
            output[panel][name] = {'mean_cost_difference': float(cases.mean()),
                'percent_cost_change': float(100 * cases.mean() / denominator) if denominator else None,
                'paired_fit_cost_differences': differences.mean(1).tolist(),
                'paired_fit_percent_changes': [float(100 * delta / divisor) if divisor else None
                                              for delta, divisor in zip(differences.mean(1), pair_denominator, strict=True)],
                'all_three_pairs_nonworse': bool((differences.mean(1) <= 0).all()),
                'episode_paired_percentile_95': np.quantile(cases[indices].mean(1), [.025, .975]).tolist(),
                'cases': len(cases), 'conditional_on_all_three_saved_fit_pairs': True,
                'primary_comparator': name in ('persistent_minus_encoded_current_gru', 'persistent_minus_cached_gru'),
                'scope': 'Conditional paired-case interval, no new training-seed uncertainty or multiplicity correction.'}
    return output


def audit_boundaries(plan, expected, execution, completed, restored):
    started, evaluation, control = (read(execution / name) for name in
        ('started.json', 'evaluation-started.json', 'control-completed.json'))
    require(set(started) == {'plan_sha256', 'unix_time'}
            and set(evaluation) == {'plan_sha256', 'all_models_restored_sha256', 'unix_time', 'elapsed_seconds'}
            and set(control) == {'rows', 'unix_time', 'plan_sha256', 'evaluation_started_sha256'}, 'Exact phase boundary schema')
    require(all(row['plan_sha256'] == expected for row in (started, evaluation, control))
            and evaluation['all_models_restored_sha256'] == sha(execution / 'all-models-restored.json')
            and control['evaluation_started_sha256'] == sha(execution / 'evaluation-started.json'), 'Hash-bound pre-evaluation restoration')
    timestamps = [finite(row['unix_time'], 'Phase timestamp', positive=True) for row in (started, restored, evaluation, control)]
    require(all(a <= b for a, b in pairwise(timestamps)), 'Every restoration precedes fresh controls')
    require(evaluation['elapsed_seconds'] == completed['evaluation_started_elapsed_seconds']
            and 0 < evaluation['elapsed_seconds'] <= completed['wall_seconds']
            and restored['wall_seconds'] <= evaluation['elapsed_seconds'] + 1e-6, 'Charged pre-evaluation setup')
    integer(control['rows'], 51, 'All control rows before completion')
    return {'timestamps_monotonic': True, 'all_twelve_restored_before_fresh_controls': True,
            'new_fits': 0, 'new_optimizer_updates': 0, 'exposed_diagnostic_roots': 0,
            'pre_evaluation_wall_seconds': evaluation['elapsed_seconds']}


def audit_costs(plan, execution, completed, restored, controls):
    saved = read(execution / 'costs.json')
    fields = {'inheritance_copy_wall_seconds', 'restore_wall_seconds', 'new_fits', 'new_optimizer_steps',
        'new_prediction_episodes', 'diagnostic_roots', 'innovation_generation_and_storage_seconds',
        'control_row_wall_seconds', 'control_setup_seconds', 'control_decision_seconds', 'control_native_step_seconds',
        'control_native_transitions', 'astra_calls', 'cache_parent_costs', 'geometry_context_costs', 'compute_matched', 'accounting'}
    require(set(saved) == fields, 'Complete charged execution cost schema')
    rows = [controls[row['panel']][row['label']] for row in protocol.execution_order(plan)]
    sums = {'restore_wall_seconds': restored['wall_seconds'],
            'control_row_wall_seconds': sum(row['row_wall_seconds'] for row in rows),
            'control_setup_seconds': sum(row['setup_seconds'] for row in rows),
            'control_decision_seconds': sum(row['decision_wall_seconds'] for row in rows),
            'control_native_step_seconds': sum(row['native_step_seconds'] for row in rows)}
    for key, expected in sums.items():
        require(math.isclose(finite(saved[key], key, positive=True), expected, rel_tol=1e-12, abs_tol=1e-8), 'Summed execution cost ' + key)
    for key, expected in (('new_fits', 0), ('new_optimizer_steps', 0), ('new_prediction_episodes', 0),
        ('diagnostic_roots', 0), ('astra_calls', 0), ('control_native_transitions', 51 * plan['control_episodes'] * 50)):
        integer(saved[key], expected, 'Execution cost scope ' + key)
    require(saved['cache_parent_costs'] == plan['cache_source']['prior_costs']
            and saved['geometry_context_costs'] == plan['geometry_source']['prior_costs']
            and saved['compute_matched'] is False and isinstance(saved['accounting'], str) and saved['accounting'],
            'Explicit nested lineage and unmatched compute accounting')
    copying = finite(saved['inheritance_copy_wall_seconds'], 'Copying time', positive=True)
    innovation = finite(saved['innovation_generation_and_storage_seconds'], 'Innovation generation/storage', positive=True)
    require(copying + innovation + saved['restore_wall_seconds'] + saved['control_row_wall_seconds'] <= completed['wall_seconds'] + 1e-6,
            'Whole wall time includes every nonoverlapping phase')
    cumulative = saved['geometry_context_costs']['cumulative_attempt_wall_seconds'] + completed['wall_seconds']
    require(math.isclose(completed['cumulative_attempt_wall_seconds'], cumulative, rel_tol=1e-12, abs_tol=1e-8),
            'Cumulative cost uses geometry lineage once, never adds its cache ancestry twice')
    learned = {key: sum(row['scoring_work'][key] for row in rows if 'scoring_work' in row) for key in (
        'candidate_evaluations', 'imagined_transitions', 'geometry_samples', 'geometry_seconds', 'model_advance_seconds',
        'selected_geometry_samples', 'selected_geometry_seconds', 'selected_model_advance_seconds')}
    physics = {key: sum(row['physics_work'][key] for row in rows if 'physics_work' in row) for key in (
        'candidate_native_transitions_replayed', 'selected_native_transitions_replayed', 'native_seconds',
        'geometry_seconds', 'operation_wall_seconds')}
    counters = {key: sum(row['physics_work']['counts'][key] for row in rows if 'physics_work' in row) for key in PHYSICS_COUNTS}
    coverage = protocol.coverage(plan)
    integer(learned['candidate_evaluations'], coverage['learned_candidate_evaluations'], 'Total learned paid candidates')
    integer(learned['imagined_transitions'], coverage['learned_imagined_transitions'], 'Total learned paid transitions')
    integer(learned['geometry_samples'], coverage['learned_imagined_transitions'], 'Every learned geometry sample')
    integer(learned['selected_geometry_samples'], coverage['learned_selected_advances'], 'Every learned selected advance')
    integer(physics['candidate_native_transitions_replayed'], coverage['physics_nominal_transitions'], 'Every nominal candidate replayed')
    integer(physics['selected_native_transitions_replayed'], coverage['physics_selected_advances'], 'Every nominal selected advance replayed')
    integer(counters['native_substeps_completed'], coverage['physics_candidate_native_substeps'] + coverage['physics_selected_native_substeps'],
            'Every candidate/selected native substep charged')
    integer(learned['geometry_samples'] + counters['geometry_samples_completed'] + learned['selected_geometry_samples'],
            coverage['geometry_candidate_samples'] + coverage['geometry_selected_samples'], 'Complete candidate plus selected geometry work')
    return {**saved, 'execution_wall_seconds': completed['wall_seconds'], 'cumulative_attempt_wall_seconds': cumulative,
            'learned_control_scoring': learned, 'physics_control_scoring': {**physics, 'counts': counters},
            'limits': 'Nested measured shared-host wall times; copied traces/hashes included. Candidate counts are not total FLOPs or equal CPU budgets.'}


def audit_saved(plan, expected_plan_sha256, execution, out, *, engineering=False):
    """Authenticate completed saved artifacts and independently replay native work."""
    execution, out = Path(execution), Path(out)
    require(not out.exists(), 'Exclusive new audit output required')
    begin = time.monotonic()
    tokens = []
    try:
        require(type(plan['audit_cap_seconds']) is int and plan['audit_cap_seconds'] > 0, 'Positive explicit audit cap')
        deadline = begin + plan['audit_cap_seconds']
        for context in (_DEADLINE, inherited._DEADLINE, previous._DEADLINE):
            tokens.append((context, context.set(deadline)))
        require(type(plan['cap_seconds']) is int and plan['cap_seconds'] > 0, 'Positive explicit execution cap')
        protocol.validate_settings(plan, engineering=engineering)
        require(plan['engineering'] is engineering, 'Explicit engineering boundary')
        validate_runtime_sources(plan)
        completed = validate_members(plan, expected_plan_sha256, execution)
        completion_hash = sha(execution / 'completed.json')
        parent, _ = validate_lineage(plan, execution)
        streams = validate_streams(plan, execution)
        fits, restored = audit_restoration(plan, parent, execution)
        inputs = [search_audit.audit_innovations(plan, execution / 'innovations' / 'control' / f'{step:03d}',
                    f'planner/control/{step}', plan['control_episodes']) for step in range(plan['steps'])]
        controls, cohorts, resets, disturbances = {panel: {} for panel in PANELS}, {}, None, None
        for row in protocol.execution_order(plan):
            check_budget()
            folder = execution / row['path']
            records = base.load_records(folder / 'episodes', plan['control_episodes'])
            cohorts[row['path']] = audit_cohort(plan, records, row['panel'])
            initial = base.stack(records, 'audit', 'integration_state')[:, 0]
            noise = base.stack(records, 'audit', 'actuator_noise')
            if resets is None:
                resets, disturbances = initial, noise
            require(np.array_equal(initial, resets) and np.array_equal(noise, disturbances), 'All51 rows share fresh reset states and exogenous noise')
            controls[row['panel']][row['label']] = (
                audit_learned_control(plan, folder, row, records, inputs, fits[row['fit']]) if 'fit' in row
                else audit_reference_control(plan, folder, row, records, inputs))
        gate = qualification(plan, controls)
        comparisons = paired_comparisons(plan, controls)
        boundary = audit_boundaries(plan, expected_plan_sha256, execution, completed, restored)
        costs = audit_costs(plan, execution, completed, restored, controls)
        require(validate_members(plan, expected_plan_sha256, execution) == completed
                and sha(execution / 'completed.json') == completion_hash, 'Execution remained unchanged throughout audit')
        validate_runtime_sources(plan)
        check_budget()
        physics = [controls[panel][name]['physics_work'] for panel in PANELS for name in PHYSICS_REFERENCES]
        observers = [controls[panel][name]['public_observer'] for panel in PANELS for name in ('particle', 'public_kinematic')]
        summary = {'status': 'completed', 'version': VERSION, 'engineering': engineering,
            'plan_sha256': expected_plan_sha256, 'execution_completed_sha256': completion_hash,
            'saved_output_only': True, 'new_model_calls': 0, 'new_policy_calls': 0, 'new_fits': 0,
            'native_control_transitions_checked': sum(row['transitions'] for row in cohorts.values()),
            'native_control_max_abs_error': max(row['max_abs_error'] for row in cohorts.values()),
            'native_nominal_candidate_transitions_checked': sum(row['candidate_native_transitions_replayed'] for row in physics),
            'native_nominal_selected_transitions_checked': sum(row['selected_native_transitions_replayed'] for row in physics),
            'native_nominal_max_abs_error': max(row['max_abs_error'] for row in physics),
            'public_observer_transitions_checked': sum(row['observer_native_transitions_replayed'] for row in observers),
            'public_observer_max_abs_error': max(row['max_abs_error'] for row in observers),
            'coverage': protocol.coverage(plan), 'random_streams': streams, 'inherited_fits': fits,
            'phase_boundary': boundary, 'control': controls, 'continuation_gate': gate,
            'paired_descriptive_comparisons': comparisons, 'cohorts': cohorts,
            'costs': {**costs, 'audit_validation_wall_seconds': time.monotonic() - begin},
            'limits': [
                'Zero new fits; all12 inherited checkpoints and51 control rows retained. Historical cache15/28 failure and geometry25/25 success remain separate claims.',
                'Before/after observed tensors are equal. No trainer/optimizer is invoked by this audit; absence of intermediate updates is source-bound, not proven by endpoints alone.',
                'Learned predictions/hidden vectors are saved source-bound evidence, not new neural inference. Public cache fields, clocks, chosen actions and actual-class configurations are checked.',
                'All three GRU arms are recurrent within imagined rollouts. Treatment is the separately trained real-assimilation update policy, including its validity gating, not recurrence in isolation.',
                'Geometry is independently derived in NumPy. Final-angle FK approximates native RK4 cached-body reward, and projected mean-angle distance is not expected distance.',
                'Native control replay, every nominal planning candidate/selected step, and causal public-observer propagation are independently checked and separately counted.',
                'Physics uses the same paid CEM256 and geometry objective but nominal zero-disturbance dynamics and supplied simulator knowledge. It is not an optimal stochastic-control oracle.',
                'All original learned heads remain executed. Geometry/cache/observer/trace work is charged; equal candidate budgets are not equal total compute.',
                'Conditional paired-case bootstrap intervals fix these three saved fits and training corpus; they do not establish training-seed population uncertainty or multiplicity-corrected confirmation.',
                'Shared-host batched wall times include recorded evidence work; they are not isolated latency, total FLOPs or cross-machine speed.',
                'Passing supports these trained persistent update policies over declared reset controls on this task; retained angles, velocity inference and explicit two-observation/action-history alternatives remain unresolved.',
                'No biological/connectome superiority, calibrated uncertainty, novel architecture or cross-environment robotics generality is established.',
            ]}
        summary['native_transitions_checked'] = (summary['native_control_transitions_checked']
            + summary['native_nominal_candidate_transitions_checked'] + summary['native_nominal_selected_transitions_checked'])
        summary['native_max_abs_error'] = max(summary['native_control_max_abs_error'], summary['native_nominal_max_abs_error'])
        out.mkdir(parents=True, exist_ok=False)
        write(out / 'summary.json', summary)
        (out / 'README.md').write_text('# Geometry-scored memory audit\n\n'
            f"Continuation: **{'PASS' if gate['passed'] else 'FAIL'}** ({sum(row['passed'] for row in gate['checks'])}/25 checks).\n\n"
            'All twelve inherited models and51 control rows retained; no new fit or learned inference. '
            'Native executed trajectories, every paid nominal candidate/selected advance and public observer estimates were replayed in separate scopes. '
            'See summary.json for the complete cost ledger, provenance, all fits and scientific limits.\n')
        receipt = {'status': 'completed', 'version': VERSION, 'engineering': engineering,
            'plan_sha256': expected_plan_sha256, 'source_sha256': plan['sources'], 'runtime': plan['runtime'],
            'execution_completed_sha256': completion_hash, 'execution_members': completed['files'],
            'saved_output_only': True, 'costs': summary['costs'],
            'files': {name: sha(out / name) for name in ('summary.json', 'README.md')}}
        check_budget()
        write(out / 'receipt.json', receipt)
        check_budget()
        return summary
    except BaseException as error:
        out.mkdir(parents=True, exist_ok=True)
        if (out / 'receipt.json').exists():
            (out / 'receipt.json').rename(out / 'over-cap-receipt.json')
        write(out / 'failed.json', {'status': 'failed', 'error': repr(error), 'plan_sha256': expected_plan_sha256,
                                  'wall_seconds': time.monotonic() - begin, 'saved_output_only': True})
        raise
    finally:
        for context, token in reversed(tokens):
            context.reset(token)


def audit_plan(path, expected_sha256, execution, out):
    require(not Path(out).exists(), 'Exclusive audit output')
    try:
        plan = read(checked(Path(path), expected_sha256))
        require(plan['engineering'] is False, 'Scored plan required by CLI')
    except BaseException as error:
        Path(out).mkdir(parents=True, exist_ok=False)
        write(Path(out) / 'failed.json', {'status': 'failed', 'error': repr(error), 'plan_sha256': expected_sha256,
                                        'saved_output_only': True, 'phase': 'authenticate-plan'})
        raise
    return audit_saved(plan, expected_sha256, execution, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summary = audit_plan(args.plan, args.expected_plan_sha256, args.execution, args.out)
    print(json.dumps({'status': summary['status'], 'native_transitions_checked': summary['native_transitions_checked'],
                      'continuation_gate_passed': summary['continuation_gate']['passed']}))


if __name__ == '__main__':
    main()
