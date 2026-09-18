# SPDX-License-Identifier: GPL-3.0-only
"""Frozen full-panel, fixed-weight WLDN pathway diagnostic and hook replay audit."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import time

import chess
import numpy as np
import torch
from torch.nn import functional as F

import chess_wldn_study as study
from openjev.research import wldn_interventions as intervention

source = study.contrast
ROOT = source.ROOT
SEEDS = (97, 109, 127)
SPLITS = ('dev', 'shift')
MODES = intervention.MODES
PLAN = 'evidence/chess-wldn-quality-v1/protocol/plan.json'
EXECUTION = 'runs/chess-wldn-quality-v1/execution'
AUDIT = 'evidence/chess-wldn-quality-v1/audit/receipt.json'
CODE = ['src/openjev/research/wldn_interventions.py', 'tests/test_wldn_interventions.py',
        'scripts/chess_wldn_interventions.py', 'tests/test_chess_wldn_interventions.py']
PROTOCOL = {
    'version': intervention.VERSION,
    'scope': 'Fixed-weight information-pathway diagnostic. No training, architecture selection gate, new engine calls or independent test panel.',
    'prior_knowledge': 'Completed WLDN, graph contrast, CGR and NBFNet development results and audits known. WLDN exceeded scalar contrast on both old panels. No intervention outcomes inspected before this freeze.',
    'seeds': list(SEEDS), 'panels': list(SPLITS), 'positions_per_panel': 2048,
    'modes': list(MODES), 'batch_size': 128,
    'native': 'Unmodified WLDN calculation, required to reproduce all12288 authenticated saved choices and NLL within1e-5.',
    'zero_delta': 'Set encoded child-minus-root node differences to zero before difference layer; retain child adjacency, flags and learned biases.',
    'root_diff_graph': 'Retain native encoded differences; replace difference-layer child graph with the corresponding root graph.',
    'zero_edge_flags': 'Retain native differences and child adjacency; zero the four difference-layer edge flags. Connectivity remains visible.',
    'zero_pooled': 'Zero32 graph channels at the readout; retain all120 original action channels, learned readout/output and base score.',
    'permuted_delta': 'Frozen child-cache permutation within each legal menu; shuffle encoded differences, retain correct child difference-layer graph and action features. Permutation may contain fixed points.',
    'references': 'Copy12288 base predictions from authenticated WLDN execution with explicit provenance; audit also freshly replays these base predictions.',
    'new_prediction_records': 73728, 'copied_reference_records': 12288,
    'primary': 'Share root/child encodings across all six modes within each batch. Frozen trained head and backbone; CPU2 deterministic.',
    'audit': 'All73728 predictions replayed using original unmodified WLDN forward with temporary hooks; all12288 copied base records freshly replayed. Check exact decisions, identity, correctness and finite target NLL error<=1e-5. Verify hook removal and unchanged head state. Fresh native root observation/menu/label checks; authenticate prior full child-cache audit.',
    'analysis': 'Retain all mode/seed/panel results. Native-minus-intervention agreement, intervention-minus-native NLL, decision disagreement, native-correct to wrong and reverse transitions.',
    'bootstrap': '2000 source-game resamples after averaging paired correctness across3 seeds; seed975101+shift_indicator. Same draws for all five interventions per panel, percentile95 intervals, position-weighted estimate. Recomputed by audit using same declared aggregation.',
    'time_cap_seconds': 900, 'nll_tolerance': 1e-5,
    'limits': 'Interventions can create out-of-distribution hidden states. Dependence under fixed weights does not establish retrained architecture performance, biological relevance, unique mechanism, new algorithm or superiority. Old exposed development panels; descriptive intervals lack adaptive or multiplicity correction. No full retraining, fresh labels, gameplay or latency study.'}


def manifest(directory, plan_hash):
    complete = json.loads((directory/'completed.json').read_text())
    assert complete['status'] == 'completed' and complete['plan_sha256'] == plan_hash
    assert {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} == set(complete['files']) | {'completed.json'}
    assert all(source.file_hash(directory/p) == h for p, h in complete['files'].items())
    return complete


def signature():
    prepared = json.loads((ROOT/PLAN).read_text())
    assert all(prepared[k] == v for k, v in study.signature().items())
    plan_hash = source.file_hash(ROOT/PLAN)
    manifest(ROOT/EXECUTION, plan_hash)
    audit = json.loads((ROOT/AUDIT).read_text())
    assert audit['status'] == 'completed' and audit['plan_sha256'] == plan_hash
    assert audit['evaluation_predictions_replayed'] == 24576 and audit['training_updates_checked'] == 4608
    assert audit['max_replay_nll_error'] <= 1e-5
    assert audit['auditor_sha256'] == source.file_hash(ROOT/'scripts/audit_chess_wldn_study.py')
    assert audit['summary_sha256'] == source.file_hash(ROOT/EXECUTION/'summary.json')
    assert audit['execution_receipt_sha256'] == source.file_hash(ROOT/EXECUTION/'completed.json')
    inputs = [PLAN, AUDIT, f'{EXECUTION}/completed.json', f'{EXECUTION}/summary.json', 'scripts/audit_chess_wldn_study.py']
    return {'protocol': PROTOCOL, 'sources': {p: source.file_hash(ROOT/p) for p in CODE},
            'inputs': {p: source.file_hash(ROOT/p) for p in inputs}, 'wldn_signature': study.signature()}


def deadline(begin):
    if time.time()-begin > PROTOCOL['time_cap_seconds']:
        raise TimeoutError('Frozen15-minute diagnostic/audit ceiling exceeded')


def prepare(out):
    result = signature(); result['prepared_unix'] = time.time()
    out.mkdir(parents=True, exist_ok=False); source.prior.write(out/'plan.json', result)
    print(json.dumps({'plan_sha256': source.file_hash(out/'plan.json'), 'new_predictions': 73728}), flush=True)


def aligned(records, rows):
    mapping = {r['id']: r for r in records}
    assert len(mapping) == len(records) == len(rows) == len({r['id'] for r in rows})
    assert set(mapping) == {r['id'] for r in rows}
    result = []
    for i, row in enumerate(rows):
        r = mapping[row['id']]
        assert r['index'] == i and r['game_id'] == row['game_id']
        assert type(r['correct']) is bool and r['correct'] == (r['choice'] == row['target_uci'])
        assert math.isfinite(r['target_nll']) and r['target_nll'] >= 0
        result.append(r)
    return result


def metrics(records):
    return {'agreement': statistics.mean(r['correct'] for r in records),
            'target_nll': statistics.mean(r['target_nll'] for r in records), 'examples': len(records)}


def paired(native, changed):
    assert len(native) == len(changed) and len(native) > 0
    for a, b in zip(native, changed):
        assert (a['index'], a['id'], a['game_id']) == (b['index'], b['id'], b['game_id'])
    return {'agreement_loss': statistics.mean(int(a['correct'])-int(b['correct']) for a, b in zip(native, changed)),
            'nll_increase': statistics.mean(b['target_nll']-a['target_nll'] for a, b in zip(native, changed)),
            'decision_disagreement': statistics.mean(a['choice'] != b['choice'] for a, b in zip(native, changed)),
            'native_correct_to_wrong': sum(a['correct'] and not b['correct'] for a, b in zip(native, changed)),
            'native_wrong_to_correct': sum(not a['correct'] and b['correct'] for a, b in zip(native, changed))}


def interval(difference, game_ids, split):
    differences = np.asarray(difference, dtype=float)
    assert differences.ndim == 2 and differences.shape[0] == 3 and differences.shape[1] == len(game_ids)
    assert np.isfinite(differences).all() and np.isin(differences, [-1, 0, 1]).all()
    value = differences.mean(0); ids = np.asarray(game_ids)
    groups = [np.flatnonzero(ids == g) for g in sorted(set(game_ids))]
    sums = np.array([value[g].sum() for g in groups]); sizes = np.array([len(g) for g in groups])
    sample = np.random.default_rng(975101+(split == 'shift')).integers(0, len(groups), (2000, len(groups)))
    return {'games': len(groups), 'point_loss': float(value.mean()),
            'percentile95': np.quantile(sums[sample].sum(1)/sizes[sample].sum(1), [.025, .975]).tolist()}


def analysis(directory):
    result = {'metrics': {}, 'paired': {}, 'conditional_game_bootstrap': [], 'panel_means': {}}
    for split in SPLITS:
        rows = source.prior.rows(ROOT/source.prior.DATA[split]); assert len(rows) == 2048
        saved = {}
        for seed in SEEDS:
            for mode in (*MODES, 'base'):
                records = aligned(source.prior.rows(directory/f'{mode}-{seed}-{split}.jsonl'), rows)
                saved[mode, seed] = records
                result['metrics'].setdefault(f'{mode}-{seed}', {})[split] = metrics(records)
            for mode in MODES[1:]:
                result['paired'].setdefault(f'{mode}-{seed}', {})[split] = paired(saved['native', seed], saved[mode, seed])
        result['panel_means'][split] = {mode: {key: statistics.mean(result['metrics'][f'{mode}-{seed}'][split][key] for seed in SEEDS)
                                             for key in ('agreement', 'target_nll')} for mode in (*MODES, 'base')}
        for mode in MODES[1:]:
            differences = [[int(a['correct'])-int(b['correct']) for a, b in zip(saved['native', seed], saved[mode, seed])] for seed in SEEDS]
            result['conditional_game_bootstrap'].append({'split': split, 'mode': mode, **interval(differences, [r['game_id'] for r in rows], split)})
    return result


def records_for(logits, targets, graph, rows, start):
    mask = graph['mask']; assert torch.isfinite(logits[mask]).all() and torch.isneginf(logits[~mask]).all()
    choices = logits.argmax(-1).tolist(); losses = F.cross_entropy(logits, targets, reduction='none').tolist()
    result = []
    for j, (choice, loss) in enumerate(zip(choices, losses)):
        row = rows[start+j]; name = graph['menus'][j][choice]
        assert math.isfinite(loss) and loss >= 0
        result.append({'index': start+j, 'id': row['id'], 'game_id': row['game_id'], 'choice': name,
                       'correct': name == row['target_uci'], 'target_nll': loss})
    return result


def compare_records(actual, expected):
    assert len(actual) == len(expected)
    error = 0.
    for a, b in zip(actual, expected):
        assert a.keys() == b.keys()
        assert all(a[k] == b[k] for k in a if k != 'target_nll')
        assert math.isfinite(b['target_nll'])
        error = max(error, abs(a['target_nll']-b['target_nll']))
    assert error <= PROTOCOL['nll_tolerance']
    return error


def native_inputs(rows, data):
    for i, row in enumerate(rows):
        board = chess.Board(row['fen']); names, features = source.prior.encode_candidates(board)
        assert torch.equal(data['observations'][i], torch.from_numpy(source.prior.encode_board(board)))
        assert torch.equal(data['candidates'][i, :len(names)], torch.from_numpy(features))
        assert data['mask'][i, :len(names)].all() and not data['mask'][i, len(names):].any()
        assert not data['candidates'][i, len(names):].count_nonzero()
        assert int(data['targets'][i]) == names.index(row['target_uci'])
        assert np.array_equal(data['edges'][i].numpy(), source.prior.root_relations(board))


@torch.no_grad()
def execute(plan_path, out, execution=None):
    plan = json.loads(plan_path.read_text()); assert all(plan[k] == v for k, v in signature().items())
    plan_hash = source.file_hash(plan_path); auditing = execution is not None
    if auditing:
        manifest(execution, plan_hash)
        summary = json.loads((execution/'summary.json').read_text())
        assert summary['status'] == 'completed' and summary['plan_sha256'] == plan_hash
        assert summary['new_prediction_records'] == 73728 and summary['copied_reference_records'] == 12288
        assert summary['new_training_updates'] == summary['new_engine_calls'] == summary['external_model_calls'] == 0
        assert summary['limits'] == PROTOCOL['limits'] and summary['original_failed_criteria_unchanged']
        assert 0 < summary['wall_seconds'] <= 900
    out.mkdir(parents=True, exist_ok=False); begin = time.time()
    source.prior.write(out/'started.json', {'unix': begin, 'pid': os.getpid(), 'plan_sha256': plan_hash, 'audit': auditing})
    try:
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        old_plan = json.loads((ROOT/source.PARENT).read_text()); checked = 0; references = []; max_error = 0.; native_error = 0.
        for split in SPLITS:
            rows = source.prior.rows(ROOT/source.prior.DATA[split]); assert len(rows) == 2048
            data = torch.load(ROOT/source.PREVIOUS/f'packed-{split}.pt', weights_only=True, map_location='cpu')
            graphs = source.ChildGraphCache(ROOT/source.CACHE/split)
            if auditing: native_inputs(rows, data)
            for seed in SEEDS:
                deadline(begin); model = source.prior.backbone(seed, old_plan); cache = source.prior.root_cache(model, data)
                head = study.load(ROOT/EXECUTION, seed, plan['inputs'][PLAN]); before = {k: v.clone() for k, v in head.state_dict().items()}
                original = aligned(source.prior.rows(ROOT/EXECUTION/f'wldn-{seed}-{split}.jsonl'), rows)
                record_map = {mode: [] for mode in MODES}
                filename = f'base-{seed}-{split}.jsonl'; origin = ROOT/EXECUTION/filename
                base_records = aligned(source.prior.rows(origin), rows)
                reference = {'file': filename, 'copied_from': str(origin.relative_to(ROOT)), 'sha256': source.file_hash(origin), 'fresh_inference': False}
                references.append(reference)
                if auditing:
                    assert source.file_hash(execution/filename) == reference['sha256']
                    expected = {mode: aligned(source.prior.rows(execution/f'{mode}-{seed}-{split}.jsonl'), rows) for mode in MODES}
                else:
                    with (out/filename).open('xb') as stream: stream.write(origin.read_bytes())
                for start in range(0, 2048, 128):
                    deadline(begin); index = torch.arange(start, start+128); graph = graphs.batch(index)
                    assert graph['fens'] == [r['fen'] for r in rows[start:start+128]]
                    args = source.arguments(model, data, cache, index, graph)
                    if auditing:
                        logits = {mode: intervention.hooked_reference(head, args, graph['permutation'], mode) for mode in MODES}
                        assert not head.difference._forward_pre_hooks and not head.readout._forward_pre_hooks
                        actual_base = records_for(args[4], data['targets'][index], graph, rows, start)
                        max_error = max(max_error, compare_records(actual_base, base_records[start:start+128])); checked += 128
                    else:
                        logits = intervention.scores(head, args, graph['permutation'])
                    for mode in MODES:
                        actual = records_for(logits[mode], data['targets'][index], graph, rows, start)
                        if mode == 'native': native_error = max(native_error, compare_records(actual, original[start:start+128]))
                        if auditing:
                            max_error = max(max_error, compare_records(actual, expected[mode][start:start+128])); checked += 128
                        else: record_map[mode].extend(actual)
                assert all(torch.equal(v, before[k]) for k, v in head.state_dict().items())
                if not auditing:
                    for mode, records in record_map.items():
                        assert len(records) == 2048
                        with (out/f'{mode}-{seed}-{split}.jsonl').open('x') as stream:
                            for record in records: stream.write(json.dumps(record, allow_nan=False)+'\n')
                        checked += len(records)
                print(json.dumps({'audit': auditing, 'panel': split, 'seed': seed, 'cumulative_predictions': checked}), flush=True)
                del head, model, cache; gc.collect()
            del data, graphs; gc.collect()
        result = analysis(execution if auditing else out); deadline(begin)
        if auditing:
            assert checked == 86016
            assert all(summary[k] == v for k, v in result.items())
            assert json.loads((execution/'reference-provenance.json').read_text()) == references
            assert summary['native_max_nll_error'] == native_error
            receipt = {'status': 'completed', 'plan_sha256': plan_hash, 'summary_sha256': source.file_hash(execution/'summary.json'),
                'execution_receipt_sha256': source.file_hash(execution/'completed.json'), 'auditor_sha256': source.file_hash(__file__),
                'new_predictions_replayed': 73728, 'copied_base_predictions_replayed': 12288, 'evaluation_predictions_replayed': checked,
                'original_native_predictions_checked': 12288, 'max_replay_nll_error': max_error, 'native_max_nll_error': native_error,
                'bootstrap_intervals_recomputed': 10, 'head_states_unchanged': True, 'temporary_hooks_removed': True,
                'wall_seconds': time.time()-begin, 'scope': PROTOCOL['audit']}
            source.prior.write(out/'receipt.json', receipt); print(json.dumps(receipt), flush=True)
        else:
            assert checked == 73728
            result.update({'status': 'completed', 'plan_sha256': plan_hash, 'new_prediction_records': checked,
                'copied_reference_records': 12288, 'native_max_nll_error': native_error,
                'new_training_updates': 0, 'new_engine_calls': 0, 'external_model_calls': 0,
                'original_failed_criteria_unchanged': True, 'wall_seconds': time.time()-begin, 'limits': PROTOCOL['limits']})
            source.prior.write(out/'reference-provenance.json', references); source.prior.write(out/'summary.json', result)
            source.prior.write(out/'completed.json', {'status': 'completed', 'plan_sha256': plan_hash,
                'files': {p.name: source.file_hash(p) for p in out.iterdir() if p.is_file()}})
            print(json.dumps({'status': 'completed', 'wall_seconds': result['wall_seconds'], 'native_max_nll_error': native_error}), flush=True)
    except BaseException as error:
        source.prior.write(out/'failed.json', {'status': 'failed', 'error': repr(error), 'wall_seconds': time.time()-begin})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'run', 'audit'])
    parser.add_argument('--plan', type=Path); parser.add_argument('--execution', type=Path)
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    if args.command == 'prepare': prepare(args.out)
    else: execute(args.plan, args.out, args.execution if args.command == 'audit' else None)
