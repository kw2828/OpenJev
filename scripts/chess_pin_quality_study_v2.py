# SPDX-License-Identifier: GPL-3.0-only
"""Versioned integer-game-ID repair; v1 partial training and frozen sources retained."""
import argparse
from contextlib import ExitStack
import gc
import json
import math
import os
from pathlib import Path
import platform
import time

import chess
import numpy as np
import torch

import chess_pin_update_profile as profile
import chess_pin_quality_analysis_v2 as analysis
import chess_union_difference_cost as union_cost
from openjev.research import chess_pin_study as mechanics
from openjev.research.chess_training_inputs import TrainingInputs

ROOT, source, original = profile.ROOT, profile.source, profile.original
selection = profile.selection
diagnostic = original.pins.diagnostic
VERSION = 'canonical-matched-pin-quality-v2'
PROFILE_PLAN = 'evidence/chess-pin-update-profile-v1/protocol/plan.json'
PROFILE_RUN = 'runs/chess-pin-update-profile-v1/execution'
PROFILE_AUDIT = 'evidence/chess-pin-update-profile-v1/audit/receipt.json'
SELECTED = 'evidence/chess-pin-comparator-selection-v1/selection/selection.json'
COST_PLAN = 'evidence/chess-union-difference-cost-v1/protocol/plan.json'
COST_RUN = 'runs/chess-union-difference-cost-v1/execution'
COST_AUDIT = 'evidence/chess-union-difference-cost-v1/audit/receipt.json'
FEATURES = 'runs/chess-native-training-cache-v1/execution'
PINS = 'runs/chess-pin-training-cache-v1/execution'
EVALUATION = 'runs/chess-native-evaluation-inputs-v1/execution'
OLD_PLAN = 'evidence/chess-pin-quality-v1/protocol/plan.json'
OLD_RUN = 'runs/chess-pin-quality-v1/execution'
OLD_REVIEW = 'evidence/chess-pin-quality-v1/failure-review.json'
CODE = ['scripts/chess_pin_quality_study_v2.py', 'tests/test_chess_pin_quality_study_v2.py',
        'scripts/chess_pin_quality_analysis_v2.py', 'tests/test_chess_pin_quality_analysis_v2.py']
SPLITS = ('dev', 'shift')
SEEDS = mechanics.SEEDS
SIZES = {'train': 32768, 'dev': 2048, 'shift': 2048}
CANDIDATES = {'train': 968036, 'dev': 60541, 'shift': 58946}
read = profile.read
write = source.prior.write


def protocol(arms, projected_seconds):
    arms = analysis.validate_arms(arms)
    if not math.isfinite(projected_seconds) or projected_seconds <= 0:
        raise ValueError('Invalid selected-arm cost projection')
    methods = ('base', *arms)
    return {
        'version': VERSION, 'arms': list(arms), 'methods': list(methods), 'seeds': list(SEEDS),
        'parameters': {a: mechanics.PARAMETERS[a] for a in arms},
        'hypothesis': 'Joint absolute-pin witness processing improves move agreement beyond the established graph-difference path, separable/pairwise factor computations, root-only/count/capacity controls and the pre-outcome-selected union comparator.',
        'prior_knowledge': 'All previous development results, including the failed union quality criterion and its full audit, are known. No joint pin quality fit or outcome has been inspected. Adaptive development, not independent confirmation.',
        'training_roots': 32768, 'epochs': 6, 'batch_size': 128, 'updates_per_fit': 1536,
        'fits': 3*len(arms), 'training_updates': 3*len(arms)*1536,
        'fit_order': 'Backbone seeds97,109,127; declared arm order within each seed. All final fits finish before reading evaluation labels or running quality evaluation.',
        'recipe': 'Frozen shared mechanics: head seed1100+backbone_seed; Adam lr.001 betas(.9,.999) eps1e-8 weight_decay0; gradient clip1; legal-menu cross entropy only. Batch permutation700000+100*seed+epoch. CPU float32,2 threads, deterministic algorithms.',
        'input_contract': 'Identical authenticated canonical one-position features, action features, complete native child graphs and absolute-pin factors for every arm. Map targets from source UCI into the exact cached legal menus. No child-backbone recomputation in training, shortlist, new positions, teacher calls or old fitted reference predictions.',
        'checkpoint_policy': 'Fresh initial state and final state only; every minibatch index/loss/gradient-norm receipt retained. No validation peek, best-checkpoint selection, resume, seed replacement or timeout extension.',
        'evaluation_roots_per_panel': 2048, 'prediction_records': 3*len(methods)*4096,
        'candidate_scores': 3*len(methods)*119487, 'native_root_backbone_reconstructions': 12288,
        'evaluation': 'Every declared method, seed and complete old panel. Save all finite legal score vectors and cached choices/NLL. The frozen backbone is freshly evaluated under the same canonical contract.',
        'native': 'For each root/backbone reconstruct FEN, sorted legal menu, single-position backbone features, all native child graphs and pin witnesses without feature/graph caches. Reuse this native input only across heads for that root. Compare every legal score and choice to cached evaluation. This is full equivalence checking, not native latency timing.',
        'score_tolerance': 1e-5,
        'numerical_gate': 'All native/cached score differences<=1e-5 and every chosen move identical. Retain every finite failure, complete remaining evaluation and report a failed numerical gate. Nonfinite/identity/structural failures stop the run with a failure receipt.',
        'quality_gate': 'Joint-minus-base mean>=0 and joint-minus-every-trained-comparator mean>=.01 on EACH panel. Every paired-seed gain>=-.005 in every comparison. Require all checks; do not relabel a stronger control as the primary treatment.',
        'quality_checks': 2*len(arms),
        'continuation': 'Both full numerical gate and all quality checks, plus the complete evidence audit, required before advancing the candidate. Earlier failed study criteria remain unchanged.',
        'bootstrap': '2000 paired source-game draws; average paired correctness across all3 seeds within each root, then position-weight sampled games. Seed996101+shift_indicator. Conditional descriptive95% intervals; no seed-population uncertainty, adaptive or multiple-testing correction.',
        'audit': 'Authenticate full source/input and execution manifests; check every initial/final checkpoint, all training-index receipts and phase ordering. Recompute every cached score/NLL exactly and every native vector/comparison exactly with the same production kernels. Recompute metrics, all gates and all intervals. Existing independent input/packing audits are authenticated; this audit is not independent neural code or full retraining.',
        'time_cap_seconds': 43200, 'audit_time_cap_seconds': 7200,
        'projected_selected_arm_update_seconds': projected_seconds,
        'budget_rationale': 'Frozen twelve-hour execution ceiling provides headroom beyond the audited short-prefix selected-arm update estimate for corpus variation, setup, saving and complete native evaluation. Separate two-hour audit ceiling. Profile is not a long-run guarantee; no extension after timeout.',
        'limits': 'Existing conditional set-encoder and graph-difference ingredients; no architecture novelty follows from a named motif. Pairwise restriction applies only inside the anchored factor MLP, not to all policy interactions. Stored parameter counts do not equalize active capacity, FLOPs or optimization. No gameplay, new engine/model calls, independent confirmation or speed claim.',
    }


def prior_failure():
    old = read(ROOT/OLD_PLAN); stopped = read(ROOT/OLD_REVIEW); directory = ROOT/OLD_RUN
    files = {str(f.relative_to(directory)):source.file_hash(f) for f in directory.rglob('*') if f.is_file()}
    failed = read(directory/'failed.json')
    if (stopped['status'] != 'stopped_input_schema_mismatch' or stopped['plan_sha256'] != source.file_hash(ROOT/OLD_PLAN)
            or stopped['failed_receipt_sha256'] != source.file_hash(directory/'failed.json')
            or files != stopped['partial_files_sha256'] or failed['error'] != 'KeyboardInterrupt()'
            or stopped['wall_seconds'] != failed['wall_seconds']
            or not 0 < failed['wall_seconds'] <= old['protocol']['time_cap_seconds']
            or stopped['completed_fits'] != 0 or stopped['quality_predictions'] != 0
            or not stopped['evaluation_outputs_absent'] or not stopped['frozen_sources_unchanged']
            or any(source.file_hash(ROOT/p) != h for p,h in old['sources'].items())
            or any((directory/n).exists() for n in ('completed.json','all-training-complete.json','predictions','native'))
            or {f.name for f in (directory/'fits').iterdir()} != {'wldn-97'}):
        raise ValueError('Stopped v1 evidence changed')
    logs = source.prior.rows(directory/'fits/wldn-97/learning.jsonl')
    if not 0 < len(logs) == stopped['logged_completed_updates'] < old['protocol']['updates_per_fit']:
        raise ValueError('Stopped v1 update prefix differs')
    order = mechanics.schedule(32768,6,128,97)
    for i, (r,(epoch,indices)) in enumerate(zip(logs,order),1):
        if (r['update'] != i or r['epoch'] != epoch or r['examples'] != 128
                or r['indices_sha256'] != mechanics.index_digest(indices)
                or not math.isfinite(r['loss']) or not math.isfinite(r['gradient_norm'])):
            raise ValueError('Stopped v1 training prefix differs')
    return old, stopped


def historical_game_metadata():
    # These are already-audited old predictions used by comparator selection.
    # Validate identity schema before fitting; do not run any new policy evaluation.
    result = {}
    for split in SPLITS:
        path = ROOT/selection.RUN/f'base-97-{split}.jsonl'
        rows = source.prior.rows(path)
        if len(rows) != 2048: raise ValueError('Incomplete historical source-game metadata')
        ids = [r['game_id'] for r in rows]; analysis.validate_game_ids(ids)
        result[split] = {'records':len(rows),'game_id_type':type(ids[0]).__name__,
                         'games':len(set(ids)),'historical_predictions_sha256':source.file_hash(path)}
    return result


def signature():
    old, stopped = prior_failure()
    frozen = read(ROOT/PROFILE_PLAN)
    if frozen != profile.signature(): raise ValueError('Frozen profile input/source chain changed')
    complete = diagnostic.manifest(ROOT/PROFILE_RUN, source.file_hash(ROOT/PROFILE_PLAN))
    prof = read(ROOT/PROFILE_RUN/'summary.json'); pa = read(ROOT/PROFILE_AUDIT)
    if (pa['status'] != 'passed' or pa['plan_sha256'] != source.file_hash(ROOT/PROFILE_PLAN)
            or pa['completed_sha256'] != source.file_hash(ROOT/PROFILE_RUN/'completed.json')
            or pa['summary_sha256'] != complete['files']['summary.json']
            or pa['replay_updates_sha256'] != source.file_hash(ROOT/Path(PROFILE_AUDIT).parent/'updates.jsonl')
            or pa['initial_and_final_states_exact'] != 66 or pa['update_values_and_indices_exact'] != 99
            or pa['head_instances'] != 33 or pa['artificial_updates'] != 99
            or not pa['timing_summaries_recomputed'] or pa['timing_replication_claimed']
            or not 0 < pa['wall_seconds'] <= 900 or not 0 < prof['wall_seconds'] <= 900
            or prof['status'] != 'completed' or prof['plan_sha256'] != source.file_hash(ROOT/PROFILE_PLAN)):
        raise ValueError('Incomplete update-profile replay')
    stats = profile.summarize(source.prior.rows(ROOT/PROFILE_RUN/'updates.jsonl'))
    if any(prof[k] != v for k,v in stats.items()): raise ValueError('Profile aggregation changed')
    selected = read(ROOT/SELECTED); previous, audit = selection.audited_outcomes()
    expected = mechanics.select_comparators(previous['metrics'])
    rule = read(ROOT/profile.SELECTION); stamp = read(ROOT/selection.RUN/'all-training-complete.json')
    if (selected['status'] != 'completed' or any(selected[k] != v for k,v in expected.items())
            or selected['rule_plan_sha256'] != source.file_hash(ROOT/profile.SELECTION)
            or selected['union_audit_sha256'] != source.file_hash(ROOT/selection.AUDIT)
            or selected['union_summary_sha256'] != audit['summary_sha256']
            or selected['union_gate_passed'] != previous['continuation_passed']
            or selected['future_fit_count'] != 3*len(expected['fit_arms'])
            or selected['quality_training_launched'] is not False
            or not 0 < rule['prepared_unix'] < stamp['unix']
            or not 0 < rule['completed_fits_at_freeze'] < 12 or not rule['evaluation_outputs_absent_at_freeze']):
        raise ValueError('Pre-outcome comparator selection identity changed')
    cp = read(ROOT/COST_PLAN)
    if not all(cp[k] == v for k,v in union_cost.signature().items()): raise ValueError('Union cost sources changed')
    diagnostic.manifest(ROOT/COST_RUN, source.file_hash(ROOT/COST_PLAN))
    ca = read(ROOT/COST_AUDIT); cs = read(ROOT/COST_RUN/'summary.json')
    if (ca['status'] != 'completed' or ca['plan_sha256'] != source.file_hash(ROOT/COST_PLAN)
            or ca['summary_sha256'] != source.file_hash(ROOT/COST_RUN/'summary.json')
            or ca['execution_receipt_sha256'] != source.file_hash(ROOT/COST_RUN/'completed.json')
            or ca['auditor_sha256'] != source.file_hash(ROOT/'scripts/chess_union_difference_cost.py')
            or ca['native_choice_replays'] != 288 or ca['timing_records_checked'] != 1728
            or not 0 < ca['wall_seconds'] <= 900 or not 0 < cs['wall_seconds'] <= 900
            or cs['inputs'][selection.AUDIT] != source.file_hash(ROOT/selection.AUDIT)):
        raise ValueError('Union native-cost evidence changed')
    arms = expected['fit_arms']; projected = sum(stats['projected_seconds_per_three_seed_arm'][a] for a in arms)
    repaired = protocol(arms,projected)
    if any(repaired[k] != v for k,v in old['protocol'].items() if k not in ('version','prior_knowledge')):
        raise ValueError('Scientific protocol changed in schema repair')
    repaired['prior_knowledge'] += (' The v1 attempt was stopped after '+str(stopped['logged_completed_updates'])+
        ' logged WLDN-97 updates because its string-only game-ID validator rejects integer corpus IDs. No fit completed or quality prediction was produced. No partial weights are reused; all24 fits restart fresh. The discarded attempt cost remains retained separately.')
    repaired['game_id_contract'] = 'Preserve homogeneous JSON integer or nonempty string source-game IDs exactly, including integer zero; reject booleans, floats and mixed types. Check authenticated historical metadata before training. No string coercion or regrouping.'
    metadata = historical_game_metadata()
    if metadata != stopped['historical_game_metadata']: raise ValueError('Historical game metadata changed')
    return {'protocol': repaired, 'sources': {p: source.file_hash(ROOT/p) for p in CODE},
            'prior_failed_plan_sha256':source.file_hash(ROOT/OLD_PLAN),
            'prior_failure_review_sha256':source.file_hash(ROOT/OLD_REVIEW),
            'historical_game_metadata':metadata,
            'profile_signature': frozen, 'profile_plan_sha256': source.file_hash(ROOT/PROFILE_PLAN),
            'profile_audit_sha256': source.file_hash(ROOT/PROFILE_AUDIT),
            'selection_sha256': source.file_hash(ROOT/SELECTED),
            'union_cost_plan_sha256': source.file_hash(ROOT/COST_PLAN), 'union_cost_audit_sha256': source.file_hash(ROOT/COST_AUDIT),
            'source_rows_sha256': {s: source.file_hash(ROOT/source.prior.DATA[s]) for s in SIZES},
            'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                        'numpy': np.__version__, 'python_chess': chess.__version__, 'threads': 2, 'device': 'cpu', 'dtype': 'float32'}}


def deadline(begin, p, audit=False):
    if time.monotonic()-begin > p['audit_time_cap_seconds' if audit else 'time_cap_seconds']:
        raise TimeoutError('Frozen pin-quality execution/audit time ceiling exceeded')


def name(arm, seed): return f'{arm.replace(":", "_")}-{seed}'


def load_inputs(split, seed):
    stop = SIZES[split]
    feature_dir = ROOT/FEATURES/str(seed) if split == 'train' else ROOT/EVALUATION/split/str(seed)
    pin_dir = ROOT/PINS/'blocks' if split == 'train' else ROOT/EVALUATION/split/'pins'
    features = [torch.load(feature_dir/f'{i:05d}.pt', weights_only=True, map_location='cpu') for i in range(0, stop, 128)]
    pins = [torch.load(pin_dir/f'{i:05d}.pt', weights_only=True, map_location='cpu') for i in range(0, stop, 128)]
    data = TrainingInputs(features, pins)
    if len(data.features['menus']) != stop or int(data.features['offsets'][-1]) != CANDIDATES[split]:
        raise AssertionError('Canonical corpus membership changed')
    return data


def targets_for(rows, data, graphs):
    if (len(rows) != len(data.features['menus']) or len(rows) != len(graphs.records)
            or len({r['id'] for r in rows}) != len(rows)):
        raise AssertionError('Source/features/graph membership differs')
    targets = []
    for i, (r, g, fen, menu) in enumerate(zip(rows, graphs.records, data.features['fens'], data.features['menus'])):
        if (g['index'] != i or r['fen'] != g['fen'] or chess.Board(r['fen']).fen(en_passant='fen') != fen
                or tuple(g['moves']) != menu or r['target_uci'] not in menu):
            raise AssertionError('Teacher target or canonical legal-menu identity differs')
        targets.append(menu.index(r['target_uci']))
    return torch.tensor(targets, dtype=torch.long)


def train_fit(seed, arm, data, graphs, targets, directory, plan_hash, p, begin):
    directory.mkdir(); started = time.time(); tick = time.monotonic()
    head = mechanics.make_head(arm, seed); opt = mechanics.optimizer(head)
    source.prior.save(directory/'initial.pt', head.state_dict())
    updates = 0
    with (directory/'learning.jsonl').open('x') as stream:
        for epoch, indices in mechanics.schedule(p['training_roots'], p['epochs'], p['batch_size'], seed):
            deadline(begin, p); args, factors = data.batch(indices, graphs.batch(indices))
            values = mechanics.update(head, arm, opt, args, factors, targets[indices]); updates += 1
            record = {'update': updates, 'epoch': epoch, 'examples': len(indices),
                      'indices_sha256': mechanics.index_digest(indices), **values}
            stream.write(json.dumps(record, allow_nan=False)+'\n'); stream.flush()
            if updates % (p['training_roots']//p['batch_size']) == 0:
                print(json.dumps({'fit': name(arm,seed), 'epoch': epoch+1, 'updates': updates,
                                  'wall_seconds': time.monotonic()-begin}), flush=True)
    deadline(begin, p)
    if updates != p['updates_per_fit']: raise AssertionError('Incomplete training schedule')
    source.prior.save(directory/'weights.pt', {'version': VERSION, 'seed': seed, 'arm': arm,
                      'plan_sha256': plan_hash, 'state_dict': head.state_dict()})
    write(directory/'training.json', {'status': 'completed', 'seed': seed, 'arm': arm, 'plan_sha256': plan_hash,
          'updates': updates, 'examples_seen': updates*p['batch_size'], 'parameters': mechanics.PARAMETERS[arm],
          'started_unix': started, 'finished_unix': time.time(), 'seconds': time.monotonic()-tick,
          'targets_sha256': mechanics.index_digest(targets),
          **{label: source.file_hash(directory/file) for label,file in
             [('initial_sha256','initial.pt'),('weights_sha256','weights.pt'),('learning_sha256','learning.jsonl')]}})


def load_head(execution, seed, arm, plan_hash):
    value = torch.load(execution/'fits'/name(arm,seed)/'weights.pt', weights_only=True, map_location='cpu')
    if (value['version'] != VERSION or value['seed'] != seed or value['arm'] != arm or value['plan_sha256'] != plan_hash):
        raise AssertionError('Final checkpoint identity differs')
    head = mechanics.make_head(arm, seed); head.load_state_dict(value['state_dict'])
    if any(not torch.isfinite(v).all() for v in head.state_dict().values()): raise AssertionError('Nonfinite final state')
    return head.eval()


def check_learning(records, seed, p):
    if len(records) != p['updates_per_fit']: raise AssertionError('Missing training updates')
    schedule = mechanics.schedule(p['training_roots'], p['epochs'], p['batch_size'], seed)
    for update, ((epoch, indices), r) in enumerate(zip(schedule, records), 1):
        if (r['update'] != update or r['epoch'] != epoch or r['examples'] != p['batch_size']
                or r['indices_sha256'] != mechanics.index_digest(indices)
                or any(not math.isfinite(r[k]) for k in ('loss','gradient_norm','update_seconds'))
                or r['loss'] < 0 or r['gradient_norm'] < 0 or r['update_seconds'] <= 0):
            raise AssertionError('Training batch receipt differs')
    if update != p['updates_per_fit']: raise AssertionError('Protocol schedule count differs')


def training_inventory(execution, p, plan_hash, *, replay_states=False):
    expected = [name(a,s) for s in p['seeds'] for a in p['arms']]
    if (p['fits'] != len(expected) or p['training_updates'] != len(expected)*p['updates_per_fit']
            or {x.name for x in (execution/'fits').iterdir()} != set(expected)):
        raise AssertionError('Incomplete or extra fitted arms')
    identities = {}; previous_end = -math.inf
    for seed in p['seeds']:
        for arm in p['arms']:
            directory = execution/'fits'/name(arm,seed); meta = read(directory/'training.json')
            if (meta['status'] != 'completed' or meta['seed'] != seed or meta['arm'] != arm
                    or meta['plan_sha256'] != plan_hash or meta['updates'] != p['updates_per_fit']
                    or meta['examples_seen'] != p['updates_per_fit']*p['batch_size']
                    or meta['parameters'] != mechanics.PARAMETERS[arm]
                    or any(not math.isfinite(meta[k]) or meta[k] <= 0 for k in ('seconds','started_unix','finished_unix'))
                    or not previous_end <= meta['started_unix'] <= meta['finished_unix']):
                raise AssertionError('Training membership, timing or update count differs')
            previous_end = meta['finished_unix']
            for key, file in [('initial_sha256','initial.pt'),('weights_sha256','weights.pt'),('learning_sha256','learning.jsonl')]:
                if source.file_hash(directory/file) != meta[key]: raise AssertionError('Training artifact changed')
            if replay_states:
                initial = torch.load(directory/'initial.pt', weights_only=True, map_location='cpu')
                original.pins.compare_states(initial, mechanics.make_head(arm,seed).state_dict(), 0)
                head = load_head(execution,seed,arm,plan_hash)
                if initial['output.weight'].count_nonzero() or torch.equal(head.output.weight, initial['output.weight']):
                    raise AssertionError('Head did not start from base and update')
                check_learning(source.prior.rows(directory/'learning.jsonl'), seed, p)
            identities[name(arm,seed)] = source.file_hash(directory/'training.json')
    return identities, previous_end


def complete_training(execution, p, plan_hash):
    if (execution/'predictions').exists() or (execution/'native').exists() or (execution/'evaluation-started.json').exists():
        raise AssertionError('Evaluation exists before all training completes')
    identities, last_end = training_inventory(execution,p,plan_hash)
    now = time.time()
    if now < last_end: raise AssertionError('Invalid training phase clock')
    stamp = {'plan_sha256': plan_hash, 'fits': p['fits'], 'updates': p['training_updates'],
             'before_any_evaluation': True, 'unix': now, 'fit_receipts_sha256': identities}
    write(execution/'all-training-complete.json', stamp)
    return stamp


def evaluation_barrier(execution, p, plan_hash):
    stamp = read(execution/'all-training-complete.json')
    identities, last_end = training_inventory(execution,p,plan_hash)
    if (stamp['plan_sha256'] != plan_hash or stamp['fits'] != p['fits'] or stamp['updates'] != p['training_updates']
            or not stamp['before_any_evaluation'] or stamp['fit_receipts_sha256'] != identities or stamp['unix'] < last_end):
        raise AssertionError('Invalid all-training-complete barrier')
    return stamp


def prediction_records(logits, targets, graph, rows, start):
    records = diagnostic.records_for(logits,targets,graph,rows,start)
    for j,r in enumerate(records):
        r['menus'] = list(graph['menus'][j]); r['scores'] = logits[j,graph['mask'][j]].tolist()
    return records


def vector_comparison(cached, native, menus, tolerance):
    a, b = torch.tensor(cached,dtype=torch.double), torch.tensor(native,dtype=torch.double)
    if (not menus or len(set(menus)) != len(menus) or tuple(sorted(menus)) != tuple(menus)
            or a.shape != (len(menus),) or b.shape != a.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all()):
        raise AssertionError('Invalid complete legal score vector')
    error = float((a-b).abs().max())
    cached_choice, native_choice = menus[int(a.argmax())], menus[int(b.argmax())]
    return {'max_score_error': error, 'score_tolerance_passed': error <= tolerance,
            'cached_choice': cached_choice, 'native_choice': native_choice, 'choice_matches': cached_choice == native_choice}


@torch.no_grad()
def native_arguments(model, fen):
    board = chess.Board(fen)
    if not board.is_valid() or board.is_game_over(claim_draw=False): raise AssertionError('Invalid native root')
    names, encoded = source.prior.encode_candidates(board)
    candidates = torch.from_numpy(encoded)[None]; mask = torch.ones(1,len(names),dtype=torch.bool)
    base, _, hidden = model(torch.from_numpy(source.prior.encode_board(board))[None],candidates,mask)
    nodes = hidden.flatten(2).transpose(1,2); actions = source.prior.candidate_features(model,nodes,candidates)
    graph = source.candidate_graphs([board]); pins = original.candidate_factors(board)
    if (list(names) != graph['menus'][0] or list(names) != [r['uci'] for r in pins['candidates']]
            or not torch.equal(mask,graph['mask'])):
        raise AssertionError('Native menus disagree')
    return list(names), [nodes,actions,candidates,mask,base,graph['root'],graph['children'][mask]], original.pack_factors([pins])


def numerical_summary(records, p):
    expected = [(split,seed,i) for split in SPLITS for seed in p['seeds'] for i in range(p['evaluation_roots_per_panel'])]
    if [(r['split'],r['seed'],r['index']) for r in records] != expected:
        raise AssertionError('Incomplete native comparison membership')
    maximum = 0.; failed = changed = scores = comparisons = 0
    by_method = {a: {'max_score_error': 0., 'failed_cases': 0, 'choice_changes': 0} for a in p['methods']}
    for r in records:
        if set(r['comparisons']) != set(p['methods']) or r['candidates'] <= 0: raise AssertionError('Incomplete method coverage')
        for arm,c in r['comparisons'].items():
            if (not math.isfinite(c['max_score_error']) or c['max_score_error'] < 0
                    or c['score_tolerance_passed'] != (c['max_score_error'] <= p['score_tolerance'])
                    or c['choice_matches'] != (c['cached_choice'] == c['native_choice'])):
                raise AssertionError('Invalid numerical comparison')
            bad = not (c['score_tolerance_passed'] and c['choice_matches'])
            comparisons += 1; scores += r['candidates']; failed += bad; changed += not c['choice_matches']
            maximum = max(maximum,c['max_score_error']); group = by_method[arm]
            group['max_score_error'] = max(group['max_score_error'],c['max_score_error'])
            group['failed_cases'] += bad; group['choice_changes'] += not c['choice_matches']
    if scores != p['candidate_scores'] or comparisons != p['prediction_records'] or len(records) != p['native_root_backbone_reconstructions']:
        raise AssertionError('Native score/record budget differs')
    return {'native_root_backbone_reconstructions': len(records), 'native_method_comparisons': comparisons,
            'candidate_score_comparisons': scores, 'max_native_cached_score_error': maximum,
            'failed_native_method_cases': failed, 'native_choice_changes': changed,
            'numerical_gate_passed': failed == 0, 'native_by_method': by_method}


def append(stream, value):
    stream.write(json.dumps(value,allow_nan=False)+'\n')


@torch.no_grad()
def evaluate_loaded(data, graphs, targets, rows, heads, model, seed, split, execution, out, p, begin, audit=False):
    if list(heads) != p['methods'] or (heads['base'] is not None): raise AssertionError('Wrong evaluation methods')
    before_model = {k:v.clone() for k,v in model.state_dict().items()}
    before_heads = {a:{k:v.clone() for k,v in h.state_dict().items()} for a,h in heads.items() if h is not None}
    captured = {a:[] for a in heads}
    expected = {a:source.prior.rows(execution/'predictions'/f'{name(a,seed)}-{split}.jsonl') for a in heads} if audit else None
    with ExitStack() as stack:
        streams = {} if audit else {a:stack.enter_context((out/'predictions'/f'{name(a,seed)}-{split}.jsonl').open('x')) for a in heads}
        for start in range(0,len(rows),p['batch_size']):
            deadline(begin,p,audit); index = torch.arange(start,min(start+p['batch_size'],len(rows)))
            graph = graphs.batch(index); args, factors = data.batch(index,graph)
            for arm,head in heads.items():
                scores = args[4] if head is None else mechanics.logits(head,arm,args,factors)
                values = prediction_records(scores,targets[index],graph,rows,start)
                if audit:
                    if values != expected[arm][start:start+len(index)]: raise AssertionError('Cached score/NLL replay differs')
                else:
                    for r in values: append(streams[arm],r)
                    streams[arm].flush()
                captured[arm].extend(values)
    if audit and any(len(expected[a]) != len(rows) for a in heads): raise AssertionError('Extra saved predictions')
    expected_native = source.prior.rows(execution/'native'/f'{seed}-{split}.jsonl') if audit else None
    if audit and len(expected_native) != len(rows): raise AssertionError('Native root membership differs')
    checks = []
    with (out/'native'/f'{seed}-{split}.jsonl').open('x') as stream:
        for i,row in enumerate(rows):
            deadline(begin,p,audit); menus, args, factors = native_arguments(model,row['fen'])
            native_scores = {}; comparisons = {}
            for arm,head in heads.items():
                scores = args[4] if head is None else mechanics.logits(head,arm,args,factors)
                native_scores[arm] = scores[0].tolist(); saved = captured[arm][i]
                if saved['menus'] != menus: raise AssertionError('Native/cached legal menus differ')
                comparisons[arm] = vector_comparison(saved['scores'],native_scores[arm],menus,p['score_tolerance'])
                if comparisons[arm]['cached_choice'] != saved['choice']: raise AssertionError('Cached choice/vector mismatch')
            item = {'split':split,'seed':seed,'index':i,'id':row['id'],'game_id':row['game_id'],
                    'fen':row['fen'],'menus':menus,'scores':native_scores,'comparisons':comparisons}
            if audit and item != expected_native[i]: raise AssertionError('Native score/comparison replay differs')
            # The audit keeps full fresh native vectors as an inspectable replay, not just counters.
            append(stream,item)
            if (i+1)%128 == 0: stream.flush()
            checks.append({'split':split,'seed':seed,'index':i,'candidates':len(menus),'comparisons':comparisons})
    original.pins.compare_states(before_model,model.state_dict(),0)
    for arm,old in before_heads.items(): original.pins.compare_states(old,heads[arm].state_dict(),0)
    return checks


def evaluate_all(execution, out, p, plan_hash, begin, audit=False):
    stamp = evaluation_barrier(execution,p,plan_hash)
    if not audit:
        if time.time() < stamp['unix']: raise AssertionError('Evaluation phase precedes training')
        write(out/'evaluation-started.json', {'unix':time.time(),'plan_sha256':plan_hash,
              'training_stamp_sha256':source.file_hash(execution/'all-training-complete.json')})
        (out/'predictions').mkdir()
    (out/'native').mkdir()
    checks = []
    for split in SPLITS:
        rows = source.prior.rows(ROOT/source.prior.DATA[split])
        if len(rows) != p['evaluation_roots_per_panel']: raise AssertionError('Wrong complete evaluation panel')
        graphs = source.ChildGraphCache(ROOT/source.CACHE/split)
        for seed in p['seeds']:
            deadline(begin,p,audit); data = load_inputs(split,seed); targets = targets_for(rows,data,graphs)
            model = source.prior.backbone(seed,read(ROOT/source.PARENT))
            heads = {'base':None, **{a:load_head(execution,seed,a,plan_hash) for a in p['arms']}}
            checks.extend(evaluate_loaded(data,graphs,targets,rows,heads,model,seed,split,execution,out,p,begin,audit))
            print(json.dumps({'panel':split,'seed':seed,'native_roots_checked':len(checks),'audit':audit}),flush=True)
            del data,targets,model,heads; gc.collect()
        del graphs,rows; gc.collect()
    deadline(begin,p,audit)
    return numerical_summary(checks,p)


def evaluation_membership(execution, p):
    predictions = {f'{name(a,s)}-{split}.jsonl' for split in SPLITS for s in p['seeds'] for a in p['methods']}
    native = {f'{s}-{split}.jsonl' for split in SPLITS for s in p['seeds']}
    for directory, expected in [('predictions',predictions),('native',native)]:
        found = list((execution/directory).iterdir())
        if {f.name for f in found} != expected or any(not f.is_file() for f in found):
            raise AssertionError('Missing or extra evaluation files')


def analyze(execution, p):
    evaluation_membership(execution,p)
    rows = {s:source.prior.rows(ROOT/source.prior.DATA[s]) for s in SPLITS}
    records = {}
    for split in SPLITS:
        for seed in p['seeds']:
            for arm in p['methods']:
                full = source.prior.rows(execution/'predictions'/f'{name(arm,seed)}-{split}.jsonl')
                records[arm,seed,split] = [{k:v for k,v in r.items() if k not in ('menus','scores')} for r in full]
    result = analysis.analyze(records,rows,p['arms'])
    if result['prediction_records'] != p['prediction_records'] or len(result['gate_checks']) != p['quality_checks']:
        raise AssertionError('Quality analysis coverage differs')
    return result


def validate_plan(plan_path):
    plan = read(plan_path); current = signature()
    if not all(plan[k] == v for k,v in current.items()): raise ValueError('Frozen pin-study signature changed')
    if not 0 < plan['prepared_unix'] <= time.time(): raise ValueError('Invalid protocol timestamp')
    return plan, source.file_hash(plan_path)


def configure():
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    if torch.get_default_dtype() != torch.float32: raise ValueError('Expected default CPU float32 tensors')


def run(plan_path, out):
    begin = time.monotonic(); plan, plan_hash = validate_plan(plan_path); p = plan['protocol']
    out.mkdir(parents=True,exist_ok=False)
    write(out/'started.json',{'unix':time.time(),'pid':os.getpid(),'plan_sha256':plan_hash})
    try:
        configure(); (out/'fits').mkdir()
        rows = source.prior.rows(ROOT/source.prior.DATA['train'])
        if len(rows) != p['training_roots']: raise AssertionError('Wrong complete training set')
        graphs = source.ChildGraphCache(ROOT/source.CACHE/'train'); target_digests = {}
        for seed in p['seeds']:
            deadline(begin,p); data = load_inputs('train',seed); targets = targets_for(rows,data,graphs)
            target_digests[str(seed)] = mechanics.index_digest(targets)
            if len(set(target_digests.values())) != 1: raise AssertionError('Targets differ between backbones')
            for arm in p['arms']:
                train_fit(seed,arm,data,graphs,targets,out/'fits'/name(arm,seed),plan_hash,p,begin)
            del data,targets; gc.collect()
        del rows,graphs; gc.collect()
        write(out/'training-targets.json',target_digests); complete_training(out,p,plan_hash)
        numerical = evaluate_all(out,out,p,plan_hash,begin); quality = analyze(out,p); deadline(begin,p)
        result = {'status':'completed','plan_sha256':plan_hash,'fits':p['fits'],'training_updates':p['training_updates'],
                  **quality,**numerical,'continuation_passed':quality['quality_gate_passed'] and numerical['numerical_gate_passed'],
                  'new_engine_calls':0,'external_model_calls':0,'copied_reference_predictions':0,
                  'original_failed_criteria_unchanged':True,'wall_seconds':time.monotonic()-begin,'limits':p['limits']}
        write(out/'summary.json',result)
        write(out/'completed.json',{'status':'completed','plan_sha256':plan_hash,
              'files':{str(f.relative_to(out)):source.file_hash(f) for f in out.rglob('*') if f.is_file()}})
        print(json.dumps({'status':'completed','quality_gate_passed':quality['quality_gate_passed'],
                          'numerical_gate_passed':numerical['numerical_gate_passed'],'wall_seconds':result['wall_seconds']}),flush=True)
    except BaseException as error:
        write(out/'failed.json',{'status':'failed','error':repr(error),'wall_seconds':time.monotonic()-begin}); raise


def audit(plan_path, execution, out):
    begin = time.monotonic(); plan, plan_hash = validate_plan(plan_path); p = plan['protocol']
    diagnostic.manifest(execution,plan_hash); saved = read(execution/'summary.json')
    out.mkdir(parents=True,exist_ok=False)
    write(out/'started.json',{'unix':time.time(),'pid':os.getpid(),'plan_sha256':plan_hash,'audit':True})
    try:
        configure()
        if (saved['status'] != 'completed' or saved['plan_sha256'] != plan_hash or saved['fits'] != p['fits']
                or saved['training_updates'] != p['training_updates'] or saved['prediction_records'] != p['prediction_records']
                or saved['new_engine_calls'] != 0 or saved['external_model_calls'] != 0 or saved['copied_reference_predictions'] != 0
                or not saved['original_failed_criteria_unchanged'] or saved['limits'] != p['limits']
                or not 0 < saved['wall_seconds'] <= p['time_cap_seconds']):
            raise AssertionError('Execution scope or budget differs')
        training_inventory(execution,p,plan_hash,replay_states=True)
        stamp = evaluation_barrier(execution,p,plan_hash)
        phase = read(execution/'evaluation-started.json'); started = read(execution/'started.json')
        if (phase['training_stamp_sha256'] != source.file_hash(execution/'all-training-complete.json')
                or phase['plan_sha256'] != plan_hash or not plan['prepared_unix'] <= started['unix'] <= stamp['unix'] <= phase['unix']):
            raise AssertionError('Training/evaluation phase ordering differs')
        for directory in ('predictions','native'):
            for f in (execution/directory).iterdir():
                if f.stat().st_mtime < phase['unix']: raise AssertionError('Evaluation output predates training barrier')
        evaluation_membership(execution,p)
        target_digests = read(execution/'training-targets.json')
        if set(target_digests) != {str(s) for s in p['seeds']}: raise AssertionError('Training target seed membership differs')
        rows = source.prior.rows(ROOT/source.prior.DATA['train']); graphs = source.ChildGraphCache(ROOT/source.CACHE/'train')
        for seed in p['seeds']:
            deadline(begin,p,True); data = load_inputs('train',seed); targets = targets_for(rows,data,graphs)
            digest = mechanics.index_digest(targets)
            if target_digests[str(seed)] != digest: raise AssertionError('Training target mapping differs')
            for arm in p['arms']:
                if read(execution/'fits'/name(arm,seed)/'training.json')['targets_sha256'] != digest:
                    raise AssertionError('Per-fit target mapping differs')
            del data,targets; gc.collect()
        del rows,graphs; gc.collect()
        numerical = evaluate_all(execution,out,p,plan_hash,begin,audit=True); quality = analyze(execution,p)
        if (any(saved[k] != v for k,v in {**quality,**numerical}.items())
                or saved['continuation_passed'] != (quality['quality_gate_passed'] and numerical['numerical_gate_passed'])):
            raise AssertionError('Recomputed quality/numerical summary differs')
        deadline(begin,p,True)
        result = {'status':'completed','plan_sha256':plan_hash,'summary_sha256':source.file_hash(execution/'summary.json'),
                  'execution_receipt_sha256':source.file_hash(execution/'completed.json'),'auditor_sha256':source.file_hash(__file__),
                  'training_updates_checked':p['training_updates'],'fresh_initial_states_exact':p['fits'],
                  'final_checkpoint_identities_checked':p['fits'],'prediction_records_replayed':p['prediction_records'],
                  'candidate_scores_replayed':p['candidate_scores'],'cached_scores_and_nll_exact':True,'native_vectors_exact':True,
                  'native_root_backbone_reconstructions':p['native_root_backbone_reconstructions'],
                  'gate_recomputed':quality['gate_checks'],'numerical_gate_recomputed':numerical,
                  'bootstrap_intervals_recomputed':len(quality['conditional_game_bootstrap']),
                  'replay_native_files_sha256':{str(f.relative_to(out)):source.file_hash(f) for f in (out/'native').iterdir()},
                  'continuation_passed':saved['continuation_passed'],'wall_seconds':time.monotonic()-begin,'scope':p['audit']}
        write(out/'receipt.json',result); print(json.dumps(result),flush=True)
    except BaseException as error:
        write(out/'failed.json',{'status':'failed','error':repr(error),'wall_seconds':time.monotonic()-begin}); raise


def prepare(out):
    configure(); plan = signature(); plan['prepared_unix'] = time.time()
    out.mkdir(parents=True,exist_ok=False); write(out/'plan.json',plan)
    p = plan['protocol']
    print(json.dumps({'plan_sha256':source.file_hash(out/'plan.json'),'fits':p['fits'],
                      'updates':p['training_updates'],'prediction_records':p['prediction_records'],
                      'candidate_scores':p['candidate_scores'],'projected_update_hours':p['projected_selected_arm_update_seconds']/3600}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command',choices=['prepare','run','audit'])
    parser.add_argument('--plan',type=Path); parser.add_argument('--execution',type=Path)
    parser.add_argument('--out',type=Path,required=True); args = parser.parse_args()
    if args.command == 'prepare': prepare(args.out)
    elif args.command == 'run': run(args.plan,args.out)
    else: audit(args.plan,args.execution,args.out)
