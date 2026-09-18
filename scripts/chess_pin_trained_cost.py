# SPDX-License-Identifier: GPL-3.0-only
"""Pre-outcome protocol for complete native cost of the trained pin policies."""
import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import statistics
import time

import chess
import torch

import chess_pin_quality_study_v2 as study
from openjev.research import chess_pin_cache

ROOT, source, original = study.ROOT, study.source, study.original
PLAN = 'evidence/chess-pin-quality-v2/protocol/plan.json'
EXECUTION = 'runs/chess-pin-quality-v2/execution'
AUDIT = 'evidence/chess-pin-quality-v2/audit/receipt.json'
METHODS = ('base', *study.mechanics.CORE_ARMS, 'union:edits')
SEEDS = study.SEEDS
CODE = ['scripts/chess_pin_trained_cost.py', 'tests/test_chess_pin_trained_cost.py']
PROTOCOL = {
    'version': 'trained-pin-complete-native-cost-v1',
    'methods': list(METHODS), 'seeds': list(SEEDS), 'roots': 128, 'repeats': 9,
    'warmups_per_method_seed': 2, 'timed_records': 31104, 'native_audit_decisions': 3456,
    'prerequisites': 'Freeze while quality training is incomplete, before any evaluation output. Run only after all24 fits, complete quality evaluation and its full replay audit. Failed scientific or numerical gates remain failed and do not exclude descriptive timing.',
    'inputs': 'First128 original dev roots in canonical source order, all legal candidates, all3 backbone seeds. No quality-dependent root, seed or method selection.',
    'timing': 'Complete FEN parse, legal candidates and board encoding, single-position backbone, required action features, native root/child graphs, required pin inputs, head, argmax and score-list serialization. Model loading, comparison and file I/O excluded. No persistent input cache between decisions.',
    'paths': 'Base skips graphs/actions/pins. WLDN and union:edits skip pins. Graph MLP uses empty pin inputs. Root-only extracts root witnesses once and repeats them as retained. Joint/separable/pairwise/counts reconstruct every legal candidate pin transition.',
    'order': 'Nine rotating repeats by(seed_index+root_index+repeat)%9, so each method occupies every slot once per seed/root. Two warmups per method/seed on root0 retained separately.',
    'verification': 'Every warmup/timed native vector compared with the full quality audit-authenticated native vector at1e-5 and exact argmax choice. Retain finite score/choice failures. Structural/nonfinite failures stop. Separate cost-path gate never repairs the quality-study gate.',
    'audit': 'Authenticate source/input/quality/cost manifests, recompute every saved score comparison and timing aggregate, then freshly repeat all3456 seed/root/method decisions and retain full vectors. Authenticate recorded durations, not rerun timings. Shared neural kernels; no independent neural implementation or full retraining.',
    'aggregation': 'Median complete milliseconds, per-seed medians and median paired method/WLDN ratios over identical seed/root/repeat keys. Also predeclared root-pin and candidate-pin-change strata; empty strata null. Retain full raw scores, timings and host loads.',
    'score_tolerance': 1e-5, 'time_cap_seconds': 1800, 'audit_time_cap_seconds': 1800,
    'budget_rationale': 'Fixed30-minute primary and separate30-minute audit ceilings for31104 timed decisions,54 warmups and3456 audit decisions, including authentication/setup. No extension or seed replacement.',
    'runtime': 'CPU2 float32 deterministic, same process, shared host; host load recorded per seed and at boundaries.',
    'limits': 'Descriptive scalar native cost on128 reused ordinary-panel roots. No throughput, cross-hardware speed, equal-FLOPs, gameplay, independent confirmation, new training, engine/model calls or novelty claim. Biological wiring is not being tested in this pin-factor study.',
}


def deadline(begin, audit=False):
    if time.monotonic()-begin > PROTOCOL['audit_time_cap_seconds' if audit else 'time_cap_seconds']:
        raise TimeoutError('Frozen trained-pin native-cost ceiling exceeded')


def order():
    return [(seed, i, repeat, method) for si, seed in enumerate(SEEDS)
            for i in range(PROTOCOL['roots']) for repeat in range(PROTOCOL['repeats'])
            for method in METHODS[(si+i+repeat)%len(METHODS):]+METHODS[:(si+i+repeat)%len(METHODS)]]


def coverage(cache):
    chess_pin_cache.validate(cache)
    roots = []
    for i, menu in enumerate(cache['menus']):
        lo, hi = cache['factor_offsets'][i:i+2].tolist(); factors = cache['factors'][lo:hi]
        changed = factors[factors[:, 5] != 0]
        roots.append({'index': i, 'candidates': len(menu), 'factor_rows': len(factors),
                      'root_pin_present': bool((factors[:, 5] != 1).any()),
                      'pin_change_present': bool(len(changed)),
                      'candidates_with_pin_change': len(set(changed[:, 0].tolist()))})
    return {'roots': roots, 'candidates': sum(r['candidates'] for r in roots),
            'factor_rows': sum(r['factor_rows'] for r in roots),
            'roots_with_root_pins': sum(r['root_pin_present'] for r in roots),
            'roots_with_candidate_pin_change': sum(r['pin_change_present'] for r in roots),
            'candidates_with_pin_change': sum(r['candidates_with_pin_change'] for r in roots)}


def inputs():
    directory = ROOT/study.EVALUATION
    rows = source.prior.rows(directory/'dev-rows.jsonl')[:PROTOCOL['roots']]
    cache = torch.load(directory/'dev/pins/00000.pt', weights_only=True, map_location='cpu')
    if (len(rows) != PROTOCOL['roots'] or len(cache['menus']) != len(rows)
            or [r['index'] for r in rows] != list(range(len(rows)))
            or len({r['id'] for r in rows}) != len(rows)
            or tuple(r['fen'] for r in rows) != cache['fens']):
        raise AssertionError('Fixed timing panel identity differs')
    return rows, coverage(cache)


def signature():
    quality, quality_hash = study.validate_plan(ROOT/PLAN)
    if quality['protocol']['methods'] != list(METHODS) or quality['protocol']['seeds'] != list(SEEDS):
        raise ValueError('Timing methods differ from matched quality study')
    rows, pin_coverage = inputs()
    return {'protocol': PROTOCOL, 'sources': {p: source.file_hash(ROOT/p) for p in CODE},
            'quality_signature': {k: v for k, v in quality.items() if k != 'prepared_unix'},
            'quality_plan_sha256': quality_hash,
            'input_sha256': {p: source.file_hash(ROOT/p) for p in
                            (study.EVALUATION+'/dev-rows.jsonl', study.EVALUATION+'/dev/pins/00000.pt')},
            'panel': rows, 'pin_input_coverage': pin_coverage}


def before_evaluation():
    directory = ROOT/EXECUTION
    forbidden = ('all-training-complete.json', 'evaluation-started.json', 'predictions',
                 'native', 'summary.json', 'completed.json', 'failed.json')
    if any((directory/f).exists() for f in forbidden) or (ROOT/AUDIT).exists():
        raise ValueError('Must freeze before quality evaluation or terminal state')
    started = study.read(directory/'started.json')
    if started['plan_sha256'] != source.file_hash(ROOT/PLAN): raise ValueError('Different running quality study')
    count = len(list((directory/'fits').glob('*/training.json')))
    if not 0 <= count < len(SEEDS)*(len(METHODS)-1): raise ValueError('Training already complete')
    return {'completed_fits': count, 'quality_started_unix': started['unix'],
            'evaluation_outputs_absent': True}


def prepare(out):
    study.configure(); before_evaluation(); plan = signature()
    plan['freeze_state'] = before_evaluation(); plan['prepared_unix'] = time.time()
    out.mkdir(parents=True, exist_ok=False); study.write(out/'plan.json', plan)
    print(json.dumps({'plan_sha256': source.file_hash(out/'plan.json'),
                      'timed_records': PROTOCOL['timed_records'], 'freeze_state': plan['freeze_state']}), flush=True)


def validate_plan(path):
    plan = study.read(path)
    if any(plan.get(k) != v for k, v in signature().items()): raise ValueError('Frozen cost signature differs')
    state = plan['freeze_state']
    if (state['evaluation_outputs_absent'] is not True
            or not 0 <= state['completed_fits'] < len(SEEDS)*(len(METHODS)-1)
            or not 0 < state['quality_started_unix'] <= plan['prepared_unix'] <= time.time()):
        raise ValueError('Invalid pre-outcome freeze evidence')
    return plan, source.file_hash(path)


def authenticate(plan):
    """Require terminal audit before reading any quality result or fitted weight."""
    directory = ROOT/EXECUTION; audit_path = ROOT/AUDIT
    if not audit_path.is_file() or not (directory/'completed.json').is_file():
        raise ValueError('Complete quality execution and full replay audit required')
    audit = study.read(audit_path)
    if audit['status'] != 'completed': raise ValueError('Quality replay audit incomplete')
    study.diagnostic.manifest(directory, plan['quality_plan_sha256'])
    saved = study.read(directory/'summary.json'); stamp = study.read(directory/'all-training-complete.json')
    started = study.read(directory/'started.json'); phase = study.read(directory/'evaluation-started.json')
    p = plan['quality_signature']['protocol']
    fields = {'plan_sha256': plan['quality_plan_sha256'],
              'summary_sha256': source.file_hash(directory/'summary.json'),
              'execution_receipt_sha256': source.file_hash(directory/'completed.json'),
              'auditor_sha256': source.file_hash(ROOT/'scripts/chess_pin_quality_study_v2.py'),
              'training_updates_checked': p['training_updates'], 'fresh_initial_states_exact': p['fits'],
              'final_checkpoint_identities_checked': p['fits'], 'prediction_records_replayed': p['prediction_records'],
              'candidate_scores_replayed': p['candidate_scores'], 'cached_scores_and_nll_exact': True,
              'native_vectors_exact': True, 'native_root_backbone_reconstructions': p['native_root_backbone_reconstructions'],
              'gate_recomputed': saved['gate_checks'], 'bootstrap_intervals_recomputed': p['quality_checks'],
              'continuation_passed': saved['continuation_passed'], 'scope': p['audit']}
    if any(audit.get(k) != v for k, v in fields.items()): raise AssertionError('Quality audit coverage or identity differs')
    if (saved['status'] != 'completed' or saved['plan_sha256'] != plan['quality_plan_sha256']
            or saved['fits'] != p['fits'] or saved['training_updates'] != p['training_updates']
            or saved['prediction_records'] != p['prediction_records']
            or not 0 < saved['wall_seconds'] <= p['time_cap_seconds']
            or not 0 < audit['wall_seconds'] <= p['audit_time_cap_seconds']
            or saved['new_engine_calls'] != 0 or saved['external_model_calls'] != 0
            or saved['copied_reference_predictions'] != 0 or saved['original_failed_criteria_unchanged'] is not True
            or any(saved.get(k) != v for k, v in audit['numerical_gate_recomputed'].items())
            or not started['unix'] == plan['freeze_state']['quality_started_unix'] <= plan['prepared_unix'] < stamp['unix'] <= phase['unix']
            or phase['training_stamp_sha256'] != source.file_hash(directory/'all-training-complete.json')):
        raise AssertionError('Quality scope, freeze ordering or numerical summary differs')
    files = {f'native/{s}-{split}.jsonl' for s in SEEDS for split in study.SPLITS}
    if set(audit['replay_native_files_sha256']) != files: raise AssertionError('Incomplete full-panel native audit')
    for name, digest in audit['replay_native_files_sha256'].items():
        if digest != source.file_hash(audit_path.parent/name) or digest != source.file_hash(directory/name):
            raise AssertionError('Native audit vectors differ from primary')
    expected = {}
    for seed in SEEDS:
        records = source.prior.rows(directory/f'native/{seed}-dev.jsonl')
        if len(records) != p['evaluation_roots_per_panel']: raise AssertionError('Incomplete native panel')
        for i, row in enumerate(plan['panel']):
            value = records[i]
            if (value['index'] != i or value['id'] != row['id'] or value['fen'] != row['fen']
                    or value['split'] != 'dev' or value['seed'] != seed or set(value['scores']) != set(METHODS)):
                raise AssertionError('Native timing reference identity differs')
            for method in METHODS:
                scores = value['scores'][method]
                check = study.vector_comparison(scores, scores, value['menus'], PROTOCOL['score_tolerance'])
                expected[seed, i, method] = {'menus': value['menus'], 'scores': scores, 'choice': check['native_choice']}
    return expected, {'quality_gate_passed': saved['quality_gate_passed'],
                      'quality_numerical_gate_passed': saved['numerical_gate_passed'],
                      'quality_continuation_passed': saved['continuation_passed'],
                      'hashes': {p: source.file_hash(ROOT/p) for p in
                                 (AUDIT, EXECUTION+'/completed.json', EXECUTION+'/summary.json')}}


@torch.no_grad()
def decision(model, head, method, fen):
    if method not in METHODS or (head is None) != (method == 'base'): raise ValueError('Invalid method/head pair')
    board = chess.Board(fen)
    if not board.is_valid() or board.is_game_over(claim_draw=False): raise ValueError('Invalid native root')
    names, encoded = source.prior.encode_candidates(board)
    candidates = torch.from_numpy(encoded)[None]; mask = torch.ones(1, len(names), dtype=torch.bool)
    scores, _, hidden = model(torch.from_numpy(source.prior.encode_board(board))[None], candidates, mask)
    if head is not None:
        nodes = hidden.flatten(2).transpose(1, 2); actions = source.prior.candidate_features(model, nodes, candidates)
        graph = source.candidate_graphs([board])
        if list(names) != graph['menus'][0] or not torch.equal(mask, graph['mask']):
            raise AssertionError('Native graph/action menus differ')
        args = [nodes, actions, candidates, mask, scores, graph['root'], graph['children'][mask]]
        factors = torch.empty(0, 6, dtype=torch.long)
        if method == 'root_only': factors = original.root_factors(board, len(names))
        elif method not in ('wldn', 'graph_mlp', 'union:edits'):
            record = original.candidate_factors(board)
            if [c['uci'] for c in record['candidates']] != list(names): raise AssertionError('Pin/action menus differ')
            factors = original.pack_factors([record])
        scores = study.mechanics.logits(head, method, args, factors)
    return original.result(names, scores)


def comparison(observed, expected):
    if observed['menus'] != expected['menus']: raise AssertionError('Prediction menus differ')
    value = study.vector_comparison(expected['scores'], observed['scores'], expected['menus'], PROTOCOL['score_tolerance'])
    if observed['choice'] != value['native_choice'] or expected['choice'] != value['cached_choice']:
        raise AssertionError('Prediction choice does not match its score vector')
    return {'max_score_error': value['max_score_error'], 'score_tolerance_passed': value['score_tolerance_passed'],
            'choice_matches': value['choice_matches']}


def checked(records, keys, expected, panel, *, timed):
    if [(r['seed'], r['root_index'], r['repeat'], r['method']) for r in records] != keys:
        raise AssertionError('Incomplete or misordered decision membership')
    for r in records:
        key = (r['seed'], r['root_index'], r['method'])
        if r['id'] != panel[r['root_index']]['id'] or r['comparison'] != comparison(r['prediction'], expected[key]):
            raise AssertionError('Decision identity or saved numerical comparison differs')
        if timed and (type(r['milliseconds']) not in (float, int) or not math.isfinite(r['milliseconds']) or r['milliseconds'] <= 0):
            raise AssertionError('Invalid timing')
    return {'records': len(records), 'failed_records': sum(not (r['comparison']['score_tolerance_passed'] and r['comparison']['choice_matches']) for r in records),
            'choice_changes': sum(not r['comparison']['choice_matches'] for r in records),
            'max_score_error': max(r['comparison']['max_score_error'] for r in records)}


def aggregates(records):
    if not records: return None
    times = {(r['seed'], r['root_index'], r['repeat'], r['method']): r['milliseconds'] for r in records}
    pairs = sorted({(s, i, r) for s, i, r, _ in times})
    return {'timed_records': len(records),
            'median_complete_ms': {m: statistics.median(r['milliseconds'] for r in records if r['method'] == m) for m in METHODS},
            'per_seed_median_ms': {str(s): {m: statistics.median(r['milliseconds'] for r in records if r['seed'] == s and r['method'] == m) for m in METHODS} for s in SEEDS},
            'median_paired_ratio_to_wldn': {m: statistics.median(times[s,i,r,m]/times[s,i,r,'wldn'] for s,i,r in pairs) for m in METHODS}}


def summarize(records, warmups, expected, plan):
    p = PROTOCOL
    membership = {(s, i, m) for s in SEEDS for i in range(p['roots']) for m in METHODS}
    if set(expected) != membership: raise AssertionError('Reference decision membership differs')
    timed = checked(records, order(), expected, plan['panel'], timed=True)
    warm_keys = [(s, 0, r, m) for s in SEEDS for m in METHODS for r in range(p['warmups_per_method_seed'])]
    warm = checked(warmups, warm_keys, expected, plan['panel'], timed=False)
    strata = {}
    for field in ('root_pin_present', 'pin_change_present'):
        for present in (False, True):
            indices = {r['index'] for r in plan['pin_input_coverage']['roots'] if r[field] == present}
            strata[f'{field}={str(present).lower()}'] = aggregates([r for r in records if r['root_index'] in indices])
    return {'timings': aggregates(records), 'strata': strata, 'timed_checks': timed, 'warmup_checks': warm,
            'cost_path_numerical_gate_passed': timed['failed_records'] == warm['failed_records'] == 0}


def execute(plan_path, out, execution=None):
    begin = time.monotonic(); auditing = execution is not None
    plan, plan_hash = validate_plan(plan_path); study.configure()
    expected, evidence = authenticate(plan); deadline(begin, auditing)
    if auditing: study.diagnostic.manifest(execution, plan_hash)
    out.mkdir(parents=True, exist_ok=False)
    study.write(out/'started.json', {'unix': time.time(), 'pid': os.getpid(), 'plan_sha256': plan_hash,
                                   'audit': auditing, 'host_load': list(os.getloadavg())})
    try:
        records, warmups, replays, loads = [], [], [], []
        with ExitStack() as stack:
            stream = stack.enter_context((out/('native-replay.jsonl' if auditing else 'timings.jsonl')).open('x'))
            warm_stream = None if auditing else stack.enter_context((out/'warmups.jsonl').open('x'))
            for seed in SEEDS:
                deadline(begin, auditing)
                model = source.prior.backbone(seed, study.read(ROOT/source.PARENT))
                heads = {'base': None, **{m: study.load_head(ROOT/EXECUTION, seed, m, plan['quality_plan_sha256']) for m in METHODS[1:]}}
                states = {m: {k:v.clone() for k,v in h.state_dict().items()} for m,h in {'backbone':model, **heads}.items() if h is not None}
                loads.append({'seed': seed, 'host_load': list(os.getloadavg())})
                keys = ([(seed,i,0,m) for i in range(PROTOCOL['roots']) for m in METHODS] if auditing
                        else [(seed,0,r,m) for m in METHODS for r in range(PROTOCOL['warmups_per_method_seed'])]
                        + [k for k in order() if k[0] == seed])
                warm_count = 0 if auditing else len(METHODS)*PROTOCOL['warmups_per_method_seed']
                for index, (s,i,repeat,method) in enumerate(keys):
                    deadline(begin, auditing); start = time.perf_counter()
                    value = decision(model, heads[method], method, plan['panel'][i]['fen'])
                    elapsed = 1000*(time.perf_counter()-start)
                    item = {'seed':s, 'root_index':i, 'repeat':repeat, 'method':method, 'id':plan['panel'][i]['id'],
                            'prediction':value, 'comparison':comparison(value,expected[s,i,method])}
                    if auditing: replays.append(item); study.append(stream,item)
                    elif index < warm_count: warmups.append(item); study.append(warm_stream,item)
                    else: item['milliseconds']=elapsed; records.append(item); study.append(stream,item)
                    if (index+1)%len(METHODS) == 0: stream.flush()
                for m,h in {'backbone':model, **heads}.items():
                    if h is not None: original.pins.compare_states(states[m],h.state_dict(),0)
                print(json.dumps({'seed':seed,'audit':auditing,'completed_decisions':len(keys)}),flush=True)
        if auditing:
            records = source.prior.rows(execution/'timings.jsonl'); warmups = source.prior.rows(execution/'warmups.jsonl')
        stats = summarize(records,warmups,expected,plan); deadline(begin,auditing)
        common = {'status':'completed','plan_sha256':plan_hash,'quality_evidence':evidence,
                  'pin_input_coverage':plan['pin_input_coverage'],'new_training_updates':0,'new_engine_calls':0,
                  'external_model_calls':0,'limits':PROTOCOL['limits']}
        if auditing:
            saved = study.read(execution/'summary.json')
            if (any(saved.get(k) != v for k,v in {**common,**stats}.items())
                    or not 0 < saved['wall_seconds'] <= PROTOCOL['time_cap_seconds']):
                raise AssertionError('Recomputed native timing summary differs')
            replay_keys = [(s,i,0,m) for s in SEEDS for i in range(PROTOCOL['roots']) for m in METHODS]
            replay = checked(replays,replay_keys,expected,plan['panel'],timed=False)
            result = {**common, 'summary_sha256':source.file_hash(execution/'summary.json'),
                      'execution_receipt_sha256':source.file_hash(execution/'completed.json'),
                      'auditor_sha256':source.file_hash(__file__),'timing_records_checked':len(records),
                      'warmup_records_checked':len(warmups),'native_audit':replay,
                      'native_replay_sha256':source.file_hash(out/'native-replay.jsonl'),
                      'cost_path_numerical_gate_passed':stats['cost_path_numerical_gate_passed'] and replay['failed_records']==0,
                      'host_load_by_seed':loads,'host_load_end':list(os.getloadavg()),'wall_seconds':time.monotonic()-begin}
            study.write(out/'receipt.json',result)
        else:
            result = {**common,**stats,'host_load_by_seed':loads,'host_load_end':list(os.getloadavg()),'wall_seconds':time.monotonic()-begin}
            study.write(out/'summary.json',result)
            study.write(out/'completed.json',{'status':'completed','plan_sha256':plan_hash,
                        'files':{str(f.relative_to(out)):source.file_hash(f) for f in out.rglob('*') if f.is_file()}})
        print(json.dumps({'status':'completed','audit':auditing,'cost_path_numerical_gate_passed':result['cost_path_numerical_gate_passed']}),flush=True)
    except BaseException as error:
        study.write(out/'failed.json',{'status':'failed','error':repr(error),'wall_seconds':time.monotonic()-begin}); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare','run','audit'])
    parser.add_argument('--plan', type=Path); parser.add_argument('--execution', type=Path)
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    if args.command == 'prepare': prepare(args.out)
    else:
        if args.plan is None or (args.command == 'audit' and args.execution is None): parser.error('Plan and audit execution required')
        execute(args.plan, args.out, args.execution if args.command == 'audit' else None)
