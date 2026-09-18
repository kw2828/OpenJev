# SPDX-License-Identifier: GPL-3.0-only
"""Native scoring and artificial-update replay for the second-order pin control."""
import argparse
import json
from pathlib import Path
import time

import chess
import torch
from torch.nn import functional as F

import chess_native_evaluation_inputs as parent
from openjev.research.chess_pin_pairwise import VERSION, ChessPinPairwiseHead

ROOT, source, original = parent.ROOT, parent.source, parent.original
PLAN = 'evidence/chess-native-evaluation-inputs-v1/protocol/plan.json'
RUN = 'runs/chess-native-evaluation-inputs-v1/execution'
AUDIT = 'evidence/chess-native-evaluation-inputs-v1/audit/receipt.json'
CODE = ['src/openjev/research/chess_pin_pairwise.py', 'tests/test_chess_pin_pairwise.py',
        'scripts/chess_pin_pairwise_preflight.py', 'tests/test_chess_pin_pairwise_preflight.py']
SEEDS = (97, 109, 127); PROBES = (211, 223, 227)
PROTOCOL = {
    'version': VERSION, 'parameters': 16658,
    'scope': 'Pairwise-control engineering only. First128 original training roots,3920 legal candidates; frozen backbones97/109/127. No teacher labels, quality fit, panel predictions or new positions.',
    'construction': 'Preserve all v1 joint-head tensors. Replace factor function by sum of three pair slices minus sum of three singleton slices plus metadata-only slice. Anchor each64-dimensional role at zero; metadata fixed. Seven factor evaluations, not compute matched.',
    'probe_seeds': list(PROBES), 'root_backbone_probe_checks': 1152, 'candidate_score_checks': 35280,
    'probe_dtype': 'float32', 'score_tolerance': 1e-5,
    'projection': 'Check zero output preserves base, then Normal(0,.1) with generator900000+probe_seed.',
    'native': 'For every probe/root/backbone, fresh single-position backbone, native graph and ray pin extraction, then the new head. Cached head batches16 roots, canonical saved features, complete menus. Retain every finite failure; require1e-5 maximum absolute score difference and exact choices for all cases.',
    'reference': 'Reconstruct canonical feature blocks and pin factors independently; rebuild128 native graph inputs. Replay native calls unchanged and compare saved native vectors exactly; cached path uses independent batch assembler and subset-sum scalar factor pooling. Cached vectors replay within1e-5 with exact choice equality.',
    'updates': 'Three artificial Adam updates per backbone, head_seed1100+backbone_seed, first legal move targets, float64 head on cast canonical float32 inputs. All128 roots per update. lr.001 betas(.9,.999) eps1e-8 weight_decay0 clip_norm1.',
    'total_artificial_updates': 9, 'update_tolerance': 1e-9,
    'update_replay': 'Initial tensors exact; every final parameter, loss and gradient norm within1e-9. All four factor parameter blocks must change; no claim of float32 optimizer equivalence.',
    'runtime': 'CPU2 deterministic,900 seconds per execution/audit. Active quality study unchanged.',
    'time_cap_seconds': 900,
    'limits': 'Anchored interaction order only inside the factor MLP with role representations and metadata held fixed. Earlier encoders, witness selection, pooling/readout can mix chess information. No causal piece isolation, architecture novelty, trained quality or speed advantage.',
}


def signature():
    plan = original.read(ROOT/PLAN)
    if plan != parent.signature(): raise ValueError('Frozen canonical-input signature changed')
    complete = original.pins.diagnostic.manifest(ROOT/RUN, source.file_hash(ROOT/PLAN))
    audit = original.read(ROOT/AUDIT)
    if (audit['status'] != 'passed' or not audit['all_features_pins_and_batch_digests_exact']
            or audit['feature_blocks'] != 96 or audit['common_pin_blocks'] != 32 or audit['alignment_batches'] != 192
            or audit['completed_sha256'] != source.file_hash(ROOT/RUN/'completed.json')
            or audit['summary_sha256'] != complete['files']['summary.json']
            or any(h != source.file_hash(ROOT/Path(AUDIT).parent/f'{name}.jsonl') for name, h in audit['reference_records_sha256'].items())):
        raise ValueError('Canonical evaluation-input audit changed')
    return {'protocol': PROTOCOL, 'sources': {p: source.file_hash(ROOT/p) for p in CODE},
            'input_signature': plan, 'input_audit_sha256': source.file_hash(ROOT/AUDIT)}


@torch.no_grad()
def native_decision(model, head, fen):
    board = chess.Board(fen); names, encoded = source.prior.encode_candidates(board)
    candidates = torch.from_numpy(encoded)[None]; mask = torch.ones(1, len(names), dtype=torch.bool)
    base, _, hidden = model(torch.from_numpy(source.prior.encode_board(board))[None], candidates, mask)
    nodes = hidden.flatten(2).transpose(1, 2); actions = source.prior.candidate_features(model, nodes, candidates)
    graph = source.candidate_graphs([board]); record = original.candidate_factors(board)
    if graph['menus'][0] != list(names) or [c['uci'] for c in record['candidates']] != list(names):
        raise AssertionError('Native menus differ')
    factors = original.pack_factors([record])
    scores = head(nodes, actions, candidates, mask, base, graph['root'], graph['children'][mask], factors)
    return original.result(names, scores)


def compare(left, right):
    if left['menus'] != right['menus']: raise AssertionError('Prediction menu mismatch')
    a, b = torch.tensor(left['scores'], dtype=torch.double), torch.tensor(right['scores'], dtype=torch.double)
    if a.shape != b.shape or a.shape != (len(left['menus']),) or not len(a) or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError('Invalid complete score vectors')
    if left['choice'] != left['menus'][int(a.argmax())] or right['choice'] != right['menus'][int(b.argmax())]:
        raise AssertionError('Invalid saved choice')
    error = float((a-b).abs().max())
    return {'max_score_error': error, 'score_tolerance_passed': error <= PROTOCOL['score_tolerance'],
            'choice_matches': left['choice'] == right['choice']}


def summarize(records):
    expected = [(b, s, i) for b in SEEDS for s in PROBES for i in range(128)]
    if [(r['backbone_seed'], r['probe_seed'], r['root_index']) for r in records] != expected:
        raise AssertionError('Incomplete pairwise probes')
    for row in records:
        if any(row[k] != v for k, v in compare(row['native'], row['cached']).items()):
            raise AssertionError('Saved numerical comparison differs')
    checks = sum(len(r['native']['scores']) for r in records)
    if checks != 35280: raise AssertionError('Wrong candidate coverage')
    failures = sum(not (r['score_tolerance_passed'] and r['choice_matches']) for r in records)
    return {'root_backbone_probe_checks': len(records), 'candidate_score_checks': checks,
            'numerical_gate_passed': failures == 0, 'failed_distinct_cases': failures,
            'max_native_cached_error': max(r['max_score_error'] for r in records),
            'choice_changes': sum(not r['choice_matches'] for r in records)}


def updates(args, factors, seed, reference):
    args = [a.double() if a.is_floating_point() else a for a in args]
    head = ChessPinPairwiseHead(seed=1100+seed).double()
    initial = {k: v.clone() for k, v in head.state_dict().items()}
    optimizer = torch.optim.Adam(head.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0)
    logs = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True); logits = head(*args, factors, reference=reference)
        loss = F.cross_entropy(logits, torch.zeros(128, dtype=torch.long))
        if not torch.isfinite(loss): raise AssertionError('Nonfinite artificial loss')
        loss.backward(); norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
            raise AssertionError('Missing/nonfinite gradient')
        optimizer.step(); logs.append({'step': step+1, 'loss': float(loss.detach()), 'gradient_norm': float(norm)})
    final = head.state_dict()
    if sum(k.startswith('factor.') and not torch.equal(v, final[k]) for k, v in initial.items()) != 4:
        raise AssertionError('Factor parameter blocks did not all update')
    return initial, final, logs


def execute(plan_path, out, execution=None):
    begin = time.monotonic(); plan = original.read(plan_path)
    if plan != signature(): raise ValueError('Frozen pairwise-control signature changed')
    plan_hash = source.file_hash(plan_path); audit = execution is not None
    if audit: original.pins.diagnostic.manifest(execution, plan_hash)
    out.mkdir(parents=True, exist_ok=False); write = source.prior.write
    write(out/'started.json', {'unix': time.time(), 'plan_sha256': plan_hash, 'audit': audit})
    try:
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        feature_root = ROOT/'runs/chess-native-training-cache-v1/execution'
        rows = source.prior.rows(feature_root/'rows.jsonl')[:128]
        graphs = source.ChildGraphCache(ROOT/source.CACHE/'train'); indices = torch.arange(128)
        graph = graphs.batch(indices); boards = [chess.Board(r['fen']) for r in rows]
        reconstructed = source.candidate_graphs(boards)
        if (graph['fens'] != [r['fen'] for r in rows] or graph['menus'] != list(map(tuple, reconstructed['menus']))
                or not torch.equal(graph['root'], reconstructed['root'])
                or not torch.equal(graph['children'], reconstructed['children'][reconstructed['mask']])):
            raise AssertionError('Native input reconstruction differs')
        pin = torch.load(ROOT/'runs/chess-pin-training-cache-v1/execution/blocks/00000.pt', weights_only=True, map_location='cpu')
        if audit:
            records = [original.candidate_factors(b, reference=True) for b in boards]
            fresh_pin = parent.pins.reference_pack(records, [b.fen(en_passant='fen') for b in boards])
            parent.pins.compare(pin, fresh_pin); pin = fresh_pin
        saved = source.prior.rows(execution/'probes.jsonl') if audit else None
        saved_summary = original.read(execution/'summary.json') if audit else None
        probes, logs, replay_errors = [], {}, []
        with (out/'probes.jsonl').open('x') as stream:
            for backbone_seed in SEEDS:
                original.deadline(begin); model = source.prior.backbone(backbone_seed, original.read(ROOT/source.PARENT))
                before = {k: v.clone() for k, v in model.state_dict().items()}
                feature = torch.load(feature_root/f'{backbone_seed}/00000.pt', weights_only=True, map_location='cpu')
                if audit:
                    fresh = parent.training.prior.parent.reference_cache(model, rows)
                    parent.training.prior.parent.compare_cache(feature, fresh); feature = fresh
                data = parent.parent.TrainingInputs([feature], [pin])
                full_args, full_factors = (parent.parent.reference_batch([feature], [pin], indices.tolist(), graph)
                                           if audit else data.batch(indices, graph))
                with torch.no_grad():
                    for probe_seed in PROBES:
                        head = ChessPinPairwiseHead(seed=probe_seed)
                        if sum(p.numel() for p in head.parameters()) != 16658: raise AssertionError('Wrong parameter count')
                        if not torch.equal(head(*full_args, full_factors, reference=audit), full_args[4]):
                            raise AssertionError('Initial backbone preservation failed')
                        head.output.weight.copy_(torch.randn(head.output.weight.shape, generator=torch.Generator().manual_seed(900000+probe_seed))*.1)
                        state = {k: v.clone() for k, v in head.state_dict().items()}
                        for start in range(0, 128, 16):
                            index = torch.arange(start, start+16); batch = graphs.batch(index)
                            args, factors = (parent.parent.reference_batch([feature], [pin], index.tolist(), batch)
                                             if audit else data.batch(index, batch))
                            logits = head(*args, factors, reference=audit)
                            for slot, i in enumerate(index.tolist()):
                                original.deadline(begin); names = batch['menus'][slot]
                                native = native_decision(model, head, rows[i]['fen'])
                                cached = original.result(names, logits[slot:slot+1, :len(names)])
                                item = {'backbone_seed': backbone_seed, 'probe_seed': probe_seed, 'root_index': i,
                                        'id': rows[i]['id'], 'native': native, 'cached': cached, **compare(native, cached)}
                                if audit:
                                    old = saved[len(probes)]
                                    if any(item[k] != old[k] for k in ('backbone_seed', 'probe_seed', 'root_index', 'id', 'native')):
                                        raise AssertionError('Probe identity/native replay changed')
                                    replay = compare(cached, old['cached'])
                                    if not replay['score_tolerance_passed'] or not replay['choice_matches']:
                                        raise AssertionError('Reference cached-score replay failed')
                                    if any(item[k] != old[k] for k in ('score_tolerance_passed', 'choice_matches')):
                                        raise AssertionError('Reference gate classification changed')
                                    replay_errors.append(replay['max_score_error'])
                                probes.append(item); stream.write(json.dumps(item, allow_nan=False)+'\n'); stream.flush()
                        original.pins.compare_states(state, head.state_dict(), 0)
                        print(json.dumps({'backbone_seed': backbone_seed, 'probe_seed': probe_seed, 'audit': audit}), flush=True)
                initial, final, steps = updates(full_args, full_factors, backbone_seed, audit)
                original.deadline(begin); logs[str(backbone_seed)] = steps
                if audit:
                    old_initial = torch.load(execution/f'{backbone_seed}-initial.pt', weights_only=True, map_location='cpu')
                    old_final = torch.load(execution/f'{backbone_seed}-final.pt', weights_only=True, map_location='cpu')
                    original.pins.compare_states(initial, old_initial, 0)
                    error = original.pins.compare_states(final, old_final, PROTOCOL['update_tolerance'])
                    old_steps = saved_summary['artificial_logs'][str(backbone_seed)]
                    if len(old_steps) != 3 or [r['step'] for r in old_steps] != [1, 2, 3]: raise AssertionError('Incomplete update log')
                    log_error = max(abs(a[k]-b[k]) for a, b in zip(steps, old_steps) for k in ('loss', 'gradient_norm'))
                    if log_error > PROTOCOL['update_tolerance']: raise AssertionError('Artificial update replay differs')
                    logs[str(backbone_seed)] = {'steps': steps, 'final_parameter_max_error': error, 'log_max_error': log_error}
                else:
                    source.prior.save(out/f'{backbone_seed}-initial.pt', initial)
                    source.prior.save(out/f'{backbone_seed}-final.pt', final)
                original.pins.compare_states(before, model.state_dict(), 0)
        original.deadline(begin); stats = summarize(probes)
        if audit:
            original_stats = summarize(saved)
            if (any(saved_summary[k] != v for k, v in original_stats.items()) or saved_summary['status'] != 'completed'
                    or saved_summary['plan_sha256'] != plan_hash or saved_summary['artificial_updates'] != 9
                    or saved_summary['limits'] != PROTOCOL['limits'] or not 0 < saved_summary['wall_seconds'] <= 900):
                raise AssertionError('Primary summary differs')
            result = {'status': 'passed', 'plan_sha256': plan_hash,
                      'completed_sha256': source.file_hash(execution/'completed.json'),
                      'summary_sha256': source.file_hash(execution/'summary.json'),
                      'reference_probes_sha256': source.file_hash(out/'probes.jsonl'),
                      'primary_numerical_gate_passed': original_stats['numerical_gate_passed'],
                      'native_vectors_exact': True, 'max_reference_cached_error': max(replay_errors),
                      'artificial_updates_replayed': 9, 'artificial_replay': logs, **stats,
                      'wall_seconds': time.monotonic()-begin, 'limits': PROTOCOL['limits']}
            write(out/'receipt.json', result)
        else:
            result = {'status': 'completed', 'plan_sha256': plan_hash, **stats,
                      'artificial_updates': 9, 'artificial_logs': logs,
                      'wall_seconds': time.monotonic()-begin, 'limits': PROTOCOL['limits']}
            write(out/'summary.json', result)
            write(out/'completed.json', {'status': 'completed', 'plan_sha256': plan_hash,
                  'files': {p.name: source.file_hash(p) for p in out.iterdir() if p.is_file()}})
        print(json.dumps(result), flush=True)
    except BaseException as error:
        write(out/'failed.json', {'error': repr(error), 'wall_seconds': time.monotonic()-begin}); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'run', 'audit'])
    parser.add_argument('--plan', type=Path); parser.add_argument('--execution', type=Path)
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    if args.command == 'prepare':
        plan = signature(); args.out.mkdir(parents=True, exist_ok=False)
        source.prior.write(args.out/'plan.json', plan)
        print(json.dumps({'plan_sha256': source.file_hash(args.out/'plan.json')}), flush=True)
    else: execute(args.plan, args.out, args.execution if args.command == 'audit' else None)
