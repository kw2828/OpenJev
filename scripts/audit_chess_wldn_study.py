# SPDX-License-Identifier: GPL-3.0-only
"""Authenticate the WLDN study and replay every saved evaluation decision."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import chess
import numpy as np
import torch
from torch.nn import functional as F

import chess_wldn_study as study

source = study.contrast


def audit(plan_path, execution, out):
    plan = json.loads(plan_path.read_text()); plan_hash = source.file_hash(plan_path)
    assert all(plan[k] == v for k, v in study.signature().items())
    completed = json.loads((execution/'completed.json').read_text())
    assert completed['status'] == 'completed' and completed['plan_sha256'] == plan_hash
    assert {str(p.relative_to(execution)) for p in execution.rglob('*') if p.is_file()} == set(completed['files']) | {'completed.json'}
    assert all(source.file_hash(execution/p) == h for p, h in completed['files'].items())
    summary = json.loads((execution/'summary.json').read_text())
    assert summary['status'] == 'completed' and summary['plan_sha256'] == plan_hash
    assert summary['fits'] == 3 and summary['training_updates'] == 4608
    assert summary['new_prediction_records'] == summary['copied_reference_records'] == 12288
    assert summary['new_engine_calls'] == summary['external_model_calls'] == 0
    assert summary['original_failed_criteria_unchanged']
    assert 0 < summary['wall_seconds'] <= study.PROTOCOL['time_cap_seconds']
    assert summary['comparison_status'] == 'Requires separately audited contrast outputs; no advantage decision in this execution'
    out.mkdir(parents=True, exist_ok=False); begin = time.time()
    source.prior.write(out/'started.json', {'started_unix': begin, 'plan_sha256': plan_hash})
    try:
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        old_plan = json.loads((source.ROOT/source.PARENT).read_text())
        stamp = json.loads((execution/'all-training-complete.json').read_text())
        assert stamp['fits'] == 3 and stamp['before_any_evaluation'] and stamp['plan_sha256'] == plan_hash
        cache_checks = json.loads((execution/'training-cache-check.json').read_text())
        assert [r['seed'] for r in cache_checks] == [97, 109, 127]
        assert all(r['training_roots'] == 32768 and 0 <= r['max_cache_error'] <= 2e-6 for r in cache_checks)
        updates = 0; changes = {}
        for seed in (97, 109, 127):
            directory = execution/f'wldn-{seed}'; meta = json.loads((directory/'training.json').read_text())
            assert meta['status'] == 'completed' and meta['updates'] == 1536 and meta['examples_seen'] == 196608
            assert math.isfinite(meta['seconds']) and meta['seconds'] > 0
            for key, name in [('weights_sha256', 'weights.pt'), ('initial_sha256', 'initial.pt'), ('learning_sha256', 'learning.jsonl')]:
                assert meta[key] == source.file_hash(directory/name)
            assert (directory/'weights.pt').stat().st_mtime <= stamp['unix']
            initial = torch.load(directory/'initial.pt', weights_only=True, map_location='cpu')
            fresh = study.ChessWLDNHead(seed=seed+1100).state_dict()
            assert initial.keys() == fresh.keys() and all(torch.equal(initial[k], v) for k, v in fresh.items())
            assert not initial['output.weight'].count_nonzero()
            head = study.load(execution, seed, plan_hash)
            assert meta['parameters'] == sum(p.numel() for p in head.parameters()) == 16638
            assert all(torch.isfinite(v).all() for v in head.state_dict().values())
            changes[f'wldn-{seed}'] = {k: int((v != initial[k]).sum()) for k, v in head.state_dict().items()}
            assert changes[f'wldn-{seed}']['output.weight'] > 0
            learning = source.prior.rows(directory/'learning.jsonl'); assert len(learning) == 1536
            for epoch in range(6):
                order = np.random.default_rng(700000+100*seed+epoch).permutation(32768)
                for j in range(256):
                    r = learning[epoch*256+j]; indices = order[j*128:(j+1)*128]
                    assert r['update'] == epoch*256+j+1 and r['epoch'] == epoch and r['examples'] == 128
                    assert r['indices_sha256'] == hashlib.sha256(indices.tobytes()).hexdigest()
                    assert math.isfinite(r['loss']) and r['loss'] >= 0
                    assert math.isfinite(r['gradient_norm']) and r['gradient_norm'] >= 0
            updates += len(learning)
        assert updates == 4608
        references = json.loads((execution/'reference-provenance.json').read_text())
        expected = {f'base-{s}-{split}.jsonl' for s in (97, 109, 127) for split in ('dev', 'shift')}
        assert len(references) == 6 and {r['file'] for r in references} == expected
        for r in references:
            assert r['copied_from'] == f"{source.PREVIOUS}/{r['file']}" and not r['fresh_inference']
            assert source.file_hash(execution/r['file']) == r['sha256'] == source.file_hash(source.ROOT/r['copied_from'])
        metrics = {}; count = 0; max_error = 0.; timed_choices = {}
        for split in ('dev', 'shift'):
            rows = source.prior.rows(source.ROOT/source.prior.DATA[split]); assert len(rows) == 2048
            data = torch.load(source.ROOT/source.PREVIOUS/f'packed-{split}.pt', weights_only=True, map_location='cpu')
            graphs = source.ChildGraphCache(source.ROOT/source.CACHE/split); menus = []
            for i, row in enumerate(rows):
                board = chess.Board(row['fen']); names, features = source.prior.encode_candidates(board); menus.append(names)
                assert torch.equal(data['observations'][i], torch.from_numpy(source.prior.encode_board(board)))
                assert torch.equal(data['candidates'][i, :len(names)], torch.from_numpy(features))
                assert data['mask'][i, :len(names)].all() and not data['mask'][i, len(names):].any()
                assert not data['candidates'][i, len(names):].count_nonzero()
                assert int(data['targets'][i]) == names.index(row['target_uci'])
                assert np.array_equal(data['edges'][i].numpy(), source.prior.root_relations(board))
            for seed in (97, 109, 127):
                model = source.prior.backbone(seed, old_plan); cache = source.prior.root_cache(model, data)
                for arm in ('base', 'wldn'):
                    head = None if arm == 'base' else study.load(execution, seed, plan_hash)
                    path = execution/f'{arm}-{seed}-{split}.jsonl'
                    assert path.stat().st_mtime >= stamp['unix']
                    records = source.prior.rows(path); assert len(records) == 2048
                    for start in range(0, 2048, 128):
                        index = torch.arange(start, start+128); graph = graphs.batch(index)
                        assert graph['menus'] == menus[start:start+128]
                        assert graph['fens'] == [r['fen'] for r in rows[start:start+128]]
                        args = source.arguments(model, data, cache, index, graph)
                        with torch.no_grad():
                            logits = args[4] if head is None else head(*args)
                            choices = logits.argmax(-1).tolist()
                            losses = F.cross_entropy(logits, data['targets'][index], reduction='none').tolist()
                        for j, (choice, loss) in enumerate(zip(choices, losses)):
                            i = start+j; r = records[i]; row = rows[i]
                            assert r['index'] == i and r['id'] == row['id'] and r['game_id'] == row['game_id']
                            assert r['choice'] == menus[i][choice] and r['correct'] == (r['choice'] == row['target_uci'])
                            assert math.isfinite(r['target_nll']) and abs(loss-r['target_nll']) <= 1e-5
                            max_error = max(max_error, abs(loss-r['target_nll'])); count += 1
                            if split == 'dev' and seed == 97 and i < 16: timed_choices[i, arm] = r['choice']
                    metrics.setdefault(f'{arm}-{seed}', {})[split] = {
                        'agreement': statistics.mean(r['correct'] for r in records),
                        'target_nll': statistics.mean(r['target_nll'] for r in records), 'examples': 2048}
            print(json.dumps({'replayed_panel': split, 'cumulative_predictions': count}), flush=True)
        assert count == 24576 and metrics == summary['metrics']
        assert metrics == json.loads((execution/'metrics.json').read_text())
        timing = json.loads((execution/'latency.json').read_text())['records']; expected = []; methods = ['base', 'wldn']
        for i in range(16):
            for repeat in range(3):
                offset = (i+repeat) % 2
                expected.extend((i, repeat, a) for a in methods[offset:]+methods[:offset])
        assert len(timing) == 96 and [(r['root_index'], r['repeat'], r['arm']) for r in timing] == expected
        assert all(math.isfinite(r['milliseconds']) and r['milliseconds'] > 0
                   and r['choice'] == timed_choices[r['root_index'], r['arm']] for r in timing)
        result = {'status': 'completed', 'plan_sha256': plan_hash,
            'summary_sha256': source.file_hash(execution/'summary.json'),
            'execution_receipt_sha256': source.file_hash(execution/'completed.json'),
            'auditor_sha256': source.file_hash(__file__), 'training_updates_checked': updates,
            'evaluation_predictions_replayed': count, 'new_head_predictions_replayed': 12288,
            'copied_reference_predictions_replayed': 12288, 'max_replay_nll_error': max_error,
            'changed_parameter_coordinates': changes, 'fresh_initial_states_exact': True,
            'training_cache_checks_authenticated': cache_checks,
            'median_complete_latency_ms': {a: statistics.median(r['milliseconds'] for r in timing if r['arm'] == a) for a in methods},
            'comparison_pending': True, 'wall_seconds': time.time()-begin,
            'scope': 'All source and artifact hashes; fixed minibatch schedule; fresh initial states; finite trained weights; copied-reference provenance; native evaluation roots, legal menus and labels; all24576 decisions replayed; independent metrics and96 timing memberships. Training-root and complete native child-cache audit receipts authenticated, not reconstructed again. No full retraining or latency rerun. The contrast comparison is a separate analysis.'}
        source.prior.write(out/'receipt.json', result); print(json.dumps(result), flush=True)
    except BaseException as error:
        source.prior.write(out/'failed.json', {'status': 'failed', 'exception': type(error).__name__, 'message': str(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--execution', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); audit(args.plan, args.execution, args.out)
