"""Invented scalar fixtures only; never open running study outputs or models."""

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('plot_continuation', ROOT/'scripts/plot_chess_continuation.py')
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)
PLAN_PATH = ROOT/'evidence/chess-continuation-v1/protocol/plan.json'


def metric(examples, nll=2.):
    return {'examples': examples, 'correct': 300, 'agreement': 300/examples, 'target_nll': nll}


def points(games):
    counts = {s: sum(g['status'] == s for g in games) for s in ('completed', 'unfinished', 'failed')}
    completed = [g for g in games if g['status'] == 'completed']
    draws = sum(g['result'] == '1/2-1/2' for g in completed)
    wins = sum(g['result'] == ('1-0' if g['white'].startswith('continuation-') else '0-1') for g in completed)
    score = wins+draws/2
    return {'games': len(games), 'status_counts': counts, 'continuation': {
        'wins': wins, 'draws': draws, 'losses': len(completed)-wins-draws,
        'completed_points': score, 'possible_points': len(games), 'score_lower_bound': score/len(games),
        'score_upper_bound': (score+len(games)-len(completed))/len(games)}}


def set_gate(summary):
    engine = {(r['configuration'], r['split']): r['mean_signed_bounded_loss'] for r in summary['engine_metrics']}
    checks = []
    for control in ('policy', 'best_value'):
        for split in ('dev', 'shift'):
            reference = math.fsum(engine[f'{control}-{s}', split] for s in (193, 211, 227))/3
            treatment = math.fsum(engine[f'continuation-{s}', split] for s in (193, 211, 227))/3
            checks.append({'control': control, 'split': split, 'metric': 'bounded_engine_loss',
                'reference': reference, 'continuation': treatment,
                'relative_reduction': (reference-treatment)/reference if reference > 0 else None,
                'passed': reference > 0 and treatment <= .8*reference+1e-12})
    arena_gate = summary['arena']['continuation_gate']
    checks.append({'metric': 'paired_games', 'details': arena_gate, 'passed': arena_gate['passed']})
    summary['continuation_gate'] = {'passed': all(c['passed'] for c in checks), 'checks': checks,
        'scope': summary['claim_scope'], 'thresholds': {'relative_loss_reduction': .20, 'lower_game_points': .60}}


def fixture_summary():
    plan = json.loads(PLAN_PATH.read_bytes())
    summary = {'version': 'chess-continuation-v1', 'synthetic_fixture': True,
        'plan_sha256': plot.PLAN_SHA256, 'claim_scope': plan['protocol']['scope'],
        'audit_scope': plan['protocol']['audit_scope'], 'training': [], 'curves': [],
        'primary': [], 'engine_metrics': [], 'native_latency': [], 'secondary_cost': {}}
    for i, config in enumerate(plan['configurations']):
        name, objective, seed = config['name'], config['objective'], config['seed']
        summary['training'].append({'status': 'completed', **config, 'plan_sha256': plot.PLAN_SHA256,
            'initial_state_sha256': plan['initial_state_sha256'][str(seed)],
            'updates': plan['updates_per_fit'], 'examples_seen': plan['examples_seen_per_fit'],
            'training_seconds': 70.+4*i, 'completed_unix': 100.+i, 'parameters': plan['parameters'][objective],
            'checkpoints': {f'epoch-{e:02d}.pt': plot.sha256(f'{name}-{e}'.encode()) for e in range(1, 9)},
            'learning_sha256': 'a'*64})
        for epoch in range(1, 9):
            for partition in ('train', 'diagnostic'):
                summary['curves'].append({'configuration': name, 'epoch': epoch, 'partition': partition,
                    'metrics': metric(2048, 2.5-.1*epoch+.02*i+(.15 if partition == 'diagnostic' else 0)),
                    'evaluation_wall_seconds': 1.})
        for split in ('dev', 'shift'):
            summary['primary'].append({'configuration': name, 'split': split,
                                       'metrics': metric(4096), 'evaluation_wall_seconds': 1.})
            loss = {'policy': .18, 'best_value': .16, 'continuation': .11}[objective]
            summary['engine_metrics'].append({'configuration': name, 'split': split, 'positions': 128,
                'mean_signed_bounded_loss': loss+(.02 if split == 'shift' else 0)+.001*(seed-193),
                'mean_cp_loss': 100., 'p95_cp_loss': 200., 'max_cp_loss': 300.})
        summary['native_latency'].append({'configuration': name})
    games = []
    for i, spec in enumerate(plan['arena']['games']):
        status = ('completed', 'completed', 'completed', 'unfinished', 'failed', 'completed')[i % 6]
        result = ('1-0', '1/2-1/2', '0-1', '*', '*', '1-0')[i % 6]
        games.append({'game_id': spec['id'], **{k: spec[k] for k in
            ('opening_id', 'opponent', 'seed', 'white', 'black')}, 'status': status, 'result': result,
            'termination': 'synthetic invented outcome', 'played_plies': 12})
    panels = {o: points([g for g in games if g['opponent'] == o]) for o in ('policy', 'best_value')}
    aggregate = points(games)
    arena_checks = [{'opponent': o, 'threshold': .60, 'lower_score_bound': p['continuation']['score_lower_bound'],
                     'passed': p['continuation']['score_lower_bound'] >= .60} for o, p in panels.items()]
    zero_failed = aggregate['status_counts']['failed'] == 0
    arena_gate = {'checks': arena_checks, 'zero_failed_games': zero_failed,
                  'passed': zero_failed and all(c['passed'] for c in arena_checks)}
    summary['arena'] = {'status': 'completed', 'games': 192, 'by_opponent': panels, 'aggregate': aggregate,
        'status_counts': aggregate['status_counts'], 'gate': copy.deepcopy(arena_gate),
        'continuation_gate': copy.deepcopy(arena_gate), 'gate_passed': arena_gate['passed'],
        'game_results': games, 'elo_estimate': None}
    set_gate(summary)
    return plan, summary


def write_audit(tmp_path, summary=None):
    plan, default = fixture_summary()
    summary = default if summary is None else summary
    audit = tmp_path/'audit'
    audit.mkdir()
    (audit/'started.json').write_text(json.dumps({'status': 'started', 'plan_sha256': plot.PLAN_SHA256,
                                                'started_unix': 100.}))
    (audit/'summary.json').write_text(json.dumps(summary))
    receipt = {'status': 'completed', 'plan_sha256': plot.PLAN_SHA256, 'execution_receipt_sha256': 'c'*64,
        'completed_unix': 200., 'audit_wall_seconds': 100., 'new_model_calls': 0, 'new_engine_calls': 0,
        'files': {name: plot.sha256((audit/name).read_bytes()) for name in ('started.json', 'summary.json')}}
    (audit/'receipt.json').write_text(json.dumps(receipt))
    return audit, plan, summary


def test_complete_source_fit_checkpoint_curve_and_game_coverage():
    plan, summary = fixture_summary()
    assert plot.validate(summary, plan, synthetic_fixture=True) is summary
    values = plot.chart_values(summary)
    assert len(values['fits']) == 9
    assert sum(len(f['checkpoints']) for f in values['fits']) == 72
    assert len(values['nll_curves']) == 48
    assert sum(g['games'] for g in values['arena']) == 192
    for row in values['nll_curves']:
        assert row['mean_nll'] == math.fsum(row['seed_nll'].values())/3
    for row in values['arena']:
        assert row['status_counts']['failed'] == 16
        assert row['status_counts']['unfinished'] == 16
        assert row['score_upper_bound']-row['score_lower_bound'] == pytest.approx(32/96)
    assert summary['continuation_gate']['passed'] is False


@pytest.mark.parametrize('corruption', [
    'missing_fit', 'duplicate_fit', 'missing_checkpoint', 'bad_checkpoint_hash', 'wrong_seed',
    'wrong_initialization', 'unfinished_fit', 'negative_time', 'over_budget', 'bool_time',
    'missing_curve', 'duplicate_curve', 'wrong_curve_count', 'nan_nll', 'bool_epoch',
    'missing_primary', 'missing_engine', 'invalid_engine_loss', 'missing_latency',
    'missing_game', 'wrong_game_identity', 'hidden_failed_game', 'imputed_draw',
    'wrong_point_denominator', 'wrong_bound', 'wrong_gate', 'wrong_scope',
])
def test_reject_incomplete_or_corrupted_evidence(corruption):
    plan, s = fixture_summary()
    if corruption == 'missing_fit':
        s['training'].pop()
    elif corruption == 'duplicate_fit':
        s['training'][1] = s['training'][0]
    elif corruption == 'missing_checkpoint':
        s['training'][0]['checkpoints'].pop('epoch-08.pt')
    elif corruption == 'bad_checkpoint_hash':
        s['training'][0]['checkpoints']['epoch-08.pt'] = 'invalid'
    elif corruption == 'wrong_seed':
        s['training'][0]['seed'] = 227
    elif corruption == 'wrong_initialization':
        s['training'][0]['initial_state_sha256'] = 'c'*64
    elif corruption == 'unfinished_fit':
        s['training'][0]['status'] = 'started'
    elif corruption == 'negative_time':
        s['training'][0]['training_seconds'] = -1
    elif corruption == 'over_budget':
        for f in s['training']:
            f['training_seconds'] = 2000.
    elif corruption == 'bool_time':
        s['training'][0]['training_seconds'] = True
    elif corruption == 'missing_curve':
        s['curves'].pop()
    elif corruption == 'duplicate_curve':
        s['curves'][1] = s['curves'][0]
    elif corruption == 'wrong_curve_count':
        s['curves'][0]['metrics']['examples'] = 1024
    elif corruption == 'nan_nll':
        s['curves'][0]['metrics']['target_nll'] = float('nan')
    elif corruption == 'bool_epoch':
        s['curves'][0]['epoch'] = True
    elif corruption == 'missing_primary':
        s['primary'].pop()
    elif corruption == 'missing_engine':
        s['engine_metrics'].pop()
    elif corruption == 'invalid_engine_loss':
        s['engine_metrics'][0]['mean_signed_bounded_loss'] = 2.1
    elif corruption == 'missing_latency':
        s['native_latency'].pop()
    elif corruption == 'missing_game':
        s['arena']['game_results'].pop()
    elif corruption == 'wrong_game_identity':
        s['arena']['game_results'][0]['seed'] = 227
    elif corruption == 'hidden_failed_game':
        s['arena']['game_results'][4]['status'] = 'unfinished'
    elif corruption == 'imputed_draw':
        s['arena']['game_results'][4]['result'] = '1/2-1/2'
    elif corruption == 'wrong_point_denominator':
        s['arena']['by_opponent']['policy']['continuation']['possible_points'] = 64
    elif corruption == 'wrong_bound':
        s['arena']['by_opponent']['policy']['continuation']['score_upper_bound'] = .9
    elif corruption == 'wrong_gate':
        s['continuation_gate']['passed'] = True
    elif corruption == 'wrong_scope':
        s['claim_scope'] = 'Confirmed Elo improvement'
    with pytest.raises(ValueError):
        plot.validate(s, plan, synthetic_fixture=True)


def test_signed_losses_are_retained_and_nonpositive_reference_fails_gate():
    plan, s = fixture_summary()
    for row in s['engine_metrics']:
        if row['configuration'].startswith('policy-'):
            row['mean_signed_bounded_loss'] = -.1
    set_gate(s)
    plot.validate(s, plan, synthetic_fixture=True)
    values = plot.chart_values(s)
    assert all(f['engine_loss']['dev'] == -.1 for f in values['fits'] if f['objective'] == 'policy')
    assert s['continuation_gate']['checks'][0]['relative_reduction'] is None
    assert s['continuation_gate']['checks'][0]['passed'] is False


@pytest.mark.parametrize('corruption', ['receipt', 'summary', 'failed', 'extra', 'plan', 'symlink',
                                      'new_calls', 'wall_budget', 'wrong_plan_binding'])
def test_authentication_fails_closed(tmp_path, corruption):
    audit, _, _ = write_audit(tmp_path)
    plan_path = PLAN_PATH
    if corruption == 'summary':
        with (audit/'summary.json').open('a') as stream:
            stream.write(' ')
    elif corruption == 'failed':
        (audit/'failed.json').write_text('{}')
    elif corruption == 'extra':
        (audit/'extra.json').write_text('{}')
    elif corruption == 'plan':
        plan_path = tmp_path/'other-plan.json'
        plan_path.write_bytes(PLAN_PATH.read_bytes()+b' ')
    elif corruption == 'symlink':
        (audit/'summary.json').rename(tmp_path/'saved-summary.json')
        (audit/'summary.json').symlink_to(tmp_path/'saved-summary.json')
    else:
        path = audit/'receipt.json'
        receipt = json.loads(path.read_text())
        if corruption == 'receipt':
            receipt['status'] = 'started'
        elif corruption == 'new_calls':
            receipt['new_model_calls'] = 1
        elif corruption == 'wall_budget':
            receipt['audit_wall_seconds'] = 2701
        elif corruption == 'wrong_plan_binding':
            receipt['plan_sha256'] = 'a'*64
        path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        plot.load_completed(audit, plan_path, synthetic_fixture=True)


@pytest.mark.parametrize('raw', ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'])
def test_ambiguous_or_nonfinite_json_rejected(raw):
    with pytest.raises(ValueError):
        plot.parse(raw)


def test_fixture_flag_cannot_mislabel_invented_data(tmp_path):
    audit, plan, summary = write_audit(tmp_path)
    with pytest.raises(ValueError, match='Synthetic'):
        plot.load_completed(audit, PLAN_PATH)
    summary.pop('synthetic_fixture')
    with pytest.raises(ValueError, match='Synthetic'):
        plot.validate(summary, plan, synthetic_fixture=True)


def test_synthetic_render_is_exclusive_and_provenance_binds_files(tmp_path):
    audit, _, _ = write_audit(tmp_path)
    out = tmp_path/'figures'
    manifest = plot.render(audit, PLAN_PATH, out, synthetic_fixture=True)
    assert manifest['synthetic_fixture'] is True
    assert manifest['new_model_or_engine_calls'] == 0
    assert {p.name for p in out.iterdir()} == {'figure.png', 'figure.pdf', 'provenance.json'}
    assert (out/'figure.png').read_bytes().startswith(b'\x89PNG')
    assert (out/'figure.pdf').read_bytes().startswith(b'%PDF')
    assert json.loads((out/'provenance.json').read_text()) == manifest
    for name, digest in manifest['files'].items():
        assert plot.sha256((out/name).read_bytes()) == digest
    with pytest.raises(FileExistsError):
        plot.render(audit, PLAN_PATH, out, synthetic_fixture=True)
    with pytest.raises(ValueError, match='outside'):
        plot.render(audit, PLAN_PATH, audit/'figures', synthetic_fixture=True)
