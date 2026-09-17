"""Synthetic publication evidence only: no engine, model, or chess scoring."""

import ast
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('_test_publish_capacity', ROOT/'scripts/publish_chess_capacity.py')
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False)+'\n')


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows))


def plan_fixture():
    tree = ast.parse((ROOT/'scripts/chess_capacity_study.py').read_text())
    protocol = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
                                    and any(isinstance(t, ast.Name) and t.id == 'PROTOCOL' for t in n.targets)))
    openings = [{'id': f'opening-{i+1:02d}', 'name': f'Synthetic {i+1}',
                 'moves': ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6']} for i in range(16)]
    schedule = []
    for o in openings:
        for seed in p.SEEDS:
            for white, black in ((128, 32), (32, 128)):
                schedule.append({'id': f'game-{len(schedule)+1:03d}', 'opening_id': o['id'],
                                 'opening_name': o['name'], 'opening_moves': o['moves'], 'seed': seed,
                                 'white': f'width{white}-{seed}', 'black': f'width{black}-{seed}',
                                 'white_width': white, 'black_width': black})
    return {'protocol': protocol,
            'parameters': {'32': {'stored': 43726, 'active': 33185},
                           '128': {'stored': 591790, 'active': 439073}},
            'configurations': [{'width': w, 'seed': s, 'name': f'width{w}-{s}'} for w in p.WIDTHS for s in p.SEEDS],
            'data_config': {'splits': [{'name': s, 'examples': n} for s, n in
                                       (('train', 65536), ('dev', 4096), ('shift', 4096))]},
            'openings': openings, 'arena_schedule': schedule,
            'arena_protocol': {'openings': openings, 'games': schedule, 'depth': 4, 'gate_lower_score': .6,
                               'device': 'cpu', 'torch_threads': 2, 'max_played_plies': 240,
                               'clock_seconds': 300, 'claim_draw': False, 'bootstrap_replicates': 2000,
                               'bootstrap_seed': 10800111, 'scope': 'Synthetic development fixture only'},
            'engine_sha256': 'c'*64, 'sources': {'synthetic.py': 'd'*64}}


def make_games(plan):
    games = []
    for i, spec in enumerate(plan['arena_schedule']):
        kind = ('win', 'win', 'win', 'draw', 'loss', 'unfinished', 'failed')[i % 7]
        unresolved = kind in ('unfinished', 'failed')
        result = '*' if unresolved else '1/2-1/2' if kind == 'draw' else (
            '1-0' if (kind == 'win') == (spec['white_width'] == 128) else '0-1')
        games.append({**{k: v for k, v in spec.items() if k != 'id'}, 'game_id': spec['id'],
                      'status': kind if unresolved else 'completed', 'result': result,
                      'termination': 'Synthetic fixture', 'played_plies': 2, 'opening_plies': 6,
                      'wall_seconds': .02, 'attempts': [{}, {}], 'moves': [{}]*8})
    counts, outcomes = Counter(g['status'] for g in games), Counter()
    for game in games:
        if game['status'] == 'completed':
            key = 'draws' if game['result'] == '1/2-1/2' else (
                'wins' if game['result'] == ('1-0' if game['white_width'] == 128 else '0-1') else 'losses')
            outcomes[key] += 1
    points = outcomes['wins']+outcomes['draws']/2
    lower, upper = points/96, (points+counts['unfinished']+counts['failed'])/96
    passed = lower >= .6 and counts['failed'] == 0
    summary = {'status': 'completed', 'games': 96, 'elo_estimate': None,
               'scope': plan['arena_protocol']['scope'],
               'status_counts': {s: counts[s] for s in ('completed', 'unfinished', 'failed')},
               'width128': {**outcomes, 'completed_points': points, 'possible_points': 96,
                            'score_lower_bound': lower, 'score_upper_bound': upper},
               'gate': {'threshold': .6, 'lower_score_bound': lower,
                        'zero_failed_games': counts['failed'] == 0, 'passed': passed}, 'gate_passed': passed,
               'opening_cluster_bootstrap': {'openings': 16, 'replicates': 2000, 'seed': 10800111,
                                             'lower_bound_interval': [0., 1.], 'upper_bound_interval': [0., 1.]},
               'policy_calls': 192, 'played_plies': 192, 'policy_wall_seconds': 1.92,
               'game_results': [{k: g[k] for k in ('game_id', 'opening_id', 'seed', 'white', 'black',
                                                  'status', 'result', 'termination', 'played_plies')} for g in games]}
    return games, summary


def compare(summary, plan):
    summary['means'] = {str(w): {s: {k: p.mean(summary['metrics'][f'width{w}-{seed}'][s][k] for seed in p.SEEDS)
                                     for k in ('agreement', 'target_nll', 'value_mae', 'mean_confidence')}
                               for s in p.SPLITS} for w in p.WIDTHS}
    summary['comparisons'], summary['checks'] = [], []
    for s in p.SPLITS:
        small, large = (p.mean(summary['regret'][f'width{w}-{seed}'][s] for seed in p.SEEDS) for w in p.WIDTHS)
        rel = (small-large)/small if small > 0 else None
        summary['comparisons'].append({'split': s, 'small_bounded_regret': small, 'large_bounded_regret': large,
                                       'bounded_regret_change': large-small, 'relative_reduction': rel,
                                       'agreement_interval': {'games': 64, 'positions': 4096,
                                          'mean': summary['means']['128'][s]['agreement']-summary['means']['32'][s]['agreement'],
                                          'lower': -.5, 'upper': .5, 'scope': plan['protocol']['uncertainty']}})
        summary['checks'].append({'metric': 'relative_regret_reduction', 'split': s, 'observed': rel,
                                  'threshold': .2, 'passed': rel is not None and (rel >= .2 or math.isclose(rel, .2, rel_tol=0, abs_tol=1e-12))})
    summary['checks'].append({'metric': 'arena', 'passed': summary['arena']['gate_passed']})
    summary['continuation_passed'] = all(c['passed'] for c in summary['checks'])


def seal(execution, report, plan_path, summary):
    complete = {'status': 'completed', 'plan_sha256': p.sha(plan_path), 'wall_seconds': 60.,
                'files': {path.relative_to(execution).as_posix(): p.sha(path) for path in sorted(execution.rglob('*'))
                          if path.is_file() and path != execution/'completed.json'}}
    write(execution/'completed.json', complete)
    summary['execution_receipt_sha256'] = p.sha(execution/'completed.json')
    write(report/'summary.json', summary)
    write(report/'completed.json', {'status': 'completed', 'plan_sha256': p.sha(plan_path),
                                   'summary_sha256': p.sha(report/'summary.json'),
                                   'execution_receipt_sha256': summary['execution_receipt_sha256']})


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    root = tmp_path_factory.mktemp('capacity-synthetic-source')
    execution, report, plan_path = root/'execution', root/'report', root/'plan.json'
    plan = plan_fixture()
    write(plan_path, plan)
    plan_hash = p.sha(plan_path)
    write(execution/'started.json', {'status': 'started', 'plan_sha256': plan_hash})
    panel = {'secondary_indices': {s: list(range(3000, 3128)) for s in p.SPLITS},
             'latency_indices': list(range(64))+list(range(4096, 4160)),
             'graded_configurations': [{'id': c['name']} for c in plan['configurations']]}
    write(execution/'panels.json', panel)
    data = {s: [{'id': f'{s}-{i}', 'game_id': i//64, 'target_uci': 'e2e4', 'target_value': 0.}
                for i in range(4096)] for s in p.SPLITS}
    for name in p.DATA_FILES:
        write(execution/'data'/name, {'synthetic': True})
    write(execution/'data/started.json', {'config': plan['data_config']})
    for s, records in data.items():
        jsonl(execution/'data'/f'{s}.jsonl', records)
    data_receipt = {'status': 'completed', 'counts': {'train': 65536, 'dev': 4096, 'shift': 4096},
                    'files': {n: p.sha(execution/'data'/n) for n in p.DATA_FILES}}
    write(execution/'data/completed.json', data_receipt)
    baseline = {s: {'greedy_material': {'metrics': {'examples': 4096, 'top1_teacher_agreement': .125}}}
                for s in p.SPLITS}
    write(execution/'baselines.json', baseline)
    summary = {'status': 'completed', 'plan_sha256': plan_hash, 'novelty_established': False, 'elo_estimate': None,
               'scope': plan['protocol']['scope'], 'metrics': {}, 'costs': {}, 'latency': {}, 'regret': {},
               'fresh_data_cost': data_receipt, 'baselines': {s: {n: v['metrics'] for n, v in b.items()} for s, b in baseline.items()}}
    learning = [{'step': i+1, 'epoch': i//768+1, 'depth': 4, 'examples': 128,
                 'policy_ce': 1., 'value_mse': .2, 'loss': 1.1, 'gradient_norm': .5} for i in range(6144)]
    for c in plan['configurations']:
        name, width = c['name'], c['width']
        directory = execution/name
        summary['metrics'][name] = {}
        for s in p.SPLITS:
            preds = []
            for i, row in enumerate(data[s]):
                correct = i < (1024 if width == 32 else 2048)
                preds.append({'id': row['id'], 'game_id': row['game_id'], 'target': 'e2e4',
                              'target_value': 0., 'choice': 'e2e4' if correct else 'd2d4' if width == 32 else 'g1f3',
                              'correct': correct, 'value': .1, 'target_nll': -math.log(.8 if correct else .2),
                              'target_probability': .8 if correct else .2, 'max_probability': .8,
                              'entropy': .5, 'hidden_rms': .6, 'logit_span': 2.})
            jsonl(directory/f'{s}.jsonl', preds)
            summary['metrics'][name][s] = p.prediction_metrics(preds, data[s])
        timing = {'device': 'cpu', 'torch_threads': 2, 'depth': 4,
                  'records': [{'id': (data['dev']+data['shift'])[i]['id'], 'index': i,
                               'choice': 'e2e4', 'wall_ms': float(width/32)} for i in panel['latency_indices']],
                  'warmup_records': [{'id': 'starting-board', 'index': i, 'choice': 'e2e4', 'wall_ms': 1.} for i in range(3)],
                  'total_wall_ms': 128.*width/32, 'warmup_wall_ms': 3.}
        write(directory/'latency.json', timing)
        jsonl(directory/'learning.jsonl', learning)
        (directory/'weights.pt').write_bytes(('Synthetic original checkpoint '+name).encode())
        cost = {'updates': 6144, 'core_iterations': 24576, 'examples_seen': 786432, 'training_seconds': width/2.}
        write(directory/'training.json', {'status': 'completed', **c, **cost, 'plan_sha256': plan_hash,
                                         'initial_state_sha256': 'a'*64, 'data_receipt_sha256': p.sha(execution/'data/completed.json'),
                                         'weights_sha256': p.sha(directory/'weights.pt'),
                                         'learning_sha256': p.sha(directory/'learning.jsonl')})
        write(directory/'completed.json', {'status': 'completed', **c, 'plan_sha256': plan_hash,
                                          'evaluation': {s: {'metrics': summary['metrics'][name][s], 'evaluation_wall_seconds': 1.} for s in p.SPLITS},
                                          'files': {n: p.sha(directory/n) for n in p.FIT_FILES}})
        summary['costs'][name], summary['latency'][name] = cost, timing
        summary['regret'][name] = dict.fromkeys(p.SPLITS, .3 if width == 32 else .24)
    analyses, grades = [], []
    for s in p.SPLITS:
        for i in panel['secondary_indices'][s]:
            for move, score, cp in ((None, .6, 400), ('d2d4', .3, 200), ('g1f3', .36, 240)):
                analyses.append({'split': s, 'panel_index': i, 'id': data[s][i]['id'], 'root_move': move,
                                 'bounded_score': score, 'score_cp': cp, 'requested_nodes': 20000,
                                 'reported_nodes': 20000, 'wall_seconds': .01})
            for c in plan['configurations']:
                width = c['width']
                grades.append({'configuration': c['name'], 'split': s, 'panel_index': i, 'id': data[s][i]['id'],
                               'choice': 'd2d4' if width == 32 else 'g1f3',
                               'bounded_regret': .3 if width == 32 else .24, 'cp_loss': 200 if width == 32 else 160})
    jsonl(execution/'regret/analyses.jsonl', analyses)
    jsonl(execution/'regret/regret.jsonl', grades)
    write(execution/'regret/engine.json', {'sha256': plan['engine_sha256'], 'id': {'name': 'Stockfish 19 synthetic'}})
    engine_cost = {'calls': len(analyses), 'requested_nodes': len(analyses)*20000,
                   'reported_nodes': len(analyses)*20000, 'wall_seconds': len(analyses)*.01}
    write(execution/'regret/completed.json', {'status': 'completed', 'cost': engine_cost,
                                            'files': {n: p.sha(execution/'regret'/n) for n in p.REGRET_FILES}})
    summary['engine_cost'] = engine_cost
    games, summary['arena'] = make_games(plan)
    write(execution/'arena/started.json', {'status': 'started', 'plan_sha256': plan_hash,
                                          'protocol': plan['arena_protocol'],
                                          'models': {c['name']: {'width': c['width'], 'seed': c['seed'], 'depth': 4,
                                                                 'recurrence': 'residual', 'state_sha256': 'e'*64}
                                                     for c in plan['configurations']}})
    for g in games:
        write(execution/'arena'/f'{g["game_id"]}.json', g)
        (execution/'arena'/f'{g["game_id"]}.pgn').write_text('Synthetic fixture, not actual PGN\n')
    write(execution/'arena/summary.json', summary['arena'])
    write(execution/'arena/completed.json', {'status': 'completed', 'plan_sha256': plan_hash, 'wall_seconds': 2.,
                                           'files': {n: p.sha(execution/'arena'/n) for n in p.arena_files(plan)}})
    compare(summary, plan)
    seal(execution, report, plan_path, summary)
    return {'plan': plan_path, 'execution': execution, 'report': report}


@pytest.fixture
def evidence(source, tmp_path, monkeypatch):
    root = tmp_path/'source'
    shutil.copytree(source['plan'].parent, root)
    result = {'plan': root/'plan.json', 'execution': root/'execution', 'report': root/'report',
              'out': tmp_path/'published', 'models': tmp_path/'models'}
    monkeypatch.setattr(p, 'load_study', lambda: SimpleNamespace(
        verify_plan=lambda path: p.read(path), report=lambda *_: p.read(result['report']/'summary.json')))
    return result


def publish(evidence):
    return p.publish(*(evidence[k] for k in ('plan', 'execution', 'report', 'out', 'models')))


def test_complete_lossless_archive_and_portable_audit(evidence):
    result = publish(evidence)
    assert result['fits'] == 6 and result['games'] == 96
    manifest = p.read(evidence['out']/'manifest.json')
    assert set(manifest['members']) == set(p.inventory(evidence['execution'], evidence['report']))
    with tarfile.open(evidence['out']/p.ARCHIVE) as archive:
        for member in archive:
            root, relative = member.name.split('/', 1)
            assert archive.extractfile(member).read() == (evidence[root]/relative).read_bytes()
            assert member.mtime == 0
    other = evidence['out'].parent/'again.tar.gz'
    p.make_archive(other, evidence['execution'], evidence['report'], manifest['members'])
    assert p.sha(other) == manifest['archive']['sha256']
    shutil.rmtree(evidence['plan'].parent)
    run = subprocess.run([sys.executable, '-S', str(ROOT/'scripts/publish_chess_capacity.py'),
                          'audit', '--publication', str(evidence['out'])], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)['status'] == 'verified'
    with pytest.raises(FileExistsError):
        publish(evidence)


@pytest.mark.parametrize('change', ['fit', 'game', 'cost', 'mean', 'gate', 'elo', 'exact20'])
def test_summary_rejects_corruption_and_accepts_exact_twenty_percent(source, change):
    plan, summary = p.read(source['plan']), p.read(source['report']/'summary.json')
    if change == 'exact20':
        assert all(c['passed'] for c in summary['checks'][:2])
        p.validate_summary(plan, summary)
        return
    if change == 'fit':
        summary['metrics'].pop('width128-83')
    elif change == 'game':
        summary['arena']['game_results'].pop()
    elif change == 'cost':
        summary['costs']['width32-53']['updates'] -= 1
    elif change == 'mean':
        summary['means']['128']['dev']['agreement'] += .01
    elif change == 'gate':
        summary['continuation_passed'] = True
    else:
        summary['elo_estimate'] = 2000
    with pytest.raises(ValueError):
        p.validate_summary(plan, summary)


@pytest.mark.parametrize('member', ['width32-53/dev.jsonl', 'width128-83/learning.jsonl',
                                    'arena/game-096.pgn', 'regret/analyses.jsonl'])
def test_complete_receipts_cannot_hide_removed_evidence(evidence, member):
    (evidence['execution']/member).unlink()
    seal(evidence['execution'], evidence['report'], evidence['plan'], p.read(evidence['report']/'summary.json'))
    with pytest.raises(ValueError, match='required full-panel'):
        publish(evidence)
    assert not evidence['out'].exists()


def test_rehashed_prediction_corruption_rejected(evidence):
    directory = evidence['execution']/'width32-53'
    path = directory/'dev.jsonl'
    rows = p.document(path.read_bytes(), path.name)
    rows[0]['target_nll'] += .1
    jsonl(path, rows)
    receipt = p.read(directory/'completed.json')
    receipt['files']['dev.jsonl'] = p.sha(path)
    write(directory/'completed.json', receipt)
    seal(evidence['execution'], evidence['report'], evidence['plan'], p.read(evidence['report']/'summary.json'))
    with pytest.raises(ValueError, match='Probability/NLL'):
        publish(evidence)


def test_rehashed_game_outcome_corruption_rejected(evidence):
    directory = evidence['execution']/'arena'
    game = p.read(directory/'game-001.json')
    game['result'] = '1/2-1/2'
    write(directory/'game-001.json', game)
    receipt = p.read(directory/'completed.json')
    receipt['files']['game-001.json'] = p.sha(directory/'game-001.json')
    write(directory/'completed.json', receipt)
    seal(evidence['execution'], evidence['report'], evidence['plan'], p.read(evidence['report']/'summary.json'))
    with pytest.raises(ValueError, match='Raw game'):
        publish(evidence)


def test_failed_execution_receipt_is_never_published(evidence):
    receipt = p.read(evidence['execution']/'completed.json')
    receipt['status'] = 'failed'
    write(evidence['execution']/'completed.json', receipt)
    with pytest.raises(ValueError, match='completion identity'):
        publish(evidence)
    assert not evidence['out'].exists()


def test_nonpositive_engine_reference_cannot_pass(source):
    plan, summary = p.read(source['plan']), p.read(source['report']/'summary.json')
    for seed in p.SEEDS:
        summary['regret'][f'width32-{seed}']['dev'] = 0.
        summary['regret'][f'width128-{seed}']['dev'] = -.1
    compare(summary, plan)
    assert summary['checks'][0]['observed'] is None and summary['checks'][0]['passed'] is False
    p.validate_summary(plan, summary)


def test_source_report_must_reproduce_before_any_output(evidence, monkeypatch):
    altered = p.read(evidence['report']/'summary.json')
    altered['extra'] = 'mismatch'
    monkeypatch.setattr(p, 'load_study', lambda: SimpleNamespace(verify_plan=p.read, report=lambda *_: altered))
    with pytest.raises(ValueError, match='does not reproduce'):
        publish(evidence)
    assert not evidence['out'].exists()


def test_copied_checkpoint_corruption_is_detected(evidence):
    publish(evidence)
    (evidence['models']/'width128-53/weights.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='original bytes'):
        p.audit(evidence['out'])


def test_rebound_compressed_hash_still_checks_raw_members(evidence):
    publish(evidence)
    manifest = p.read(evidence['out']/'manifest.json')
    (evidence['execution']/'width32-53/weights.pt').write_bytes(b'Synthetic original checkpoint width32-54')
    archive = evidence['out']/p.ARCHIVE
    archive.unlink()
    p.make_archive(archive, evidence['execution'], evidence['report'], manifest['members'])
    manifest['archive'].update(sha256=p.sha(archive), size=archive.stat().st_size)
    write(evidence['out']/'manifest.json', manifest)
    receipt = p.read(evidence['out']/'completed.json')
    receipt['manifest_sha256'] = p.sha(evidence['out']/'manifest.json')
    write(evidence['out']/'completed.json', receipt)
    with pytest.raises(ValueError, match='member hash'):
        p.audit(evidence['out'])


def test_figure_all_seeds_unresolved_games_and_frozen_gates(source, tmp_path):
    import matplotlib.pyplot as plt
    fig = p.figure(p.read(source['plan']), p.read(source['report']/'summary.json'))
    assert len(fig.axes) == 4
    assert len(fig.axes[0].collections) == len(fig.axes[1].collections) == 12
    assert len(fig.axes[2].collections) == 6
    assert sum(rect.get_height() for rect in fig.axes[3].patches) == 96
    labels = [t.get_text() for t in fig.axes[3].get_legend().get_texts()]
    assert any(s.startswith('Unfinished: 13') for s in labels)
    assert any(s.startswith('Failed: 13') for s in labels)
    assert any('engine score PASS' in t.get_text() and 'games FAIL' in t.get_text() for t in fig.texts)
    fig.savefig(tmp_path/'synthetic.png')
    plt.close(fig)


def test_render_uses_audit_and_binds_figure_hashes(evidence, tmp_path):
    publish(evidence)
    prefix = tmp_path/'figure'
    result = p.render(evidence['out'], prefix)
    receipt = p.read(result['json'])
    assert receipt['verification']['status'] == 'verified'
    assert receipt['plots'] == {ext: p.sha(result[ext]) for ext in ('png', 'svg')}
    with pytest.raises(FileExistsError):
        p.render(evidence['out'], prefix)
