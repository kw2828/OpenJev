"""Synthetic receipts and charts only. No model, engine, or real-study scoring."""

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('_test_publish_chess_anchor', ROOT/'scripts/publish_chess_anchor.py')
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False)+'\n')


def plan_fixture():
    return {
        'protocol': {'version': 'chess-anchor-v1', 'seeds': list(p.SEEDS),
                     'arms': {a: list(v) for a, v in p.ARMS.items()}, 'evaluation_depths': list(p.DEPTHS),
                     'train_examples': 32768, 'epochs': 6, 'batch_size': 128, 'updates_per_fit': 1536,
                     'core_iterations_per_fit': 6144, 'mixed_depths': [2, 4, 6], 'width': 32, 'default_depth': 4,
                     'torch_threads': 2, 'latency_positions_per_split': 32, 'latency_warmups': 3,
                     'regret_call_ceiling': 3328, 'regret_node_ceiling': 66560000, 'regret_nodes': 20000,
                     'gate': {'primary_gain': .02, 'worst_seed_deficit': .01, 'extra_depth_gain': .01},
                     'scope': 'Synthetic development fixture; no results.',
                     'uncertainty': 'Conditional on three fitted seeds, synthetic fixture.'},
        'fresh_data': {'splits': [{'name': split, 'examples': 2048} for split in p.SPLITS]},
        'active_parameters': 33185, 'stored_parameters': 43726,
        'configurations': [{'name': f'{a}-{s}', 'arm': a, 'seed': s, 'recurrence': r, 'regime': g}
                           for a, (r, g) in p.ARMS.items() for s in p.SEEDS],
        'engine_sha256': 'c'*64,
        'sources': {'src/openjev/research/chess_anchor.py': 'd'*64,
                    'src/openjev/research/chess_spatial.py': 'e'*64},
    }


def summary_fixture(plan, plan_hash='a'*64):
    metrics, costs, latency = {}, {}, {}
    for index, arm in enumerate(p.ARMS):
        for offset, seed in enumerate(p.SEEDS):
            name = f'{arm}-{seed}'
            metrics[name], latency[name] = {}, {}
            costs[name] = {'updates': 1536, 'core_iterations': 6144, 'examples_seen': 196608, 'training_seconds': 2.}
            for d_index, depth in enumerate(p.DEPTHS):
                for split in p.SPLITS:
                    correct = 256+16*index+4*offset+d_index*(6 if arm == 'anchor_mixed' else 1)
                    metrics[name][f'{split}-d{depth}'] = {
                        'examples': 2048, 'correct': correct, 'agreement': correct/2048,
                        'target_nll': 3., 'entropy': 2., 'hidden_rms': .5, 'logit_span': 4.,
                        'value_mae': .4, 'mean_confidence': .3, 'mismatch_confidence': .35+index*.02,
                    }
                records = [{'id': f'position-{i}', 'index': i if i < 32 else i+2016,
                            'choice': 'e2e4', 'wall_ms': float(depth+i*.001)} for i in range(64)]
                warmups = [{'id': 'starting-board', 'index': i, 'choice': 'e2e4', 'wall_ms': 1.} for i in range(3)]
                latency[name][str(depth)] = {
                    'device': 'cpu', 'depth': depth, 'torch_threads': 2,
                    'timing_scope': 'Synthetic timing fixture', 'warmup_policy': 'Synthetic three warmups',
                    'records': records, 'warmup_records': warmups,
                    'total_wall_ms': sum(r['wall_ms'] for r in records), 'warmup_wall_ms': 3.,
                }
    means = {arm: {str(d): {split: p.mean(metrics[f'{arm}-{s}'][f'{split}-d{d}']['agreement'] for s in p.SEEDS)
                           for split in p.SPLITS} for d in p.DEPTHS} for arm in p.ARMS}
    regret = {f'{arm}-{s}-d{d}': dict.fromkeys(p.SPLITS, .5-(.1 if arm.startswith('anchor') else 0)-d*.001)
              for arm in p.ARMS if arm.endswith('_mixed') for s in p.SEEDS for d in (4, 8)}
    comparisons, checks = [], []
    for split in p.SPLITS:
        for label, candidate, cd, reference, rd in (
            ('primary', 'anchor_mixed', 4, 'residual_mixed', 4),
            ('extra_depth', 'anchor_mixed', 8, 'anchor_mixed', 4),
            ('depth8_architecture', 'anchor_mixed', 8, 'residual_mixed', 8),
        ):
            ds = [metrics[f'{candidate}-{s}'][f'{split}-d{cd}']['agreement']
                  - metrics[f'{reference}-{s}'][f'{split}-d{rd}']['agreement'] for s in p.SEEDS]
            delta = p.mean(regret[f'{candidate}-{s}-d{cd}'][split]-regret[f'{reference}-{s}-d{rd}'][split]
                           for s in p.SEEDS)
            comparisons.append({'comparison': label, 'split': split, 'seed_agreement_deltas': ds,
                                'bounded_regret_delta': delta,
                                'game_cluster_interval': {'mean': p.mean(ds), 'lower': -.01, 'upper': .1,
                                                          'positions': 2048, 'games': 100,
                                                          'scope': plan['protocol']['uncertainty']}})
            threshold = .02 if label == 'primary' else .01 if label == 'extra_depth' else 0.
            checks.extend([
                {'gate': label, 'split': split, 'metric': 'agreement_gain', 'observed': p.mean(ds),
                 'threshold': threshold, 'passed': p.mean(ds) >= threshold if label != 'depth8_architecture'
                 else p.mean(ds) > 0},
                {'gate': label, 'split': split, 'metric': 'bounded_regret_change', 'observed': delta,
                 'threshold': 0., 'passed': delta < 0 if label == 'primary' else delta <= 0},
            ])
            if label == 'primary':
                checks.append({'gate': label, 'split': split, 'metric': 'worst_seed_gain',
                               'observed': min(ds), 'threshold': -.01, 'passed': min(ds) >= -.01})
    return {'status': 'completed', 'plan_sha256': plan_hash, 'novelty_established': False, 'elo_estimate': None,
            'scope': plan['protocol']['scope'], 'metrics': metrics, 'costs': costs, 'latency': latency,
            'means': means, 'regret': regret, 'comparisons': comparisons, 'checks': checks,
            'primary_passed': all(c['passed'] for c in checks if c['gate'] == 'primary'),
            'extra_compute_passed': all(c['passed'] for c in checks if c['gate'] != 'primary'),
            'interaction_at_depth4': {split: means['anchor_mixed']['4'][split]-means['residual_mixed']['4'][split]
                                     - means['anchor_fixed']['4'][split]+means['residual_fixed']['4'][split]
                                     for split in p.SPLITS}}


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    source = tmp_path/'source'
    execution, report, plan_path = source/'execution', source/'report', source/'plan.json'
    plan = plan_fixture()
    write(plan_path, plan)
    plan_hash = p.sha(plan_path)
    summary = summary_fixture(plan, plan_hash)
    write(execution/'started.json', {'status': 'started', 'plan_sha256': plan_hash})
    write(execution/'panels.json', {'latency_indices': list(range(32))+list(range(2048, 2080))})
    write(execution/'baselines.json', {'synthetic': 'No measured results'})
    for name in p.DATA_FILES:
        write(execution/'data'/name, {'synthetic': name})
    data = {'status': 'completed', 'files': {n: p.sha(execution/'data'/n) for n in p.DATA_FILES}}
    write(execution/'data/completed.json', data)
    summary['fresh_data_cost'] = data
    for name in p.REGRET_FILES:
        write(execution/'regret'/name, {'synthetic': name})
    write(execution/'regret/engine.json', {'sha256': plan['engine_sha256'], 'id': {'name': 'Stockfish 19 synthetic fixture'}})
    cost = {'calls': 256, 'requested_nodes': 5120000, 'reported_nodes': 5120000, 'wall_seconds': 1.}
    write(execution/'regret/completed.json', {'status': 'completed', 'cost': cost,
          'files': {n: p.sha(execution/'regret'/n) for n in p.REGRET_FILES}})
    summary['engine_cost'] = cost
    for config in plan['configurations']:
        name = config['name']
        directory = execution/name
        directory.mkdir()
        (directory/'weights.pt').write_bytes(b'Synthetic bytes, not a model: '+name.encode())
        write(directory/'learning.jsonl', {'synthetic': name})
        for key in p.EVALUATIONS:
            write(directory/f'{key}.jsonl', {'synthetic': name+key})
        write(directory/'latency.json', summary['latency'][name])
        write(directory/'training.json', {
            'status': 'completed', **config, 'plan_sha256': plan_hash,
            'data_receipt_sha256': p.sha(execution/'data/completed.json'),
            'initial_state_sha256': str(config['seed'])[0]*64, **summary['costs'][name],
            'weights_sha256': p.sha(directory/'weights.pt'), 'learning_sha256': p.sha(directory/'learning.jsonl'),
        })
        write(directory/'completed.json', {
            'status': 'completed', **config, 'plan_sha256': plan_hash,
            'evaluation': {key: {'metrics': m, 'evaluation_wall_seconds': 1.}
                           for key, m in summary['metrics'][name].items()},
            'files': {n: p.sha(directory/n) for n in p.FIT_FILES},
        })
    write(execution/'completed.json', {
        'status': 'completed', 'plan_sha256': plan_hash,
        'fits': {c['name']: p.sha(execution/c['name']/'completed.json') for c in plan['configurations']},
        'data_receipt_sha256': p.sha(execution/'data/completed.json'),
        'panels_sha256': p.sha(execution/'panels.json'), 'baselines_sha256': p.sha(execution/'baselines.json'),
        'regret_receipt_sha256': p.sha(execution/'regret/completed.json'), 'wall_seconds': 30.,
    })
    summary['execution_receipt_sha256'] = p.sha(execution/'completed.json')
    write(report/'summary.json', summary)
    write(report/'completed.json', {'status': 'completed', 'plan_sha256': plan_hash,
          'summary_sha256': p.sha(report/'summary.json'),
          'execution_receipt_sha256': p.sha(execution/'completed.json')})
    (execution/'extra-raw.log').write_bytes(b'Every execution file must be preserved.\n')
    (report/'extra-report.txt').write_bytes(b'Every report file must be preserved.\n')
    calls = []

    def verify(path):
        assert Path(path) == plan_path
        calls.append('verify')
        return plan

    def reproduce(path, original, out):
        assert Path(path) == plan_path and Path(original) == execution
        assert not Path(out).exists()
        calls.append('report')
        return copy.deepcopy(summary)

    study = SimpleNamespace(verify_plan=verify, report=reproduce)
    monkeypatch.setattr(p, 'load_study', lambda: study)
    return {'plan': plan_path, 'execution': execution, 'report': report, 'summary': summary,
            'project': tmp_path/'published', 'calls': calls, 'study': study}


def publish_fixture(evidence, project=None):
    project = project or evidence['project']
    return p.publish(evidence['plan'], evidence['execution'], evidence['report'], project/'results', project/'models')


def rebind_manifest(out):
    receipt = p.read(out/'completed.json')
    receipt['manifest_sha256'] = p.sha(out/'manifest.json')
    write(out/'completed.json', receipt)


def test_lossless_deterministic_package_and_portable_audit(evidence, tmp_path, monkeypatch):
    result = publish_fixture(evidence)
    assert result['status'] == 'verified' and result['fits'] == 12 and result['evaluations'] == 96
    assert evidence['calls'] == ['verify', 'report']
    out = evidence['project']/'results'
    manifest = p.read(out/'manifest.json')
    assert 'execution/extra-raw.log' in manifest['members'] and 'report/extra-report.txt' in manifest['members']
    assert set(manifest['weights']) == p.expected_names()
    with tarfile.open(out/p.ARCHIVE, 'r:gz') as archive:
        for member in archive:
            prefix, relative = member.name.split('/', 1)
            assert archive.extractfile(member).read() == (evidence[prefix]/relative).read_bytes()
            assert member.mtime == member.uid == member.gid == 0
    other = tmp_path/'second'
    publish_fixture(evidence, other)
    assert p.sha(out/p.ARCHIVE) == p.sha(other/'results'/p.ARCHIVE)
    assert p.sha(out/'manifest.json') == p.sha(other/'results/manifest.json')

    def forbidden():
        raise AssertionError('Portable audit must not import the study or access engines/GPU')

    monkeypatch.setattr(p, 'load_study', forbidden)
    shutil.rmtree(evidence['execution'])
    shutil.rmtree(evidence['report'])
    assert p.audit(out) == result
    with pytest.raises(FileExistsError):
        publish_fixture(evidence)


def test_import_and_portable_audit_use_only_standard_library(evidence):
    publish_fixture(evidence)
    # -S removes all site packages, proving that torch/numpy/chess are unnecessary.
    result = subprocess.run([sys.executable, '-S', str(ROOT/'scripts/publish_chess_anchor.py'),
                             'audit', '--publication', str(evidence['project']/'results')],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['fits'] == 12


@pytest.mark.parametrize('change', ['missing_fit', 'depth', 'count', 'mean', 'gate', 'scope', 'nonfinite', 'regret'])
def test_summary_semantic_corruption_rejected(change):
    plan = plan_fixture()
    summary = summary_fixture(plan)
    if change == 'missing_fit':
        del summary['metrics']['anchor_mixed-43']
    elif change == 'depth':
        del summary['metrics']['anchor_mixed-43']['dev-d16']
    elif change == 'count':
        summary['metrics']['anchor_mixed-43']['dev-d16']['correct'] += 1
    elif change == 'mean':
        summary['means']['anchor_mixed']['4']['dev'] = .8
    elif change == 'gate':
        summary['primary_passed'] = not summary['primary_passed']
    elif change == 'scope':
        summary['novelty_established'] = True
    elif change == 'nonfinite':
        summary['metrics']['anchor_mixed-43']['dev-d16']['hidden_rms'] = float('nan')
    else:
        del summary['regret']['anchor_mixed-43-d8']
    with pytest.raises(ValueError):
        p.validate_summary(plan, summary)


@pytest.mark.parametrize('change', ['failed', 'missing_raw', 'symlink', 'report_binding', 'fit_identity', 'different_report'])
def test_bad_source_evidence_rejected_before_publication(evidence, change):
    if change == 'failed':
        write(evidence['execution']/'anchor_fixed-17/failed.json', {'status': 'failed'})
    elif change == 'missing_raw':
        (evidence['execution']/'data/analyses.jsonl').unlink()
    elif change == 'symlink':
        (evidence['execution']/'link').symlink_to(evidence['execution']/'extra-raw.log')
    elif change == 'report_binding':
        value = p.read(evidence['report']/'completed.json')
        value['summary_sha256'] = '0'*64
        write(evidence['report']/'completed.json', value)
    elif change == 'fit_identity':
        path = evidence['execution']/'anchor_fixed-17/completed.json'
        value = p.read(path)
        value['recurrence'] = 'residual'
        write(path, value)
        complete = p.read(evidence['execution']/'completed.json')
        complete['fits']['anchor_fixed-17'] = p.sha(path)
        write(evidence['execution']/'completed.json', complete)
    else:
        evidence['study'].report = lambda *args: {'different': 'report'}
    with pytest.raises(ValueError):
        publish_fixture(evidence)
    assert not evidence['project'].exists()


@pytest.mark.parametrize('change', ['checkpoint', 'archive', 'manifest', 'coverage', 'alias', 'model_card', 'gate'])
def test_portable_audit_rejects_changed_publication(evidence, change):
    publish_fixture(evidence)
    project = evidence['project']
    out = project/'results'
    if change == 'checkpoint':
        (project/'models/anchor_mixed-17/weights.pt').write_bytes(b'changed')
    elif change == 'archive':
        with (out/p.ARCHIVE).open('ab') as stream:
            stream.write(b'changed')
    elif change in ('manifest', 'coverage', 'alias'):
        manifest = p.read(out/'manifest.json')
        if change == 'manifest':
            manifest['member_count'] += 1
        elif change == 'coverage':
            del manifest['weights']['anchor_mixed-17']
        else:
            manifest['weights']['anchor_mixed-17']['recurrence'] = 'residual'
        write(out/'manifest.json', manifest)
        rebind_manifest(out)
    elif change == 'model_card':
        (project/'models/README.md').write_text('Only publish a selected winning checkpoint.')
    else:
        summary = p.read(out/'summary.json')
        summary['primary_passed'] = not summary['primary_passed']
        write(out/'summary.json', summary)
    with pytest.raises(ValueError):
        p.audit(out)


def test_required_raw_evidence_cannot_be_removed_with_rehashed_manifest(evidence):
    publish_fixture(evidence)
    out = evidence['project']/'results'
    manifest = p.read(out/'manifest.json')
    del manifest['members']['execution/data/analyses.jsonl']
    (out/p.ARCHIVE).unlink()
    p.make_archive(out/p.ARCHIVE, evidence['execution'], evidence['report'], manifest['members'])
    manifest['archive'].update(sha256=p.sha(out/p.ARCHIVE), size=(out/p.ARCHIVE).stat().st_size)
    manifest['member_count'] = len(manifest['members'])
    manifest['raw_bytes'] = sum(row['size'] for row in manifest['members'].values())
    write(out/'manifest.json', manifest)
    rebind_manifest(out)
    with pytest.raises(ValueError, match='lacks required'):
        p.audit(out)


def test_archive_member_hashes_checked_even_when_compressed_hash_is_rebound(evidence):
    publish_fixture(evidence)
    out = evidence['project']/'results'
    manifest = p.read(out/'manifest.json')
    path = evidence['execution']/'extra-raw.log'
    path.write_bytes(b'x'*path.stat().st_size)
    (out/p.ARCHIVE).unlink()
    p.make_archive(out/p.ARCHIVE, evidence['execution'], evidence['report'], manifest['members'])
    manifest['archive'].update(sha256=p.sha(out/p.ARCHIVE), size=(out/p.ARCHIVE).stat().st_size)
    write(out/'manifest.json', manifest)
    rebind_manifest(out)
    with pytest.raises(ValueError, match='member hash mismatch'):
        p.audit(out)


def test_chart_shows_every_seed_training_depths_and_failed_gate():
    pytest.importorskip('matplotlib')
    plan = plan_fixture()
    summary = summary_fixture(plan)
    fig = p.figure(plan, summary)
    assert len(fig.axes) == 4
    for ax in fig.axes[:2]:
        thin = [line for line in ax.lines if line.get_linewidth() == .7]
        assert len(thin) == 12 and sum(len(line.get_xdata()) for line in thin) == 48
        assert {line.get_marker() for line in thin} == {'o', 's', '^'}
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ['2', '4', '8*', '16*']
    text = '\n'.join(t.get_text() for t in fig.texts)
    assert 'primary FAILED' in text and 'Fixed training: depth 4' in text
    assert 'Mixed training: depths 2, 4, 6' in text and 'no cross-move memory' in text
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_render_from_audited_synthetic_summary_preserves_source_and_receipt(evidence, tmp_path, monkeypatch):
    pytest.importorskip('matplotlib')
    publish_fixture(evidence)
    out = evidence['project']/'results'
    before = p.sha(out/'summary.json')
    monkeypatch.setattr(p, 'load_study', lambda: (_ for _ in ()).throw(AssertionError('No model access')))
    files = p.render(out, tmp_path/'figures/synthetic-test')
    assert p.sha(out/'summary.json') == before
    receipt = p.read(files['json'])
    for extension in ('png', 'svg'):
        assert p.sha(files[extension]) == receipt['plots'][extension]
    with pytest.raises(FileExistsError):
        p.render(out, tmp_path/'figures/synthetic-test')
