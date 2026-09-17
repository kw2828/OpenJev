"""Synthetic publication receipts only; no training, engine or model calls."""

import copy
import importlib.util
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('_test_publish_chess_spatial',
                                             ROOT/'scripts/publish_chess_spatial.py')
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False)+'\n')


def summary_fixture(plan_hash='a'*64):
    fits = []
    counts = {row['name']: row['examples'] for row in publisher.STUDY.PROTOCOL['splits']}
    for split in publisher.SPLITS:
        for index, mode in enumerate(publisher.MODES):
            for seed_index, seed in enumerate(publisher.SEEDS):
                fits.append({'mode': mode, 'seed': seed, 'split': split, 'examples': counts[split],
                             'top1_teacher_agreement': .15+index*.005+seed_index*.001,
                             'target_nll': 3., 'value_mae': .5, 'future_changed_accuracy': .6,
                             'future_square_accuracy': .94, 'future_exact_board_accuracy': .25,
                             'parameter_count': 100, 'training_wall_seconds': 2.,
                             'checkpoint_sha256': 'b'*64})
    means = {split: {mode: {metric: float(np.mean([row[metric] for row in fits
                                                  if row['split'] == split and row['mode'] == mode]))
                            for metric in publisher.METRICS} for mode in publisher.MODES}
             for split in publisher.SPLITS}
    return {'status': 'completed', 'protocol': copy.deepcopy(publisher.STUDY.PROTOCOL),
            'plan_sha256': plan_hash, 'fits': fits, 'means': means,
            'novelty_established': False, 'elo_estimate': None,
            'claim_scope': publisher.STUDY.PROTOCOL['claim_scope'],
            'continuation_gate': publisher.STUDY.continuation_gate(fits),
            'training_wall_seconds': 24.,
            'baselines': {split: {'greedy_material': {'metrics': {'top1_teacher_agreement': .12}}}
                          for split in publisher.SPLITS}}


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    source = tmp_path/'source'
    execution, report = source/'execution', source/'report'
    execution.mkdir(parents=True)
    report.mkdir()
    plan = source/'plan.json'
    write(plan, {'protocol': publisher.STUDY.PROTOCOL, 'parameters': dict.fromkeys(publisher.MODES, 100)})
    plan_hash = publisher.sha(plan)
    summary = summary_fixture(plan_hash)
    write(execution/'started.json', {'plan_sha256': plan_hash})
    write(execution/'engine.json', {'name': 'synthetic fixture, never executed'})
    write(execution/'baselines.json', summary['baselines'])
    data_files = ('started.json', 'excluded-states.json', 'games.jsonl', 'analyses.jsonl',
                  'train.jsonl', 'dev.jsonl', 'shift.jsonl')
    for name in data_files:
        write(execution/'data'/name, {'synthetic': name})
    data = {'status': 'completed', 'files': {name: publisher.sha(execution/'data'/name)
                                          for name in data_files}}
    write(execution/'data/completed.json', data)
    summary['teacher_cost'] = data
    regret_files = ('selection.json', 'analyses.jsonl', 'predictions.json')
    for name in regret_files:
        write(execution/'regret'/name, {'synthetic': name})
    regret = {'status': 'completed', 'files': {name: publisher.sha(execution/'regret'/name)
                                            for name in regret_files}}
    write(execution/'regret/completed.json', regret)
    summary['regret'] = regret
    count = publisher.STUDY.PROTOCOL['splits'][0]['examples']
    updates = count//publisher.STUDY.PROTOCOL['batch_size']*publisher.STUDY.PROTOCOL['epochs']
    fit_order = []
    for mode in publisher.MODES:
        for seed in publisher.SEEDS:
            name = f'{mode}-{seed}'
            directory = execution/name
            write(directory/'started.json', {'plan_sha256': plan_hash})
            (directory/'weights.pt').write_bytes(b'Not a model. Synthetic test bytes: '+name.encode())
            for file in ('learning.jsonl', 'dev.json', 'shift.json'):
                write(directory/file, {'synthetic': name+file})
            files = {file: publisher.sha(directory/file)
                     for file in ('weights.pt', 'learning.jsonl', 'dev.json', 'shift.json')}
            write(directory/'completed.json', {'status': 'completed', 'mode': mode, 'seed': seed,
                  'plan_sha256': plan_hash, 'data_receipt_sha256': publisher.sha(execution/'data/completed.json'),
                  'updates': updates, 'examples_seen': count*publisher.STUDY.PROTOCOL['epochs'],
                  'parameter_count': 100, 'training_wall_seconds': 2., 'files': files})
            for row in summary['fits']:
                if row['mode'] == mode and row['seed'] == seed:
                    row['checkpoint_sha256'] = files['weights.pt']
            fit_order.append([mode, seed])
    write(execution/'completed.json', {'status': 'completed', 'fits': 12, 'plan_sha256': plan_hash,
          'fit_order': fit_order, 'total_updates': updates*12, 'training_wall_seconds': 24.,
          'fit_receipts': {f'{mode}-{seed}': publisher.sha(execution/f'{mode}-{seed}/completed.json')
                           for mode, seed in fit_order},
          'baseline_sha256': publisher.sha(execution/'baselines.json'),
          'regret_receipt_sha256': publisher.sha(execution/'regret/completed.json')})
    write(report/'summary.json', summary)
    write(report/'completed.json', {'status': 'completed', 'plan_sha256': plan_hash,
          'summary_sha256': publisher.sha(report/'summary.json'),
          'execution_receipt_sha256': publisher.sha(execution/'completed.json')})
    (execution/'additional-raw-note.txt').write_bytes(b'Preserve every input file, including this one.\n')
    (report/'additional-note.txt').write_bytes(b'All report files too.\n')
    calls = []

    def verify(path):
        assert Path(path) == plan
        calls.append('verify')
        return publisher.read_json(path)

    def reproduce(plan_path, original_execution, out):
        assert Path(plan_path) == plan and Path(original_execution) == execution
        calls.append('reproduce')
        return copy.deepcopy(summary)

    monkeypatch.setattr(publisher.STUDY, 'verify_plan', verify)
    monkeypatch.setattr(publisher.STUDY, 'report', reproduce)
    return {'plan': plan, 'execution': execution, 'report': report, 'summary': summary,
            'project': tmp_path/'published', 'calls': calls}


def publish_fixture(evidence, project=None):
    return publisher.publish(evidence['plan'], evidence['execution'], evidence['report'],
                             project or evidence['project'])


def rebind_manifest(project):
    out = project/publisher.RESULTS
    receipt = publisher.read_json(out/'completed.json')
    receipt['manifest_sha256'] = publisher.sha(out/'manifest.json')
    write(out/'completed.json', receipt)


def test_complete_archive_is_lossless_deterministic_and_portable(evidence, monkeypatch, tmp_path):
    result = publish_fixture(evidence)
    assert result['status'] == 'verified' and result['fits'] == 12
    assert evidence['calls'] == ['verify', 'reproduce']
    out = evidence['project']/publisher.RESULTS
    manifest = publisher.read_json(out/'manifest.json')
    assert 'execution/additional-raw-note.txt' in manifest['members']
    assert 'report/additional-note.txt' in manifest['members']
    assert set(manifest['weights']) == publisher._expected_names()
    with tarfile.open(out/publisher.ARCHIVE, 'r:gz') as archive:
        for member in archive:
            prefix, relative = member.name.split('/', 1)
            original = evidence[prefix]/relative
            assert archive.extractfile(member).read() == original.read_bytes()
            assert member.mtime == member.uid == member.gid == 0
    other = tmp_path/'published-again'
    publish_fixture(evidence, other)
    assert publisher.sha(out/publisher.ARCHIVE) == publisher.sha(other/publisher.RESULTS/publisher.ARCHIVE)
    assert publisher.sha(out/'manifest.json') == publisher.sha(other/publisher.RESULTS/'manifest.json')

    def forbidden(*args, **kwargs):
        raise AssertionError('Offline audit must not access original engine or model execution')

    monkeypatch.setattr(publisher.STUDY, 'verify_plan', forbidden)
    monkeypatch.setattr(publisher.STUDY, 'report', forbidden)
    assert publisher.audit(evidence['project']) == result
    with pytest.raises(FileExistsError):
        publish_fixture(evidence)


@pytest.mark.parametrize('change', ['missing_fit', 'duplicate_fit', 'mean', 'gate', 'scope', 'nonfinite'])
def test_inconsistent_summary_is_rejected(change):
    summary = summary_fixture()
    if change == 'missing_fit':
        summary['fits'].pop()
    elif change == 'duplicate_fit':
        summary['fits'][0] = copy.deepcopy(summary['fits'][1])
    elif change == 'mean':
        summary['means']['dev']['predict']['top1_teacher_agreement'] = .9
    elif change == 'gate':
        summary['continuation_gate']['passed'] = not summary['continuation_gate']['passed']
    elif change == 'scope':
        summary['novelty_established'] = True
    else:
        summary['fits'][0]['target_nll'] = float('nan')
    with pytest.raises(ValueError):
        publisher.validate_summary(summary)


@pytest.mark.parametrize('change', ['failed', 'missing_raw', 'summary_hash', 'execution_binding', 'symlink'])
def test_publication_rejects_incomplete_or_changed_inputs_before_writing(evidence, change):
    if change == 'failed':
        write(evidence['execution']/'recurrent-17/failed.json', {'status': 'failed'})
    elif change == 'missing_raw':
        (evidence['execution']/'predict-17/learning.jsonl').unlink()
    elif change == 'summary_hash':
        (evidence['report']/'summary.json').write_text(json.dumps(evidence['summary'])+'\n\n')
    elif change == 'execution_binding':
        value = publisher.read_json(evidence['report']/'completed.json')
        value['execution_receipt_sha256'] = '0'*64
        write(evidence['report']/'completed.json', value)
    else:
        (evidence['execution']/'alias.txt').symlink_to(evidence['execution']/'additional-raw-note.txt')
    with pytest.raises(ValueError):
        publish_fixture(evidence)
    assert not (evidence['project']/publisher.RESULTS).exists()


def test_changed_summary_must_reproduce_even_with_fresh_receipt(evidence, monkeypatch):
    monkeypatch.setattr(publisher.STUDY, 'report', lambda *args: {'different': 'summary'})
    with pytest.raises(ValueError, match='does not reproduce'):
        publish_fixture(evidence)


@pytest.mark.parametrize('change', ['weights', 'archive', 'manifest', 'summary', 'weight_coverage'])
def test_offline_audit_rejects_changed_publication(evidence, change):
    publish_fixture(evidence)
    project = evidence['project']
    out = project/publisher.RESULTS
    if change == 'weights':
        with (project/publisher.WEIGHTS/'predict-17/weights.pt').open('ab') as stream:
            stream.write(b'tampering')
    elif change == 'archive':
        with (out/publisher.ARCHIVE).open('ab') as stream:
            stream.write(b'tampering')
    elif change == 'manifest':
        value = publisher.read_json(out/'manifest.json')
        value['member_count'] += 1
        write(out/'manifest.json', value)
    elif change == 'weight_coverage':
        value = publisher.read_json(out/'manifest.json')
        value['weights'].pop('predict-17')
        write(out/'manifest.json', value)
        rebind_manifest(project)
    else:
        with (out/'summary.json').open('a') as stream:
            stream.write('\n')
    with pytest.raises(ValueError):
        publisher.audit(project)


def test_required_archive_coverage_cannot_be_removed_with_rehashed_manifest(evidence):
    publish_fixture(evidence)
    out = evidence['project']/publisher.RESULTS
    manifest = publisher.read_json(out/'manifest.json')
    manifest['members'].pop('execution/predict-17/dev.json')
    (out/publisher.ARCHIVE).unlink()
    publisher.make_archive(out/publisher.ARCHIVE, evidence['execution'], evidence['report'], manifest['members'])
    manifest['archive'].update(sha256=publisher.sha(out/publisher.ARCHIVE),
                               size=(out/publisher.ARCHIVE).stat().st_size)
    manifest['member_count'] = len(manifest['members'])
    manifest['raw_bytes'] = sum(row['size'] for row in manifest['members'].values())
    write(out/'manifest.json', manifest)
    rebind_manifest(evidence['project'])
    with pytest.raises(ValueError, match='lacks required'):
        publisher.audit(evidence['project'])


def test_compressed_hash_does_not_replace_member_verification(evidence):
    publish_fixture(evidence)
    out = evidence['project']/publisher.RESULTS
    manifest = publisher.read_json(out/'manifest.json')
    (evidence['execution']/'additional-raw-note.txt').write_bytes(b'X'*len(
        (evidence['execution']/'additional-raw-note.txt').read_bytes()))
    (out/publisher.ARCHIVE).unlink()
    publisher.make_archive(out/publisher.ARCHIVE, evidence['execution'], evidence['report'], manifest['members'])
    manifest['archive'].update(sha256=publisher.sha(out/publisher.ARCHIVE),
                               size=(out/publisher.ARCHIVE).stat().st_size)
    write(out/'manifest.json', manifest)
    rebind_manifest(evidence['project'])
    with pytest.raises(ValueError, match='member hash mismatch'):
        publisher.audit(evidence['project'])


def test_chart_exposes_each_seed_and_preserves_scope():
    pytest.importorskip('matplotlib')
    fig = publisher.figure(summary_fixture())
    assert len(fig.axes) == 2
    assert sum(len(collection.get_offsets()) for collection in fig.axes[0].collections) == 24
    assert sum(len(collection.get_offsets()) for collection in fig.axes[1].collections) == 24
    assert len(fig.axes[0].lines) == 2
    assert all(axis.get_ylim()[0] == 0 for axis in fig.axes)
    assert any('No Elo or novelty' in text.get_text() for text in fig.texts)
    assert any('FAILED' in text.get_text() for text in fig.texts)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_render_is_bound_to_verified_package_and_refuses_overwrite(evidence):
    pytest.importorskip('matplotlib')
    publish_fixture(evidence)
    outputs = publisher.render(evidence['project'])
    receipt = publisher.read_json(outputs['json'])
    for extension in ('png', 'svg'):
        assert receipt['plots'][extension] == publisher.sha(outputs[extension])
    with pytest.raises(FileExistsError):
        publisher.render(evidence['project'])
