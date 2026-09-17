"""Synthetic evidence packaging tests; no training, engine or model scoring."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('_test_chess_refinement_publisher',
                                             ROOT/'scripts/publish_chess_refinement.py')
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False)+'\n')


def fixture(tmp_path, kind, monkeypatch):
    source = tmp_path/kind
    source.mkdir(parents=True)
    execution = source/'execution'
    execution.mkdir()
    plan_path = source/'plan.json'
    report = source/'report' if kind == 'compute' else source/'report.json'
    if kind == 'compute':
        variants = ('policy_d4', 'policy_d8', 'policy_d16', 'successor_2_d4', 'successor_4_d4', 'successor_all_d4')
        configs = [{'id': f'predict-{seed}-{variant}', 'mode': 'predict', 'seed': seed, 'variant': variant}
                   for variant in variants for seed in (17, 29, 43)]
        plan = {'configurations': configs, 'panels': {'dev': [1, 2], 'shift': [3, 4]}, 'protocol': {}}
    else:
        write(source/'selection.json', {'splits': 'synthetic selection, no actual source rows'})
        plan = {'selection_sha256': publisher.sha(source/'selection.json'),
                'protocol': {'gate': {'mate_gain_over_frozen': .2, 'max_ordinary_mean_drop': .01,
                                       'set_gain_over_single': .05}}}
    write(plan_path, plan)
    plan_hash = publisher.sha(plan_path)
    write(source/'prepared.json', {'status': 'prepared', 'plan_sha256': plan_hash})
    write(execution/'started.json', {'status': 'started', 'plan_sha256': plan_hash})
    (execution/'extra-original.bin').write_bytes(b'\x00\xfforiginal bytes\n')
    if kind == 'compute':
        complete = {'status': 'completed', 'plan_sha256': plan_hash, 'decisions': len(configs)*4,
                    'files': {'extra-original.bin': publisher.sha(execution/'extra-original.bin')}}
        write(execution/'completed.json', complete)
        records = [{**config, 'configuration': config['id'], 'split': split, 'examples': 2,
                    'agreement': .5, 'latency_mean_ms': 2., 'latency_p50_ms': 1., 'latency_p95_ms': 3.}
                   for config in configs for split in ('dev', 'shift')]
        summary = {'status': 'completed', 'plan_sha256': plan_hash, 'configurations': records,
                   'execution_receipt_sha256': publisher.sha(execution/'completed.json'),
                   'novelty_established': False, 'elo_estimate': None}
        write(report/'summary.json', summary)
        write(report/'completed.json', {'status': 'completed', 'plan_sha256': plan_hash,
                                        'summary_sha256': publisher.sha(report/'summary.json'),
                                        'execution_receipt_sha256': publisher.sha(execution/'completed.json')})
    else:
        results, receipts = {}, {}
        splits = ('mate_dev', 'mate_confirm', 'dev', 'shift')
        for arm, correct in (('frozen', 1), ('single', 2), ('set', 3)):
            for seed in (17, 29, 43):
                name = f'{arm}-{seed}'
                directory = execution/name
                directory.mkdir()
                filenames = [f'{split}.jsonl' for split in splits]
                if arm != 'frozen':
                    filenames += ['weights.pt', 'learning.jsonl']
                for filename in filenames:
                    (directory/filename).write_bytes(f'Original synthetic {name}/{filename}\n'.encode())
                metrics = {split: {'examples': 4, 'correct': correct, 'accuracy': correct/4} for split in splits}
                results[name] = {'metrics': metrics}
                write(directory/'completed.json', {'status': 'completed', 'arm': arm, 'seed': seed,
                                                   'plan_sha256': plan_hash, 'metrics': metrics,
                                                   'files': {name: publisher.sha(directory/name) for name in filenames}})
                receipts[name] = publisher.sha(directory/'completed.json')
        write(execution/'schedule.json', {'synthetic': True})
        write(execution/'completed.json', {'status': 'completed', 'plan_sha256': plan_hash, 'fits': receipts,
                                           'schedule_sha256': publisher.sha(execution/'schedule.json')})
        summary = {'plan_sha256': plan_hash, 'execution_receipt_sha256': publisher.sha(execution/'completed.json'),
                   'results': results,
                   'mean_accuracy': {arm: dict.fromkeys(splits, value) for arm, value in
                                     (('frozen', .25), ('single', .5), ('set', .75))},
                   'set_minus_frozen': dict.fromkeys(splits, .5), 'set_minus_single': dict.fromkeys(splits, .25),
                   'gates': {'targeted_skill': True, 'ordinary_retention': True, 'set_objective': True},
                   'new_training_engine_calls': 0}
        write(report, summary)
    calls = []

    def revalidate(received_plan, received_execution, destination):
        assert received_plan == plan_path and received_execution == execution
        assert not destination.exists()
        calls.append(destination)
        if kind == 'compute':
            write(destination/'summary.json', summary)
        else:
            write(destination, summary)
        return copy.deepcopy(summary)

    monkeypatch.setattr(publisher, 'load_study', lambda requested: SimpleNamespace(report=revalidate))
    return {'kind': kind, 'plan': plan_path, 'execution': execution, 'report': report,
            'out': source/'published', 'models': source/'models'}, summary, calls


@pytest.mark.parametrize('kind', ['compute', 'mate'])
def test_publish_revalidates_once_and_preserves_every_original_file(tmp_path, monkeypatch, kind):
    arguments, summary, calls = fixture(tmp_path, kind, monkeypatch)
    verified = publisher.publish(**arguments)
    assert verified['status'] == 'verified' and verified['kind'] == kind and len(calls) == 1
    assert not calls[0].exists()
    out = arguments['out']
    manifest = publisher.read(out/'manifest.json')
    assert 'execution/extra-original.bin' in manifest['members']
    assert publisher.read(out/'summary.json') == summary
    assert (out/'plan.json').read_bytes() == arguments['plan'].read_bytes()
    assert (out/'prepared.json').read_bytes() == (arguments['plan'].parent/'prepared.json').read_bytes()
    if kind == 'mate':
        assert verified['published_weights'] == 6
        assert (out/'selection.json').read_bytes() == (arguments['plan'].parent/'selection.json').read_bytes()
        assert (arguments['models']/'LICENSE').read_bytes() == (ROOT/'LICENSE').read_bytes()
        for name in publisher.MATE_NAMES:
            assert (arguments['models']/name/'weights.pt').read_bytes() == (
                arguments['execution']/name/'weights.pt').read_bytes()
    else:
        assert verified['published_weights'] == 0 and not arguments['models'].exists()
    with pytest.raises(FileExistsError):
        publisher.publish(**arguments)


def test_archive_is_deterministic_and_roundtrips_binary_bytes(tmp_path, monkeypatch):
    arguments, _, _ = fixture(tmp_path, 'compute', monkeypatch)
    paths, members = publisher.inventory(arguments['execution'], arguments['report'], 'compute')
    publisher.make_archive(tmp_path/'one.tar.gz', paths, members)
    publisher.make_archive(tmp_path/'two.tar.gz', paths, members)
    assert (tmp_path/'one.tar.gz').read_bytes() == (tmp_path/'two.tar.gz').read_bytes()
    documents = publisher.inspect_archive(tmp_path/'one.tar.gz', members)
    assert documents['execution/completed.json']['status'] == 'completed'


@pytest.mark.parametrize('corruption', ['summary', 'prepared', 'failed', 'symlink'])
def test_preflight_rejects_inconsistent_or_unsafe_source_evidence(tmp_path, monkeypatch, corruption):
    arguments, summary, _ = fixture(tmp_path, 'compute', monkeypatch)
    if corruption == 'summary':
        changed = copy.deepcopy(summary)
        changed['configurations'][0]['agreement'] = .6
        write(arguments['report']/'summary.json', changed)
    elif corruption == 'prepared':
        write(arguments['plan'].parent/'prepared.json', {'status': 'prepared', 'plan_sha256': '0'*64})
    elif corruption == 'failed':
        write(arguments['execution']/'failed.json', {'status': 'failed'})
    else:
        (arguments['execution']/'link.bin').symlink_to(arguments['execution']/'extra-original.bin')
    with pytest.raises(ValueError):
        publisher.publish(**arguments)
    assert not arguments['out'].exists()


@pytest.mark.parametrize('corruption', ['archive', 'summary', 'weights', 'license', 'manifest'])
def test_portable_audit_rejects_changed_publication_artifacts(tmp_path, monkeypatch, corruption):
    arguments, _, _ = fixture(tmp_path, 'mate', monkeypatch)
    publisher.publish(**arguments)
    targets = {'archive': arguments['out']/publisher.ARCHIVE, 'summary': arguments['out']/'summary.json',
               'weights': arguments['models']/'single-17/weights.pt', 'license': arguments['models']/'LICENSE',
               'manifest': arguments['out']/'manifest.json'}
    with targets[corruption].open('ab') as stream:
        stream.write(b' ')
    with pytest.raises(ValueError):
        publisher.audit(arguments['out'], models=arguments['models'])


def test_missing_fit_and_bad_aggregate_are_rejected(tmp_path, monkeypatch):
    arguments, summary, _ = fixture(tmp_path, 'mate', monkeypatch)
    summary['results'].pop('set-43')
    with pytest.raises(ValueError, match='nine-model'):
        publisher.validate_summary('mate', publisher.read(arguments['plan']), summary)
    arguments, summary, _ = fixture(tmp_path/'second', 'mate', monkeypatch)
    summary['mean_accuracy']['set']['mate_confirm'] = .8
    with pytest.raises(ValueError, match='arithmetic'):
        publisher.validate_summary('mate', publisher.read(arguments['plan']), summary)


def test_render_uses_audited_published_summaries_and_refuses_overwrite(tmp_path, monkeypatch):
    compute_args, _, _ = fixture(tmp_path, 'compute', monkeypatch)
    publisher.publish(**compute_args)
    mate_args, _, _ = fixture(tmp_path, 'mate', monkeypatch)
    publisher.publish(**mate_args)
    prefix = tmp_path/'figures/refinement'
    result = publisher.render(compute_args['out'], mate_args['out'], prefix, mate_args['models'])
    assert set(result) == {'png', 'svg', 'json'}
    assert Path(result['png']).read_bytes().startswith(b'\x89PNG')
    assert b'<svg' in Path(result['svg']).read_bytes()
    receipt = publisher.read(result['json'])
    assert receipt['verification']['mate']['published_weights'] == 6
    with pytest.raises(FileExistsError):
        publisher.render(compute_args['out'], mate_args['out'], prefix, mate_args['models'])
