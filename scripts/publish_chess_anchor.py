"""Lossless publication and portable integrity audit of the anchor chess study.

Publish reproduces the frozen report without training or inference. Audit and
render require no engine, GPU, PyTorch or original execution directory. The
portable audit verifies bytes, receipt chains and summary arithmetic; it does
not independently rerun model decisions or Stockfish analyses.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    'residual_fixed': ('residual', 'fixed'),
    'residual_mixed': ('residual', 'mixed'),
    'anchor_fixed': ('anchor', 'fixed'),
    'anchor_mixed': ('anchor', 'mixed'),
}
SEEDS = (17, 29, 43)
DEPTHS = (2, 4, 8, 16)
SPLITS = ('dev', 'shift')
ARCHIVE = 'execution-and-report.tar.gz'
COLORS = {'residual_fixed': '#64748b', 'residual_mixed': '#426aa4',
          'anchor_fixed': '#b87730', 'anchor_mixed': '#168277'}
LABELS = {arm: arm.replace('_', ' ').capitalize() for arm in ARMS}
EVALUATIONS = {f'{split}-d{depth}' for split in SPLITS for depth in DEPTHS}
DATA_FILES = {'started.json', 'excluded-states.json', 'games.jsonl', 'analyses.jsonl',
              'dev.jsonl', 'shift.jsonl'}
REGRET_FILES = {'engine.json', 'analyses.jsonl', 'regret.jsonl'}
FIT_FILES = {'weights.pt', 'learning.jsonl', 'training.json', 'latency.json'} | {
    f'{key}.jsonl' for key in EVALUATIONS}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result

    def invalid(_):
        raise ValueError('Nonfinite JSON number')

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def read(path):
    return decode(Path(path).read_bytes())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def number(value, low=None, high=None):
    require(type(value) in (float, int) and math.isfinite(value), 'Invalid finite numeric metric')
    require((low is None or value >= low) and (high is None or value <= high), 'Metric outside bounds')
    return value


def close(actual, expected):
    number(actual)
    require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10), 'Inconsistent summary aggregate')


def mean(values):
    values = list(values)
    require(bool(values), 'Cannot aggregate an empty panel')
    return math.fsum(values)/len(values)


def expected_names():
    return {f'{arm}-{seed}' for arm in ARMS for seed in SEEDS}


def validate_plan(plan):
    p = plan['protocol']
    require(p['version'] == 'chess-anchor-v1' and p['seeds'] == list(SEEDS)
            and p['arms'] == {a: list(v) for a, v in ARMS.items()}
            and p['evaluation_depths'] == list(DEPTHS), 'Unexpected anchor protocol panel')
    require(p['train_examples'] == 32768 and p['epochs'] == 6 and p['batch_size'] == 128
            and p['updates_per_fit'] == 1536 and p['core_iterations_per_fit'] == 6144
            and p['mixed_depths'] == [2, 4, 6] and p['width'] == 32 and p['default_depth'] == 4,
            'Unexpected matched training budget')
    require(plan['active_parameters'] == 33185 and plan['stored_parameters'] == 43726,
            'Parameter counts mismatch')
    configurations = plan['configurations']
    require(len(configurations) == 12 and {c['name'] for c in configurations} == expected_names(),
            'Plan must contain all 12 fits')
    for c in configurations:
        require(c == {'name': f"{c['arm']}-{c['seed']}", 'arm': c['arm'], 'seed': c['seed'],
                      'recurrence': ARMS[c['arm']][0], 'regime': ARMS[c['arm']][1]}
                and c['seed'] in SEEDS, 'Configuration identity mismatch')


def validate_summary(plan, summary):
    validate_plan(plan)
    p = plan['protocol']
    require(summary['status'] == 'completed' and summary['novelty_established'] is False
            and summary['elo_estimate'] is None and summary['scope'] == p['scope'],
            'Expected a scoped completed development summary')
    names = expected_names()
    for field in ('metrics', 'costs', 'latency'):
        require(set(summary[field]) == names, 'Summary requires every fit')
    counts = {s['name']: s['examples'] for s in plan['fresh_data']['splits']}
    require(counts == {'dev': 2048, 'shift': 2048}, 'Fresh evaluation coverage mismatch')
    for name in names:
        require(set(summary['metrics'][name]) == EVALUATIONS, 'Incomplete depth/split summary')
        for key, row in summary['metrics'][name].items():
            n = counts[key.split('-')[0]]
            require(row['examples'] == n and type(row['correct']) is int
                    and 0 <= row['correct'] <= n, 'Invalid evaluation count')
            close(row['agreement'], row['correct']/n)
            for field in ('target_nll', 'entropy', 'hidden_rms', 'logit_span'):
                number(row[field], 0)
            number(row['value_mae'], 0, 2)
            number(row['mean_confidence'], 0, 1)
            if row['correct'] == n:
                require(row['mismatch_confidence'] is None, 'Perfect agreement has no mismatches')
            else:
                number(row['mismatch_confidence'], 0, 1)
        cost = summary['costs'][name]
        require(cost['updates'] == p['updates_per_fit'] and cost['core_iterations'] == p['core_iterations_per_fit']
                and cost['examples_seen'] == p['train_examples']*p['epochs'], 'Training budget mismatch')
        number(cost['training_seconds'], 0)
        require(set(summary['latency'][name]) == {str(d) for d in DEPTHS}, 'Incomplete latency depths')
        for depth in DEPTHS:
            row = summary['latency'][name][str(depth)]
            require(row['device'] == 'cpu' and row['depth'] == depth
                    and row['torch_threads'] == p['torch_threads'], 'Latency identity mismatch')
            require(len(row['records']) == 2*p['latency_positions_per_split']
                    and len(row['warmup_records']) == p['latency_warmups'], 'Incomplete latency calls')
            for key, total in (('records', 'total_wall_ms'), ('warmup_records', 'warmup_wall_ms')):
                for r in row[key]:
                    number(r['wall_ms'], 0)
                close(row[total], math.fsum(r['wall_ms'] for r in row[key]))
    require(set(summary['means']) == set(ARMS), 'Incomplete arm means')
    for arm in ARMS:
        require(set(summary['means'][arm]) == {str(d) for d in DEPTHS}, 'Incomplete mean depths')
        for depth in DEPTHS:
            require(set(summary['means'][arm][str(depth)]) == set(SPLITS), 'Incomplete mean splits')
            for split in SPLITS:
                close(summary['means'][arm][str(depth)][split], mean(
                    summary['metrics'][f'{arm}-{seed}'][f'{split}-d{depth}']['agreement'] for seed in SEEDS))
    expected_regret = {f'{arm}-{seed}-d{d}' for arm in ARMS if arm.endswith('_mixed')
                       for seed in SEEDS for d in (4, 8)}
    require(set(summary['regret']) == expected_regret, 'Incomplete graded configurations')
    for row in summary['regret'].values():
        require(set(row) == set(SPLITS), 'Incomplete regret splits')
        for value in row.values():
            number(value, -2, 2)
    comparisons, checks = [], []
    for split in SPLITS:
        for label, candidate, cd, reference, rd in (
            ('primary', 'anchor_mixed', 4, 'residual_mixed', 4),
            ('extra_depth', 'anchor_mixed', 8, 'anchor_mixed', 4),
            ('depth8_architecture', 'anchor_mixed', 8, 'residual_mixed', 8),
        ):
            deltas = [summary['metrics'][f'{candidate}-{s}'][f'{split}-d{cd}']['agreement']
                      - summary['metrics'][f'{reference}-{s}'][f'{split}-d{rd}']['agreement'] for s in SEEDS]
            regret = mean(summary['regret'][f'{candidate}-{s}-d{cd}'][split]
                          - summary['regret'][f'{reference}-{s}-d{rd}'][split] for s in SEEDS)
            comparisons.append((label, split, deltas, regret))
            threshold = p['gate']['primary_gain'] if label == 'primary' else (
                p['gate']['extra_depth_gain'] if label == 'extra_depth' else 0.)
            checks.extend([
                {'gate': label, 'split': split, 'metric': 'agreement_gain', 'observed': mean(deltas),
                 'threshold': threshold, 'passed': mean(deltas) >= threshold if label != 'depth8_architecture'
                 else mean(deltas) > 0},
                {'gate': label, 'split': split, 'metric': 'bounded_regret_change', 'observed': regret,
                 'threshold': 0., 'passed': regret < 0 if label == 'primary' else regret <= 0},
            ])
            if label == 'primary':
                checks.append({'gate': label, 'split': split, 'metric': 'worst_seed_gain',
                               'observed': min(deltas), 'threshold': -p['gate']['worst_seed_deficit'],
                               'passed': min(deltas) >= -p['gate']['worst_seed_deficit']})
    require(len(summary['checks']) == len(checks), 'Gate coverage mismatch')
    for actual, expected in zip(summary['checks'], checks, strict=True):
        close(actual['observed'], expected['observed'])
        require({k: v for k, v in actual.items() if k != 'observed'}
                == {k: v for k, v in expected.items() if k != 'observed'}, 'Gate rule or verdict mismatch')
    require(summary['primary_passed'] is all(c['passed'] for c in checks if c['gate'] == 'primary')
            and summary['extra_compute_passed'] is all(c['passed'] for c in checks if c['gate'] != 'primary'),
            'Gate verdict mismatch')
    require(len(summary['comparisons']) == len(comparisons), 'Comparison coverage mismatch')
    for row, (label, split, deltas, regret) in zip(summary['comparisons'], comparisons, strict=True):
        require(row['comparison'] == label and row['split'] == split, 'Comparison identity mismatch')
        require(len(row['seed_agreement_deltas']) == 3, 'Incomplete paired seeds')
        for actual, expected in zip(row['seed_agreement_deltas'], deltas, strict=True):
            close(actual, expected)
        close(row['bounded_regret_delta'], regret)
        interval = row['game_cluster_interval']
        close(interval['mean'], mean(deltas))
        require(interval['positions'] == counts[split] and type(interval['games']) is int
                and 1 <= interval['games'] <= counts[split] and interval['scope'] == p['uncertainty'],
                'Bootstrap scope mismatch')
        require(-1 <= number(interval['lower']) <= number(interval['upper']) <= 1, 'Invalid interval bounds')
    require(set(summary['interaction_at_depth4']) == set(SPLITS), 'Incomplete factorial interaction')
    for split in SPLITS:
        m = summary['means']
        close(summary['interaction_at_depth4'][split], m['anchor_mixed']['4'][split]-m['residual_mixed']['4'][split]
              - m['anchor_fixed']['4'][split]+m['residual_fixed']['4'][split])
    return summary


def inventory(execution, report):
    result = {}
    for prefix, directory in (('execution', Path(execution)), ('report', Path(report))):
        require(directory.is_dir() and not directory.is_symlink(), 'Evidence roots must be real directories')
        for path in sorted(directory.rglob('*')):
            require(not path.is_symlink() and (path.is_file() or path.is_dir()), 'Nonregular evidence artifact')
            if path.is_file():
                require(path.name != 'failed.json', 'Failed evidence cannot be published')
                result[f'{prefix}/{path.relative_to(directory).as_posix()}'] = {
                    'sha256': sha(path), 'size': path.stat().st_size}
    return result


def required_members():
    result = {'execution/started.json', 'execution/completed.json', 'execution/baselines.json',
              'execution/panels.json', 'report/summary.json', 'report/completed.json'}
    result |= {f'execution/data/{name}' for name in DATA_FILES | {'completed.json'}}
    result |= {f'execution/regret/{name}' for name in REGRET_FILES | {'completed.json'}}
    result |= {f'execution/{fit}/{name}' for fit in expected_names() for name in FIT_FILES | {'completed.json'}}
    return result


def keep_document(name):
    return name.endswith(('/completed.json', '/training.json', '/latency.json')) or name in {
        'execution/started.json', 'execution/panels.json', 'execution/regret/engine.json'}


def validate_chain(plan, plan_hash, summary, members, documents):
    require(required_members() <= set(members), 'Archive lacks required full-panel evidence')
    require(summary['plan_sha256'] == plan_hash, 'Summary plan mismatch')

    def digest(name):
        return members[name]['sha256']

    def files(prefix, receipt, required):
        require(receipt['status'] == 'completed' and set(receipt['files']) == required, 'Receipt file coverage mismatch')
        for name, value in receipt['files'].items():
            require(digest(f'{prefix}/{name}') == value, 'Archived receipt hash mismatch')

    complete, report = documents['execution/completed.json'], documents['report/completed.json']
    started = documents['execution/started.json']
    require(complete['status'] == 'completed' and complete['plan_sha256'] == plan_hash
            and set(complete['fits']) == expected_names() and started['status'] == 'started'
            and started['plan_sha256'] == plan_hash, 'Execution completion identity mismatch')
    require(report['status'] == 'completed' and report['plan_sha256'] == plan_hash
            and report['summary_sha256'] == digest('report/summary.json')
            and report['execution_receipt_sha256'] == digest('execution/completed.json')
            and summary['execution_receipt_sha256'] == digest('execution/completed.json'), 'Report chain mismatch')
    for field, name in (('data_receipt_sha256', 'data/completed.json'),
                        ('panels_sha256', 'panels.json'), ('baselines_sha256', 'baselines.json'),
                        ('regret_receipt_sha256', 'regret/completed.json')):
        require(complete[field] == digest('execution/'+name), 'Execution component binding mismatch')
    data, regret = documents['execution/data/completed.json'], documents['execution/regret/completed.json']
    files('execution/data', data, DATA_FILES)
    files('execution/regret', regret, REGRET_FILES)
    require(summary['fresh_data_cost'] == data and summary['engine_cost'] == regret['cost'], 'Summary cost receipt mismatch')
    cost, p = regret['cost'], plan['protocol']
    require(type(cost['calls']) is int and 0 < cost['calls'] <= p['regret_call_ceiling']
            and cost['requested_nodes'] == cost['calls']*p['regret_nodes']
            and cost['requested_nodes'] <= p['regret_node_ceiling'], 'Engine call budget mismatch')
    engine = documents['execution/regret/engine.json']
    require(engine['sha256'] == plan['engine_sha256'] and engine['id']['name'].startswith('Stockfish 19'),
            'Engine identity mismatch')
    latency_indices = documents['execution/panels.json']['latency_indices']
    require(len(latency_indices) == 2*p['latency_positions_per_split']
            and len(set(latency_indices)) == len(latency_indices), 'Latency panel identity mismatch')
    initial = {}
    for config in plan['configurations']:
        name, prefix = config['name'], f"execution/{config['name']}"
        receipt, trained = documents[prefix+'/completed.json'], documents[prefix+'/training.json']
        require(complete['fits'][name] == digest(prefix+'/completed.json'), 'Fit receipt binding mismatch')
        files(prefix, receipt, FIT_FILES)
        for identity in (receipt, trained):
            require(identity['status'] == 'completed' and identity['plan_sha256'] == plan_hash
                    and all(identity[k] == v for k, v in config.items()), 'Fit identity mismatch')
        require(trained['weights_sha256'] == digest(prefix+'/weights.pt')
                and trained['learning_sha256'] == digest(prefix+'/learning.jsonl')
                and trained['data_receipt_sha256'] == digest('execution/data/completed.json'), 'Training input binding mismatch')
        require({k: trained[k] for k in summary['costs'][name]} == summary['costs'][name], 'Fit budget summary mismatch')
        state = trained['initial_state_sha256']
        require(isinstance(state, str) and len(state) == 64 and set(state) <= set('0123456789abcdef'),
                'Invalid initialization digest')
        require(initial.setdefault(config['seed'], state) == state, 'Same-seed initial weights differ')
        require(set(receipt['evaluation']) == EVALUATIONS, 'Fit evaluation membership mismatch')
        for key in EVALUATIONS:
            require(receipt['evaluation'][key]['metrics'] == summary['metrics'][name][key], 'Fit metrics differ from summary')
            number(receipt['evaluation'][key]['evaluation_wall_seconds'], 0)
        require(documents[prefix+'/latency.json'] == summary['latency'][name], 'Latency summary mismatch')
        for measurement in summary['latency'][name].values():
            require([r['index'] for r in measurement['records']] == latency_indices,
                    'Latency records differ from selected panel')
            require([r['index'] for r in measurement['warmup_records']] == list(range(p['latency_warmups']))
                    and all(r['id'] == 'starting-board' for r in measurement['warmup_records']),
                    'Warmup identity mismatch')


def make_archive(path, execution, report, members):
    roots = {'execution': Path(execution), 'report': Path(report)}
    with (Path(path).open('xb') as raw,
          gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as compressed,
          tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as archive):
        for name, metadata in sorted(members.items()):
            prefix, relative = name.split('/', 1)
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = metadata['size'], 0o644, 0
            with (roots[prefix]/relative).open('rb') as stream:
                archive.addfile(info, stream)


def inspect_archive(path, expected):
    actual, documents = {}, {}
    with tarfile.open(path, mode='r|gz') as archive:
        for member in archive:
            name, parts = member.name, PurePosixPath(member.name).parts
            require(member.isfile() and name not in actual and name in expected and parts
                    and parts[0] in ('execution', 'report') and '..' not in parts
                    and str(PurePosixPath(name)) == name and 'failed.json' not in parts,
                    'Unexpected, duplicated or unsafe archive member')
            require(member.size == expected[name]['size'], 'Archive member size mismatch')
            stream, digest, chunks = archive.extractfile(member), hashlib.sha256(), []
            keep = keep_document(name)
            while chunk := stream.read(1024*1024):
                digest.update(chunk)
                if keep:
                    chunks.append(chunk)
            actual[name] = {'sha256': digest.hexdigest(), 'size': member.size}
            require(actual[name] == expected[name], 'Archive member hash mismatch')
            if keep:
                documents[name] = decode(b''.join(chunks))
    require(actual == expected, 'Archive membership differs from manifest')
    return actual, documents


def load_study():
    spec = importlib.util.spec_from_file_location('_anchor_publication_study', ROOT/'scripts/chess_anchor_study.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_card(plan, plan_hash, weights):
    lines = [
        '# OpenJev anchor chess checkpoints', '', 'License: MIT. All 12 final checkpoints are retained.', '',
        ('This is a fresh-initialization supervised ablation on fixed Stockfish-labeled positions. '
        'Residual and fixed-encoder-anchor recurrence share the same architecture, heads and seed-specific initial tensors. '
        'Fixed training uses depth 4; mixed training uses equal counts at depths 2, 4 and 6. '
        'There is no cross-move memory, search, reinforcement learning, Elo estimate or novelty claim.'), '',
        ('Each model has 33,185 active parameters and 43,726 stored parameters. The auxiliary head is unused. '
        'Training uses legal-move cross-entropy plus 0.5 bounded-value MSE. '
        'These are original final weights, without selection or a promoted winner alias.'), '',
        f'Frozen plan SHA-256: `{plan_hash}`.',
        f'Anchor source SHA-256: `{plan["sources"]["src/openjev/research/chess_anchor.py"]}`.',
        f'Spatial initializer source SHA-256: `{plan["sources"]["src/openjev/research/chess_spatial.py"]}`.', '',
        '| Checkpoint | Recurrence | Training depths | Seed | Paired residual control |',
        '| --- | --- | --- | --- | --- |',
    ]
    for name, row in sorted(weights.items()):
        depths = '4' if row['regime'] == 'fixed' else '2 / 4 / 6'
        lines.append(f'| [{name}]({row["path"]}) | {row["recurrence"]} | {depths} | {row["seed"]} '
                     f'| {row["paired_residual"]} |')
    lines.extend(['', ('Load with `AnchorChess.load(path, expected_plan_sha256=..., '
                  'expected_recurrence=..., expected_seed=...)`. The distinct checkpoint format rejects old spatial models.'), ''])
    return '\n'.join(lines)


def publish(plan_path, execution, report, out, models=None):
    plan_path, execution, report, out = map(Path, (plan_path, execution, report, out))
    models = ROOT/'models/chess-anchor-v1' if models is None else Path(models)
    if out.exists() or models.exists():
        raise FileExistsError('Publication or models directory already exists')
    for destination in (out.resolve(), models.resolve()):
        require(not any(destination.is_relative_to(root.resolve()) for root in (execution, report)),
                'Publication destinations must be outside source evidence')
    require(not out.resolve().is_relative_to(models.resolve()) and not models.resolve().is_relative_to(out.resolve()),
            'Publication and model directories must not overlap')
    study = load_study()
    plan = study.verify_plan(plan_path)
    summary = validate_summary(plan, read(report/'summary.json'))
    members = inventory(execution, report)
    documents = {name: read((execution if name.startswith('execution/') else report)/name.split('/', 1)[1])
                 for name in members if keep_document(name)}
    validate_chain(plan, sha(plan_path), summary, members, documents)
    with tempfile.TemporaryDirectory(prefix='openjev-anchor-publication-') as temporary:
        reproduced = study.report(plan_path, execution, Path(temporary)/'report')
    require(summary == reproduced, 'Supplied summary does not reproduce from the frozen report')
    out.mkdir(parents=True, exist_ok=False)
    models.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(plan_path, out/'plan.json')
    shutil.copyfile(report/'summary.json', out/'summary.json')
    make_archive(out/ARCHIVE, execution, report, members)
    require(members == inventory(execution, report), 'Evidence changed while being packaged')
    weights = {}
    for config in sorted(plan['configurations'], key=lambda c: c['name']):
        name = config['name']
        original = f'execution/{name}/weights.pt'
        target = models/name/'weights.pt'
        target.parent.mkdir()
        shutil.copyfile(execution/name/'weights.pt', target)
        require(sha(target) == members[original]['sha256'] and target.stat().st_size == members[original]['size'],
                'Copied checkpoint differs from original')
        weights[name] = {**config, **members[original], 'path': f'{name}/weights.pt', 'original_member': original,
                         'paired_residual': f"residual_{config['regime']}-{config['seed']}",
                         'same_initialization_group': [f'{arm}-{config["seed"]}' for arm in ARMS],
                         'plan_sha256': sha(plan_path)}
    (models/'README.md').write_text(model_card(plan, sha(plan_path), weights))
    shutil.copyfile(ROOT/'LICENSE', models/'LICENSE')
    models_relative = Path(os.path.relpath(models.resolve(), out.resolve())).as_posix()
    primary, extra = ('PASSED' if summary[key] else 'FAILED' for key in ('primary_passed', 'extra_compute_passed'))
    (out/'README.md').write_text(
        '# Anchor chess development evidence\n\n'
        f'Frozen primary gate: **{primary}**. Extra-compute gate: **{extra}**.\n\n'
        f'[Summary](summary.json) · [Plan](plan.json) · [All raw evidence]({ARCHIVE}) · '
        f'[Checkpoint model card]({models_relative}/README.md) · [Hash manifest](manifest.json)\n\n'
        'The archive preserves every execution and report file, including fresh positions, generator traces, '
        'raw engine calls, training logs, predictions and original weights. All 12 checkpoint aliases are retained. '
        'This is a development ablation with three seeds, not an Elo estimate or proof of novelty.\n')
    manifest = {'version': 1, 'plan_sha256': sha(out/'plan.json'),
                'archive': {'path': ARCHIVE, 'sha256': sha(out/ARCHIVE), 'size': (out/ARCHIVE).stat().st_size},
                'members': members, 'member_count': len(members), 'raw_bytes': sum(r['size'] for r in members.values()),
                'models_directory': models_relative, 'weights': weights,
                'model_files': {name: sha(models/name) for name in ('README.md', 'LICENSE')},
                'publication_readme_sha256': sha(out/'README.md')}
    write_new(out/'manifest.json', manifest)
    actual, documents = inspect_archive(out/ARCHIVE, members)
    validate_chain(plan, sha(out/'plan.json'), summary, actual, documents)
    write_new(out/'completed.json', {'status': 'completed', 'fits': 12,
                                    'plan_sha256': sha(out/'plan.json'), 'summary_sha256': sha(out/'summary.json'),
                                    'manifest_sha256': sha(out/'manifest.json'), 'publisher_sha256': sha(__file__),
                                    'source_report_reproduced': True, 'archive_roundtrip_verified': True,
                                    'copied_weights_match_originals': True})
    return audit(out)


def audit(publication):
    out = Path(publication)
    receipt, manifest = read(out/'completed.json'), read(out/'manifest.json')
    plan = read(out/'plan.json')
    summary = validate_summary(plan, read(out/'summary.json'))
    plan_hash = sha(out/'plan.json')
    require(receipt['status'] == 'completed' and receipt['fits'] == 12
            and receipt['plan_sha256'] == plan_hash == manifest['plan_sha256']
            and receipt['summary_sha256'] == sha(out/'summary.json')
            and receipt['manifest_sha256'] == sha(out/'manifest.json') and not (out/'failed.json').exists(),
            'Publication completion receipt mismatch')
    require(all(receipt.get(key) is True for key in ('source_report_reproduced', 'archive_roundtrip_verified',
                                                    'copied_weights_match_originals')),
            'Missing publication verification receipt')
    archive = manifest['archive']
    require(archive['path'] == ARCHIVE and archive['sha256'] == sha(out/ARCHIVE)
            and archive['size'] == (out/ARCHIVE).stat().st_size, 'Compressed archive checksum mismatch')
    members, documents = inspect_archive(out/ARCHIVE, manifest['members'])
    require(manifest['member_count'] == len(members)
            and manifest['raw_bytes'] == sum(r['size'] for r in members.values())
            and members['report/summary.json']['sha256'] == sha(out/'summary.json')
            and set(manifest['weights']) == expected_names(), 'Manifest totals or checkpoint coverage mismatch')
    validate_chain(plan, plan_hash, summary, members, documents)
    require(not Path(manifest['models_directory']).is_absolute(), 'Model directory must be a portable relative path')
    models = out/manifest['models_directory']
    require(models.is_dir() and not models.is_symlink(), 'Model directory must be a real directory')
    for name, row in manifest['weights'].items():
        arm, seed = name.rsplit('-', 1)
        original = f'execution/{name}/weights.pt'
        relative = f'{name}/weights.pt'
        path = models/relative
        require(row['name'] == name and row['arm'] == arm and row['seed'] == int(seed)
                and (row['recurrence'], row['regime']) == ARMS[arm]
                and row['paired_residual'] == f"residual_{row['regime']}-{seed}"
                and row['same_initialization_group'] == [f'{a}-{seed}' for a in ARMS]
                and row['original_member'] == original and row['path'] == relative
                and row['plan_sha256'] == plan_hash
                and row['sha256'] == members[original]['sha256'] and row['size'] == members[original]['size']
                and not path.is_symlink() and not path.parent.is_symlink()
                and sha(path) == row['sha256'] and path.stat().st_size == row['size'],
                'Published checkpoint identity or original bytes mismatch')
    require(set(manifest['model_files']) == {'README.md', 'LICENSE'}, 'Missing model card or MIT license')
    for name, digest in manifest['model_files'].items():
        require(not (models/name).is_symlink() and sha(models/name) == digest, 'Model documentation checksum mismatch')
    require(sha(out/'README.md') == manifest['publication_readme_sha256'], 'Publication index changed')
    return {'status': 'verified', 'fits': 12, 'evaluations': 96, 'members': len(members),
            'plan_sha256': plan_hash, 'archive_sha256': archive['sha256'], 'summary_sha256': sha(out/'summary.json')}


def mismatch_mean(summary, arm, depth):
    rows = [summary['metrics'][f'{arm}-{s}'][f'{split}-d{depth}'] for s in SEEDS for split in SPLITS]
    count = sum(row['examples']-row['correct'] for row in rows)
    if not count:
        return float('nan')
    return math.fsum(row['mismatch_confidence']*(row['examples']-row['correct'])
                     for row in rows if row['mismatch_confidence'] is not None)/count


def figure(plan, summary):
    validate_summary(plan, summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.hashsalt': 'chess-anchor-v1'})
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.patch.set_facecolor('#fbfcfe')
    fig.subplots_adjust(left=.075, right=.98, top=.83, bottom=.19, hspace=.43, wspace=.23)
    fig.text(.075, .955, 'OpenJev: fixed-anchor recurrence', fontsize=21, weight='bold', color='#192b3c')
    fig.text(.075, .918, 'Fresh development positions | 4 arms x 3 seeds | all final checkpoints', color='#536272')
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xscale('log', base=2)
        ax.set_xticks(DEPTHS, ['2', '4', '8*', '16*'])
        ax.set_xlabel('Internal recurrent iterations')
        ax.axvspan(6, 17, color='#dce4ee', alpha=.3, zorder=0)
    for split, ax in zip(SPLITS, axes[0], strict=True):
        ax.set_title('Teacher top-move agreement: '+('development' if split == 'dev' else 'shifted development'), loc='left')
        ax.set_ylabel('Agreement (%)')
        for arm in ARMS:
            for index, seed in enumerate(SEEDS):
                values = [summary['metrics'][f'{arm}-{seed}'][f'{split}-d{d}']['agreement']*100 for d in DEPTHS]
                ax.plot(DEPTHS, values, color=COLORS[arm], linewidth=.7, alpha=.45,
                        marker=('o', 's', '^')[index], markersize=4, markerfacecolor='none')
            ax.plot(DEPTHS, [summary['means'][arm][str(d)][split]*100 for d in DEPTHS],
                    color=COLORS[arm], linewidth=2.3, linestyle='--' if arm.endswith('mixed') else '-')
        ax.set_ylim(bottom=0)
    ax = axes[1, 0]
    ax.set_title('Confidence when disagreeing with teacher', loc='left')
    ax.set_ylabel('Mean maximum probability (%)')
    ax.set_ylim(0, 100)
    for arm in ARMS:
        ax.plot(DEPTHS, [mismatch_mean(summary, arm, d)*100 for d in DEPTHS], color=COLORS[arm],
                marker='o', linestyle='--' if arm.endswith('mixed') else '-')
    ax = axes[1, 1]
    ax.set_title('Single-position CPU cost versus agreement', loc='left')
    ax.set_xlabel('Mean decision time (ms, 2 CPU threads)')
    ax.set_ylabel('Pooled agreement (%)')
    for arm in ARMS:
        x, y = [], []
        for depth in DEPTHS:
            rows = [summary['latency'][f'{arm}-{seed}'][str(depth)] for seed in SEEDS]
            x.append(sum(r['total_wall_ms'] for r in rows)/sum(len(r['records']) for r in rows))
            y.append(mean(summary['means'][arm][str(depth)].values())*100)
        ax.plot(x, y, color=COLORS[arm], marker='o', linestyle='--' if arm.endswith('mixed') else '-')
        for depth, xx, yy in zip(DEPTHS, x, y, strict=True):
            ax.annotate(str(depth), (xx, yy), xytext=(4, 3), textcoords='offset points', fontsize=8, color=COLORS[arm])
    handles = [Line2D([], [], color=COLORS[a], linewidth=2, linestyle='--' if a.endswith('mixed') else '-',
                      label=LABELS[a]) for a in ARMS]
    handles += [Line2D([], [], color='#536272', marker=m, linestyle='', markerfacecolor='none', label=f'Seed {s}')
                for s, m in zip(SEEDS, ('o', 's', '^'), strict=True)]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.067, .895), ncol=7, frameon=False, fontsize=9)
    primary, extra = ('PASSED' if summary[key] else 'FAILED' for key in ('primary_passed', 'extra_compute_passed'))
    fig.text(.075, .13, f'Frozen gates: primary {primary}; extra compute {extra}. Thin curves show every seed; thick curves show means.', fontsize=10)
    fig.text(.075, .095, 'Fixed training: depth 4. Mixed training: depths 2, 4, 6. * Depths 8 and 16 were never used in training.', fontsize=9)
    fig.text(.075, .063, 'Confidence pools mismatches across both splits and seeds. CPU latency pools the fixed timing panels; warmups excluded.', fontsize=9)
    fig.text(.075, .031, 'Internal refinement of one visible board; no cross-move memory, search, Elo or novelty claim.', fontsize=9, color='#536272')
    return fig


def render(publication, prefix):
    publication, prefix = Path(publication), Path(prefix)
    verification = audit(publication)
    paths = {ext: prefix.with_suffix('.'+ext) for ext in ('png', 'svg', 'json')}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Figure output already exists')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig = figure(read(publication/'plan.json'), read(publication/'summary.json'))
    fig.savefig(paths['png'], dpi=180)
    fig.savefig(paths['svg'], metadata={'Date': None})
    import matplotlib.pyplot as plt
    plt.close(fig)
    write_new(paths['json'], {'status': 'completed', 'verification': verification,
                             'publisher_sha256': sha(__file__),
                             'plots': {ext: sha(paths[ext]) for ext in ('png', 'svg')}})
    return {ext: str(path) for ext, path in paths.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('publish')
    for name in ('plan', 'execution', 'report', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--models', type=Path, default=ROOT/'models/chess-anchor-v1')
    for command in ('audit', 'render'):
        p = sub.add_parser(command)
        p.add_argument('--publication', type=Path, required=True)
        if command == 'render':
            p.add_argument('--prefix', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'publish':
        result = publish(args.plan, args.execution, args.report, args.out, args.models)
    elif args.command == 'audit':
        result = audit(args.publication)
    else:
        result = render(args.publication, args.prefix)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
