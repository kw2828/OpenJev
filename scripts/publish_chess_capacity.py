"""Lossless capacity-study publication and standard-library portable audit.

Publish reproduces the frozen report without inference or engine calls. Portable
checks validate archived bytes, receipts and arithmetic, not chess legality or
checkpoint tensor contents. The full source report performs those checks first.
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
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = (32, 128)
SEEDS = (53, 67, 83)
SPLITS = ('dev', 'shift')
ARCHIVE = 'execution-and-report.tar.gz'
DATA_FILES = {'started.json', 'excluded-states.json', 'games.jsonl', 'analyses.jsonl',
              'train.jsonl', 'dev.jsonl', 'shift.jsonl'}
REGRET_FILES = {'engine.json', 'analyses.jsonl', 'regret.jsonl'}
FIT_FILES = {'weights.pt', 'learning.jsonl', 'training.json', 'latency.json', 'dev.jsonl', 'shift.jsonl'}
COLORS = {32: '#64748b', 128: '#168277'}

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


def expected_names():
    return {f'width{width}-{seed}' for width in WIDTHS for seed in SEEDS}


def arena_files(plan):
    return {'started.json', 'summary.json'} | {
        f'{game["id"]}.{ext}' for game in plan['arena_schedule'] for ext in ('json', 'pgn')}


def validate_plan(plan):
    p = plan['protocol']
    require(p['version'] == 'chess-capacity-v1' and p['widths'] == list(WIDTHS)
            and p['seeds'] == list(SEEDS) and p['depth'] == 4, 'Unexpected capacity panel')
    require(p['train_examples'] == 98304 and p['epochs'] == 8 and p['batch_size'] == 128
            and p['updates_per_fit'] == 6144 and p['core_iterations_per_fit'] == 24576,
            'Unexpected training budget')
    require(p['relative_regret_reduction'] == .2 and p['arena_score_threshold'] == .6
            and p['gate_absolute_tolerance'] == 1e-12, 'Unexpected continuation thresholds')
    require(plan['parameters'] == {'32': {'stored': 43726, 'active': 33185},
                                   '128': {'stored': 591790, 'active': 439073}}, 'Parameter counts differ')
    require(len(plan['configurations']) == 6 and {c['name'] for c in plan['configurations']} == expected_names(),
            'Plan must retain all six fits')
    for config in plan['configurations']:
        require(config['width'] in WIDTHS and config['seed'] in SEEDS
                and config == {'width': config['width'], 'seed': config['seed'],
                                'name': f'width{config["width"]}-{config["seed"]}'}, 'Fit identity mismatch')
    require({s['name']: s['examples'] for s in plan['data_config']['splits']}
            == {'train': 65536, 'dev': 4096, 'shift': 4096}, 'Data quotas differ')
    openings, schedule = plan['openings'], plan['arena_schedule']
    require(len(openings) == 16 and len({o['id'] for o in openings}) == 16, 'Incomplete opening panel')
    expected = []
    for opening in openings:
        require(len(opening['moves']) == 6, 'Opening prefix length differs')
        for seed in SEEDS:
            for white, black in ((128, 32), (32, 128)):
                expected.append({'id': f'game-{len(expected)+1:03d}', 'opening_id': opening['id'],
                                 'opening_name': opening['name'], 'opening_moves': opening['moves'], 'seed': seed,
                                 'white': f'width{white}-{seed}', 'black': f'width{black}-{seed}',
                                 'white_width': white, 'black_width': black})
    require(schedule == expected, 'Arena schedule is not the full paired 96-game panel')
    rules = plan['arena_protocol']
    require(rules['games'] == schedule and rules['openings'] == openings and rules['depth'] == 4
            and rules['gate_lower_score'] == .6 and rules['device'] == 'cpu'
            and rules['torch_threads'] == 2 and rules['max_played_plies'] == 240
            and rules['clock_seconds'] == 300 and rules['claim_draw'] is False,
            'Arena protocol differs from frozen scope')


def validate_arena(plan, arena):
    require(arena['status'] == 'completed' and arena['games'] == 96 and arena['elo_estimate'] is None
            and arena['scope'] == plan['arena_protocol']['scope'], 'Incomplete or mis-scoped arena')
    games, specs = arena['game_results'], plan['arena_schedule']
    require(len(games) == len(specs), 'Missing arena results')
    statuses, outcomes = Counter(), Counter()
    for game, spec in zip(games, specs, strict=True):
        require(game['game_id'] == spec['id'] and all(game[key] == spec[key]
                for key in ('opening_id', 'seed', 'white', 'black')), 'Arena game identity mismatch')
        status, result = game['status'], game['result']
        require(status in ('completed', 'unfinished', 'failed'), 'Unknown game status')
        require(type(game['played_plies']) is int and 0 <= game['played_plies'] <= 240, 'Invalid game length')
        statuses[status] += 1
        if status == 'completed':
            require(result in ('1-0', '0-1', '1/2-1/2'), 'Completed game lacks chess result')
            if result == '1/2-1/2':
                outcomes['draws'] += 1
            elif result == ('1-0' if spec['white_width'] == 128 else '0-1'):
                outcomes['wins'] += 1
            else:
                outcomes['losses'] += 1
        else:
            require(result == '*', 'Unresolved game incorrectly assigned points')
    require(arena['status_counts'] == {s: statuses[s] for s in ('completed', 'unfinished', 'failed')},
            'Arena status counts do not match all games')
    width = arena['width128']
    require(all(width[key] == outcomes[key] for key in ('wins', 'draws', 'losses'))
            and width['possible_points'] == 96, 'Arena outcome counts differ')
    points = outcomes['wins'] + outcomes['draws']/2
    lower, upper = points/96, (points+statuses['unfinished']+statuses['failed'])/96
    close(width['completed_points'], points)
    close(width['score_lower_bound'], lower)
    close(width['score_upper_bound'], upper)
    gate = {'threshold': .6, 'lower_score_bound': lower, 'zero_failed_games': statuses['failed'] == 0,
            'passed': lower >= .6 and statuses['failed'] == 0}
    require(arena['gate'] == gate and arena['gate_passed'] is gate['passed'], 'Arena gate mismatch')
    interval = arena['opening_cluster_bootstrap']
    require(interval['openings'] == 16
            and interval['replicates'] == plan['arena_protocol']['bootstrap_replicates']
            and interval['seed'] == plan['arena_protocol']['bootstrap_seed'], 'Arena interval protocol mismatch')
    for key in ('lower_bound_interval', 'upper_bound_interval'):
        require(len(interval[key]) == 2 and 0 <= number(interval[key][0]) <= number(interval[key][1]) <= 1,
                'Invalid arena interval')
    require(arena['played_plies'] == sum(g['played_plies'] for g in games), 'Arena plies aggregate differs')
    require(type(arena['policy_calls']) is int and arena['policy_calls'] >= arena['played_plies'],
            'Invalid policy call count')
    number(arena['policy_wall_seconds'], 0)


def validate_summary(plan, summary):
    validate_plan(plan)
    p = plan['protocol']
    require(summary['status'] == 'completed' and summary['novelty_established'] is False
            and summary['elo_estimate'] is None and summary['scope'] == p['scope'], 'Summary scope mismatch')
    for field in ('metrics', 'costs', 'latency', 'regret'):
        require(set(summary[field]) == expected_names(), 'Incomplete fit panel')
    for name in expected_names():
        require(set(summary['metrics'][name]) == set(SPLITS) and set(summary['regret'][name]) == set(SPLITS),
                'Incomplete fit split coverage')
        for split in SPLITS:
            row = summary['metrics'][name][split]
            require(row['examples'] == 4096 and type(row['correct']) is int and 0 <= row['correct'] <= 4096,
                    'Evaluation quota/count mismatch')
            close(row['agreement'], row['correct']/4096)
            for field in ('target_nll', 'entropy', 'hidden_rms', 'logit_span'):
                number(row[field], 0)
            number(row['value_mae'], 0, 2)
            number(row['mean_confidence'], 0, 1)
            if row['correct'] == 4096:
                require(row['mismatch_confidence'] is None, 'Unexpected mismatch confidence')
            else:
                number(row['mismatch_confidence'], 0, 1)
            number(summary['regret'][name][split], -2, 2)
        cost = summary['costs'][name]
        require(cost['updates'] == 6144 and cost['core_iterations'] == 24576
                and cost['examples_seen'] == 786432, 'Training budget differs')
        number(cost['training_seconds'], 0)
        timing = summary['latency'][name]
        require(timing['device'] == 'cpu' and timing['depth'] == 4 and timing['torch_threads'] == p['torch_threads'],
                'Latency configuration differs')
        for key, count, total in (('records', 2*p['latency_positions_per_split'], 'total_wall_ms'),
                                  ('warmup_records', p['latency_warmups'], 'warmup_wall_ms')):
            require(len(timing[key]) == count, 'Incomplete timing panel')
            close(timing[total], math.fsum(number(r['wall_ms'], 0) for r in timing[key]))
    require(set(summary['means']) == {'32', '128'}, 'Incomplete capacity means')
    for width in WIDTHS:
        require(set(summary['means'][str(width)]) == set(SPLITS), 'Incomplete mean splits')
        for split in SPLITS:
            for metric in ('agreement', 'target_nll', 'value_mae', 'mean_confidence'):
                close(summary['means'][str(width)][split][metric], mean(
                    summary['metrics'][f'width{width}-{seed}'][split][metric] for seed in SEEDS))
    validate_arena(plan, summary['arena'])
    checks = []
    require(len(summary['comparisons']) == 2, 'Incomplete capacity comparisons')
    for split, comparison in zip(SPLITS, summary['comparisons'], strict=True):
        small = mean(summary['regret'][f'width32-{seed}'][split] for seed in SEEDS)
        large = mean(summary['regret'][f'width128-{seed}'][split] for seed in SEEDS)
        reduction = (small-large)/small if small > 0 else None
        require(comparison['split'] == split, 'Comparison split differs')
        for key, expected in (('small_bounded_regret', small), ('large_bounded_regret', large),
                              ('bounded_regret_change', large-small)):
            close(comparison[key], expected)
        if reduction is None:
            require(comparison['relative_reduction'] is None, 'Nonpositive reference has no relative reduction')
        else:
            close(comparison['relative_reduction'], reduction)
        interval = comparison['agreement_interval']
        close(interval['mean'], summary['means']['128'][split]['agreement']-summary['means']['32'][split]['agreement'])
        require(interval['positions'] == 4096 and type(interval['games']) is int and 0 < interval['games'] <= 4096
                and interval['scope'] == p['uncertainty'] and -1 <= number(interval['lower']) <= number(interval['upper']) <= 1,
                'Invalid comparison interval scope')
        passed = reduction is not None and (reduction >= .2 or math.isclose(reduction, .2, rel_tol=0, abs_tol=p['gate_absolute_tolerance']))
        checks.append({'metric': 'relative_regret_reduction', 'split': split, 'observed': reduction,
                       'threshold': .2, 'passed': passed})
    checks.append({'metric': 'arena', 'passed': summary['arena']['gate_passed']})
    require(len(summary['checks']) == len(checks), 'Incomplete continuation checks')
    for actual, expected in zip(summary['checks'], checks, strict=True):
        if expected.get('observed') is not None:
            close(actual['observed'], expected['observed'])
        require({k: v for k, v in actual.items() if k != 'observed'}
                == {k: v for k, v in expected.items() if k != 'observed'}, 'Continuation check differs')
        if 'observed' in expected and expected['observed'] is None:
            require(actual.get('observed') is None, 'Undefined reduction must stay null')
    require(summary['continuation_passed'] is all(c['passed'] for c in checks), 'Continuation verdict differs')
    return summary


def required_members(plan):
    result = {'execution/started.json', 'execution/completed.json', 'execution/baselines.json',
              'execution/panels.json', 'report/summary.json', 'report/completed.json'}
    result |= {f'execution/data/{name}' for name in DATA_FILES | {'completed.json'}}
    result |= {f'execution/regret/{name}' for name in REGRET_FILES | {'completed.json'}}
    result |= {f'execution/arena/{name}' for name in arena_files(plan) | {'completed.json'}}
    result |= {f'execution/{fit}/{name}' for fit in expected_names() for name in FIT_FILES | {'completed.json'}}
    return result


def keep_document(name):
    if name.endswith('.json'):
        return name != 'execution/data/excluded-states.json'
    return (name in {'execution/data/dev.jsonl', 'execution/data/shift.jsonl', 'execution/regret/regret.jsonl',
                     'execution/regret/analyses.jsonl'}
            or any(name == f'execution/{fit}/{suffix}' for fit in expected_names()
                   for suffix in ('dev.jsonl', 'shift.jsonl', 'learning.jsonl')))


def document(raw, name):
    if name.endswith('.jsonl'):
        require(raw.endswith(b'\n'), 'Incomplete JSONL record')
        return [decode(line) for line in raw.splitlines()]
    result = decode(raw)
    if name.startswith('execution/arena/game-'):
        result = dict(result)
        result['attempt_count'] = len(result.pop('attempts'))
        result['move_count'] = len(result.pop('moves'))
    return result


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
                documents[name] = document(b''.join(chunks), name)
    require(actual == expected, 'Archive membership differs from manifest')
    return actual, documents


def prediction_metrics(predictions, rows):
    require(len(predictions) == len(rows) == 4096, 'Incomplete prediction panel')
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate evaluation IDs')
    for pred, row in zip(predictions, rows, strict=True):
        require(pred['id'] == row['id'] and pred['game_id'] == row['game_id']
                and type(pred['game_id']) is type(row['game_id'])
                and pred['target'] == row['target_uci'] and pred['target_value'] == row['target_value'],
                'Prediction identity or target differs')
        require(type(pred['correct']) is bool and pred['correct'] == (pred['choice'] == pred['target']),
                'Prediction correctness differs')
        require(type(pred['choice']) is str and bool(pred['choice']), 'Missing move choice')
        for key in ('target_nll', 'entropy', 'hidden_rms', 'logit_span'):
            number(pred[key], 0)
        number(pred['value'], -1, 1)
        number(pred['target_value'], -1, 1)
        number(pred['target_probability'], 0, 1)
        number(pred['max_probability'], 0, 1)
        require(pred['max_probability'] > 0, 'Zero maximum probability')
        require(math.isclose(pred['target_probability'], math.exp(-pred['target_nll']),
                             rel_tol=1e-10, abs_tol=1e-14), 'Probability/NLL mismatch')
        require(pred['target_probability'] <= pred['max_probability']+1e-12
                and pred['entropy']+1e-10 >= -math.log(pred['max_probability']), 'Probability bound mismatch')
        if pred['correct']:
            require(math.isclose(pred['target_probability'], pred['max_probability'],
                                 rel_tol=1e-10, abs_tol=1e-14), 'Correct choice is not the maximum')
    wrong = [r['max_probability'] for r in predictions if not r['correct']]
    correct = sum(r['correct'] for r in predictions)
    return {'examples': len(rows), 'correct': correct, 'agreement': correct/len(rows),
            'target_nll': mean(r['target_nll'] for r in predictions),
            'value_mae': mean(abs(r['value']-r['target_value']) for r in predictions),
            'mean_confidence': mean(r['max_probability'] for r in predictions),
            'mismatch_confidence': mean(wrong) if wrong else None,
            **{k: mean(r[k] for r in predictions) for k in ('entropy', 'hidden_rms', 'logit_span')}}


def validate_chain(plan, plan_hash, summary, members, documents):
    require(required_members(plan) <= set(members), 'Archive lacks required full-panel evidence')
    require(summary['plan_sha256'] == plan_hash, 'Summary plan mismatch')

    def digest(name):
        return members[name]['sha256']

    def files(prefix, receipt, required):
        require(receipt['status'] == 'completed' and set(receipt['files']) == required,
                'Receipt file coverage mismatch')
        require(all(digest(f'{prefix}/{n}') == h for n, h in receipt['files'].items()),
                'Archived receipt hash mismatch')

    complete, report = documents['execution/completed.json'], documents['report/completed.json']
    started = documents['execution/started.json']
    require(complete['status'] == 'completed' and complete['plan_sha256'] == plan_hash
            and started['status'] == 'started' and started['plan_sha256'] == plan_hash,
            'Execution completion identity mismatch')
    number(complete['wall_seconds'], 0)
    files('execution', complete, {n.removeprefix('execution/') for n in members
                                 if n.startswith('execution/') and n != 'execution/completed.json'})
    require(report['status'] == 'completed' and report['plan_sha256'] == plan_hash
            and report['summary_sha256'] == digest('report/summary.json')
            and report['execution_receipt_sha256'] == digest('execution/completed.json')
            and summary['execution_receipt_sha256'] == digest('execution/completed.json')
            and documents['report/summary.json'] == summary, 'Report chain mismatch')
    data, regret = documents['execution/data/completed.json'], documents['execution/regret/completed.json']
    files('execution/data', data, DATA_FILES)
    files('execution/regret', regret, REGRET_FILES)
    require(data['counts'] == {'train': 65536, 'dev': 4096, 'shift': 4096}
            and documents['execution/data/started.json']['config'] == plan['data_config'], 'Data quotas/config differ')
    require(summary['fresh_data_cost'] == data and summary['engine_cost'] == regret['cost'],
            'Summary cost receipt mismatch')
    cost, p = regret['cost'], plan['protocol']
    require(type(cost['calls']) is int and 0 < cost['calls'] <= p['regret_call_ceiling']
            and cost['requested_nodes'] == cost['calls']*p['regret_nodes']
            and cost['requested_nodes'] <= p['regret_node_ceiling'], 'Engine call budget mismatch')
    engine = documents['execution/regret/engine.json']
    require(engine['sha256'] == plan['engine_sha256'] and engine['id']['name'].startswith('Stockfish 19'),
            'Engine identity mismatch')
    panel = documents['execution/panels.json']
    indices = panel['latency_indices']
    require(len(indices) == 128 and len(set(indices)) == 128
            and all(type(i) is int and 0 <= i < 8192 for i in indices)
            and sum(i < 4096 for i in indices) == 64, 'Latency panel differs')
    require(panel['graded_configurations'] == [{'id': c['name']} for c in plan['configurations']],
            'Graded fit identities differ')
    eval_rows = {s: documents[f'execution/data/{s}.jsonl'] for s in SPLITS}
    for split in SPLITS:
        selected = panel['secondary_indices'][split]
        require(len(selected) == 128 and selected == sorted(set(selected))
                and all(type(i) is int and 0 <= i < 4096 for i in selected), 'Grading panel differs')
    combined = eval_rows['dev']+eval_rows['shift']
    predictions = {}
    for config in plan['configurations']:
        name, prefix = config['name'], f"execution/{config['name']}"
        receipt, trained = documents[prefix+'/completed.json'], documents[prefix+'/training.json']
        files(prefix, receipt, FIT_FILES)
        for identity in (receipt, trained):
            require(identity['status'] == 'completed' and identity['plan_sha256'] == plan_hash
                    and all(identity[k] == v for k, v in config.items()), 'Fit identity mismatch')
        require(trained['weights_sha256'] == digest(prefix+'/weights.pt')
                and trained['learning_sha256'] == digest(prefix+'/learning.jsonl')
                and trained['data_receipt_sha256'] == digest('execution/data/completed.json'),
                'Training input binding mismatch')
        require({k: trained[k] for k in summary['costs'][name]} == summary['costs'][name], 'Fit cost mismatch')
        state = trained['initial_state_sha256']
        require(type(state) is str and len(state) == 64 and set(state) <= set('0123456789abcdef'),
                'Invalid initialization digest')
        learning = documents[prefix+'/learning.jsonl']
        require(len(learning) == 6144, 'Incomplete optimizer steps')
        for i, row in enumerate(learning):
            require(row['step'] == i+1 and row['epoch'] == i//768+1 and row['depth'] == 4
                    and row['examples'] == 128, 'Learning journal schedule differs')
            for field in ('policy_ce', 'value_mse', 'loss', 'gradient_norm'):
                number(row[field], 0)
            require(math.isclose(row['loss'], row['policy_ce']+.5*row['value_mse'],
                                 rel_tol=2e-6, abs_tol=1e-6), 'Learning loss arithmetic differs')
        require(set(receipt['evaluation']) == set(SPLITS), 'Fit split coverage differs')
        predictions[name] = {}
        for split in SPLITS:
            preds = documents[f'{prefix}/{split}.jsonl']
            predictions[name][split] = preds
            metrics = prediction_metrics(preds, eval_rows[split])
            require(metrics == summary['metrics'][name][split] == receipt['evaluation'][split]['metrics'],
                    'Recorded predictions do not reproduce metrics')
            number(receipt['evaluation'][split]['evaluation_wall_seconds'], 0)
        timing = documents[prefix+'/latency.json']
        require(timing == summary['latency'][name], 'Latency summary differs')
        require([r['index'] for r in timing['records']] == indices
                and all(r['id'] == combined[i]['id'] for r, i in zip(timing['records'], indices, strict=True)),
                'Latency position identity mismatch')
        require([r['index'] for r in timing['warmup_records']] == [0, 1, 2]
                and all(r['id'] == 'starting-board' for r in timing['warmup_records']), 'Warmup identity differs')
    baseline = documents['execution/baselines.json']
    require(summary['baselines'] == {s: {n: v['metrics'] for n, v in baseline[s].items()} for s in SPLITS},
            'Baseline summary differs')
    for split in SPLITS:
        require(summary['comparisons'][SPLITS.index(split)]['agreement_interval']['games']
                == len({r['game_id'] for r in eval_rows[split]}), 'Interval game-cluster count differs')
    analyses = documents['execution/regret/analyses.jsonl']
    require(len(analyses) == cost['calls'], 'Engine call count differs')
    close(cost['wall_seconds'], math.fsum(number(a['wall_seconds'], 0) for a in analyses))
    require(cost['reported_nodes'] == sum(a['reported_nodes'] for a in analyses), 'Engine node sum differs')
    lookup = {}
    for row in analyses:
        split, i = row['split'], row['panel_index']
        require(split in SPLITS and i in panel['secondary_indices'][split]
                and row['id'] == eval_rows[split][i]['id'] and row['requested_nodes'] == 20000,
                'Engine analysis identity differs')
        number(row['bounded_score'], -1, 1)
        key = (split, i, row['root_move'])
        require(key not in lookup, 'Duplicate engine analysis')
        lookup[key] = row
    records = documents['execution/regret/regret.jsonl']
    require(len(records) == 1536, 'Incomplete engine grading records')
    expected, seen, required_calls = {}, set(), set()
    for name in expected_names():
        for split in SPLITS:
            expected[(name, split)] = []
    for row in records:
        name, split, i = row['configuration'], row['split'], row['panel_index']
        require((name, split) in expected and i in panel['secondary_indices'][split], 'Grading selection differs')
        key = name, split, i
        require(key not in seen, 'Repeated graded decision')
        seen.add(key)
        pred = predictions[name][split][i]
        require(row['id'] == pred['id'] and row['choice'] == pred['choice'], 'Graded decision differs')
        ref, chosen = (split, i, None), (split, i, row['choice'])
        require(ref in lookup and chosen in lookup, 'Missing paired engine analysis')
        close(row['bounded_regret'], lookup[ref]['bounded_score']-lookup[chosen]['bounded_score'])
        close(row['cp_loss'], lookup[ref]['score_cp']-lookup[chosen]['score_cp'])
        required_calls.update((ref, chosen))
        expected[(name, split)].append(row['bounded_regret'])
    require(required_calls == set(lookup), 'Unexpected engine calls')
    for (name, split), values in expected.items():
        require(len(values) == 128, 'Incomplete per-fit grading coverage')
        close(summary['regret'][name][split], mean(values))
    arena_prefix = 'execution/arena'
    arena_receipt, arena_start = documents[arena_prefix+'/completed.json'], documents[arena_prefix+'/started.json']
    files(arena_prefix, arena_receipt, arena_files(plan))
    require(arena_receipt['plan_sha256'] == plan_hash and arena_start['plan_sha256'] == plan_hash
            and arena_start['status'] == 'started' and arena_start['protocol'] == plan['arena_protocol']
            and set(arena_start['models']) == expected_names(), 'Arena source binding differs')
    for c in plan['configurations']:
        model = arena_start['models'][c['name']]
        require(model['width'] == c['width'] and model['seed'] == c['seed'] and model['depth'] == 4
                and model['recurrence'] == 'residual' and len(model['state_sha256']) == 64
                and set(model['state_sha256']) <= set('0123456789abcdef'), 'Arena model identity differs')
    require(documents[arena_prefix+'/summary.json'] == summary['arena'], 'Arena summary differs')
    games = [documents[f'{arena_prefix}/{spec["id"]}.json'] for spec in plan['arena_schedule']]
    for game, result, spec in zip(games, summary['arena']['game_results'], plan['arena_schedule'], strict=True):
        require(all(game[k] == v for k, v in result.items())
                and game['opening_moves'] == spec['opening_moves']
                and game['white_width'] == spec['white_width'] and game['black_width'] == spec['black_width']
                and game['opening_plies'] == 6 and game['move_count'] == game['played_plies']+6,
                'Raw game and summary identity differ')
    require(sum(g['attempt_count'] for g in games) == summary['arena']['policy_calls'], 'Arena call sum differs')
    close(summary['arena']['policy_wall_seconds'], math.fsum(number(g['wall_seconds'], 0) for g in games))


def load_study():
    spec = importlib.util.spec_from_file_location('_capacity_publication_study', ROOT/'scripts/chess_capacity_study.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_card(plan, plan_hash, weights):
    lines = ['# OpenJev capacity chess checkpoints', '', 'License: MIT. All six final checkpoints are retained.', '',
             ('These residual spatial policies use four internal updates on a fully visible board. '
             'Training uses 98,304 fixed Stockfish-labeled positions, eight epochs, legal-move cross-entropy '
             'and 0.5 bounded-value MSE. There is no cross-move memory, search, RL, Elo estimate or novelty claim.'), '',
             ('Width 32: 33,185 active / 43,726 stored parameters. Width 128: 439,073 active / 591,790 stored. '
             'The unused auxiliary head remains in both checkpoints. Data, batches, optimizer updates and depth '
             'are matched; parameter count, FLOPs and wall time are not. Equal seed numbers do not imply '
             'identical tensors across different widths.'), '', f'Frozen plan SHA-256: `{plan_hash}`.', '',
             '| Checkpoint | Width | Seed |', '| --- | --- | --- |']
    lines.extend(f'| [{name}]({row["path"]}) | {row["width"]} | {row["seed"]} |' for name, row in sorted(weights.items()))
    lines.extend(['', ('Load with `AnchorChess.load(path, expected_plan_sha256=..., '
                  'expected_recurrence="residual", expected_seed=...)`. Weights are copied byte for byte '
                  'from the completed execution; no winner alias or selected checkpoint is substituted.'), '',
                  'Source hashes:', *[f'- `{name}`: `{digest}`' for name, digest in sorted(plan['sources'].items())], ''])
    return '\n'.join(lines)


def publish(plan_path, execution, report, out, models=None):
    plan_path, execution, report, out = map(Path, (plan_path, execution, report, out))
    models = ROOT/'models/chess-capacity-v1' if models is None else Path(models)
    if out.exists() or models.exists():
        raise FileExistsError('Publication or models directory already exists')
    for destination in (out.resolve(), models.resolve()):
        require(not any(destination.is_relative_to(root.resolve()) or root.resolve().is_relative_to(destination)
                        for root in (execution, report)), 'Publication destinations must be outside source evidence')
    require(not out.resolve().is_relative_to(models.resolve()) and not models.resolve().is_relative_to(out.resolve()),
            'Publication and model directories must not overlap')
    study = load_study()
    plan = study.verify_plan(plan_path)
    summary = validate_summary(plan, read(report/'summary.json'))
    members = inventory(execution, report)
    documents = {name: document(((execution if name.startswith('execution/') else report)/name.split('/', 1)[1]).read_bytes(), name)
                 for name in members if keep_document(name)}
    validate_chain(plan, sha(plan_path), summary, members, documents)
    with tempfile.TemporaryDirectory(prefix='openjev-capacity-publication-') as temporary:
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
        original, target = f'execution/{name}/weights.pt', models/name/'weights.pt'
        target.parent.mkdir()
        shutil.copyfile(execution/name/'weights.pt', target)
        require(sha(target) == members[original]['sha256'] and target.stat().st_size == members[original]['size'],
                'Copied checkpoint differs from original')
        weights[name] = {**config, **members[original], 'path': f'{name}/weights.pt',
                         'original_member': original, 'plan_sha256': sha(plan_path)}
    (models/'README.md').write_text(model_card(plan, sha(plan_path), weights))
    shutil.copyfile(ROOT/'LICENSE', models/'LICENSE')
    models_relative = Path(os.path.relpath(models.resolve(), out.resolve())).as_posix()
    verdict = lambda value: 'PASSED' if value else 'FAILED'
    (out/'README.md').write_text(
        '# Capacity chess development evidence\n\n'
        f'Engine-score gate: **{verdict(all(c["passed"] for c in summary["checks"][:2]))}**. '
        f'Game gate: **{verdict(summary["arena"]["gate_passed"])}**. '
        f'Combined continuation: **{verdict(summary["continuation_passed"])}**.\n\n'
        f'[Summary](summary.json) · [Plan](plan.json) · [All raw evidence]({ARCHIVE}) · '
        f'[All six checkpoints]({models_relative}/README.md) · [Hash manifest](manifest.json)\n\n'
        'The archive preserves every execution and report file, including fresh positions, generator traces, '
        'raw engine calls, training logs, predictions, original weights and all 96 game JSON/PGN traces. '
        'Unfinished or failed games retain unresolved points; they are never relabeled draws. '
        'This is a development capacity comparison on known generators, conditional on three training seeds. '
        'No Elo estimate, new architecture or novelty claim is established.\n')
    manifest = {'version': 1, 'plan_sha256': sha(out/'plan.json'),
                'archive': {'path': ARCHIVE, 'sha256': sha(out/ARCHIVE), 'size': (out/ARCHIVE).stat().st_size},
                'members': members, 'member_count': len(members), 'raw_bytes': sum(r['size'] for r in members.values()),
                'models_directory': models_relative, 'weights': weights,
                'model_files': {name: sha(models/name) for name in ('README.md', 'LICENSE')},
                'publication_readme_sha256': sha(out/'README.md')}
    write_new(out/'manifest.json', manifest)
    actual, documents = inspect_archive(out/ARCHIVE, members)
    validate_chain(plan, sha(out/'plan.json'), summary, actual, documents)
    write_new(out/'completed.json', {'status': 'completed', 'fits': 6, 'games': 96,
                                    'plan_sha256': sha(out/'plan.json'), 'summary_sha256': sha(out/'summary.json'),
                                    'manifest_sha256': sha(out/'manifest.json'), 'publisher_sha256': sha(__file__),
                                    'source_report_reproduced': True, 'archive_roundtrip_verified': True,
                                    'copied_weights_match_originals': True})
    return audit(out)


def audit(publication):
    out = Path(publication)
    receipt, manifest = read(out/'completed.json'), read(out/'manifest.json')
    plan, summary = read(out/'plan.json'), read(out/'summary.json')
    validate_summary(plan, summary)
    plan_hash = sha(out/'plan.json')
    require(receipt['status'] == 'completed' and receipt['fits'] == 6 and receipt['games'] == 96
            and receipt['plan_sha256'] == plan_hash == manifest['plan_sha256']
            and receipt['summary_sha256'] == sha(out/'summary.json')
            and receipt['manifest_sha256'] == sha(out/'manifest.json') and not (out/'failed.json').exists(),
            'Publication completion receipt mismatch')
    require(all(receipt.get(k) is True for k in ('source_report_reproduced', 'archive_roundtrip_verified',
                                               'copied_weights_match_originals')), 'Missing verification receipt')
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
    for config in plan['configurations']:
        name = config['name']
        row = manifest['weights'][name]
        original, relative = f'execution/{name}/weights.pt', f'{name}/weights.pt'
        path = models/relative
        require(all(row[k] == v for k, v in config.items())
                and row['original_member'] == original and row['path'] == relative
                and row['plan_sha256'] == plan_hash and row['sha256'] == members[original]['sha256']
                and row['size'] == members[original]['size'] and not path.is_symlink() and not path.parent.is_symlink()
                and sha(path) == row['sha256'] and path.stat().st_size == row['size'],
                'Published checkpoint identity or original bytes mismatch')
    require(set(manifest['model_files']) == {'README.md', 'LICENSE'}, 'Missing model card or MIT license')
    for name, digest in manifest['model_files'].items():
        require(not (models/name).is_symlink() and sha(models/name) == digest, 'Model documentation checksum mismatch')
    require(sha(out/'README.md') == manifest['publication_readme_sha256'], 'Publication index changed')
    return {'status': 'verified', 'fits': 6, 'games': 96, 'evaluations': 12, 'members': len(members),
            'plan_sha256': plan_hash, 'archive_sha256': archive['sha256'], 'summary_sha256': sha(out/'summary.json'),
            'scope': 'Portable byte, receipt and scalar arithmetic audit. Chess legality, tensor identities and '
                     'bootstrap resampling are checked by frozen report reproduction at publication time.'}


def figure(plan, summary):
    validate_summary(plan, summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titleweight': 'bold',
                         'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    fig.subplots_adjust(left=.08, right=.97, bottom=.16, top=.86, hspace=.52, wspace=.3)
    fig.suptitle('OpenJev: does more capacity improve chess decisions?', x=.08, ha='left', y=.97,
                 fontsize=19, fontweight='bold')
    fig.text(.08, .925, '98,304 training positions · 3 fits per width · 96 paired games · development study',
             color='#475569', fontsize=11)
    markers = ('o', 's', '^')
    labels = ('Ordinary', 'Shift')
    for ax, key, title, ylabel, scale in (
        (axes[0, 0], 'agreement', 'A   Teacher agreement', 'Top-1 agreement (%)', 100),
        (axes[0, 1], 'regret', 'B   Engine score loss', 'Signed bounded score loss (lower is better)', 1),
    ):
        for j, split in enumerate(SPLITS):
            for width, offset in ((32, -.17), (128, .17)):
                values = [summary['regret'][f'width{width}-{s}'][split] if key == 'regret'
                          else summary['metrics'][f'width{width}-{s}'][split][key] for s in SEEDS]
                ax.bar(j+offset, mean(values)*scale, width=.3, color=COLORS[width], alpha=.32)
                for i, value in enumerate(values):
                    ax.scatter(j+offset+(i-1)*.035, value*scale, marker=markers[i],
                               color=COLORS[width], edgecolor='white', linewidth=.5, s=45, zorder=3)
            if key == 'agreement':
                greedy = summary['baselines'][split]['greedy_material']['top1_teacher_agreement']*100
                ax.plot([j-.36, j+.36], [greedy]*2, color='#a16207', ls=':', lw=1.5)
        ax.set(xticks=range(2), xticklabels=labels, ylabel=ylabel, title=title)
        ax.axhline(0, color='#cbd5e1', lw=.8)
        ax.grid(axis='y', alpha=.16)
    axes[0, 0].legend(handles=[Patch(facecolor=COLORS[w], label=f'Width {w}') for w in WIDTHS]
                     +[Line2D([], [], color='#a16207', ls=':', label='Greedy material')],
                     fontsize=8.5, frameon=False, loc='upper left')
    ax = axes[1, 0]
    for width in WIDTHS:
        for i, seed in enumerate(SEEDS):
            name = f'width{width}-{seed}'
            ms = summary['latency'][name]['total_wall_ms']/128
            quality = mean(summary['metrics'][name][s]['agreement'] for s in SPLITS)*100
            ax.scatter(ms, quality, s=65, marker=markers[i], color=COLORS[width], edgecolor='white', lw=.5)
    ax.set(title='C   Measured CPU cost and agreement', xlabel='Full decision latency (ms, mean of 128 calls)',
           ylabel='Mean ordinary / shift agreement (%)')
    ax.grid(alpha=.16)
    ax.legend(handles=[Line2D([], [], marker=m, color='#475569', ls='', label=f'Seed {s}')
                       for m, s in zip(markers, SEEDS, strict=True)], frameon=False, fontsize=8.5)
    ax = axes[1, 1]
    categories = ('Win', 'Draw', 'Loss', 'Unfinished', 'Failed')
    palette = ('#168277', '#a3c7bf', '#be5e5e', '#cbd5e1', '#7c3aed')
    totals = Counter()
    for j, seed in enumerate(SEEDS):
        counts = Counter()
        for g in summary['arena']['game_results']:
            if g['seed'] != seed:
                continue
            category = g['status'].title() if g['status'] != 'completed' else (
                'Draw' if g['result'] == '1/2-1/2' else
                'Win' if g['result'] == ('1-0' if g['white'].startswith('width128-') else '0-1') else 'Loss')
            counts[category] += 1
        bottom = 0
        for category, color in zip(categories, palette, strict=True):
            count = counts[category]
            ax.bar(j, count, bottom=bottom, width=.58, color=color,
                   hatch='///' if category in ('Unfinished', 'Failed') else None, edgecolor='white', linewidth=.5)
            if count:
                ax.text(j, bottom+count/2, str(count), ha='center', va='center', fontsize=9)
            bottom += count
        totals.update(counts)
    ax.set(title='D   All games, from width 128’s side', xticks=range(3),
           xticklabels=[f'Seed {s}' for s in SEEDS], ylabel='Games (32 per seed)', ylim=(0, 35))
    ax.set_title('D   All games, from width 128’s side', pad=29)
    ax.legend(handles=[Patch(facecolor=c, label=f'{k}: {totals[k]}',
                             hatch='///' if k in ('Unfinished', 'Failed') else None)
                       for k, c in zip(categories, palette, strict=True)],
              loc='upper center', bbox_to_anchor=(.5, -.15), ncol=3, fontsize=8.5, frameon=False)
    width = summary['arena']['width128']
    ax.text(.5, 1.01, f'All-game point bounds: {100*width["score_lower_bound"]:.1f}% to '
            f'{100*width["score_upper_bound"]:.1f}% (unresolved games retained)',
            ha='center', transform=ax.transAxes, color='#475569', fontsize=8.3)
    gate = lambda value: 'PASS' if value else 'FAIL'
    fig.text(.08, .065, f'Frozen gates: engine score {gate(all(c["passed"] for c in summary["checks"][:2]))} · '
             f'games {gate(summary["arena"]["gate_passed"])} · combined {gate(summary["continuation_passed"])}',
             fontsize=11, fontweight='bold')
    fig.text(.08, .027, 'Matched data, updates and depth; capacity and compute differ. Dots show every fit. '
             'No Elo estimate or novel architecture claim.', color='#475569', fontsize=9)
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
    p.add_argument('--models', type=Path, default=ROOT/'models/chess-capacity-v1')
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
