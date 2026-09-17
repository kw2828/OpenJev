"""Separate external transfer panel for all completed candidate-v2 final fits.

Prepare freezes source, checkpoint and data bytes before decoding ChessBench.
This study does not modify the candidate experiment's continuation criteria.
"""
import argparse
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import time
from pathlib import Path

import chess
import torch

from openjev.research import chess_candidate_data as history
from openjev.research import chessbench_data as data
from openjev.research import chessbench_eval as evaluation
from openjev.research.chess_candidate import CandidateChess
from openjev.research.chess_spatial_data import state_key

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PLAN = 'evidence/chess-candidate-v2/protocol/plan.json'
CANDIDATE_PLAN_SHA = '0cdcf5b9f9e64b5dde6f0a951fa4249475d050470e9036ce5c8ab76cf181c106'
PUBLICATION = 'evidence/chess-candidate-v2/results'
EXECUTION = 'runs/chess-candidate-v2/execution'
INPUT = 'runs/chessbench-inputs-v1/behavioral_cloning_data.bag'
DOWNLOAD = 'runs/chessbench-inputs-v1/download.json'
INPUT_SHA = 'fc1d27fc98baae3a7f15fee85439b499f4a6f1815dffde26d047a77f02cc50b1'
SOURCES = [
    'scripts/chessbench_transfer_study.py', 'tests/test_chessbench_transfer_study.py',
    'src/openjev/research/chessbench_data.py', 'tests/test_chessbench_data.py',
    'src/openjev/research/chessbench_eval.py', 'tests/test_chessbench_eval.py',
    'scripts/publish_chess_candidate.py',
]
PROTOCOL = {
    'version': 'chessbench-transfer-v1', 'positions': 4096, 'selection_seed': 11400001,
    'batch_size': 16, 'device': 'cpu', 'torch_threads': 2,
    'root_depth': 4, 'branch_depth': 2,
    'selection': 'All twelve candidate-v2 final checkpoints; no best seed or model selection, retries or replacement panel.',
    'metrics': 'Target-move agreement and legal-menu NLL; per-seed and equal-seed means. No value target, value MAE, engine regret or winning probability.',
    'exposure': 'Prior candidate-data exclusions plus every visited candidate-v2 generator and arena root and every legal successor, with mirrored-state exclusion. FEN-available automatic terminal roots excluded; repetition history unavailable.',
    'uncertainty': 'Descriptive paired differences over this public selected panel. Source-game IDs are unavailable; no invented independent-game intervals.',
    'timing': 'Uncached CPU batch construction, forward and output writing; not single-decision latency. No timing pass concurrent with candidate-v2 execution.',
    'scope': 'Separate public development transfer panel. Does not alter the candidate-v2 gates. No Elo, calibration or novelty established.',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def write_rows(path, values):
    with Path(path).open('x') as stream:
        stream.writelines(json.dumps(value, allow_nan=False) + '\n' for value in values)


def tree(directory):
    result = {}
    for path in sorted(Path(directory).rglob('*')):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError('Nonregular evidence file')
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = sha(path)
    return result


def publisher():
    spec = importlib.util.spec_from_file_location('_transfer_candidate_publisher', ROOT / 'scripts/publish_chess_candidate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prerequisite():
    """Require a completed, audited publication and identical local raw evidence."""
    plan = read(ROOT / CANDIDATE_PLAN)
    if sha(ROOT / CANDIDATE_PLAN) != CANDIDATE_PLAN_SHA:
        raise ValueError('Candidate source plan changed')
    publisher().audit(ROOT / PUBLICATION)
    manifest = read(ROOT / PUBLICATION / 'manifest.json')
    complete = read(ROOT / EXECUTION / 'completed.json')
    actual = tree(ROOT / EXECUTION)
    actual.pop('completed.json')
    if (complete['status'] != 'completed' or complete['plan_sha256'] != CANDIDATE_PLAN_SHA
            or complete['files'] != actual or (ROOT / EXECUTION / 'failed.json').exists()):
        raise ValueError('Candidate execution is incomplete or changed')
    for name, digest in tree(ROOT / EXECUTION).items():
        if manifest['members'][f'execution/{name}']['sha256'] != digest:
            raise ValueError('Candidate raw evidence differs from its audited publication')
    for path, digest in plan['sources'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Candidate source changed')
    weights = {}
    models = (ROOT / PUBLICATION / manifest['models_directory']).resolve()
    for config in plan['configurations']:
        name = config['name']
        item = manifest['weights'][name]
        path = models / item['path']
        if sha(path) != item['sha256']:
            raise ValueError('Published checkpoint changed')
        weights[name] = {**config, 'path': path.relative_to(ROOT).as_posix(), 'sha256': item['sha256']}
    return plan, weights


def collect_exclusions(candidate_plan):
    """Keep all observed roots and conservative native branch exposures."""
    prior = history.collect_exclusions(ROOT)
    if (prior['files'] != candidate_plan['exclusion_files']
            or prior['counts'] != candidate_plan['exclusion_counts']
            or hashlib.sha256(json.dumps(prior['states'], separators=(',', ':')).encode()).hexdigest()
            != candidate_plan['exclusion_states_sha256']):
        raise ValueError('Historical exposure boundary differs from candidate plan')
    states = set(prior['states'])
    roots = {r['fen'] for r in rows(ROOT / EXECUTION / 'data/analyses.jsonl')}
    for spec in candidate_plan['arena_schedule']:
        game = read(ROOT / EXECUTION / 'arena' / f"{spec['id']}.json")
        roots.update((game['initial_fen'], game['final_fen']))
        roots.update(move[k] for move in game['moves'] for k in ('fen_before', 'fen_after'))
        roots.update(move['fen_before'] for move in game['attempts'])
    visited, distinct, children = set(), 0, 0
    for fen in sorted(roots):
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError('Invalid historical root')
        key = state_key(board)
        if key in visited:
            continue
        visited.add(key)
        distinct += 1
        states.add(key)
        for move in board.legal_moves:
            child = board.copy(stack=False)
            child.push(move)
            children += 1
            states.add(state_key(child))
    return sorted(states), {
        'prior_states': len(prior['states']), 'candidate_distinct_roots': distinct,
        'candidate_native_successor_visits': children, 'total_natural_states': len(states),
        'scope': PROTOCOL['exposure'],
    }


def environment():
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'dependencies': {name: importlib.metadata.version(name) for name in ('torch', 'chess', 'numpy')},
            'lock_sha256': sha(ROOT / 'uv.lock'), 'pyproject_sha256': sha(ROOT / 'pyproject.toml')}


def prepare(out):
    candidate, weights = prerequisite()
    if sha(ROOT / INPUT) != INPUT_SHA or read(ROOT / DOWNLOAD)['sha256'] != INPUT_SHA:
        raise ValueError('Opaque source bytes differ from acquired file')
    sources = {name: sha(ROOT / name) for name in SOURCES}
    excluded, exposure = collect_exclusions(candidate)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    raw = (json.dumps(excluded, separators=(',', ':')) + '\n').encode()
    with (out / 'excluded-states.json.gz').open('xb') as stream:
        stream.write(gzip.compress(raw, mtime=0))
    plan = {
        'protocol': PROTOCOL, 'sources': sources, 'environment': environment(),
        'candidate_plan': CANDIDATE_PLAN, 'candidate_plan_sha256': CANDIDATE_PLAN_SHA,
        'publication_manifest_sha256': sha(ROOT / PUBLICATION / 'manifest.json'),
        'publication_receipt_sha256': sha(ROOT / PUBLICATION / 'completed.json'),
        'input': INPUT, 'input_sha256': INPUT_SHA, 'input_bytes': (ROOT / INPUT).stat().st_size,
        'download_receipt': read(ROOT / DOWNLOAD), 'download_receipt_sha256': sha(ROOT / DOWNLOAD),
        'exclusion_file': 'excluded-states.json.gz', 'exclusion_sha256': sha(out / 'excluded-states.json.gz'),
        'exclusion_uncompressed_sha256': hashlib.sha256(raw).hexdigest(), 'exposure': exposure,
        'configurations': [weights[c['name']] for c in candidate['configurations']],
        'decoded_benchmark_before_freeze': False,
    }
    write(out / 'plan.json', plan)
    write(out / 'prepared.json', {'status': 'prepared', 'plan_sha256': sha(out / 'plan.json'), 'created_unix': time.time(), 'neural_evaluation_started': False})
    return out / 'plan.json'


def verify(plan_path):
    plan = read(plan_path)
    if (plan['protocol'] != PROTOCOL or plan['environment'] != environment()
            or plan['input'] != INPUT or plan['input_sha256'] != INPUT_SHA
            or sha(ROOT / INPUT) != INPUT_SHA
            or plan['sources'] != {name: sha(ROOT / name) for name in SOURCES}
            or plan['download_receipt_sha256'] != sha(ROOT / DOWNLOAD)
            or plan['download_receipt'] != read(ROOT / DOWNLOAD)
            or plan['decoded_benchmark_before_freeze'] is not False):
        raise ValueError('Frozen source, input or environment changed')
    candidate, weights = prerequisite()
    if (plan['candidate_plan'] != CANDIDATE_PLAN or plan['candidate_plan_sha256'] != CANDIDATE_PLAN_SHA
            or plan['publication_manifest_sha256'] != sha(ROOT / PUBLICATION / 'manifest.json')
            or plan['publication_receipt_sha256'] != sha(ROOT / PUBLICATION / 'completed.json')
            or plan['configurations'] != [weights[c['name']] for c in candidate['configurations']]):
        raise ValueError('Candidate prerequisite binding changed')
    path = Path(plan_path).parent / 'excluded-states.json.gz'
    if plan['exclusion_file'] != path.name or sha(path) != plan['exclusion_sha256']:
        raise ValueError('Exclusion snapshot changed')
    raw = gzip.decompress(path.read_bytes())
    if hashlib.sha256(raw).hexdigest() != plan['exclusion_uncompressed_sha256']:
        raise ValueError('Exclusion contents changed')
    excluded = json.loads(raw)
    if excluded != sorted(set(excluded)) or len(excluded) != plan['exposure']['total_natural_states']:
        raise ValueError('Exclusion membership inconsistent')
    return plan, excluded


def selection(excluded):
    decoded = data.decode_bc_bag(ROOT / INPUT)
    return data.select(decoded, excluded, limit=PROTOCOL['positions'], seed=PROTOCOL['selection_seed'])


def run(plan_path, out):
    plan, excluded = verify(plan_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    plan_hash = sha(plan_path)
    write(out / 'started.json', {'status': 'started', 'plan_sha256': plan_hash, 'started_unix': time.time()})
    begun = time.perf_counter()
    try:
        chosen = selection(excluded)
        del excluded
        write(out / 'selection.json', chosen['selection_receipt'])
        write_rows(out / 'rejections.jsonl', chosen['rejection_log'])
        write_rows(out / 'selected.jsonl', chosen['selected'])
        if chosen['selection_receipt']['status'] != 'completed':
            raise ValueError('Fixed-source panel quota not met')
        torch.set_num_threads(PROTOCOL['torch_threads'])
        for config in plan['configurations']:
            name = config['name']
            model = CandidateChess.load(ROOT / config['path'], expected_plan_sha256=CANDIDATE_PLAN_SHA,
                expected_arm=config['arm'], expected_seed=config['seed'], expected_width=config['width'],
                expected_root_depth=PROTOCOL['root_depth'], expected_branch_depth=PROTOCOL['branch_depth'])
            receipt = evaluation.evaluate(model, chosen['selected'], out / f'{name}.jsonl', batch_size=PROTOCOL['batch_size'])
            if sha(ROOT / config['path']) != config['sha256']:
                raise ValueError('Checkpoint changed during evaluation')
            write(out / f'{name}.json', {'configuration': config, 'evaluation': receipt,
                'predictions_sha256': sha(out / f'{name}.jsonl'), 'plan_sha256': plan_hash})
            print(json.dumps({'evaluated': name, 'metrics': receipt['metrics']}), flush=True)
        write(out / 'completed.json', {'status': 'completed', 'plan_sha256': plan_hash,
            'wall_seconds': time.perf_counter() - begun, 'files': tree(out)})
    except Exception as error:
        write(out / 'failed.json', {'status': 'failed', 'plan_sha256': plan_hash,
            'error': str(error), 'wall_seconds': time.perf_counter() - begun})
        raise
    return out / 'completed.json'


def aggregate(metrics, configurations):
    mean = lambda values: math.fsum(values) / len(values)
    arms = ('direct', 'action_only', 'delta', 'full_afterstate')
    means = {arm: {key: mean([metrics[c['name']][key] for c in configurations if c['arm'] == arm])
                  for key in ('agreement', 'mean_nll')} for arm in arms}
    paired = []
    for other in ('direct', 'action_only', 'full_afterstate'):
        for seed in (97, 109, 127):
            a, b = metrics[f'delta-{seed}'], metrics[f'{other}-{seed}']
            paired.append({'reference': other, 'seed': seed,
                'delta_agreement_minus_reference': a['agreement'] - b['agreement'],
                'delta_nll_minus_reference': a['mean_nll'] - b['mean_nll']})
    return {'means': means, 'paired_differences': paired}



def audit_evaluation_receipt(receipt, path, selected, config, calculated):
    """Bind saved logits and cost arithmetic to the actual immutable checkpoint."""
    expected_fields = {
        'status', 'version', 'model', 'metrics', 'input_rows_sha256', 'predictions_sha256',
        'model_state_unchanged', 'device', 'torch_threads', 'batch_size', 'batches',
        'evaluation_wall_seconds', 'timing_scope', 'root_encodings', 'candidate_evaluations',
        'native_successors', 'native_board_copies', 'native_pushes', 'successor_encodings',
        'probability_semantics',
    }
    model = CandidateChess.load(ROOT / config['path'], expected_plan_sha256=CANDIDATE_PLAN_SHA,
        expected_arm=config['arm'], expected_seed=config['seed'], expected_width=config['width'],
        expected_root_depth=PROTOCOL['root_depth'], expected_branch_depth=PROTOCOL['branch_depth'])
    identity = {'arm': config['arm'], 'seed': config['seed'], 'width': config['width'],
        'root_depth': PROTOCOL['root_depth'], 'branch_depth': PROTOCOL['branch_depth'],
        'state_sha256': evaluation.state_sha256(model)}
    count = sum(chess.Board(row['fen']).legal_moves.count() for row in selected)
    native = count if config['arm'] in ('delta', 'full_afterstate') else 0
    if (set(receipt) != expected_fields or receipt['status'] != 'completed'
            or receipt['version'] != evaluation.VERSION or receipt['model'] != identity
            or receipt['metrics'] != calculated
            or receipt['input_rows_sha256'] != evaluation.input_rows_sha256(selected)
            or receipt['predictions_sha256'] != sha(path) or receipt['model_state_unchanged'] is not True
            or receipt['device'] != 'cpu' or receipt['torch_threads'] != 2
            or receipt['batch_size'] != PROTOCOL['batch_size']
            or receipt['batches'] != math.ceil(len(selected) / PROTOCOL['batch_size'])
            or type(receipt['evaluation_wall_seconds']) not in (int, float)
            or not math.isfinite(receipt['evaluation_wall_seconds']) or receipt['evaluation_wall_seconds'] < 0
            or receipt['timing_scope'] != evaluation.TIMING_SCOPE
            or receipt['root_encodings'] != len(selected) or receipt['candidate_evaluations'] != count
            or any(receipt[k] != native for k in ('native_successors', 'native_board_copies', 'native_pushes', 'successor_encodings'))
            or receipt['probability_semantics'] != 'Uncalibrated softmax over the complete native legal menu; not absolute move quality.'):
        raise ValueError('Evaluation identity, cost or checkpoint binding differs')
    if any({key: row[key] for key in identity} != identity for row in rows(path)):
        raise ValueError('Prediction identity differs from actual checkpoint')


def report(plan_path, execution, out):
    plan, excluded = verify(plan_path)
    execution = Path(execution)
    complete = read(execution / 'completed.json')
    actual = tree(execution)
    actual.pop('completed.json')
    expected = {'started.json', 'selection.json', 'rejections.jsonl', 'selected.jsonl'} | {f"{c['name']}.{ext}" for c in plan['configurations'] for ext in ('json', 'jsonl')}
    if (set(actual) != expected or type(complete['wall_seconds']) not in (int, float)
            or not math.isfinite(complete['wall_seconds']) or complete['wall_seconds'] < 0
            or read(execution / 'started.json')['status'] != 'started'
            or (execution / 'failed.json').exists() or complete['status'] != 'completed'
            or complete['files'] != actual or complete['plan_sha256'] != sha(plan_path)
            or read(execution / 'started.json')['plan_sha256'] != sha(plan_path)):
        raise ValueError('Execution is incomplete or altered')
    chosen = selection(excluded)
    if (chosen['selection_receipt']['status'] != 'completed'
            or chosen['selection_receipt'] != read(execution / 'selection.json')
            or chosen['rejection_log'] != rows(execution / 'rejections.jsonl')
            or chosen['selected'] != rows(execution / 'selected.jsonl')):
        raise ValueError('Selection does not reproduce')
    metrics, costs = {}, {}
    for config in plan['configurations']:
        name = config['name']
        receipt = read(execution / f'{name}.json')
        if (receipt['configuration'] != config or receipt['plan_sha256'] != sha(plan_path)
                or receipt['predictions_sha256'] != sha(execution / f'{name}.jsonl')):
            raise ValueError('Evaluation configuration changed')
        calculated = evaluation.audit_predictions(execution / f'{name}.jsonl', chosen['selected'], expected_arm=config['arm'])
        audit_evaluation_receipt(receipt['evaluation'], execution / f'{name}.jsonl', chosen['selected'], config, calculated)
        metrics[name] = calculated
        costs[name] = receipt['evaluation']
    summary = {'status': 'completed', 'plan_sha256': sha(plan_path),
        'execution_receipt_sha256': sha(execution / 'completed.json'), 'selection': chosen['selection_receipt'],
        'metrics': metrics, 'evaluation_receipts': costs, **aggregate(metrics, plan['configurations']),
        'scope': PROTOCOL['scope'], 'uncertainty': PROTOCOL['uncertainty'],
        'candidate_gates_changed': False, 'elo_estimate': None, 'novelty_established': False}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'summary.json', summary)
    write(out / 'completed.json', {'status': 'completed', 'summary_sha256': sha(out / 'summary.json'),
        'plan_sha256': sha(plan_path), 'execution_receipt_sha256': sha(execution / 'completed.json')})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--out', type=Path, required=True)
    for name in ('run', 'report'):
        p = sub.add_parser(name)
        p.add_argument('--plan', type=Path, required=True)
        p.add_argument('--out', type=Path, required=True)
        if name == 'report':
            p.add_argument('--execution', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = str(prepare(args.out))
    elif args.command == 'run':
        result = str(run(args.plan, args.out))
    else:
        result = report(args.plan, args.execution, args.out)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
