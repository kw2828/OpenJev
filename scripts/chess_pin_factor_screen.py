"""Frozen input coverage and independent replay for absolute-pin factors."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import chess
from openjev.research.chess_pin_factors import VERSION, candidate_factors

ROOT = Path(__file__).resolve().parents[1]
CACHE = 'runs/chess-child-graph-cache-v1/execution'
SOURCE = CACHE+'/train/roots.jsonl'
SOURCE_SHA = '6c8b551cd157e379dae0cf93c7131832b56affd19ee562a8fb6324c0ddad72a6'
PARENT_AUDIT = 'evidence/chess-child-graph-cache-v1/audit/receipt.json'
CODE = ('src/openjev/research/chess_pin_factors.py', 'tests/test_chess_pin_factors.py',
        'scripts/chess_pin_factor_screen.py', 'tests/test_chess_pin_factor_screen.py')
PROTOCOL = {
    'version': VERSION, 'source_roots': 32768, 'stride': 32, 'selected_roots': 1024,
    'indices': '0,32,...,32736 in the original training-root metadata, fixed before coverage outcomes.',
    'scope': 'All native legal candidates of 1024 fixed old training roots; label-free pin-witness input screen. No dev/shift panel, teacher, model or quality outcome is read.',
    'witness': 'Absolute pin only: same-color king and first blocker followed by an opposing compatible bishop, rook or queen. Both colors, fixed root-player square frame. Square-identity tuple(owner,king,blocker,attacker); factor role0 retained,1 added,2 removed.',
    'primary': 'Eight-ray geometric scan from both kings. Every child is a native legal successor. Check mirrored root and child witness equality after UCI rank reflection, source menu identity and board/history preservation.',
    'reference': 'Full native replay using python-chess is_pinned and a blocker-removed occupancy attack query, with independent coordinate conversion and factor classification. Shares board/rule library but not the primary ray detector.',
    'coverage': 'Report root pins, candidate pins, changed witnesses, candidate variation per root, counts-only versus identity-only changes, and retained/added/removed witnesses by owner. Descriptive; no post-hoc coverage threshold or quality inference.',
    'engineering_gate': 'Every primary root/candidate tuple and menu exactly reproduces in reference; every primary mirror and board-preservation check passes.',
    'time_cap_seconds': 900,
    'limits': 'Pins are established chess information and factor graphs are established architectures. This is not a new trained network, a legal-attack mask, search, tactical correctness proof or generic novelty claim. Relative pins, double-vacancy en-passant constraints and other king-safety rules are not encoded. All positions are previously exposed roots or legal children, plus their already-excluded mirrors.'}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def signature():
    if sha(ROOT/SOURCE) != SOURCE_SHA:
        raise ValueError('Original training metadata changed')
    audit = read(ROOT/PARENT_AUDIT)
    complete = read(ROOT/CACHE/'completed.json')
    if (audit['status'] != 'completed' or audit['execution_receipt_sha256'] != sha(ROOT/CACHE/'completed.json')
            or audit['reconstructed']['train'] != {'roots': 32768, 'children': 968036}
            or complete['files']['train/roots.jsonl'] != SOURCE_SHA):
        raise ValueError('Inherited native-input audit identity differs')
    return {'protocol': PROTOCOL, 'chess_version': chess.__version__,
            'sources': {p: sha(ROOT/p) for p in CODE}, 'input': SOURCE, 'input_sha256': SOURCE_SHA,
            'parent_audit_sha256': sha(ROOT/PARENT_AUDIT),
            'parent_completed_sha256': sha(ROOT/CACHE/'completed.json')}


def indices():
    return list(range(PROTOCOL['source_roots']))[::PROTOCOL['stride']]


def mirror_uci(uci):
    move = chess.Move.from_uci(uci)
    return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), promotion=move.promotion).uci()


def summarize(records):
    counts = Counter({k: 0 for k in ('roots', 'roots_with_pins', 'roots_with_changed_candidates',
                                   'roots_with_candidate_variation', 'candidates', 'candidates_with_pins',
                                   'changed_candidates', 'counts_changed_candidates', 'identity_only_changed_candidates')})
    roles = [[0, 0, 0], [0, 0, 0]]
    for row in records:
        before = row['before']; rows = row['candidates']
        counts['roots'] += 1; counts['roots_with_pins'] += bool(before)
        changed = 0; variants = set()
        before_counts = [sum(x[0] == owner for x in before) for owner in (0, 1)]
        for child in rows:
            after = child['after']
            after_counts = [sum(x[0] == owner for x in after) for owner in (0, 1)]
            different = after != before; changed += different
            counts['candidates'] += 1; counts['candidates_with_pins'] += bool(after)
            counts['changed_candidates'] += different
            counts['counts_changed_candidates'] += before_counts != after_counts
            counts['identity_only_changed_candidates'] += different and before_counts == after_counts
            variants.add(tuple(tuple(x) for x in after))
            for factor in child['factors']:
                roles[factor[0]][factor[4]] += 1
        counts['roots_with_changed_candidates'] += changed > 0
        counts['roots_with_candidate_variation'] += len(variants) > 1
    return {**counts, 'witness_roles_by_owner_retained_added_removed': roles}


def execute(plan_path, out, execution=None):
    begin = time.monotonic(); plan = read(plan_path)
    if plan != signature():
        raise ValueError('Frozen protocol/input/source changed')
    out.mkdir(parents=True, exist_ok=False)
    write(out/'started.json', {'plan_sha256': sha(plan_path), 'reference': execution is not None})
    try:
        all_rows = [json.loads(line) for line in (ROOT/SOURCE).open()]
        if len(all_rows) != PROTOCOL['source_roots'] or any(row['index'] != i for i, row in enumerate(all_rows)):
            raise ValueError('Wrong source membership')
        records = []
        for index in indices():
            source = all_rows[index]; board = chess.Board(source['fen'])
            original = board.fen(en_passant='fen'), list(board.move_stack)
            result = candidate_factors(board, reference=execution is not None)
            if [x['uci'] for x in result['candidates']] != source['moves']:
                raise AssertionError('Native menu differs from audited metadata')
            if execution is None:
                mirrored = candidate_factors(board.mirror())
                if result['before'] != mirrored['before']:
                    raise AssertionError('Root mirror differs')
                lookup = {x['uci']: x for x in mirrored['candidates']}
                for row in result['candidates']:
                    other = lookup[mirror_uci(row['uci'])]
                    if row['after'] != other['after'] or row['factors'] != other['factors']:
                        raise AssertionError('Candidate mirror differs')
            if original != (board.fen(en_passant='fen'), list(board.move_stack)):
                raise AssertionError('Caller board changed')
            records.append({'index': index, 'fen_sha256': hashlib.sha256(source['fen'].encode()).hexdigest(), **result})
            if time.monotonic()-begin > PROTOCOL['time_cap_seconds']:
                raise TimeoutError('Frozen pin input screen ceiling exceeded')
        records = json.loads(json.dumps(records))
        result = summarize(records)
        if result['roots'] != 1024 or plan != signature():
            raise ValueError('Membership or input authentication differs')
        if execution is None:
            with (out/'records.jsonl').open('x') as f:
                for row in records:
                    f.write(json.dumps(row, separators=(',', ':'))+'\n')
            write(out/'summary.json', result)
            files = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
            if time.monotonic()-begin > PROTOCOL['time_cap_seconds']:
                raise TimeoutError('Frozen pin input screen ceiling exceeded')
            write(out/'completed.json', {'status': 'completed', 'plan_sha256': sha(plan_path),
                  'files': files, 'mirrors_exact': True, 'boards_preserved': True,
                  'wall_seconds': time.monotonic()-begin})
        else:
            complete = read(execution/'completed.json')
            if complete['status'] != 'completed' or complete['plan_sha256'] != sha(plan_path):
                raise ValueError('Wrong primary receipt')
            if not complete['mirrors_exact'] or not complete['boards_preserved']:
                raise ValueError('Primary invariants failed')
            if {p.name for p in execution.iterdir()} != set(complete['files']) | {'completed.json'}:
                raise ValueError('Primary inventory changed')
            if any(sha(execution/name) != digest for name, digest in complete['files'].items()):
                raise ValueError('Primary bytes changed')
            saved = [json.loads(line) for line in (execution/'records.jsonl').open()]
            if saved != records or read(execution/'summary.json') != result:
                raise AssertionError('Exact witness/coverage replay failed')
            if time.monotonic()-begin > PROTOCOL['time_cap_seconds']:
                raise TimeoutError('Frozen pin audit ceiling exceeded')
            write(out/'receipt.json', {'status': 'passed', 'plan_sha256': sha(plan_path),
                  'completed_sha256': sha(execution/'completed.json'), 'summary': result,
                  'all_root_candidate_witnesses_exact': True, 'wall_seconds': time.monotonic()-begin})
        print(json.dumps(result), flush=True)
    except Exception as error:
        write(out/'failed.json', {'status': 'failed', 'error': repr(error), 'wall_seconds': time.monotonic()-begin})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('prepare', 'run', 'audit'))
    parser.add_argument('--plan', type=Path); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execution', type=Path); args = parser.parse_args()
    if args.command == 'prepare':
        plan = signature(); args.out.mkdir(parents=True, exist_ok=False); write(args.out/'plan.json', plan)
        print(json.dumps({'plan_sha256': sha(args.out/'plan.json'), 'roots': len(indices())}))
    else:
        if args.plan is None or (args.command == 'audit' and args.execution is None):
            parser.error('Expected --plan and, for audit, --execution')
        execute(args.plan, args.out, args.execution if args.command == 'audit' else None)
