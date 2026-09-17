import csv
import importlib.util
import json
from pathlib import Path

import chess
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('chess_publish_tested', ROOT/'scripts/publish_chess_results.py')
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


@pytest.fixture
def execution(tmp_path):
    study = publisher.load_runner()
    csv_path = tmp_path/'puzzles.csv'
    with csv_path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['PuzzleId', 'FEN', 'Moves', 'Rating', 'GameUrl'])
        writer.writeheader()
        for i in range(24):
            writer.writerow({'PuzzleId': f'fixture-{i}', 'FEN': chess.STARTING_FEN, 'Moves': 'e2e4 e7e5',
                             'Rating': [1000, 1500, 2000][i % 3], 'GameUrl': 'https://example.org/fixture'})
    plan = tmp_path/'plan.json'
    study.prepare({'policies': {'fixture': {'kind': 'greedy'}}, 'max_plies': 2,
                   'games': [{'id': 'fixture', 'white': 'fixture', 'black': 'fixture'}]}, csv_path, plan)
    out = tmp_path/'execution'
    study.run(plan, out)
    return out, plan


def test_verification_preserves_unfinished_game(execution):
    plan, summary, games = publisher.verified_results(*execution)
    assert summary['unfinished_game_count'] == 1
    assert games[0]['result'] == '*'
    assert plan['config']['max_plies'] == 2


def test_corrupt_attempt_cannot_be_published(execution):
    out, plan = execution
    with (out/'puzzles.jsonl').open('a') as stream:
        stream.write('{}\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        publisher.verified_results(out, plan)


def test_summary_edit_detected_even_if_receipt_rehashed(execution):
    out, plan = execution
    summary = json.loads((out/'summary.json').read_text())
    summary['scored_game_count'] = 1
    (out/'summary.json').write_text(json.dumps(summary))
    completed = json.loads((out/'completed.json').read_text())
    completed['outputs']['summary.json'] = publisher.sha(out/'summary.json')
    (out/'completed.json').write_text(json.dumps(completed))
    with pytest.raises(ValueError, match='Summary does not match'):
        publisher.verified_results(out, plan)


def test_subset_cannot_be_published_as_full_panel(execution):
    out, plan = execution
    summary = json.loads((out/'summary.json').read_text())
    summary['entire_prepared_panel'] = False
    (out/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='subset'):
        publisher.verified_results(out, plan)


def test_empty_output_manifest_cannot_bypass_hash_checks(execution):
    out, plan = execution
    completed = json.loads((out/'completed.json').read_text())
    completed['outputs'] = {}
    (out/'completed.json').write_text(json.dumps(completed))
    with pytest.raises(ValueError, match='omits required output hashes'):
        publisher.verified_results(out, plan)


def test_plan_edit_cannot_keep_old_identity(execution):
    out, plan = execution
    data = json.loads(plan.read_text())
    data['config']['clock_seconds'] += 1
    plan.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Canonical plan'):
        publisher.verified_results(out, plan)


def test_failure_receipt_prevents_publication(execution):
    out, plan = execution
    (out/'failed.json').write_text('{}')
    with pytest.raises(ValueError, match='failure receipt'):
        publisher.verified_results(out, plan)
