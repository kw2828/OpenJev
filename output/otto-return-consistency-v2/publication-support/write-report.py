"""Publish already audited saved arithmetic; no model or simulator calls."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'output/otto-return-consistency-v2'
WORKER_PIN = '64cbe0ad150bc574d336d8d4472949ca80f136dfcfb2117d66b795234a06c212'
PLAN_PIN = '9bc2015d762c8d63df4f6bc51ee687f9281be7d855815ac3f475498dd0ced16f'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def closed(directory, pin):
    receipt = directory / 'receipt.json'
    assert digest(receipt) == pin
    data = json.loads(receipt.read_text())
    assert data['status'] == 'completed'
    assert {p.name for p in directory.iterdir()} == set(data['files']) | {'receipt.json'}
    for name, record in data['files'].items():
        path = directory / name
        assert digest(path) == record['sha256'] and path.stat().st_size == record['bytes']
    return data


parser = argparse.ArgumentParser()
parser.add_argument('--audit-sha256', required=True)
args = parser.parse_args()
worker = closed(BASE / 'run-01', WORKER_PIN)
audit = closed(BASE / 'audit-01', args.audit_sha256)
assert audit['agreement'] is True and audit['input_receipt_sha256'] == WORKER_PIN
assert audit['plan_sha256'] == worker['plan_sha256'] == PLAN_PIN
summary = json.loads((BASE / 'run-01/summary.json').read_text())
assert summary['model_prefix_rows'] == 144 and summary['unique_prefixes'] == 16
assert summary['unique_episodes'] == 2 and not summary['original_decisions_changed']
assert all(worker[key] == 0 for key in ('model_calls', 'training_calls', 'simulator_calls', 'policy_calls'))

text = [
    '# Saved branch consistency: a narrow diagnostic', '',
    '**All 144 saved model-prefix rows were processed and independently checked.** The minimum-linear model has larger average greedy-backup inconsistencies on these selected states than either MLP control. This motivates a learning-objective comparison; it does not establish why the autonomous study failed.', '',
    'The scope is exactly steps 0 through 7 of two length-three, initial-hit-one teacher episodes: one TRAIN episode lasting 79 steps and one VALID episode lasting 14 steps. All nine fixed models see the same 16 prefixes. Only eight of the full 1,109 VALID states occur here; this is not representative validation coverage or 144 independent episodes.', '',
    '[Original design](otto-return-consistency-design.md) · [Terminal-packet correction](otto-return-consistency-repair.md) · [All rows](../output/otto-return-consistency-v2/run-01/rows.jsonl) · [Summary](../output/otto-return-consistency-v2/run-01/summary.json) · [Independent audit](../output/otto-return-consistency-v2/audit-01/receipt.json)', '',
    '## What was measured', '',
    'Current error is the predicted physical remaining cost minus the realized teacher return. Teacher and greedy residuals subtract the current value from the corresponding one-step deployed backup. Switching advantage is the teacher-action backup minus the chosen-action backup. Branch values already use physical units; only the stored current scalar is multiplied by 64.', '',
    'All sixteen observation branches, the 1e-10 mass floor, biased zero-input values, signed predictions and strict eligible-action near-tie rule are retained. These are consistency measurements of the deployed floored arithmetic, not exact physical Bellman errors or observed counterfactual returns. Lower backup cost can indicate possible policy improvement or inaccurate successor predictions; this diagnostic cannot distinguish them.', '',
    '## Family means across all three fitting seeds', '',
    '| Family | Split | Current return MAE | Teacher residual MAE | Greedy residual MAE | Mean greedy residual | Mean switching advantage | Mean teacher disagreements / 8 |',
    '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
]
for family in ('min8', 'mlp8', 'homogeneous8'):
    for split in ('train', 'valid'):
        group = summary['family_means'][family][split]
        stats = group['statistics']
        text.append(f'| {family} | {split} | {stats["current_return_error"]["mean_absolute"]:.3f} | '
                    f'{stats["teacher_action_residual"]["mean_absolute"]:.3f} | '
                    f'{stats["greedy_action_residual"]["mean_absolute"]:.3f} | '
                    f'{stats["greedy_action_residual"]["signed_mean"]:.3f} | '
                    f'{group["mean_switching_advantage"]:.3f} | {group["mean_teacher_disagreement_count"]:.3f} |')
text += ['', 'Values are in movement-cost units. Means equally weight the three fitting seeds. Disagreement is not an error rate. The noisy realized teacher returns are not optimal values or expected action-value labels.', '',
         '## Every fit and split', '',
         '| Fit | Split | Current MAE | Teacher residual MAE | Greedy residual MAE | Mean greedy residual | Switching advantage | Teacher disagreements / 8 |',
         '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for fit, splits in sorted(summary['fit_split'].items()):
    for split in ('train', 'valid'):
        group = splits[split]
        stats = group['statistics']
        text.append(f'| {fit} | {split} | {stats["current_return_error"]["mean_absolute"]:.3f} | '
                    f'{stats["teacher_action_residual"]["mean_absolute"]:.3f} | '
                    f'{stats["greedy_action_residual"]["mean_absolute"]:.3f} | '
                    f'{stats["greedy_action_residual"]["signed_mean"]:.3f} | '
                    f'{group["mean_switching_advantage"]:.3f} | {group["teacher_disagreement_count"]} |')
text += ['', 'Signed means, MAE, per-seed RMS and all branch contributions are retained in the linked complete records. The published family RMS is the mean of per-seed RMS values, not a pooled RMS. There is no significance test or efficacy gate.', '',
         '## Execution and next experiment', '',
         f'The corrected saved diagnostic took {worker["wall_seconds"]:.3f} seconds with {worker["peak_rss_bytes"]:,} bytes peak RSS. It made zero model, training, policy or simulator calls. All 44 current-study payloads, 41 teacher payloads and two original-audit payloads were authenticated before reading numerical records. The separate independent audit checks the joins, arithmetic and reporting without generating new model predictions.', '',
         'The first attempt failed on terminal packets with an empty legal-action list, before producing any diagnostic rows. Its original source, plan and failed receipt remain preserved. The separately frozen correction changes only that terminal-packet contract. [Failed attempt](../output/otto-return-consistency-v1/execution-witness.json) · [Completed execution](../output/otto-return-consistency-v2/execution-witness.json).', '',
         'The next proposed learning control compares continued Monte Carlo fitting with delayed observation-backup targets in the same ordinary MLP, plus unchanged-checkpoint references. It requires runtime qualification and fresh evaluation cases. [Concrete design](otto-bellman-control-design.md) · [Research rationale and prior art](otto-learning-direction.md).', '',
         'No new policy has been trained or evaluated by this diagnostic. The original scalar-return study remains **0/54**, and no recurrent-memory, biological-wiring or novel-architecture advantage is established.', '']
out = ROOT / 'research/otto-return-consistency-results.md'
with out.open('x') as stream:
    stream.write('\n'.join(text))
receipt = {'status': 'completed', 'scope': 'Presentation of independently audited saved arithmetic only.',
           'worker_sha256': WORKER_PIN, 'audit_sha256': args.audit_sha256, 'plan_sha256': PLAN_PIN,
           'source_sha256': digest(Path(__file__)), 'report_path': str(out.relative_to(ROOT)),
           'report_sha256': digest(out), 'family_split_rows': 6, 'fit_split_rows': 18, 'scientific_calls': 0}
with (BASE / 'publication-support/receipt.json').open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps(receipt))
