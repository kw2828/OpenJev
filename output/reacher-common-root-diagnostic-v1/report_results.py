"""Frozen saved-output report; no simulator, RNG or learned-model calls."""
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from openjev.research.reacher_tracking_rollout import sha, write

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
RUN = BASE/'attempt-01'
OUT = BASE/'review-01'


def load(path):
    with np.load(path, allow_pickle=False) as value:
        return {name: value[name].copy() for name in value.files}


def useful(rows, contrast, rule):
    assert len(rows) == rule['slot_count'] == 12
    assert {r['role'] for r in rows} == {'nominal', 'public_gain', 'true_state'}
    assert all(sum(r['role'] == role for r in rows) == 4 for role in ('nominal', 'public_gain', 'true_state'))
    baseline = np.array([r['contrasts'][contrast]['baseline'] for r in rows])
    candidate = np.array([r['contrasts'][contrast]['candidate'] for r in rows])
    assert np.isfinite(baseline).all() and np.isfinite(candidate).all() and np.all(baseline > 0) and np.all(candidate >= 0)
    def mean(values):
        return math.fsum(values)/len(values)
    role_means = {role: {'baseline': mean([r['contrasts'][contrast]['baseline'] for r in rows if r['role'] == role]),
                        'candidate': mean([r['contrasts'][contrast]['candidate'] for r in rows if r['role'] == role])}
                  for role in ('nominal', 'public_gain', 'true_state')}
    baseline_sum = math.fsum(baseline); candidate_sum = math.fsum(candidate)
    exact_baseline = sum(Fraction(str(float(x))) for x in baseline)
    exact_candidate = sum(Fraction(str(float(x))) for x in candidate)
    improvement = (baseline_sum-candidate_sum)/baseline_sum
    absolute = (baseline_sum-candidate_sum)/len(rows)
    count = int(np.sum(candidate <= baseline))
    checks = {'relative': exact_candidate <= (1-Fraction(str(rule['pooled_cost_reduction_fraction_min'])))*exact_baseline,
              'absolute': exact_candidate <= exact_baseline-Fraction(str(rule['pooled_cost_reduction_per_action_min']))*len(rows),
              'trajectory_means': all(sum(Fraction(str(float(r['contrasts'][contrast]['candidate']))) for r in rows if r['role'] == role)
                                     <= sum(Fraction(str(float(r['contrasts'][contrast]['baseline']))) for r in rows if r['role'] == role)
                                     for role in role_means),
              'nonworse_slots': count >= rule['nonworse_slot_means_min']}
    return {'pass': all(checks.values()), 'checks': checks, 'improvement_percent': 100*improvement,
            'absolute_cost_reduction': absolute, 'baseline_mean': baseline_sum/len(rows),
            'candidate_mean': candidate_sum/len(rows), 'nonworse_slots': count, 'trajectory_means': role_means}


def main():
    protocol = json.loads((BASE/'protocol.json').read_text())
    arithmetic = json.loads((BASE/'report-arithmetic.json').read_text())
    assert arithmetic['version'] == 'exact-canonical-decimal-gates-v1'
    completed = json.loads((RUN/'completed.json').read_text())
    execution = json.loads((RUN/'execution-completed.json').read_text())
    assert completed['status'] == execution['status'] == 'completed' and not (RUN/'failed.json').exists()
    assert sha(RUN/'execution-completed.json') == completed['execution_completed_sha256']
    assert execution['rows'] == completed['rows'] and execution['work'] == completed['work']
    assert len(completed['rows']) == len(completed['audits']) == 12
    expected = [f'{r}--{t:03d}' for r in protocol['roles'] for t in protocol['root_steps']]
    assert [r['name'] for r in completed['rows']] == expected
    assert [r['name'] for r in completed['audits']] == expected
    for name, digest in completed['source_sha256'].items():
        assert sha(ROOT/name) == sha(RUN/'source-snapshot'/name) == digest
    for item in completed['audits']:
        path = RUN/f'audit-{item["name"]}.json'
        assert sha(path) == item['sha256']
        assert json.loads(path.read_text())['status'] == 'completed'
    rows = []; root_ids = {}; startup = None; payloads = {}
    for item in completed['rows']:
        folder = RUN/'slots'/item['name']; receipt = json.loads((folder/'completed.json').read_text())
        assert sha(folder/'completed.json') == item['completed_sha256']
        assert receipt['status'] == 'completed' and receipt['work'] == item['work']
        members = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
        assert members == set(receipt['files']) | {'completed.json'}
        for name, digest in receipt['files'].items():
            assert sha(folder/name) == digest
            payloads[(folder/name).relative_to(ROOT).as_posix()] = digest
        root = json.loads((folder/'root.json').read_text()); union = load(folder/'union.npz')
        branches = load(folder/'branches/data.npz')
        native24 = -branches['rewards'].mean(axis=2)
        native12 = -branches['rewards'][:, :, :12].mean(axis=2)
        pool24 = native24.mean(axis=0); mapping = union['slot_to_unique']
        cross = {g: load(folder/'cross'/f'{g}.npz') for g in ('nominal', 'actual')}
        ids = {g: int(np.argmax(cross[g]['scores_24'])) for g in cross}
        winners = {'nominal_short': int(mapping[64]), 'nominal_restarts': int(mapping[65]),
                   'nominal_long': int(mapping[66]), 'actual_short': int(mapping[67]),
                   'actual_restarts': int(mapping[68]), 'actual_long': int(mapping[69]),
                   'pool_nominal': ids['nominal'], 'pool_actual': ids['actual']}
        measures = {name: {'unique_id': i, 'cost24': float(pool24[i]),
                          'cost12': float(native12[:, i].mean()), 'branches24': native24[:, i].tolist()}
                    for name, i in winners.items()}
        contrasts = {name: {'candidate': measures[c]['cost24'], 'baseline': measures[b]['cost24']}
                     for name, c, b in [('search', 'actual_restarts', 'actual_short'),
                                       ('horizon', 'actual_long', 'actual_restarts'),
                                       ('gain', 'pool_actual', 'pool_nominal')]}
        root_identity = hashlib.sha256(json.dumps(root['root'], sort_keys=True).encode()).hexdigest()
        root_ids.setdefault(root_identity, []).append(item['name'])
        if root['step'] == 0:
            if startup is None:
                startup = branches
            else:
                assert set(startup) == set(branches) and all(np.array_equal(startup[k], branches[k]) for k in startup)
        rows.append({'name': item['name'], 'role': root['source_role'], 'step': root['step'],
            'root_identity': root_identity, 'true_gain': root['true_gain'], 'unique_sequences': len(pool24),
            'measures': measures, 'contrasts': contrasts,
            'posthoc_pool_best_cost24': float(pool24.min()),
            'search_prefix_gap': float(cross['actual']['scores_12'][winners['actual_long']]-cross['actual']['scores_12'][winners['actual_restarts']]),
            'wall_seconds': item['wall_seconds']})
    checks = {k: useful(rows, k, protocol['engineering_usefulness_rule']) for k in ('search', 'horizon', 'gain')}
    if checks['gain']['pass'] and checks['horizon']['pass']:
        decision = 'Prepare only a longer-horizon closed-loop engineering correction; no fresh scientific launch.'
    elif checks['gain']['pass'] and checks['search']['pass']:
        decision = 'Prepare only the extra-short-search closed-loop engineering correction; horizon contrast still failed.'
    else:
        decision = 'Stop this task/controller recipe as evidence for learned adaptation. No additional tuning sweep.'
    summary = {'status': 'completed', 'classification': protocol['classification'], 'decision': decision,
        'checks': checks, 'rows': rows, 'root_identity_groups': root_ids,
        'distinct_native_roots': len(root_ids), 'startup_branches_bit_exact': True,
        'work': completed['work'], 'execution_seconds': execution['wall_seconds'],
        'audit_seconds': completed['audit_wall_seconds'], 'all_payload_sha256': payloads,
        'run_completed_sha256': sha(RUN/'completed.json'), 'protocol_sha256': sha(BASE/'protocol.json'),
        'report_arithmetic': arithmetic, 'report_arithmetic_sha256': sha(BASE/'report-arithmetic.json')}
    OUT.mkdir(exist_ok=False); write(OUT/'summary.json', summary)
    lines = ['# Common-root planning diagnostic', '',
        '**Twelve exposed root slots, including repeated startup states. No learned model, scientific qualification or architecture claim.**', '',
        'Selection rules were fixed before execution and depend only on modeled scores, never native branch outcomes. Native cost is distance plus applied-command effort per action, averaged over four shared noise branches. Short plans hold their final command through step 24.', '',
        '| Contrast | Mean baseline | Mean candidate | Improvement | Nonworse slots | Fixed rule |',
        '|---|---:|---:|---:|---:|---|']
    for name, result in checks.items():
        lines.append(f"| {name} | {result['baseline_mean']:.6f} | {result['candidate_mean']:.6f} | {result['improvement_percent']:+.3f}% | {result['nonworse_slots']}/12 | {'PASS' if result['pass'] else 'FAIL'} |")
    lines += ['', 'The rule requires at least 3% and 0.001/action pooled improvement, no worse mean on each of the three source trajectories, and at least 8/12 nonworse slots. Each contrast stands alone.', '',
        '| Source trajectory | Root | Short A | Short A+B | Long A | Union nominal | Union actual |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        vals = [row['measures'][k]['cost24'] for k in ('actual_short', 'actual_restarts', 'actual_long', 'pool_nominal', 'pool_actual')]
        lines.append(f"| {row['role']} | {row['step']} | "+' | '.join(f'{x:.6f}' for x in vals)+' |')
    lines += ['', decision, '',
        f"Execution {execution['wall_seconds']:.3f}s; independent replay {completed['audit_wall_seconds']:.3f}s. Work: {json.dumps(completed['work'], sort_keys=True)}.", '',
        '![All root costs and fixed contrasts](figure.png)', '',
        'These are open-loop branches from reused states. The held-tail convention, privileged state/gain and small exposed root set limit the inference. H12 MPC would normally replan after acting, unlike these held-tail branches. The gain contrast ranks a common union populated by both gain-conditioned searches; it measures dynamics-dependent ranking, not a deployable online estimator or its compute value. No contrast establishes closed-loop improvement, generalization or a useful recurrent architecture. The minimum four-branch mean cost within the union is post-hoc descriptive context, never a deployable selected method. Slot timers include payload hashes but exclude their final completion write; the enclosing execution time includes it.', '']
    (OUT/'report.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    x = np.arange(12)
    for name, label in [('actual_short', 'Short A'), ('actual_restarts', 'Short A+B'), ('actual_long', 'Long A')]:
        axes[0].plot(x, [r['measures'][name]['cost24'] for r in rows], marker='o', label=label)
    axes[0].set_xticks(x, [r['role'].replace('public_gain', 'gain').replace('true_state', 'state')+'\n'+str(r['step']) for r in rows], rotation=45, ha='right')
    axes[0].set_ylabel('Native cost / action (lower is better)'); axes[0].set_title('Every root: correct gain, four shared branches'); axes[0].legend()
    values = [checks[k]['improvement_percent'] for k in checks]
    axes[1].bar(list(checks), values, color=['#2563eb' if checks[k]['pass'] else '#64748b' for k in checks])
    axes[1].axhline(3, linestyle='--', color='#c2410c', label='3% margin (other criteria also required)')
    axes[1].axhline(0, color='#111827', linewidth=.8); axes[1].set_ylabel('Pooled native-cost improvement (%)'); axes[1].legend()
    axes[1].set_title('Separate fixed engineering contrasts')
    fig.suptitle('Common states: search, horizon and dynamics knowledge\nExposed engineering diagnostic; no architectural claim')
    fig.savefig(OUT/'figure.png', dpi=140); plt.close(fig)
    write(OUT/'receipt.json', {'status': 'completed', 'report_source_sha256': sha(Path(__file__)),
        'run_completed_sha256': sha(RUN/'completed.json'),
        'files': {name: sha(OUT/name) for name in ('summary.json', 'report.md', 'figure.png')}})
    print(json.dumps({'status': 'completed', 'checks': checks, 'decision': decision}))


if __name__ == '__main__':
    existed = OUT.exists()
    try:
        main()
    except BaseException as error:
        if not existed and OUT.exists():
            if (OUT/'receipt.json').exists():
                try:
                    (OUT/'receipt.json').rename(OUT/'partial-receipt.json')
                except BaseException as secondary:  # noqa: BLE001 - preserve original failure.
                    error.add_note('Receipt demotion failed: '+repr(secondary))
            try:
                write(OUT/'failed.json', {'status': 'failed', 'error': repr(error)})
            except BaseException as secondary:  # noqa: BLE001 - preserve original failure.
                error.add_note('Report failure receipt failed: '+repr(secondary))
        raise
