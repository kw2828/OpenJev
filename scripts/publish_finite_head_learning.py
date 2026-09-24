"""Publish closed, audited saved JSON and opaque evidence without replaying learning."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'output/finite-head-learning-v2'
FAILED_STUDY = ROOT / 'output/finite-head-learning-v1'
PLAN = STUDY / 'study-registration.json'
PLAN_SHA = '0ccd7cacca49f939b50fe58475b7ade84cdf23bd37c298b88a7aa6bb1b202963'
ENGINEERING_SHA = '74cd97a9b28895e655b2b69329e24e04941b818f849300ddb214b696efe56434'
OUTPUT = ROOT / 'research/finite-head-learning-results'
OVERVIEW = ROOT / 'research/finite-head-learning-results.md'
VERSION = 'finite-head-learning-publication-v1'
ARMS = ('rounded_anchor', 'rounded_random', 'matched_free_random')
SEEDS = (436261001, 436261002, 436261003, 436261004, 436261005)
CRITERIA = ('SHORT_HORIZON_LEARNING', 'BLIND_EXTRAPOLATION',
            'OBSERVED_FILTERING_EXTRAPOLATION')
LABELS = ('Rounded anchor', 'Rounded random', 'Matched free random')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def descriptor(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve() == path.absolute(),
            'canonical ordinary evidence file')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text())


def authenticate():
    """Authenticate all original processes and bytes before current metric JSON."""
    require(re.fullmatch(r'[0-9a-f]{64}', PLAN_SHA) and re.fullmatch(r'[0-9a-f]{64}', ENGINEERING_SHA),
            'publication remains unbound until original registrations exist')
    require(descriptor(PLAN)['sha256'] == PLAN_SHA, 'exact scientific registration')
    plan = read(PLAN)
    require(len(plan['sources']) == 137 and all(descriptor(ROOT / name) == expected
            for name, expected in plan['sources'].items()), 'all frozen sources before helper imports')
    from finite_head_learning_worker_v2 import (
        admit_qualification,
        closed_producer,
        files,
        validate_launch_binding,
        validate_registered_plan,
    )
    validate_registered_plan(PLAN, plan)
    # The registered validator authenticates the original failed process, exact
    # test-only repair, all 131 original sources and the complete old snapshot.
    failed = plan['head_prerequisite']['failed_qualification']
    require(plan['version'] == 'finite-head-learning-v2' and plan['mode'] == 'study'
            and failed['folder'] == str(FAILED_STUDY) and failed['status'] == 'FAILED'
            and failed['tests_passed'] == 372 and failed['tests_failed'] == 3
            and failed['exposure_started'] is False and failed['science_started'] is False
            and len(failed['sources']) == 131 and files(FAILED_STUDY) == failed['files'],
            'complete closed v1 qualification preserved without exposure or science')
    engineering_path = STUDY / 'engineering-registration-01.json'
    require(descriptor(engineering_path)['sha256'] == ENGINEERING_SHA,
            'exact original engineering registration')
    require(sorted(STUDY.glob('engineering-registration-*.json')) == [engineering_path],
            'one original engineering attempt')
    qualify_terminal = admit_qualification(plan, plan['qualification']['path'])
    require(descriptor(plan['qualification']['path']) == plan['qualification']['descriptor']
            and descriptor(qualify_terminal) == plan['qualification']['terminal'],
            'original qualification joins scientific plan')
    phases = {}
    for phase in ('qualify', 'fit', 'audit'):
        spec = plan['phases'][phase]
        receipt_path = Path(spec['output'] + '.receipt.json')
        receipt = read(receipt_path)
        original_plan = engineering_path if phase == 'qualify' else PLAN
        require(receipt['phase'] == phase and receipt['status'] == 'PASS'
                and receipt['plan'] == str(original_plan)
                and receipt['plan_sha256'] == descriptor(original_plan)['sha256']
                and receipt['sources_before'] == receipt['sources_after'] == plan['sources'],
                'original successful phase and unchanged complete sources')
        validate_launch_binding(plan, receipt)
        require(read(spec['supervision']) == receipt['launch'], 'persisted original launch')
        terminal_path = closed_producer(receipt)
        require(files(spec['output']) == receipt['files'], 'complete original phase payload inventory')
        phases[phase] = {'receipt': receipt, 'receipt_path': receipt_path,
                         'terminal': read(terminal_path), 'terminal_path': terminal_path}
    for before, after in (('qualify', 'fit'), ('fit', 'audit')):
        first, second = phases[before]['terminal'], phases[after]['terminal']
        require(first['clock_backend'] == second['clock_backend']
                and first['finished_ns'] <= second['started_ns'], 'original phase chronology')
    require(phases['audit']['receipt']['producer_receipt'] == descriptor(phases['fit']['receipt_path'])
            and phases['audit']['receipt']['producer_terminal'] == descriptor(phases['fit']['terminal_path']),
            'audit binds the original closed producer')
    require(phases['fit']['receipt']['result'] == {'fits': 15, 'rows': 60}, 'all fifteen fits and endpoint rows')
    study_files = files(STUDY)
    # All current scientific results are first read after the checks above.
    producer = read(Path(plan['phases']['fit']['output']) / 'summary.json')
    audit = read(Path(plan['phases']['audit']['output']) / 'audit.json')
    from audit_finite_observation_learning import compare
    require(producer['version'] == 'finite-head-learning-v1' and producer['implementation'] == 'reuse'
            and producer['config'] == plan['config'] and audit['version'] == 'finite-head-learning-audit-v1'
            and audit['profile'] == 'science' and audit['agreement'] is True
            and audit['technical_complete'] is False and audit['requires_original_supervisor_closure'] is True
            and audit['exact_oracle_agreement'] is True, 'original independently audited scientific record')
    expected_fits = {(arm, seed) for arm in ARMS for seed in SEEDS}
    require(len(producer['fits']) == 15 and {(r['arm'], r['seed']) for r in producer['fits']} == expected_fits,
            'complete fixed fit roster')
    require(len(audit['rows']) == 60 and {(r['arm'], r['seed'], r['horizon']) for r in audit['rows']}
            == {(arm, seed, h) for arm, seed in expected_fits for h in (1, 2, 4, 8)}, 'complete metric roster')
    row_key = lambda row: (row['arm'], row['seed'], row['horizon'])
    compare(sorted(producer['rows'], key=row_key), sorted(audit['rows'], key=row_key))
    compare(sorted(producer['baseline_rows'], key=lambda r: r['horizon']), audit['baseline_rows'])
    compare(producer['prefix_rows'], audit['prefix_rows'])
    require(len(audit['prefix_rows']) == 15 and producer['structural_work'] == audit['structural_work'],
            'all prefix rows and disjoint actual work')
    require(audit['metadata']['fits'] == 15 and audit['metadata']['payloads'] == 136
            and audit['counts'] == {'array_decodes': 69, 'checkpoint_decodes': 45,
                'optimizer_json_decodes': 45, 'model_calls': 0, 'optimizer_calls': 0,
                'world_or_generator_calls': 0, 'native_calls': 0, 'initializer_reconstructions': 5}, 'complete saved-output audit scope')
    require(set(audit['gates']) == set(ARMS), 'all three arm criteria')
    for arm in ARMS:
        require(set(audit['gates'][arm]) == set(CRITERIA), 'all three absolute criteria')
        for gate in audit['gates'][arm].values():
            require(all(type(v) is bool for v in gate['conditions'].values())
                    and gate['passed'] == all(gate['conditions'].values()), 'saved absolute gate conjunction')
    advance = audit['advance']
    require(len(advance['conditions']) == 15 and all(type(v) is bool for v in advance['conditions'].values())
            and advance['passed'] == all(advance['conditions'].values())
            and advance['status'] == 'HEAD_INDEPENDENT_ADVANCE_' + ('PASS' if advance['passed'] else 'FAIL')
            and audit['allocation_comparisons']['advance'] == advance
            and len(audit['allocation_comparisons']['paired']) == 10
            and len(audit['allocation_comparisons']['mean_regret']) == 2
            and len(audit['allocation_comparisons']['mean_fit_time']) == 1
            and len(audit['allocation_comparisons']['anchor_comparisons']) == 20
            and audit['allocation_comparisons']['anchor_is_descriptive_only'] is True
            and phases['audit']['receipt']['result'] == {'gates': audit['gates'], 'advance': advance},
            'all fifteen primary conditions and descriptive anchor contrasts and saved audit receipt')
    require(producer['counts']['fit_count'] == 15 and producer['counts']['accepted_prefix_steps'] == 15360
            and producer['counts']['accepted_joint_steps'] == 46080
            and producer['counts']['accepted_optimizer_steps'] == 61440
            and producer['checkpoint_barrier']['dev_generation_count'] == 0,
            'exact scientific update counts and pre-DEV barrier')
    require(producer['counts']['dev_generation_count'] == producer['counts']['oracle_model_constructions'] == 1
            and producer['regimes'] == {'train': {'split_id': 0, 'epsilon': .12}, 'base': {'split_id': 1, 'epsilon': .12}},
            'one BASE-only cohort and explicit epsilon-.12 reference')
    require(producer['prefix_pair_checks'] == audit['metadata']['prefix_pair_checks']
            == producer['checkpoint_barrier']['prefix_pair_checks']
            and len(producer['prefix_pair_checks']) == 5
            and all(row['same_initial_and_prefix_dynamics'] is True
                    and row['same_prefix_optimizer_states'] is True
                    and row['all_heads_unchanged_after_prefix'] is True
                    for row in producer['prefix_pair_checks']), 'all pre-DEV head and prefix integrity checks')
    require(audit['metadata']['dynamics_boundary_hash_exports'] == audit['metadata']['head_boundary_hash_exports'] == 45,
            'complete charged head and dynamics boundary exports')
    historical = plan['head_prerequisite']['head_prerequisite']['parent_advance']
    require(historical['passed'] is False and len(historical['conditions']) == 54
            and sum(historical['conditions'].values()) == 45,
            'published replication failure remains a failed historical prerequisite')
    log = (Path(plan['phases']['qualify']['output']) / 'command-1.log').read_text()
    completions = re.findall(r'(?m)^\s*(\d+) passed(?:, (\d+) warnings?)? in [^\n]+$', log)
    require(len(completions) == 1 and int(completions[0][0]) > 0, 'one complete passing selected-test result')
    qualification = {'tests_passed': int(completions[0][0]), 'warnings': int(completions[0][1] or 0)}
    return {'plan': plan, 'phases': phases, 'producer': producer, 'audit': audit,
            'study_files': study_files, 'failed_qualification': failed, 'qualification': qualification}


def chart(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5), layout='constrained')
    colors = ('#2864a3', '#c17622', '#248372', '#9264a8', '#bd4957')
    labels = ('Rounded\nanchor', 'Rounded\nrandom', 'Matched free\nrandom')
    keyed = {(r['arm'], r['seed'], r['horizon']): r for r in summary['rows']}
    fitted = {(r['arm'], r['seed']): r for r in summary['fits']}
    for panel, axis in enumerate(axes):
        horizon = (4, 8)[panel] if panel < 2 else None
        for j, seed in enumerate(SEEDS):
            ys = [keyed[arm, seed, horizon]['blind_regret'] if horizon is not None
                  else fitted[arm, seed]['seconds'] for arm in ARMS]
            xs = [i + (j - 2) * .04 for i in range(3)]
            axis.plot(xs, ys, color=colors[j], marker='o', linewidth=1, markersize=5,
                      alpha=.85, label=str(seed))
        for i, arm in enumerate(ARMS):
            values = [keyed[arm, seed, horizon]['blind_regret'] if horizon is not None
                      else fitted[arm, seed]['seconds'] for seed in SEEDS]
            axis.scatter(i, math.fsum(values) / len(SEEDS), color='#152130', marker='D', s=35,
                         zorder=4, label='Mean' if i == 0 else None)
        axis.set_xticks(range(3), labels)
        axis.set_xlim(-.35, 2.35)
        if horizon is not None and all(keyed[arm, seed, horizon]['blind_regret'] > 0
                                       for arm in ARMS for seed in SEEDS):
            axis.set_yscale('log')
            axis.set_ylabel('Regret, log scale (lower is better)')
        else:
            axis.set_ylim(bottom=0)
            axis.set_ylabel('Regret (lower is better)' if horizon else 'Seconds (lower is faster)')
        axis.set_title(f'H{horizon} blind decision regret' if horizon else 'Complete fit time', fontsize=12)
        axis.grid(axis='y', color='#e4e8ed')
        axis.spines[['top', 'right']].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='outside lower center', ncol=3, frameon=False, fontsize=9)
    fig.suptitle('BASE-only head-prior diagnostic: all five seeds and three arms', fontsize=14, weight='bold')
    fig.savefig(path, dpi=180, metadata={'Description':
        'Saved audited JSON only. All fifteen fits; paired lines identify fit seeds. '
        'Positive regret panels use labeled log axes; zero-containing panels use a linear axis. '
        'Diamonds are arithmetic means without confidence intervals. No model or checkpoint decoding.'})
    plt.close(fig)


def document(summary):
    audit, comparison = summary['audit'], summary['audit']['allocation_comparisons']
    lines = ['# Does performance depend on the starting cost head?', '',
        f"**{summary['status']}**. {sum(audit['advance']['conditions'].values())}/15 prospectively fixed primary conditions pass.", '',
        ('Fifteen fresh fits compare rounded transport with its original task-derived head, rounded transport '
         'with a task-independent random head, and initially function-matched free transport with the same random head. '
         'The random-head rounded arm is the candidate; random-head matched free is its sole primary control. '
         'The original-head rounded anchor is descriptive and cannot rescue a failed candidate.'), '',
        '![All five seeds: H4/H8 blind regret and complete fit time](benchmark.png)', '',
        ('Both random arms use the same local standard-normal 4-by-8 logits from PCG64 with '
         'SeedSequence([fit_seed, 436, 1]). Draws are not centered, rescaled, selected or tuned. '
         'The inherited constructor computes its original head first, then discards it in the random arms. '
         'All arms retain 352 float64 parameters, the same losses, 1,024 prefix updates and 3,072 joint updates. '
         'Generic random logits change scale and shape as well as removing the cost-template prior; '
         'the comparison does not isolate latent-label alignment alone.'), '',
        '| Model | Short horizon | Blind extrapolation | Observed filtering |', '|---|---:|---:|---:|']
    for arm, label in zip(ARMS, LABELS, strict=True):
        values = []
        for criterion in CRITERIA:
            gate = audit['gates'][arm][criterion]
            values.append(f"{'PASS' if gate['passed'] else 'FAIL'} {sum(gate['conditions'].values())}/{len(gate['conditions'])}")
        lines.append('| ' + ' | '.join([label, *values]) + ' |')
    lines += ['', ('The candidate must pass all three absolute criteria, all ten nonpositive paired H4/H8 regret '
        'differences and both 10% mean-regret improvements against its primary control. Every seed remains binding. '
        'An anchor result or favorable average cannot rescue a failed condition.'), '',
        '| Primary control | H | Candidate mean regret | Control mean regret | Relative reduction |',
        '|---|---:|---:|---:|---:|']
    for row in comparison['mean_regret']:
        reduction = 'undefined (zero control)' if row['relative_reduction'] is None else f"{100 * row['relative_reduction']:.2f}%"
        lines.append(f"| {row['control']} | {row['horizon']} | {row['candidate_mean_regret']:.8g} | {row['control_mean_regret']:.8g} | {reduction} |")
    lines += ['', ('Relative reductions are ratios of the five fit means; positive values favor rounded random. '
        'All seed values remain visible because weak controls can dominate averages. Five seeds share one TRAIN '
        'cohort and are not five independent population studies. No confidence interval or significance claim is made. '
        'The figure labels log axes when all regrets are positive and uses a linear axis when any is zero.'), '',
        '| Control | Candidate mean fit seconds | Control mean fit seconds | Candidate / control time |',
        '|---|---:|---:|---:|']
    for row in comparison['mean_fit_time']:
        lines.append(f"| {row['control']} | {row['candidate_mean_seconds']:.6f} | {row['control_mean_seconds']:.6f} | {row['ratio']:.6f} |")
    lines += ['', '| Model | Seed | Prefix updates | Joint updates | Prefix stage seconds | Joint stage seconds | Complete fit seconds | Controller seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    stages = {(r['arm'], r['seed']): r['stages'] for r in audit['allocation']['stages']}
    for row in sorted(summary['fits'], key=lambda r: (ARMS.index(r['arm']), r['seed'])):
        times = [s['stopped_elapsed'] - s['start_elapsed'] for s in stages[row['arm'], row['seed']]]
        lines.append(f"| {row['arm']} | {row['seed']} | {row['accepted_prefix_updates']} | {row['updates']} | {times[0]:.6f} | {times[1]:.6f} | {row['seconds']:.6f} | {row['timed_seconds']:.6f} |")
    phase, qualification = summary['phase_seconds'], summary['qualification']
    lines += ['', (f"Original native phases: qualification {phase['qualify']:.6f}s, fit/evaluation {phase['fit']:.6f}s, "
        f"audit {phase['audit']:.6f}s. Selected qualification: {qualification['tests_passed']} passing tests and "
        f"{qualification['warnings']} warning(s), plus the retained independently audited exposure probe."), '',
        (f"The original v1 engineering qualification remains FAILED: 372 tests passed, three failed and one "
         f"warning was recorded; its native phase lasted {summary['failed_qualification']['seconds']:.9f}s. "
         'All three failures were the same public-keyword rejection test expecting ValueError instead of the '
         'actual TypeError. Exposure and scientific fitting never started in v1. A separate v2 registration '
         'corrected that assertion and its admission plumbing while preserving all original 131 sources and '
         'failure evidence. The model, trainer, runner, audit, science configuration and numerical criteria '
         'remain byte-identical; the failed qualification is not relabeled as a pass.'), '',
        ('Complete fit time includes construction, random initialization, validation, updates, boundary checkpoints '
         'and durable allocation logs. Stage and controller times are nested and must not be added again to full '
         'fit or native-phase time. All six restricted head/dynamics hashes per fit are charged inside checkpoints. '
         'Timing is descriptive. Equal parameters and updates do not imply equal compute or optimization geometry.'), '',
        (f"TRAIN retains {summary['data_cases']['train']}/512 attempted prefixes and BASE retains "
         f"{summary['data_cases']['base']}/512. All attempts, including found events, contribute to prefix likelihood; "
         'endpoint metrics use surviving prefixes without replacements. TRAIN labels cover H1/H2; H4/H8 are '
         'extrapolation. All fifteen final checkpoints precede BASE generation. The explicit epsilon-0.12 '
         'reference validates targets but never supplies oracle states to learned models.'), '',
        'Pre-DEV integrity, independently checked again from saved boundary bytes:', '',
        '| Seed | Rounded initial/prefix dynamics identical | Prefix Adam identical | All three heads unchanged in prefix stage |',
        '|---|---|---|---|']
    for row in summary['prefix_pair_checks']:
        lines.append(f"| {row['seed']} | {row['same_initial_and_prefix_dynamics']} | {row['same_prefix_optimizer_states']} | {row['all_heads_unchanged_after_prefix']} |")
    lines += ['', '| Model | Seed | Actual transport | Effective head policy | Extra random entries | Extra copied bytes |',
        '|---|---:|---|---|---:|---:|']
    for row in summary['fits']:
        meta, work = row['model_metadata'], row['head_initialization_work']
        lines.append(f"| {row['arm']} | {row['seed']} | {meta['transport_arm']} | {meta['head_initialization']['policy']} | {work['standard_normal_entries']} | {work['parameter_copy_bytes']} |")
    lines += ['', ('The independent audit decodes 69 saved NPZ files, including all 45 model boundaries, plus 45 Adam '
         'JSONs. It reconstructs the declared random initializer once per seed, five times, separately from its zero '
         'model, optimizer and world-generator calls. It rebuilds targets, metrics, schedules and shared forward work '
         'without replaying learning. Publication uses audited JSON and opaque evidence bytes only. These zero-call '
         'statements do not describe the fitting phase.'), '',
        ('This BASE-only diagnostic does not reopen the prior replication, which failed its combined rule with '
         '45/54 conditions passing, or repair its observation-noise-transfer failure. The uniform reset and known '
         'doubly stochastic world structure remain favorable priors. Supervised cost targets and the head family '
         'still provide task information. A local pass is not architectural novelty, latent identification, calibrated '
         'text decisions, biological benefit, native transfer or ICLR readiness. Follow-up work requires a separate '
         'frozen protocol; no further study is executed or admitted here.'), '',
        'All primary conditions:', '', '| Condition | Result |', '|---|---|']
    lines.extend(f"| {name} | {'PASS' if value else 'FAIL'} |" for name, value in audit['advance']['conditions'].items())
    lines += ['', 'Every failed absolute condition:', '']
    failures = [f"- {arm} / {criterion}: {name}" for arm in ARMS for criterion in CRITERIA
        for name, value in audit['gates'][arm][criterion]['conditions'].items() if not value]
    lines.extend(failures or ['None.'])
    lines += ['', 'Every primary paired contrast:', '',
        '| Seed | H | Rounded random regret | Matched free random regret | Candidate minus control |',
        '|---|---:|---:|---:|---:|']
    for row in comparison['paired']:
        value = row['blind_regret']
        lines.append(f"| {row['seed']} | {row['horizon']} | {value['candidate']:.8g} | {value['control']:.8g} | {value['difference']:.8g} |")
    lines += ['', 'Every anchor contrast, descriptive only:', '',
        '| Seed | H | Random regret | Anchor regret | Regret difference | Blind MSE difference | Observed MSE difference | KL difference |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in comparison['anchor_comparisons']:
        value = row['blind_regret']
        differences = ' | '.join(f"{row[key]['difference']:.8g}" for key in ('blind_cost_mse', 'observed_cost_mse', 'observed_kl'))
        lines.append(f"| {row['seed']} | {row['horizon']} | {value['candidate']:.8g} | {value['anchor']:.8g} | {value['difference']:.8g} | {differences} |")
    lines += ['', 'All endpoint results:', '',
        '| Model | Seed | H | Blind regret | Blind cost MSE | Observed cost MSE | Observed KL | Blind survival MAE | Observed survival MAE | Shuffled regret |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    metrics = ('blind_regret', 'blind_cost_mse', 'observed_cost_mse', 'observed_kl',
               'blind_survival_mae', 'observed_survival_mae', 'shuffled_blind_regret')
    for row in sorted(summary['rows'], key=lambda r: (ARMS.index(r['arm']), r['seed'], r['horizon'])):
        lines.append(f"| {row['arm']} | {row['seed']} | {row['horizon']} | " + ' | '.join(f'{row[key]:.8g}' for key in metrics) + ' |')
    lines += ['', 'All-attempt prefix diagnostics:', '', '| Model | Seed | Attempts | Events | Mean event NLL |', '|---|---:|---:|---:|---:|']
    for row in summary['prefix_rows']:
        lines.append(f"| {row['arm']} | {row['seed']} | {row['attempts']} | {row['valid_events']} | {row['mean_nll']:.8g} |")
    lines += ['', ('[Protocol](../finite-head-learning-protocol-v2.md) · [Complete saved summary](summary.json) · '
        '[Current-study evidence](evidence.tar.gz) · [Manifest](manifest.json) · [Publication receipt](receipt.json)'), '',
        ('The archive preserves the complete v2 study, both 137-source snapshots, the engineering exposure '
         'probe, original closures, all predictions and parameter/optimizer boundaries, and these publication '
         'artifacts. It also includes the complete failed v1 qualification folder with its 131-source snapshot, '
         'registration, original logs and closure. Earlier linked studies, interpreter and installed packages remain external. Their original '
         'evidence is reauthenticated and its descriptors remain in the current plans and summary.'), '']
    return '\n'.join(lines)


def archive(output, study_files, failed):
    from finite_head_learning_worker_v2 import publish
    paths = {'study/' + name: (STUDY / name, pin) for name, pin in study_files.items()}
    paths.update({'failed-qualification-v1/' + name: (FAILED_STUDY / name, pin)
                  for name, pin in failed['files'].items()})
    for name in ('summary.json', 'report.md', 'benchmark.png'):
        paths['publication/' + name] = (output / name, descriptor(output / name))
    paths['publication/overview.md'] = (OVERVIEW, descriptor(OVERVIEW))
    paths['publication/publisher.py'] = (Path(__file__).resolve(), descriptor(Path(__file__).resolve()))
    manifest = {'version': VERSION, 'registration': {'path': str(PLAN), **descriptor(PLAN)},
                'files': {name: {'original_path': str(path), **pin} for name, (path, pin) in sorted(paths.items())},
                'failed_qualification': failed,
                'scope': 'Complete current v2 study, original failed v1 qualification and this publication; earlier linked evidence and installed runtime remain external.',
                'self_exclusion': 'Manifest is included in the archive but excludes its own hash.'}
    publish(output / 'manifest.json', manifest)
    paths['publication/manifest.json'] = (output / 'manifest.json', descriptor(output / 'manifest.json'))
    with ((output / 'evidence.tar.gz').open('xb') as raw,
          gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as zipped,
          tarfile.open(fileobj=zipped, mode='w', format=tarfile.PAX_FORMAT) as saved):
        for name, (path, pin) in sorted(paths.items()):
            require(descriptor(path) == pin, 'unchanged archive input')
            data = path.read_bytes()
            require({'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} == pin,
                    'opaque archive bytes match pin')
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            saved.addfile(info, io.BytesIO(data))
    with tarfile.open(output / 'evidence.tar.gz', 'r:gz') as saved:
        require(saved.getnames() == sorted(paths), 'complete exact archive roster')
        for member in saved.getmembers():
            path, pin = paths[member.name]
            require(member.isfile() and saved.extractfile(member).read() == path.read_bytes()
                    and descriptor(path) == pin, 'opaque archive byte roundtrip')
    return len(paths)


def main():
    require(not OUTPUT.exists() and not OVERVIEW.exists(), 'exclusive publication destinations')
    publisher_pin = descriptor(Path(__file__).resolve())
    auth = authenticate()
    from finite_head_learning_worker_v2 import files, publish
    producer, audit, plan = auth['producer'], auth['audit'], auth['plan']
    summary = {'version': VERSION, 'status': audit['advance']['status'], 'technical_complete': True,
        'registration': {'path': str(PLAN), **descriptor(PLAN)}, 'config': plan['config'],
        'rows': audit['rows'], 'baseline_rows': audit['baseline_rows'], 'prefix_rows': audit['prefix_rows'],
        'fits': producer['fits'], 'data_cases': audit['data_cases'], 'audit': audit,
        'producer_counts': producer['counts'], 'dataset_counts': producer['dataset_counts'],
        'structural_work_scope': producer['structural_work_scope'],
        'phase_seconds': {k: v['terminal']['wall_seconds'] for k, v in auth['phases'].items()},
        'generation_seconds': producer['generation_seconds'], 'prediction_times': producer['prediction_times'],
        'prefix_prediction_times': producer['prefix_prediction_times'], 'runtime': plan['runtime'],
        'qualification': auth['qualification'],
        'failed_qualification': auth['failed_qualification'],
        'transport_arms': producer['transport_arms'], 'regimes': producer['regimes'],
        'prefix_pair_checks': producer['prefix_pair_checks'],
        'head_scope': producer['head_scope'], 'oracle_metadata': producer['oracle_metadata'],
        'counts_scope': 'Producer counts describe learning and inference; audit.counts describe independent saved-output reconstruction only.',
        'head_prerequisite': plan['head_prerequisite'],
        'model_selection': False, 'architecture_claim': False, 'statistical_significance_claim': False}
    OUTPUT.mkdir()
    publish(OUTPUT / 'summary.json', summary)
    chart(summary, OUTPUT / 'benchmark.png')
    report = document(summary)
    with (OUTPUT / 'report.md').open('x') as stream:
        stream.write(report)
    overview = report.replace('(benchmark.png)', '(finite-head-learning-results/benchmark.png)')
    overview = overview.replace('(../finite-head-learning-protocol-v2.md)', '(finite-head-learning-protocol-v2.md)')
    for name in ('summary.json', 'evidence.tar.gz', 'manifest.json', 'receipt.json'):
        overview = overview.replace('(' + name + ')', '(finite-head-learning-results/' + name + ')')
    with OVERVIEW.open('x') as stream:
        stream.write(overview)
    members = archive(OUTPUT, auth['study_files'], auth['failed_qualification'])
    after = authenticate()
    require(after['study_files'] == auth['study_files']
            and after['failed_qualification'] == auth['failed_qualification']
            and descriptor(Path(__file__).resolve()) == publisher_pin,
            'all original inputs and publisher unchanged after output')
    publish(OUTPUT / 'receipt.json', {'version': VERSION, 'status': 'PASS',
        'registration': summary['registration'], 'publisher': publisher_pin, 'files': files(OUTPUT),
        'overview': {'path': str(OVERVIEW), **descriptor(OVERVIEW)}, 'archive_members': members,
        'study_files': auth['study_files'], 'sources': plan['sources'],
        'failed_qualification': auth['failed_qualification'],
        'phase_closures': {phase: {'receipt': descriptor(row['receipt_path']),
                                  'terminal': descriptor(row['terminal_path'])}
                           for phase, row in auth['phases'].items()},
        'head_prerequisite': plan['head_prerequisite'],
        'publication_counts': dict.fromkeys(('array_decodes', 'checkpoint_decodes', 'model_calls',
                                            'optimizer_calls', 'generator_calls', 'audit_reruns'), 0),
        'plotting_scope': 'Matplotlib uses saved JSON scalars after original closure; no checkpoint or scientific array is decoded.',
        'result_reads_after_original_closure': True, 'inputs_unchanged': True,
        'model_selection': False, 'visual_review_required': True})
    print(json.dumps({'output': str(OUTPUT), 'status': summary['status'], 'archive_members': members}))


if __name__ == '__main__':
    main()
