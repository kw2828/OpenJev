"""Publish an originally closed reliability pilot from audited JSON scalars.

The pinned study module is imported only for its runtime/source/process
admission functions. No model is constructed or called, no generator or
optimizer runs, and no scientific NPZ is opened. Archives handle opaque bytes.
Publication is exclusive and requires both reviewed source and plan hashes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'output/reliability-memory-v1'
OUTPUT = ROOT / 'research/reliability-memory-results'
OVERVIEW = ROOT / 'research/reliability-memory-results.md'
PARENT = ROOT / 'research/finite-action-range-results'
PLAN_SHA = '6ccfbc7cb0c7204e9a6c922c5c6a1d7cf4ceca799aba98813f3b1f5a0fce446c'
VERSION = 'reliability-memory-publication-v1'
ARMS = ('unchanged', 'global', 'static_bank', 'markov_bank', 'recurrent_bank', 'reset_bank')
TRAINED = ('global', 'markov_bank', 'recurrent_bank', 'reset_bank')
STRATA = ('base', 'shift', 'switch', 'stress')
LABELS = ('Unchanged', 'Global', 'Static\nbank', 'Markov\nbank', 'Recurrent\nbank', 'Reset\nbank')
PARAMETERS = dict(zip(ARMS, (0, 1, 0, 4, 164, 164), strict=True))
STATES = dict(zip(ARMS, (8, 8, 24, 24, 28, 28), strict=True))
MAX_ARCHIVE_BYTES, RAW_PART_BYTES = 95_000_000, 85_000_000


def require(value, message):
    if not value:
        raise ValueError(message)


def descriptor(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve() == path.absolute(),
            'canonical ordinary evidence file: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def relative(name):
    path = PurePosixPath(name)
    require(isinstance(name, str) and str(path) == name and not path.is_absolute()
            and '..' not in path.parts and path.parts, 'portable relative evidence path')
    return name


def inventory(folder):
    return {str(p.relative_to(folder)): descriptor(p)
            for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def close(actual, expected, name):
    require(type(actual) in (int, float) and type(expected) in (int, float)
            and math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), name)


def authenticate(plan_sha):
    """Close every provenance check before opening the current audit result."""
    require(Path.cwd() == ROOT and re.fullmatch('[0-9a-f]{64}', PLAN_SHA)
            and plan_sha == PLAN_SHA, 'explicit frozen publication plan in original checkout')
    require(descriptor(STUDY / 'registration.json')['sha256'] == plan_sha, 'pinned original registration')
    plan = read(STUDY / 'registration.json')
    require(plan['version'] == 'reliability-memory-v1' and plan['output'] == str(STUDY), 'fixed study identity')
    source_files = {}
    for name, pin in plan['sources'].items():
        relative(name)
        require(descriptor(ROOT / name) == descriptor(STUDY / 'sources' / name) == pin,
                'original source and snapshot: ' + name)
        source_files['sources/' + name] = pin
    require(inventory(STUDY / 'sources') == plan['sources'], 'complete exact source snapshot')
    # Imports load the registered runtime, but these calls perform metadata-only
    # source/closure checks. Model, generator and numerical audit APIs are unused.
    from reliability_memory_study import closed, validate
    require(validate(STUDY, plan_sha) == plan, 'registered original runtime and parent provenance')
    phases, files = {}, {'registration.json': descriptor(STUDY / 'registration.json'), **source_files}
    for phase in ('qualify', 'run', 'audit'):
        receipt, terminal = closed(STUDY, phase, plan_sha)
        require(receipt['phase'] == phase and receipt['error'] is None
                and receipt['runtime'] == plan['runtime'], 'original successful phase identity')
        phases[phase] = {'receipt': receipt, 'terminal': terminal}
        for name, pin in receipt['files'].items():
            relative(name)
            require(descriptor(STUDY / phase / name) == pin, 'original phase member hash')
            files[phase + '/' + name] = pin
        for suffix in ('.launch.json', '.terminal.json', '.log'):
            name = phase + '-native-01' + suffix
            files[name] = descriptor(STUDY / name)
        files[phase + '.receipt.json'] = descriptor(STUDY / (phase + '.receipt.json'))
    for before, after in (('qualify', 'run'), ('run', 'audit')):
        left, right = phases[before]['terminal'], phases[after]['terminal']
        require(left['clock_backend'] == right['clock_backend']
                and left['finished_ns'] <= right['started_ns'], 'original chronological process closure')
    require(phases['qualify']['receipt']['result'] == {'fits': 4, 'train_only': True}
            and phases['run']['receipt']['result'] == {'fits': 20, 'rows': 120}, 'complete qualification and scientific run')
    require(len(phases['run']['receipt']['files']) == 299
            and phases['audit']['receipt']['files'].keys() == {'audit.json'}, 'complete registered producer/audit inventories')
    require(inventory(STUDY) == files, 'no omitted or unassigned current study artifacts')
    # The parent is an external, unchanged failed study. Archive descriptors are
    # verified as opaque bytes; its old archives are never copied into this one.
    parent_receipt = PARENT / 'receipt.json'
    require(descriptor(parent_receipt) == plan['parent']['receipt'], 'fixed parent publication receipt')
    parent = read(parent_receipt)
    require(parent['status'] == 'PASS' and plan['parent']['status'] == 'ACTION_RANGE_ADVANCE_FAIL'
            and plan['parent']['conditions_passed'] == 17, 'parent remains scientifically failed17/54')
    require(descriptor(PARENT / 'summary.json') == parent['files']['summary.json'], 'parent summary byte binding')
    external_files = {name: {'path': str(PARENT / name), **pin}
                      for name, pin in parent['files'].items()
                      if name in ('summary.json', 'manifest.json', 'archive-index.json') or name.endswith('.tar.gz')}
    require(any(name.endswith('.tar.gz') for name in external_files), 'complete parent archive references')
    for name, pin in external_files.items():
        relative(name)
        require(descriptor(PARENT / name) == {key: pin[key] for key in ('sha256', 'bytes')}, 'unchanged external parent evidence')
    external = {'receipt': {'path': str(parent_receipt), **descriptor(parent_receipt)},
                'files': external_files, 'selected_family': 'rounded_mse, all five original cohorts',
                'checkpoints': plan['parent']['checkpoints'], 'parent_status': 'ACTION_RANGE_ADVANCE_FAIL',
                'passing_conditions': 17, 'conditions': 54, 'archives_included': False}
    # Only now read current result scalars. Never import or rerun the auditor.
    audit = read(STUDY / 'audit/audit.json')
    verify_saved_result(audit, phases['audit']['receipt']['result'])
    completion = re.findall(r'(?m)^\s*(\d+) passed(?:, (\d+) warnings?)? in [^\n]+$',
                            (STUDY / 'qualify/command-1.log').read_text())
    require(len(completion) == 1 and int(completion[0][0]) > 0, 'one complete original passing pytest command')
    tests = {'passed': int(completion[0][0]), 'warnings': int(completion[0][1] or 0)}
    feasibility = read(STUDY / 'qualify/feasibility.json')
    require(len(feasibility) == 4 and [r['arm'] for r in feasibility] == list(TRAINED), 'complete original feasibility probe')
    return {'plan': plan, 'phases': phases, 'study_files': files, 'external': external,
            'audit': audit, 'tests': tests, 'feasibility': feasibility}


def verify_saved_result(audit, receipt_result):
    """Check roster and scalar joins, not raw targets, inference or gate replay."""
    require(audit['version'] == 'reliability-memory-audit-v1' and audit['agreement'] is True, 'independent original audit agreement')
    counts = {'npz_decodes': 225, 'model_state_decodes': 50, 'optimizer_state_decodes': 20,
              'target_reconstructions': 25, 'prediction_checks': 120, 'update_records': 5120,
              'model_calls': 0, 'optimizer_calls': 0, 'generator_calls': 0, 'rng_replays': 0}
    require(audit['checkcounts'] == counts, 'complete independent audit check roster')
    rows = audit['rows']
    expected = {(cohort, arm, split) for cohort in range(5) for arm in ARMS for split in STRATA}
    require(len(rows) == 120 and {(r['cohort'], r['arm'], r['split']) for r in rows} == expected,
            'every declared cohort arm and stratum retained')
    for row in rows:
        require(row['parameters'] == PARAMETERS[row['arm']] and row['state_scalars'] == STATES[row['arm']], 'model capacity disclosure')
        for name in ('primary_regret', 'h4_regret', 'h8_regret', 'post_regret', 'event_log_loss', 'inference_seconds'):
            require(type(row[name]) in (int, float) and math.isfinite(row[name]) and row[name] >= 0, 'finite nonnegative reported scalar/' + name)
        require(row['late_episode_support'] >= 256
                and row['late_episode_support'] + row['late_unsupported'] == 512, 'all late support cases disclosed')
    fits = audit['fits']
    require(len(fits) == 20 and {(r['cohort'], r['arm']) for r in fits}
            == {(c, a) for c in range(5) for a in TRAINED}, 'all20 trainable fits')
    for fit in fits:
        require(fit['seed'] == 438261001 + fit['cohort'] and fit['updates'] == 256
                and fit['parameters'] == PARAMETERS[fit['arm']] and fit['state_scalars'] == STATES[fit['arm']]
                and 0 < fit['seconds'] < 180, 'complete fixed training recipe')
    result = audit['result']
    conditions = {f'{s}/{c}/{criterion}' for c in ('markov_bank', 'reset_bank')
                  for s in ('shift', 'switch') for criterion in ('mean_gain10pct', 'paired_wins4of5')}
    conditions.update(('base/markov_bank/noninferiority', 'base/reset_bank/noninferiority', 'base/unchanged/noninferiority',
                       'all_strata/inference_time2x', 'base/event_nll_plus001'))
    require(set(result['conditions']) == conditions and len(conditions) == 13
            and all(type(v) is bool for v in result['conditions'].values())
            and audit['conditions'] == result['conditions'], 'unchanged complete13-condition rule')
    require(type(result['passed']) is bool and result['passed'] == all(result['conditions'].values())
            and result['status'] == ('PASS' if result['passed'] else 'FAIL'), 'scientific status and all-condition conjunction')
    require(receipt_result == {'agreement': True, 'status': result['status'],
            'conditions_passed': sum(result['conditions'].values()), 'conditions_total': 13, 'checkcounts': counts}, 'audit receipt scalar join')
    require(len(result['means']) == 24 and {(r['arm'], r['split']) for r in result['means']}
            == {(a, s) for a in ARMS for s in STRATA}, 'all saved equal-cohort means')
    means = {(r['arm'], r['split']): r['primary_regret'] for r in result['means']}
    for identity, mean in means.items():
        close(mean, math.fsum(r['primary_regret'] for r in rows if (r['arm'], r['split']) == identity) / 5,
              'saved mean joins all five reported cohort scalars')
    require(len(result['comparisons']) == 4 and {(r['control'], r['split']) for r in result['comparisons']}
            == {(c, s) for c in ('markov_bank', 'reset_bank') for s in ('shift', 'switch')}, 'all primary control comparisons')
    for row in result['comparisons']:
        close(row['candidate_mean'], means['recurrent_bank', row['split']], 'candidate mean join')
        close(row['control_mean'], means[row['control'], row['split']], 'control mean join')
        require(type(row['paired_wins']) is int and 0 <= row['paired_wins'] <= 5, 'recorded paired-win count')


def chart(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rows = summary['audit']['rows']
    lookup = {(r['cohort'], r['arm'], r['split']): r for r in rows}
    colors = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#6F5AA5')
    fig, axes = plt.subplots(3, 2, figsize=(14, 14), layout='constrained')
    for ax, split in zip(axes.flat, STRATA, strict=False):
        for index, arm in enumerate(ARMS):
            values = [lookup[c, arm, split]['primary_regret'] for c in range(5)]
            for cohort, value in enumerate(values):
                ax.scatter(index + (cohort - 2) * .07, value, color=colors[cohort], s=35, zorder=3)
            ax.scatter(index, math.fsum(values) / 5, color='black', marker='D', s=45, zorder=4)
        ax.set_title(split.upper() + (' (descriptive stress test)' if split == 'stress' else ''))
        ax.set_xticks(range(6), LABELS)
        ax.set_ylabel('Primary H4/H8 regret; lower is better')
        ax.set_ylim(bottom=0)
        ax.margins(y=.14)
        ax.grid(axis='y', alpha=.22)
    ax = axes[2, 0]
    pairs = [(s, a) for s in ('shift', 'switch') for a in ('markov_bank', 'reset_bank')]
    for i, (split, control) in enumerate(pairs):
        values = [lookup[c, 'recurrent_bank', split]['primary_regret'] - lookup[c, control, split]['primary_regret'] for c in range(5)]
        for c, value in enumerate(values):
            ax.scatter(i + (c - 2) * .07, value, color=colors[c], s=35)
        ax.scatter(i, math.fsum(values) / 5, color='black', marker='D', s=45)
    ax.axhline(0, color='#555555', linewidth=1)
    ax.set_xticks(range(4), [s.upper() + '\nvs ' + a.replace('_bank', '') for s, a in pairs])
    ax.set_ylabel('Recurrent minus control regret')
    ax.set_title('All paired primary comparisons; negative favors recurrent')
    ax.grid(axis='y', alpha=.22)
    ax = axes[2, 1]
    for i, arm in enumerate(ARMS):
        values = []
        for c in range(5):
            for j, split in enumerate(STRATA):
                value = lookup[c, arm, split]['inference_seconds']
                values.append(value)
                ax.scatter(i + (c - 2) * .06 + (j - 1.5) * .012, value, color=colors[c], s=19, alpha=.7)
        ax.scatter(i, math.fsum(values) / 20, color='black', marker='D', s=45)
    ax.set_xticks(range(6), LABELS)
    ax.set_ylabel('Seconds per 512-episode inference call')
    ax.set_title('All 120 inference calls, including blind forks')
    ax.set_ylim(bottom=0)
    ax.margins(y=.14)
    ax.grid(axis='y', alpha=.22)
    handles = [Line2D([], [], marker='o', color=color, linestyle='None', label=f'Cohort {i+1}') for i, color in enumerate(colors)]
    handles.append(Line2D([], [], marker='D', color='black', linestyle='None', label='Equal-cohort mean'))
    fig.legend(handles=handles, loc='outside lower center', ncol=6, frameon=False)
    result = summary['audit']['result']
    fig.suptitle(f"Reliability memory: {result['status']} ({sum(result['conditions'].values())}/13 conditions)\n"
                 'Privileged private-noise-path risk reference; linear axes; all cohorts retained', fontsize=15)
    fig.savefig(path, dpi=170, metadata={'Software': VERSION})
    plt.close(fig)


def fmt(value):
    return f'{value:.6g}'


def document(summary):
    audit, phases = summary['audit'], summary['phase_seconds']
    result = audit['result']
    means = {(r['arm'], r['split']): r['primary_regret'] for r in result['means']}
    text = [
        '# Causal observation-reliability memory', '',
        f"**{result['status']}: {sum(result['conditions'].values())}/13 prospective continuation conditions passed.** "
        + ('This pilot met the fixed rule; independent validation is still required.' if result['passed'] else
           'This recipe did not meet the fixed continuation rule. No control, cohort or threshold was changed after evaluation.'), '',
        ('The candidate adds a four-unit recurrent reliability gate to a three-mode Bayesian filter. '
        'The primary controls are a learned constant-switching Markov filter and a matched stored-size gate whose GRU memory is reset at each event. '
        'The Markov control performs exact filtering for its assumed mode-transition law.'), '',
        '![All cohorts, strata and arms](benchmark.png)', '',
        ('Primary regret averages H4/H8 blind-fork regret within each episode over alive boundaries at event index 8 or later, '
        'then over supported episodes and equally over five cohorts. Lower is better. '
        'The reference knows the private realized noise path: this is conditional expected-cost risk, not an attainable public-history Bayes floor or a calibration guarantee.'), '',
        '| Stratum | Unchanged | Global | Static bank | Markov bank | Recurrent bank | Reset bank |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for split in STRATA:
        text.append('| ' + split.upper() + ' | ' + ' | '.join(fmt(means[a, split]) for a in ARMS) + ' |')
    text += ['', '| Primary comparison | Recurrent mean | Control mean | Relative reduction | Paired wins |',
             '|---|---:|---:|---:|---:|']
    for row in result['comparisons']:
        gain = f"{100*(1-row['candidate_mean']/row['control_mean']):.2f}%" if row['control_mean'] > 0 else 'undefined (zero control)'
        text.append(f"| {row['split'].upper()} vs {row['control']} | {fmt(row['candidate_mean'])} | {fmt(row['control_mean'])} | {gain} | {row['paired_wins']}/5 |")
    text += ['', (f"Mean inference calls: recurrent **{fmt(result['candidate_inference_seconds'])}s**, "
             f"Markov **{fmt(result['markov_inference_seconds'])}s**. BASE event NLL: recurrent "
             f"**{fmt(result['candidate_base_event_nll'])}**, Markov **{fmt(result['markov_base_event_nll'])}** nats."), '',
             ('Every original condition is retained below. SHIFT and SWITCH each require a 10% equal-cohort mean reduction and strict wins in at least 4/5 cohorts against both primary controls. '
             'BASE noninferiority against both primary controls and the unchanged backbone, inference time and BASE event NLL are additional requirements; STRESS is descriptive.'), '',
             '| Fixed condition | Outcome |', '|---|---|']
    text += [f"| `{name}` | {'PASS' if passed else 'FAIL'} |" for name, passed in result['conditions'].items()]
    text += ['', ('All arms use the same five frozen `rounded_mse` backbones, one from every original parent cohort. '
             'Each backbone stores 352 learned parameters. The supplied emission family and bank are structural prior knowledge, not inferred task structure. '
             'No active noise level, switch index or privileged target enters a learner. Precommitted fork actions are public controls.'), '',
             '| Arm | Added trainable parameters | Persistent state scalars |', '|---|---:|---:|']
    text += [f'| {arm} | {PARAMETERS[arm]} | {STATES[arm]} |' for arm in ARMS]
    text += ['', ('The reset control stores 164 parameters, but 48 recurrent matrix entries always multiply zero. '
             'Stored size is matched; effective capacity is not. State counts exclude flags and temporary work.'), '',
             ('Four arms were fitted per cohort, for 20 fits total. Every fit used 256 Adam updates of 64 public episodes, '
             'learning rate 0.01 and gradient clipping 5, with CPU float64 and one numerical thread. '
             'The only training loss was actual observed-event NLL, including first found and excluding absorbing padding. '
             'All 30 final model states preceded evaluation generation. Each cohort had 512 TRAIN episodes and 512 fresh episodes per evaluation stratum, '
             'with 32 actions plus reset; all first-found episodes were retained.'), '',
             (f"Original native wall time: qualification **{fmt(phases['qualify']['native_seconds'])}s**, scientific run "
             f"**{fmt(phases['run']['native_seconds'])}s**, audit **{fmt(phases['audit']['native_seconds'])}s**. "
             f"Original qualification passed **{summary['qualification']['passed']} tests** "
             f"with **{summary['qualification']['warnings']} warnings**."), '',
             ('Fit times include initial/final state and final Adam persistence, update logging and final checks. '
             'They exclude outer model construction and the final fits journal. Inference times include the episode filter and all blind forks, '
             'but exclude array conversion, prediction file writing and metric calculation. Native phase times include those surrounding costs. '
             'These are single-machine timings, not matched FLOPs or independently retimed audit measurements.'), '',
             '<details><summary>Every trained fit and complete measured fit time</summary>', '',
             '| Cohort | Arm | Seed | Updates | Seconds |', '|---:|---|---:|---:|---:|']
    text += [f"| {r['cohort']+1} | {r['arm']} | {r['seed']} | {r['updates']} | {fmt(r['seconds'])} |" for r in audit['fits']]
    text += ['', '</details>', '', '<details><summary>All 120 audited cohort/arm/stratum rows</summary>', '',
             '| Cohort | Arm | Stratum | Primary | H4 | H8 | Immediate | Event NLL | Supported / 512 | Late boundaries | Inference s |',
             '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in audit['rows']:
        text.append(f"| {row['cohort']+1} | {row['arm']} | {row['split']} | "
                    + ' | '.join(fmt(row[k]) for k in ('primary_regret', 'h4_regret', 'h8_regret', 'post_regret', 'event_log_loss'))
                    + f" | {row['late_episode_support']} | {row['late_boundaries']} | {fmt(row['inference_seconds'])} |")
    text += ['', '</details>', '',
             ('The saved summary also retains support exclusions and every descriptive SWITCH-delay row. Early-found episodes remain in event NLL; '
             'episodes without an alive boundary at index 8 or later do not enter the primary regret population. '
             'The independent audit checked 225 NPZ loads, 50 model-state loads and 20 optimizer-state loads, 5120 update records, '
             '25 independently reconstructed target populations and 120 prediction checks. Model-state and optimizer-state loads are subsets of 225, not additional loads. '
             'It did not replay training, model likelihoods or RNG; process chronology and execution remain authenticated source/receipt evidence.'), '',
             ('The five backbones come from the prior action-range study, whose **FAIL 17/54** remains unchanged. '
             'Histories use random public actions: this tests estimation and counterfactual decision scoring, not autonomous gameplay or a learned control policy. '
             'This pilot is not a significance test, proof of architectural novelty, equal-capacity comparison or evidence of native/robotics transfer. '
             'A failed continuation is not rescued by a favorable secondary metric or descriptive stratum.'), '',
             ('[Protocol](../reliability-memory-protocol.md) · [Full saved summary](summary.json) · '
             '[Archive index](archive-index.json) · [Member manifest](manifest.json) · [Publication receipt](receipt.json)'), '',
             ('Complete original current-study snapshots, registration, qualification probe, datasets, trained states, Adam states, update logs, '
             'predictions and native closures are archived as opaque bytes. Parent archives remain external SHA-256 references; no historical evidence is discarded. '
             'The publication helper performs no scientific array decoding, model execution, optimization, generation or audit replay.'), '']
    return '\n'.join(text)


def archives(auth):
    paths = {'study/' + name: (STUDY / name, pin) for name, pin in auth['study_files'].items()}
    for name in ('summary.json', 'report.md', 'benchmark.png'):
        paths['publication/' + name] = (OUTPUT / name, descriptor(OUTPUT / name))
    paths.update({'publication/overview.md': (OVERVIEW, descriptor(OVERVIEW)),
                  'publication/publisher.py': (Path(__file__).resolve(), descriptor(Path(__file__).resolve()))})
    manifest = {'version': VERSION, 'files': {name: {'original_path': str(path), **pin}
                for name, (path, pin) in sorted(paths.items())}, 'external_parent': auth['external'],
                'self_exclusion': 'Manifest excludes itself but is archived; archive index and final receipt remain outside archives.',
                'scope': 'Every current registration, source snapshot, closed phase and original opaque payload; no external parent archive duplication.'}
    write(OUTPUT / 'manifest.json', manifest)
    paths['publication/manifest.json'] = (OUTPUT / 'manifest.json', descriptor(OUTPUT / 'manifest.json'))
    groups = {'shared': []}
    for name in sorted(paths):
        match = re.match(r'study/run/(cohort-\d\d)/', name)
        groups.setdefault(match[1] if match else 'shared', []).append(name)
    parts = []
    for group, names in sorted(groups.items()):
        batches, current, size = [], [], 0
        for name in names:
            estimate = paths[name][1]['bytes'] + 2048
            require(estimate <= RAW_PART_BYTES, 'one whole member exceeds supported archive part size')
            if current and size + estimate > RAW_PART_BYTES:
                batches.append(current); current, size = [], 0
            current.append(name); size += estimate
        if current:
            batches.append(current)
        for index, members in enumerate(batches):
            filename = 'evidence-' + group + (f'-part-{index+1:02d}' if len(batches) > 1 else '') + '.tar.gz'
            destination = OUTPUT / filename
            with destination.open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as zipped, tarfile.open(fileobj=zipped, mode='w', format=tarfile.PAX_FORMAT) as archive:
                for name in members:
                    source, pin = paths[name]
                    require(descriptor(source) == pin, 'unchanged opaque evidence before archiving')
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = pin['bytes'], 0o644, 0
                    with source.open('rb') as stream:
                        archive.addfile(info, stream)
            pin = descriptor(destination)
            require(pin['bytes'] < MAX_ARCHIVE_BYTES, 'archive part below95MB')
            with tarfile.open(destination, 'r:gz') as archive:
                require(archive.getnames() == members, 'exact portable member inventory')
                for member in archive.getmembers():
                    require(member.isfile(), 'ordinary archive member')
                    digest, size = hashlib.sha256(), 0
                    with archive.extractfile(member) as stream:
                        for block in iter(lambda: stream.read(1048576), b''):
                            digest.update(block); size += len(block)
                    require({'sha256': digest.hexdigest(), 'bytes': size} == paths[member.name][1]
                            and descriptor(paths[member.name][0]) == paths[member.name][1], 'opaque byte-exact archive roundtrip')
            parts.append({'path': filename, **pin, 'members': members, 'member_count': len(members)})
    flattened = [name for part in parts for name in part['members']]
    require(len(flattened) == len(set(flattened)) == len(paths) and set(flattened) == set(paths), 'every evidence member exactly once')
    index = {'version': VERSION, 'archives': parts, 'members': {name: part['path'] for part in parts for name in part['members']},
             'member_count': len(paths), 'manifest': descriptor(OUTPUT / 'manifest.json'),
             'opaque_hash_roundtrip': True, 'maximum_archive_bytes': MAX_ARCHIVE_BYTES}
    write(OUTPUT / 'archive-index.json', index)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--publisher-sha256', required=True)
    args = parser.parse_args()
    require(not OUTPUT.exists() and not OVERVIEW.exists(), 'exclusive publication without replacement')
    publisher = descriptor(Path(__file__).resolve())
    require(publisher['sha256'] == args.publisher_sha256, 'explicitly reviewed publisher source')
    auth = authenticate(args.plan_sha256)
    summary = {'version': VERSION, 'status': auth['audit']['result']['status'], 'technical_complete': True,
               'registration': {'path': str(STUDY / 'registration.json'), **descriptor(STUDY / 'registration.json')},
               'audit': auth['audit'], 'config': auth['plan']['config'], 'runtime': auth['plan']['runtime'],
               'phase_seconds': {phase: {'native_seconds': row['terminal']['wall_seconds'],
                   'worker_seconds': row['receipt']['seconds'], 'cap_seconds': row['terminal']['cap_seconds']}
                   for phase, row in auth['phases'].items()}, 'qualification': auth['tests'],
               'feasibility_probe': auth['feasibility'], 'external_parent': auth['external'],
               'metric_source': 'Original independently audited saved scalars; no raw prediction or model replay.',
               'architecture_novelty_claim': False, 'effective_capacity_match_claim': False,
               'statistical_significance_claim': False, 'attainable_public_bayes_floor_claim': False,
               'private_target_calibration_claim': False, 'prior_failure_rescued': False}
    OUTPUT.mkdir()
    write(OUTPUT / 'summary.json', summary)
    chart(summary, OUTPUT / 'benchmark.png')
    report = document(summary)
    with (OUTPUT / 'report.md').open('x') as stream:
        stream.write(report)
    overview = report.replace('(benchmark.png)', '(reliability-memory-results/benchmark.png)')
    overview = overview.replace('(../reliability-memory-protocol.md)', '(reliability-memory-protocol.md)')
    for name in ('summary.json', 'archive-index.json', 'manifest.json', 'receipt.json'):
        overview = overview.replace('(' + name + ')', '(reliability-memory-results/' + name + ')')
    with OVERVIEW.open('x') as stream:
        stream.write(overview)
    index = archives(auth)
    after = authenticate(args.plan_sha256)
    require(after == auth and descriptor(Path(__file__).resolve()) == publisher, 'all authenticated inputs unchanged after publication')
    receipt = {'version': VERSION, 'status': 'PASS', 'scientific_status': summary['status'],
               'registration': summary['registration'], 'publisher': publisher,
               'overview': {'path': str(OVERVIEW), **descriptor(OVERVIEW)}, 'files': inventory(OUTPUT),
               'sources': auth['plan']['sources'], 'study_files': auth['study_files'], 'external_parent': auth['external'],
               'original_closures': {phase: {name: descriptor(STUDY / filename) for name, filename in
                   (('receipt', phase + '.receipt.json'), ('launch', phase + '-native-01.launch.json'),
                    ('terminal', phase + '-native-01.terminal.json'))} for phase in auth['phases']},
               'archive_parts': len(index['archives']), 'archive_members': index['member_count'],
               'publication_counts': dict.fromkeys(('scientific_npz_decodes', 'checkpoint_decodes', 'model_calls',
                   'generator_calls', 'optimizer_calls', 'audit_replays'), 0),
               'all_inputs_unchanged': True, 'opaque_hash_roundtrip': True, 'visual_review_required': True}
    write(OUTPUT / 'receipt.json', receipt)
    print(json.dumps({'output': str(OUTPUT), 'scientific_status': summary['status'],
                     'conditions_passed': sum(auth['audit']['conditions'].values()), 'conditions_total': 13,
                     'archive_parts': len(index['archives']), 'archive_members': index['member_count']}))


if __name__ == '__main__':
    main()
