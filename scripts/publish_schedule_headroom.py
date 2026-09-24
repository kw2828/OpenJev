"""Publish a closed schedule diagnostic from original audited JSON only.

Only registered source/runtime/native-closure helpers are called. No scientific
NPZ, model, generator, optimizer, or numerical audit execution is permitted.
All current evidence is archived as opaque bytes; parent archives stay external.
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
STUDY = ROOT / 'output/schedule-headroom-v1'
OUTPUT = ROOT / 'research/schedule-headroom-results'
OVERVIEW = ROOT / 'research/schedule-headroom-results.md'
PARENT = ROOT / 'research/reliability-memory-results'
PLAN_SHA = '41ed08db3af8299512fed045400a4869d5eb7a96d15a243993aa267acdbe758e'
VERSION = 'schedule-headroom-publication-v1'
OLD_ARMS = ('unchanged', 'global', 'static_bank', 'markov_bank', 'recurrent_bank', 'reset_bank')
ARMS = OLD_ARMS + ('learned_exact', 'true_exact', 'learned_static2', 'true_static2')
CONTROLS = OLD_ARMS + ('learned_static2',)
CANDIDATES = ('learned_exact', 'true_exact')
LABELS = ('Original', 'Global', 'Static\n3-mode', 'Markov\n3-mode', 'Recurrent\n3-mode', 'Reset\n3-mode',
          'Learned\nexact 36', 'True\nexact 36', 'Learned\nstatic 2', 'True\nstatic 2')
FAMILIES = ('Static 0.12', 'Static 0.30', 'Switch 0.12 to 0.30', 'Switch 0.30 to 0.12')
PARAMETERS = dict(zip(ARMS, (0, 1, 0, 4, 164, 164, 0, 0, 0, 0), strict=True))
STATES = dict(zip(ARMS, (8, 8, 24, 24, 28, 28, 288, 288, 16, 16), strict=True))
MAX_ARCHIVE_BYTES, RAW_PART_BYTES = 45_000_000, 40_000_000
CLASSIFICATIONS = {
    'LEARNED_MODEL_HEADROOM': 'The learned-field schedule reference meets the registered screen. A separate approximation study could be considered with matched prior and training access.',
    'MODEL_MISMATCH_HEADROOM': 'Only the true-field reference establishes the required headroom and schedule-specific benefit. Investigate learned-model mismatch before another reliability GRU.',
    'TRUE_MODEL_ADVANTAGE_WITHOUT_SCHEDULE_MEMORY': 'Supplied world laws help, but the required benefit over true static-two filtering is not established.',
    'NO_REGISTERED_HEADROOM': 'Neither reference meets every frozen requirement. This label records a failed screen; it does not mean that descriptive decision improvements are absent. Overall gains cannot override the BASE-family guard or authorize an architecture advance.',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def descriptor(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve() == path.absolute(), 'canonical ordinary file')
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
    return {str(p.relative_to(folder)): descriptor(p) for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def close(left, right, name):
    require(type(left) in (int, float) and type(right) in (int, float)
            and math.isfinite(left) and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12), name)


def authenticate(plan_sha):
    require(Path.cwd() == ROOT and re.fullmatch('[0-9a-f]{64}', PLAN_SHA)
            and plan_sha == PLAN_SHA, 'reviewed fixed plan in original checkout')
    require(descriptor(STUDY / 'registration.json')['sha256'] == plan_sha, 'original frozen registration')
    plan = read(STUDY / 'registration.json')
    require(plan['version'] == 'schedule-headroom-v1' and plan['output'] == str(STUDY)
            and plan['arms'] == list(ARMS), 'fixed study and full arm roster')
    require({'scripts/schedule_headroom_study.py', 'scripts/audit_schedule_headroom.py',
             'research/schedule-headroom-protocol.md'} <= set(plan['sources']), 'required registered admission sources')
    files = {'registration.json': descriptor(STUDY / 'registration.json')}
    for name, pin in plan['sources'].items():
        relative(name)
        require(descriptor(ROOT / name) == descriptor(STUDY / 'sources' / name) == pin, 'unchanged source and snapshot')
        files['sources/' + name] = pin
    require(inventory(STUDY / 'sources') == plan['sources'], 'complete exact source snapshot')
    # This imports the pinned NumPy/Torch runtime for metadata identity only;
    # no construct(), predict(), generator, or scientific audit call is made.
    from schedule_headroom_study import closed, validate
    require(validate(STUDY, plan_sha) == plan, 'original registered runtime and parent evidence')
    phases = {}
    for phase in ('qualify', 'run', 'audit'):
        receipt, terminal = closed(STUDY, phase, plan_sha)
        require(receipt['phase'] == phase and receipt['error'] is None
                and receipt['runtime'] == plan['runtime'], 'original successful phase identity')
        phases[phase] = {'receipt': receipt, 'terminal': terminal}
        for name, pin in receipt['files'].items():
            relative(name)
            require(descriptor(STUDY / phase / name) == pin, 'original closed payload bytes')
            files[phase + '/' + name] = pin
        for name in (phase + '.receipt.json', *(phase + '-native-01' + suffix for suffix in ('.launch.json', '.terminal.json', '.log'))):
            files[name] = descriptor(STUDY / name)
    for before, after in (('qualify', 'run'), ('run', 'audit')):
        left, right = phases[before]['terminal'], phases[after]['terminal']
        require(left['clock_backend'] == right['clock_backend'] and left['finished_ns'] <= right['started_ns'], 'original process chronology')
    require(phases['qualify']['receipt']['result'] == {'probe_episodes': 64, 'models': 10, 'effectiveness_scored': False}
            and phases['run']['receipt']['result'] == {'rows': 50, 'models': 50, 'training_calls': 0}, 'complete original probe and diagnostic')
    require(len(phases['run']['receipt']['files']) == 98 and phases['audit']['receipt']['files'].keys() == {'audit.json'}, 'complete 98-file producer and audit')
    require(inventory(STUDY) == files, 'every current artifact assigned to original snapshot or closed phase')
    parent_path = PARENT / 'receipt.json'
    require(descriptor(parent_path) == plan['parent']['publication'], 'pinned parent publication including layout repair')
    parent = read(parent_path)
    require(parent['status'] == 'PASS' and parent['scientific_status'] == 'FAIL'
            and plan['parent']['scientific_status'] == 'FAIL' and plan['parent']['conditions_passed'] == 8, 'parent remains FAIL 8/13')
    external_files = {name: {'path': str(PARENT / name), **pin} for name, pin in parent['files'].items()
                      if name in ('summary.json', 'manifest.json', 'archive-index.json') or name.endswith('.tar.gz')}
    require(any(name.endswith('.tar.gz') for name in external_files), 'all external parent archive references')
    for name, pin in external_files.items():
        relative(name)
        require(descriptor(PARENT / name) == {k: pin[k] for k in ('sha256', 'bytes')}, 'external parent bytes unchanged')
    external = {'receipt': {'path': str(parent_path), **descriptor(parent_path)}, 'files': external_files,
                'parent_plan_sha256': plan['parent']['plan_sha256'], 'parent_input_files': plan['parent']['inputs'],
                'scientific_status': 'FAIL', 'passing_conditions': 8, 'conditions': 13, 'archives_included': False,
                'scope': 'All five original backbones and thirty reliability states reused without selection. Original parent archives and earlier lineage remain external hash references.'}
    # Current outcome JSON is opened only after all original closures pass.
    audit = read(STUDY / 'audit/audit.json')
    verify_saved_result(audit, phases['audit']['receipt']['result'], plan['config'])
    completion = re.findall(r'(?m)^\s*(\d+) passed(?:, (\d+) warnings?)? in [^\n]+$',
                            (STUDY / 'qualify/command-1.log').read_text())
    require(len(completion) == 1 and int(completion[0][0]) > 0, 'one original complete passing test command')
    feasibility = read(STUDY / 'qualify/feasibility.json')
    require(feasibility['episodes'] == 64 and feasibility['effectiveness_scored'] is False
            and set(feasibility['inference_seconds']) == set(ARMS)
            and 0 < feasibility['conservative_projected_seconds'] < 450, 'original fixed ten-arm feasibility screen')
    return {'plan': plan, 'phases': phases, 'study_files': files, 'external': external, 'audit': audit,
            'tests': {'passed': int(completion[0][0]), 'warnings': int(completion[0][1] or 0)}, 'feasibility': feasibility}


def verify_saved_result(audit, receipt, config):
    require(audit['version'] == 'schedule-headroom-audit-v1' and audit['agreement'] is True
            and audit['config'] == config, 'independent closed diagnostic agreement')
    counts = {'npz_decodes': 90, 'model_state_decodes': 30, 'backbone_decodes': 5, 'data_decodes': 5,
              'prediction_decodes': 50, 'private_target_reconstructions': 5, 'exact_public_reconstructions': 20,
              'prediction_checks': 50, 'copied_files_checked': 35, 'model_calls': 0, 'generator_calls': 0,
              'optimizer_calls': 0, 'rng_replays': 0}
    require(audit['checkcounts'] == counts, 'complete independent computation/decode disclosure')
    rows = audit['rows']
    require(len(rows) == 50 and {(r['cohort'], r['arm']) for r in rows} == {(c, a) for c in range(5) for a in ARMS}, 'all 50 diagnostic rows')
    for row in rows:
        require(row['source_info']['added_stored_parameters'] == PARAMETERS[row['arm']]
                and row['state_scalars'] == STATES[row['arm']], 'capacity metadata')
        for key in ('primary_regret', 'h4_regret', 'h8_regret', 'post_regret', 'event_nll', 'inference_seconds'):
            require(type(row[key]) in (int, float) and math.isfinite(row[key]) and row[key] >= 0, 'finite nonnegative saved metric')
        require(len(row['per_episode_regret']) == 512 and all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                for v in row['per_episode_regret']), 'all 512 episode risks retained')
        require([r['family'] for r in row['family_regret']] == list(range(4))
                and sum(r['episodes'] for r in row['family_regret']) == 512
                and all(r['episodes'] >= 64 and math.isfinite(r['primary_regret']) and r['primary_regret'] >= 0
                        for r in row['family_regret']), 'all four family means/support')
    result = audit['result']
    require(set(result['candidates']) == set(CANDIDATES) and set(result['means']) == set(ARMS), 'all candidate groups and arm means')
    conditions = {f'{a}/{kind}' for a in CONTROLS for kind in ('mean_gain10pct', 'paired_wins4of5')}
    conditions.add('base/unchanged/noninferiority')
    for candidate in CANDIDATES:
        group = result['candidates'][candidate]
        require(set(group['conditions']) == conditions and len(conditions) == 15
                and all(type(v) is bool for v in group['conditions'].values())
                and group['passed'] is all(group['conditions'].values()), 'exact 15-condition candidate conjunction')
        for arm, key in ((candidate, 'candidate_base'), ('unchanged', 'unchanged_base')):
            close(group[key], math.fsum(row['family_regret'][0]['primary_regret'] for row in rows if row['arm'] == arm)/5, 'saved BASE family mean join')
        require(len(group['comparisons']) == 7 and [r['control'] for r in group['comparisons']] == list(CONTROLS), 'all seven original comparisons')
        for row in group['comparisons']:
            close(row['candidate_mean'], result['means'][candidate], 'saved candidate mean join')
            close(row['control_mean'], result['means'][row['control']], 'saved control mean join')
            require(type(row['paired_wins']) is int and 0 <= row['paired_wins'] <= 5, 'saved paired win count')
    history = result['true_history_conditions']
    require(set(history) == {'mean_gain10pct', 'paired_wins4of5'} and all(type(v) is bool for v in history.values()), 'two separate history conditions')
    learned, true = (result['candidates'][a]['passed'] for a in CANDIDATES)
    expected = ('LEARNED_MODEL_HEADROOM' if learned else 'MODEL_MISMATCH_HEADROOM' if true and all(history.values())
                else 'TRUE_MODEL_ADVANTAGE_WITHOUT_SCHEDULE_MEMORY' if true else 'NO_REGISTERED_HEADROOM')
    require(result['classification'] == expected, 'literal prospective ordered classification, not a new all 32 conjunction')
    require(receipt == {'agreement': True, 'classification': expected, 'rows': 50,
        'learned_conditions_passed': sum(result['candidates']['learned_exact']['conditions'].values()),
        'true_conditions_passed': sum(result['candidates']['true_exact']['conditions'].values()),
        'true_history_conditions_passed': sum(history.values()), 'checkcounts': counts}, 'original audit receipt joins')
    for arm, mean in result['means'].items():
        close(mean, math.fsum(r['primary_regret'] for r in rows if r['arm'] == arm)/5, 'saved mean joins every cohort scalar')


def chart(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rows = {(r['cohort'], r['arm']): r for r in summary['audit']['rows']}
    colors = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#6F5AA5')
    fig, axes = plt.subplots(3, 2, figsize=(16, 14), layout='constrained')
    for panel, ax in enumerate(axes.flat):
        for index, arm in enumerate(ARMS):
            values = [rows[c, arm]['family_regret'][panel]['primary_regret'] if panel < 4
                      else rows[c, arm]['primary_regret' if panel == 4 else 'inference_seconds'] for c in range(5)]
            for cohort, value in enumerate(values):
                ax.scatter(index+(cohort-2)*.07, value, color=colors[cohort], s=26, zorder=3)
            ax.scatter(index, math.fsum(values)/5, color='black', marker='D', s=32, zorder=4)
        ax.set_xticks(range(10), LABELS, fontsize=8)
        ax.set_ylim(bottom=0); ax.margins(y=.15); ax.grid(axis='y', alpha=.22)
        ax.set_title(FAMILIES[panel] if panel < 4 else 'Primary mixture endpoint: all 50 cohort/arm points' if panel == 4 else 'Complete inference: all 50 calls')
        ax.set_ylabel('Fixed-denominator H4/H8 regret' if panel < 5 else 'Seconds per 512-episode call, including blind forks')
    handles = [Line2D([], [], marker='o', color=color, linestyle='None', label=f'Cohort {i+1}') for i, color in enumerate(colors)]
    handles.append(Line2D([], [], marker='D', color='black', linestyle='None', label='Equal-cohort mean'))
    fig.legend(handles=handles, loc='outside lower center', ncol=6, frameon=False)
    result = summary['audit']['result']
    fig.suptitle('Public-history schedule diagnostic\n'+result['classification']+
                 '\nSupplied schedule prior; private-path risk reference; linear axes; no new training', fontsize=14)
    fig.savefig(path, dpi=170, metadata={'Software': VERSION})
    plt.close(fig)


def fmt(value):
    return f'{value:.6g}'


def document(summary):
    audit = summary['audit']; result = audit['result']; rows = audit['rows']
    learned, true = (result['candidates'][a] for a in CANDIDATES)
    family_means = {(arm, f): math.fsum(r['family_regret'][f]['primary_regret'] for r in rows if r['arm'] == arm)/5 for arm in ARMS for f in range(4)}
    lines = ['# Public-history schedule headroom', '', '**'+result['classification']+'**. '+CLASSIFICATIONS[result['classification']], '',
        (f"Learned-field screen: **{sum(learned['conditions'].values())}/15**. True-field screen: **{sum(true['conditions'].values())}/15**. "
        f"Separate true-field history comparison: **{sum(result['true_history_conditions'].values())}/2**. These are the original ordered classification rules, not a single 32-condition pass gate."), '',
        ('This diagnostic trains no model. It compares all six frozen reliability models with exact 36-schedule and static-two references under both learned and supplied true fields. '
        'The schedule prior is correct for the new 0.12/0.30 population; old trained controls learned on 0.12/0.48. '
        'That information advantage is intentional for diagnosing available decision improvement, and prevents a fair architecture or robustness claim.'), '',
        '![All cohorts, families and references](benchmark.png)', '',
        '| Reference | Overall mixture conditions | BASE guard | BASE regret | Allowed BASE ceiling |',
        '|---|---:|---|---:|---:|',
        *[f"| {candidate} | {sum(passed for name, passed in group['conditions'].items() if name != 'base/unchanged/noninferiority')}/14 | {'PASS' if group['conditions']['base/unchanged/noninferiority'] else 'FAIL'} | {fmt(group['candidate_base'])} | {fmt(1.05*group['unchanged_base']+1e-6)} |" for candidate, group in result['candidates'].items()], '',
        'The BASE-family ceiling is 1.05 times the unchanged control mean plus 0.000001. Passing overall-mixture gains cannot compensate for failing this separate requirement.', '',
        ('Primary risk divides the sum of H4/H8 regret at boundaries 8..32 by 50 for every episode, then averages all 512 episodes and the five cohorts equally. '
        'Found and later boundaries contribute zero. The divisor does not depend on survival. This differs from the parent alive-boundary endpoint, so the values are not a same-metric replication. '
        'The private-path reference has more information than public filters and is not an attainable public-information floor.'), '',
        '| Arm | Primary mean | Static 0.12 | Static 0.30 | Switch up | Switch down |', '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        lines.append('| '+arm+' | '+fmt(result['means'][arm])+' | '+' | '.join(fmt(family_means[arm, f]) for f in range(4))+' |')
    lines += ['', 'Family columns are equal-cohort means conditional on that family; family counts vary because families are sampled independently. All four families have at least 64 episodes in every cohort.', '',
              '| Candidate | Control | Candidate mean | Control mean | Relative reduction | Strict paired wins |', '|---|---|---:|---:|---:|---:|']
    for candidate in CANDIDATES:
        for row in result['candidates'][candidate]['comparisons']:
            gain = f"{100*(1-row['candidate_mean']/row['control_mean']):.2f}%" if row['control_mean'] > 0 else 'undefined (zero control)'
            lines.append(f"| {candidate} | {row['control']} | {fmt(row['candidate_mean'])} | {fmt(row['control_mean'])} | {gain} | {row['paired_wins']}/5 |")
    lines += ['', (f"The separate true_exact vs true_static2 history comparison has **{result['true_history_paired_wins']}/5** strict cohort wins; "
              f"means are **{fmt(result['means']['true_exact'])}** and **{fmt(result['means']['true_static2'])}**."), '',
              'All 32 precommitted conditions remain visible. Each candidate must pass all 15 conditions for its own screen; the true-history pair affects only the specified branch of the classification.', '',
              '| Screen | Fixed condition | Outcome |', '|---|---|---|']
    for candidate in CANDIDATES:
        lines += [f"| {candidate} | `{name}` | {'PASS' if passed else 'FAIL'} |" for name, passed in result['candidates'][candidate]['conditions'].items()]
    lines += [f"| True history | `{name}` | {'PASS' if passed else 'FAIL'} |" for name, passed in result['true_history_conditions'].items()]
    lines += ['', ('The exact filter maintains joint schedule/state mass and chooses actions after averaging costs. True-law filtering minimizes expected cost for this fixed additive endpoint under the supplied population prior, in expectation. '
              'It need not win each finite sample or conditional family. Learned-field exactness is conditional on its possibly wrong model. '
              'True-versus-learned contrasts combine transition, hazard, emission and cost-head mismatch; they do not identify a component or align learned coordinates to true states.'), '',
              '| Arm | Existing added parameters | Persistent numerical state |', '|---|---:|---:|']
    lines += [f'| {arm} | {PARAMETERS[arm]} | {STATES[arm]} |' for arm in ARMS]
    lines += ['', ('No new parameters were fitted. Learned-field references reuse the same 352-parameter backbone in each cohort. '
              'State counts exclude fixed tables, terminal flags and temporary work. The reset GRU stores 164 parameters, but 48 recurrent entries always multiply zero; stored-size matching is not effective-capacity matching. '
              'Exact 36 filtering uses 288 joint masses versus 16 for static-two, so it is not a matched-state-memory comparison.'), '',
              ('Five fresh cohorts each contain 512 episodes with 32 precommitted actions plus reset. All 35 original parent backbone/model files were copied before the first new scientific episode. '
              'Every first-found episode is retained. Histories use random public actions; counterfactual decisions do not control those histories. This is not autonomous control, RL or robotics evidence.'), '',
              '| Original phase | Native elapsed seconds | Cap seconds |', '|---|---:|---:|']
    lines += [f"| {phase} | {fmt(value['native_seconds'])} | {value['cap_seconds']} |" for phase, value in summary['phase_seconds'].items()]
    lines += ['', (f"Original qualification passed **{summary['qualification']['passed']} tests**, with **{summary['qualification']['warnings']} warnings**. "
              'Its 64-episode ten-reference probe scored no effectiveness metric. Inference timings include filtering and all eight blind-fork steps; they exclude model construction, surrounding torch conversions, persistence and metric calculations. '
              'Native phase times include those surrounding costs. This is one-machine elapsed time, not equal FLOPs or a speed superiority claim.'), '',
              '<details><summary>All 50 audited rows, costs and support</summary>', '',
              '| Cohort | Arm | Primary | H4 | H8 | Immediate | Event NLL | Found / 512 | Family supports 0/1/2/3 | Inference s |',
              '|---:|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for row in rows:
        lines.append(f"| {row['cohort']+1} | {row['arm']} | "+' | '.join(fmt(row[k]) for k in ('primary_regret', 'h4_regret', 'h8_regret', 'post_regret', 'event_nll'))
                     +f" | {row['found_episodes']} | "+'/'.join(str(r['episodes']) for r in row['family_regret'])+f" | {fmt(row['inference_seconds'])} |")
    lines += ['', '</details>', '', ('The summary retains all 512 per-episode risks for every row, every family mean and all 32 conditions. '
              'The independent audit loaded 90 NPZs: 30 copied model states, five backbones, five episode files and 50 predictions. It reconstructed five private-path target populations and 20 public exact-reference outputs. '
              'It did not replay old neural gates, sampling, training or optimization. Exact-reference reconstruction is disclosed audit computation. Historical ordering remains authenticated producer evidence.'), '',
              ('The parent reliability study remains **FAIL 8/13**. This pilot screen is conditional on five fixed backbones, not a significance test, a novel architecture, unknown-prior robustness or a revised prior verdict. '
              'Any future mechanism requires separately registered evidence and controls with matched prior/training access.'), '',
              '[Protocol](../schedule-headroom-protocol.md) · [Full saved summary](summary.json) · [Archive index](archive-index.json) · [Member manifest](manifest.json) · [Publication receipt](receipt.json)', '',
              ('Archives preserve the complete current registration, source snapshot, original three native closures, qualification probes, copied states, raw episodes and all predictions. '
              'Parent archives remain externally hash-pinned. Publication reads saved audited JSON only and performs no scientific NPZ decoding, model execution or audit replay.'), '']
    return '\n'.join(lines)


def archives(auth):
    paths = {'study/'+name: (STUDY/name, pin) for name, pin in auth['study_files'].items()}
    for name in ('summary.json', 'report.md', 'benchmark.png'):
        paths['publication/'+name] = (OUTPUT/name, descriptor(OUTPUT/name))
    paths.update({'publication/overview.md': (OVERVIEW, descriptor(OVERVIEW)),
                  'publication/publisher.py': (Path(__file__).resolve(), descriptor(Path(__file__).resolve()))})
    manifest = {'version': VERSION, 'files': {name: {'original_path': str(path), **pin} for name, (path, pin) in sorted(paths.items())},
                'external_parent': auth['external'], 'maximum_archive_bytes': MAX_ARCHIVE_BYTES,
                'raw_part_budget_bytes': RAW_PART_BYTES,
                'scope': 'Complete current study and publication as opaque whole files; no parent archive duplication.',
                'self_exclusion': 'Manifest excludes itself but is archived; index and receipt remain outside archives.'}
    write(OUTPUT/'manifest.json', manifest)
    paths['publication/manifest.json'] = (OUTPUT/'manifest.json', descriptor(OUTPUT/'manifest.json'))
    groups = {'shared': []}
    for name in sorted(paths):
        match = re.match(r'study/run/(cohort-\d\d)/', name)
        groups.setdefault(match[1] if match else 'shared', []).append(name)
    parts = []
    for group, names in sorted(groups.items()):
        batches, current, size = [], [], 0
        for name in names:
            estimate = paths[name][1]['bytes']+2048
            require(estimate <= RAW_PART_BYTES, 'whole evidence member fits supported part budget')
            if current and size+estimate > RAW_PART_BYTES:
                batches.append(current); current, size = [], 0
            current.append(name); size += estimate
        if current:
            batches.append(current)
        for index, members in enumerate(batches):
            filename = 'evidence-'+group+(f'-part-{index+1:02d}' if len(batches) > 1 else '')+'.tar.gz'
            destination = OUTPUT/filename
            with destination.open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as zipped, tarfile.open(fileobj=zipped, mode='w', format=tarfile.PAX_FORMAT) as archive:
                for name in members:
                    source, pin = paths[name]
                    require(descriptor(source) == pin, 'unchanged opaque member before archive')
                    info = tarfile.TarInfo(name); info.size, info.mode, info.mtime = pin['bytes'], 0o644, 0
                    with source.open('rb') as stream:
                        archive.addfile(info, stream)
            pin = descriptor(destination)
            require(pin['bytes'] < MAX_ARCHIVE_BYTES, 'each compressed part below 45 MB')
            with tarfile.open(destination, 'r:gz') as archive:
                require(archive.getnames() == members, 'complete portable member list')
                for member in archive.getmembers():
                    require(member.isfile(), 'ordinary archive member')
                    digest, length = hashlib.sha256(), 0
                    with archive.extractfile(member) as stream:
                        for block in iter(lambda: stream.read(1048576), b''):
                            digest.update(block); length += len(block)
                    require({'sha256': digest.hexdigest(), 'bytes': length} == paths[member.name][1]
                            and descriptor(paths[member.name][0]) == paths[member.name][1], 'byte-exact opaque hash roundtrip')
            parts.append({'path': filename, **pin, 'members': members, 'member_count': len(members)})
    flattened = [name for part in parts for name in part['members']]
    require(len(flattened) == len(set(flattened)) == len(paths) and set(flattened) == set(paths), 'all evidence archived exactly once')
    index = {'version': VERSION, 'archives': parts, 'members': {name: part['path'] for part in parts for name in part['members']},
             'member_count': len(paths), 'manifest': descriptor(OUTPUT/'manifest.json'), 'maximum_archive_bytes': MAX_ARCHIVE_BYTES,
             'raw_part_budget_bytes': RAW_PART_BYTES, 'opaque_hash_roundtrip': True}
    write(OUTPUT/'archive-index.json', index)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-sha256', required=True); parser.add_argument('--publisher-sha256', required=True)
    args = parser.parse_args()
    require(not OUTPUT.exists() and not OVERVIEW.exists(), 'exclusive original publication')
    publisher = descriptor(Path(__file__).resolve())
    require(publisher['sha256'] == args.publisher_sha256, 'explicit reviewed publication source')
    auth = authenticate(args.plan_sha256)
    summary = {'version': VERSION, 'classification': auth['audit']['result']['classification'], 'technical_complete': True,
               'registration': {'path': str(STUDY/'registration.json'), **descriptor(STUDY/'registration.json')},
               'audit': auth['audit'], 'config': auth['plan']['config'], 'runtime': auth['plan']['runtime'],
               'phase_seconds': {phase: {'native_seconds': row['terminal']['wall_seconds'], 'worker_seconds': row['receipt']['seconds'],
                                        'cap_seconds': row['terminal']['cap_seconds']} for phase, row in auth['phases'].items()},
               'qualification': auth['tests'], 'feasibility_probe': auth['feasibility'], 'external_parent': auth['external'],
               'metric_source': 'Original independent audit JSON scalars; no current raw prediction decoding or replay.',
               'architecture_novelty_claim': False, 'matched_information_claim': False, 'matched_state_capacity_claim': False,
               'statistical_significance_claim': False, 'prior_failure_rescued': False, 'new_training_calls': 0}
    OUTPUT.mkdir(); write(OUTPUT/'summary.json', summary)
    chart(summary, OUTPUT/'benchmark.png')
    report = document(summary)
    with (OUTPUT/'report.md').open('x') as stream:
        stream.write(report)
    overview = report.replace('(benchmark.png)', '(schedule-headroom-results/benchmark.png)')
    overview = overview.replace('(../schedule-headroom-protocol.md)', '(schedule-headroom-protocol.md)')
    for name in ('summary.json', 'archive-index.json', 'manifest.json', 'receipt.json'):
        overview = overview.replace('('+name+')', '(schedule-headroom-results/'+name+')')
    with OVERVIEW.open('x') as stream:
        stream.write(overview)
    index = archives(auth)
    require(authenticate(args.plan_sha256) == auth and descriptor(Path(__file__).resolve()) == publisher, 'all original inputs unchanged after publication')
    receipt = {'version': VERSION, 'status': 'PASS', 'classification': summary['classification'],
               'registration': summary['registration'], 'publisher': publisher,
               'overview': {'path': str(OVERVIEW), **descriptor(OVERVIEW)}, 'files': inventory(OUTPUT),
               'sources': auth['plan']['sources'], 'study_files': auth['study_files'], 'external_parent': auth['external'],
               'original_closures': {phase: {name: descriptor(STUDY/filename) for name, filename in
                   (('receipt', phase+'.receipt.json'), ('launch', phase+'-native-01.launch.json'), ('terminal', phase+'-native-01.terminal.json'))} for phase in auth['phases']},
               'archive_parts': len(index['archives']), 'archive_members': index['member_count'],
               'publication_counts': dict.fromkeys(('scientific_npz_decodes', 'checkpoint_decodes', 'model_calls', 'generator_calls', 'optimizer_calls', 'audit_replays'), 0),
               'all_inputs_unchanged': True, 'opaque_hash_roundtrip': True, 'visual_review_required': True}
    write(OUTPUT/'receipt.json', receipt)
    print(json.dumps({'output': str(OUTPUT), 'classification': summary['classification'],
                      'archive_parts': len(index['archives']), 'archive_members': index['member_count']}))


if __name__ == '__main__':
    main()
