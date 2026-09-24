"""Publish authenticated saved timings and evidence; never replay learning."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

from qualify_finite_joint_reuse_throughput_v2 import (
    FAILED,
    FOLDER,
    PLAN_PATH,
    ROOT,
    closed_phase,
    descriptor,
    files,
    read,
    require,
)
from supervise_dialogue_observation_v2 import publish

PLAN_SHA = '2ebc7bda8b49b0c0ac9684a15bcdd614c17ed5b2f4e7f60d9b3329e92c6eebc6'
OUTPUT = ROOT / 'research/finite-joint-reuse-throughput-results'
OVERVIEW = ROOT / 'research/finite-joint-reuse-throughput-results.md'
ARMS = ('original_free', 'matched_free', 'rounded')


def pin(payload):
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}


def chart(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.8), layout='constrained')
    colors = ('#2463a6', '#d27816', '#248567')
    for i, arm in enumerate(ARMS):
        rows = [r for r in summary['throughput']['paired'] if r['arm'] == arm and not r['warmup']]
        for j, row in enumerate(rows):
            ax.scatter(row['ratio'], i + (j - 1) * .13, color=colors[j], s=55,
                       label=f'Round {j + 1}' if i == 0 else None, zorder=3)
        median = summary['throughput']['median_ratios'][arm]
        ax.scatter(median, i, color='#182331', marker='D', s=28, zorder=4,
                   label='Median' if i == 0 else None)
        ax.annotate(f'{median:.3f}x', (median, i), xytext=(7, 10), textcoords='offset points', fontsize=10)
    ax.axvline(1, color='#788492', linewidth=1)
    ax.axvline(1.05, color='#788492', linestyle=':', linewidth=1, label='Median gate: >1.05x')
    ax.set_yticks(range(3), ['Original free', 'Matched free', 'Rounded recurrent'])
    ax.set_ylim(2.55, -.6)
    ax.set_xlim(.97, max(r['ratio'] for r in summary['throughput']['paired'] if not r['warmup']) + .08)
    ax.set_xlabel('Separate / reuse full training-call time (higher is faster)')
    ax.set_title('Shared computation improves training throughput', loc='left', fontsize=14, weight='bold')
    ax.grid(axis='x', color='#e5e9ee', zorder=0)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.18), ncol=3, frameon=False, fontsize=9)
    fig.savefig(path, dpi=180, metadata={'Description': 'All nine measured pairs; six warmup fits excluded by prior protocol. One machine, no significance claim.'})
    plt.close(fig)


def main():
    require(descriptor(PLAN_PATH)['sha256'] == PLAN_SHA, 'exact original throughput registration')
    plan = read(PLAN_PATH)
    closures = {phase: closed_phase(plan, phase) for phase in ('qualify', 'run', 'audit')}
    current_before, failed_before = files(FOLDER), files(FAILED)
    audit_path = Path(plan['phases']['audit']['output']) / 'audit.json'
    audit = read(audit_path)
    require(audit['agreement'] is True and audit['scientific_admission'] is False
            and audit['counts']['fits'] == 24 and audit['counts']['measured_pairs'] == 9
            and audit['sources'] == plan['sources'], 'complete original audited comparison')
    require(audit['throughput']['passed'] is True and len(audit['throughput']['conditions']) == 12
            and all(audit['throughput']['conditions'].values()), 'all original throughput conditions pass')
    log = (Path(plan['phases']['qualify']['output']) / 'command-1.log').read_text()
    require('78 passed in 2.48s' in log and 'failed' not in log, 'original qualification result')
    model_error = max(v['maximum_absolute_error'] for p in audit['parity']
                      for b in p['boundaries'] for v in b['parameters'].values())
    optimizer_error = max(b['optimizer']['maximum_absolute_error'] for p in audit['parity'] for b in p['boundaries'])
    summary = {'version': 'finite-joint-reuse-throughput-publication-v1',
        'registration': {'path': str(PLAN_PATH), **descriptor(PLAN_PATH)},
        'status': audit['throughput']['status'], 'technical_complete': True,
        'scientific_admission': False, 'qualification_tests_passed': 78,
        'throughput': audit['throughput'], 'counts': audit['counts'],
        'counts_scope': 'Independent saved-output audit. Zero model/optimizer/generator calls describes the audit only, not training or qualification.',
        'maximum_absolute_model_difference': model_error,
        'maximum_absolute_optimizer_difference': optimizer_error, 'tolerance': audit['tolerance'],
        'timings': {**audit['timings'], 'audit_native_seconds': closures['audit']['terminal']['wall_seconds'],
                    'prior_failed_qualification_native_seconds': 3.106916791},
        'runtime': plan['runtime'], 'sources': len(plan['sources']),
        'prior_failure': {'tests_passed': 77, 'tests_failed': 1, 'timed_fits': 0,
                          'reason': 'Test selector accidentally matched the rounded model name.',
                          'production_gate_changed': False, 'files': len(failed_before)},
        'limitations': audit['limitations'],
        'archive_scope': 'Complete current study and original failed qualification, including all source snapshots and saved model/optimizer boundaries. Earlier linked studies, interpreter and installed packages remain external.'}
    require(not OUTPUT.exists() and not OVERVIEW.exists(), 'exclusive publication outputs')
    OUTPUT.mkdir()
    publish(OUTPUT / 'summary.json', summary)
    chart(summary, OUTPUT / 'benchmark.png')
    table = '\n'.join(f'| {arm} | {audit["throughput"]["median_ratios"][arm]:.3f}x | '
                       f'{min(r["ratio"] for r in audit["throughput"]["paired"] if r["arm"] == arm and not r["warmup"]):.3f}x | '
                       f'{max(r["ratio"] for r in audit["throughput"]["paired"] if r["arm"] == arm and not r["warmup"]):.3f}x |'
                       for arm in ARMS)
    report = f'''# Shared computation: measured training throughput

**{summary['status']}**. All nine measured pairs favor reuse, and all three
median ratios exceed the prospectively fixed 1.05x threshold. These are complete
training-call timings on one machine, not an inference benchmark or an
architectural/task-performance gain.

| Model | Median separate / reuse time | Smallest pair | Largest pair |
|---|---:|---:|---:|
{table}

![All nine measured paired runtime ratios](finite-joint-reuse-throughput-results/benchmark.png)

The comparison retains 24 fresh fits: six warmups and 18 measured fits, with
the same 512 attempted training histories, seed, initialization, optimizer,
minibatch order and 32-prefix/64-joint update schedule. The primary timer
includes construction, all safeguards, updates, model/Adam checkpoints and
training-log writes. Warmups were excluded by the original protocol. One
original-free pair improves by only 0.9%; three pairs per model are not a
statistical-significance result.

All 78 qualification tests pass. The independent saved-output audit checks
every update exposure, 72 model checkpoints and 72 optimizer checkpoints.
Initial and prefix-boundary states match exactly. Maximum final model and
optimizer differences are {model_error:.3g} and {optimizer_error:.3g}, within
the fixed symmetric absolute/relative tolerance 1e-7. No training is replayed
by the audit. Agreement over this short schedule does not establish identical
long training trajectories.

Native qualification, producer and audit phases took
{summary['timings']['qualification_native_seconds']:.3f},
{summary['timings']['producer_native_seconds']:.3f} and
{summary['timings']['audit_native_seconds']:.3f} seconds respectively.
Generation took {summary['timings']['generation_seconds']:.3f} seconds and is
nested within producer time. Complete warmup/measured training calls total
{audit['throughput']['warmup_call_seconds']:.3f}/{audit['throughput']['measured_call_seconds']:.3f}
seconds, also nested within producer time. These scopes must not be added twice.

The first qualification stopped at a test-selector bug: 77 tests passed and
one failed before any timed fit. Its complete evidence and 88 frozen sources
remain intact. A separate [v2 registration](finite-joint-reuse-throughput-v2-protocol.md)
corrected only the test selector and registration binding. The workload,
production gate, tolerances and caps were unchanged. The prior
[equal-update feasibility stop](finite-update-learning-stop-results.md) also
remains closed. This engineering result does not reopen or admit that study.

The next research step is a fresh equal-update effectiveness comparison using
this implementation. Any claim that recurrent constraints improve decisions
still needs new held-out results and scenario-shift validation.

[Original protocol](finite-joint-reuse-throughput-protocol.md) ·
[Summary and every paired timing](finite-joint-reuse-throughput-results/summary.json) ·
[Complete current-study and failure evidence](finite-joint-reuse-throughput-results/evidence.tar.gz) ·
[Archive manifest](finite-joint-reuse-throughput-results/manifest.json) ·
[Publication receipt](finite-joint-reuse-throughput-results/receipt.json)
'''
    with OVERVIEW.open('x') as stream:
        stream.write(report)
    payloads = {}
    for label, folder, pins in (('study', FOLDER, current_before), ('failed-qualification', FAILED, failed_before)):
        for name, expected in pins.items():
            data = (folder / name).read_bytes()
            require(pin(data) == expected, 'unchanged archived evidence')
            payloads[label + '/' + name] = data
    payloads['publication/report.md'] = OVERVIEW.read_bytes()
    payloads['publication/summary.json'] = (OUTPUT / 'summary.json').read_bytes()
    payloads['publication/benchmark.png'] = (OUTPUT / 'benchmark.png').read_bytes()
    payloads['publication/publisher.py'] = Path(__file__).read_bytes()
    manifest = {'files': {name: pin(data) for name, data in sorted(payloads.items())},
                'scope': summary['archive_scope'], 'self_exclusion': 'Manifest is archived but excludes its own hash.'}
    publish(OUTPUT / 'manifest.json', manifest)
    payloads['publication/manifest.json'] = (OUTPUT / 'manifest.json').read_bytes()
    with ((OUTPUT / 'evidence.tar.gz').open('xb') as raw,
          gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as zipped,
          tarfile.open(fileobj=zipped, mode='w', format=tarfile.USTAR_FORMAT) as archive):
        for name, data in sorted(payloads.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            archive.addfile(info, io.BytesIO(data))
    with tarfile.open(OUTPUT / 'evidence.tar.gz', 'r:gz') as archive:
        require(archive.getnames() == sorted(payloads), 'exact archive roster')
        for member in archive.getmembers():
            require(member.isfile() and archive.extractfile(member).read() == payloads[member.name], 'opaque archive roundtrip')
    require(files(FOLDER) == current_before and files(FAILED) == failed_before, 'all current and failed evidence unchanged')
    for phase in closures:
        closed_phase(plan, phase)
    publish(OUTPUT / 'receipt.json', {'status': 'PASS', 'registration': summary['registration'],
        'publisher': descriptor(Path(__file__).resolve()), 'files': files(OUTPUT),
        'overview': {'path': str(OVERVIEW), **descriptor(OVERVIEW)}, 'archive_members': len(payloads),
        'study_files': current_before, 'failed_qualification_files': failed_before,
        'model_calls': 0, 'checkpoint_decodes': 0, 'audit_reruns': 0, 'inputs_unchanged': True,
        'scientific_admission': False})
    print(json.dumps({'output': str(OUTPUT), 'status': summary['status'], 'archive_members': len(payloads)}))


if __name__ == '__main__':
    main()
