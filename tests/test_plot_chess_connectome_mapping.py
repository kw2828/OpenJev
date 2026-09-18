"""Invented scalar fixtures only. Never open mapping experiment outputs."""

import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('plot_mapping', ROOT / 'scripts/plot_chess_connectome_mapping.py')
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


def fixture_plan():
    protocol = {'version': plot.STUDY, 'paired_seeds': list(plot.SEEDS), 'variants': list(plot.VARIANTS),
                'mapping_policies': list(plot.POLICIES), 'train_examples': 32768, 'epochs': 6,
                'updates_per_fit': 1536, 'warmup_updates': 256, 'proposal_interval': 4,
                'proposal_seed_base': 11600041, 'proposals_per_fit': 320,
                'all_final_fits_before_neural_evaluation': True, 'regret_positions_per_split': 128,
                'regret_nodes': 20000, 'relative_loss_reduction': .1, 'gate_absolute_tolerance': 1e-12,
                'primary_wall_seconds': 7200, 'audit_wall_seconds': 1800,
                'regret_call_ceiling': 8704, 'regret_node_ceiling': 174080000}
    for key in ('matching', 'data', 'pretraining', 'selection', 'timing', 'scope', 'arena', 'asset_policy', 'audit_scope'):
        protocol[key] = f'Synthetic fixture {key}; not real results'
    return {'protocol': protocol, 'configurations': list(reversed(plot.configurations())),
            'baselines': plot.baselines(), 'sources': {p: plot.sha256(p.encode()) for p in plot.CORE_SOURCES},
            'training_settings': {'device': 'mps', 'synthetic': 'settings'},
            'original_inputs': {'synthetic': 'only'}, 'binding': {'backbone_pretraining_costs': {'synthetic': 1}}}


def fixture_summary(plan=None, plan_hash='a' * 64):
    plan = fixture_plan() if plan is None else plan
    s = {'status': 'completed', 'version': plot.STUDY, 'synthetic_fixture': True,
         'plan_sha256': plan_hash, 'execution_receipt_sha256': 'e' * 64,
         'metrics': {}, 'engine_metrics': {}, 'engine_loss': {}, 'latency': {}, 'training_costs': {},
         'backbone_pretraining_costs': plan['binding']['backbone_pretraining_costs'],
         'original_inputs': plan['original_inputs'], 'original_data_costs': {'status': 'completed'},
         'grading_costs': {'calls': 900, 'requested_nodes': 18000000, 'reported_nodes': 18000100, 'wall_seconds': 50.},
         'wall_seconds': 2000., 'report_new_model_or_engine_calls': 0,
         'limits': [plan['protocol'][k] for k in ('matching', 'data', 'pretraining', 'selection', 'timing',
                                                'scope', 'arena', 'asset_policy', 'audit_scope')]}
    for index, c in enumerate(plot.configurations() + plot.baselines()):
        name = c['name']
        s['engine_metrics'][name], s['engine_loss'][name], s['metrics'][name] = {}, {}, {}
        for split in plot.SPLITS:
            loss = .14 + .002 * (index % 3) + (.06 if split == 'shift' else 0)
            if c['variant'] == 'biological' and c['mapping_policy'] == 'learned':
                loss -= .035 if split == 'dev' else .006
            elif c.get('mapping_policy') == 'learned':
                loss -= .012
            elif c['variant'] == 'direct':
                loss -= .008
            s['engine_loss'][name][split] = loss
            s['engine_metrics'][name][split] = {'positions': 128, 'mean_signed_bounded_loss': loss,
                'mean_cp_loss': 130. + index, 'p95_cp_loss': 1000. + 10 * index, 'max_cp_loss': 8000 + index}
            s['metrics'][name][split] = {'examples': 2048, 'correct': 600, 'agreement': 600 / 2048, 'target_nll': 2.4}
        s['latency'][name] = {'mean_ms': 2., 'total_wall_ms': 256., 'warmup_wall_ms': 5.,
                              'host_load_average_before_latency': [1., 1., 1.],
                              'host_load_average_after_latency': [2., 2., 2.]}
        if c['variant'] == 'direct':
            continue
        permutation = list(range(64))
        if c['mapping_policy'] == 'learned':
            permutation[0], permutation[1] = 1, 0
        f = {'status': 'completed', 'version': 'openjev-connectome-hard-mapping-study-v1',
             'configuration': c, 'plan_sha256': plan_hash, 'settings': plan['training_settings'],
             'updates': 1536, 'examples_seen': 196608, 'proposal_count': 320, 'proposal_forward_calls': 640,
             'proposal_optimizer_updates': 0, 'proposal_examples_seen': 81920, 'training_forward_calls': 1536,
             'backward_calls': 1536, 'total_forward_calls': 2176, 'accepted_proposals': int(c['mapping_policy'] == 'learned'),
             'final_permutation': permutation, 'proposal_wall_seconds': 10. + index / 20,
             'step_wall_seconds': 30. + index / 10, 'training_seconds': 32. + index / 10,
             'fit_wall_seconds': 34. + index / 10,
             'source_sha256': plan['sources'], 'backbone_unchanged': True, 'mps_fallback_environment': '0'}
        for key in ('initial_state_sha256', 'state_sha256', 'checkpoint_sha256', 'learning_sha256',
                    'proposals_sha256', 'backbone_sha256', 'immutable_buffers_sha256', 'graph_sha256'):
            f[key] = plot.sha256(f'{name}-{key}'.encode())
        s['training_costs'][name] = f
    s['mapping_comparison'] = plot.recompute_comparison(s['engine_loss'], plan['protocol']['scope'])
    return plan, s


def write_fixture(directory):
    directory.mkdir(parents=True, exist_ok=True)
    plan = fixture_plan()
    source_root = directory / 'sources'
    for path in plan['sources']:
        file = source_root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(path)  # Inert fixture bytes, never imported.
    plan_dir = directory / 'protocol'
    plan_dir.mkdir()
    plan_path = plan_dir / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    plan_hash = plot.sha256(plan_path.read_bytes())
    _, summary = fixture_summary(plan, plan_hash)
    audit = directory / 'audit'
    audit.mkdir()
    (audit / 'started.json').write_text(json.dumps({'status': 'started', 'plan_sha256': plan_hash, 'started_unix': 10.}))
    (audit / 'summary.json').write_text(json.dumps(summary))
    receipt = {'status': 'completed', 'plan_sha256': plan_hash, 'execution_receipt_sha256': 'e' * 64,
               'audit_wall_seconds': 50., 'completed_unix': 60., 'new_model_calls': 0, 'new_engine_calls': 0,
               'files': {name: plot.sha256((audit / name).read_bytes()) for name in ('started.json', 'summary.json')}}
    (audit / 'receipt.json').write_text(json.dumps(receipt))
    options = {'expected_plan_sha256': plan_hash, 'expected_receipt_sha256': plot.sha256((audit / 'receipt.json').read_bytes()),
               'source_root': source_root, 'synthetic_fixture': True}
    return audit, plan_path, options


def test_full_member_coverage_signed_means_nested_costs_and_all_criteria():
    plan, summary = fixture_summary()
    assert plot.validate(summary, plan, expected_plan_sha256='a' * 64, synthetic_fixture=True) is summary
    v = plot.chart_values(summary)
    assert len(v['loss_points']) == len(v['engine_tail_rows']) == 66
    assert len(v['fit_costs']) == 30 and len(v['criteria']) == 40
    assert {c['split'] for c in v['criteria']} == {'dev', 'shift'}
    assert sum(c['passed'] for c in v['criteria']) == 20
    for arm, costs in v['mean_fit_costs'].items():
        assert costs['fit_wall_seconds'] == pytest.approx(costs['proposal_wall_seconds'] + costs['nonproposal_fit_wall_seconds'])
        assert costs['fit_wall_seconds'] == math.fsum(f['fit_wall_seconds'] for f in v['fit_costs']
                                                      if f"{f['variant']}-{f['mapping_policy']}" == arm) / 3
    # Independent closed-form examples rather than only trusting gate generation.
    first = v['criteria'][0]
    assert first['reference'] == pytest.approx(.142)
    assert first['treatment'] == pytest.approx(.107)
    assert first['reduction'] == pytest.approx(.035 / .142)
    interaction = next(c for c in v['criteria'] if c['split'] == 'dev'
                       and c['criterion'] == 'mapping_difference_in_differences')
    assert interaction['difference'] == pytest.approx(.035 - .012)


@pytest.mark.parametrize('fault', ['missing_fit', 'duplicate_config', 'missing_baseline', 'wrong_seed', 'partial_fit',
    'wrong_updates', 'free_fixed_proposals', 'fixed_accept', 'fixed_mapping', 'negative_time', 'bool_time',
    'double_count_time', 'aggregate_wall', 'missing_checkpoint_hash', 'changed_fit_source', 'empty_fit_source',
    'missing_eval', 'missing_split', 'missing_engine', 'wrong_positions', 'bounded_disagreement', 'bad_cp_tail',
    'missing_latency', 'wrong_latency', 'changed_gate', 'missing_check', 'reordered_checks', 'boolean_gate',
    'false_mean', 'bool_engine_loss', 'wrong_grading', 'wrong_scope', 'new_calls', 'wrong_protocol'])
def test_parser_rejects_incomplete_or_inconsistent_summary(fault):
    plan, s = fixture_summary()
    name = plan['configurations'][0]['name']
    fixed = next(c['name'] for c in plan['configurations'] if c['mapping_policy'] == 'fixed')
    if fault == 'missing_fit': s['training_costs'].pop(name)
    elif fault == 'duplicate_config': plan['configurations'][0] = plan['configurations'][1]
    elif fault == 'missing_baseline': plan['baselines'].pop()
    elif fault == 'wrong_seed': plan['configurations'][0]['seed'] = 7
    elif fault == 'partial_fit': s['training_costs'][name]['status'] = 'started'
    elif fault == 'wrong_updates': s['training_costs'][name]['updates'] = 1535
    elif fault == 'free_fixed_proposals': s['training_costs'][fixed]['proposal_forward_calls'] = 0
    elif fault == 'fixed_accept': s['training_costs'][fixed]['accepted_proposals'] = 1
    elif fault == 'fixed_mapping': s['training_costs'][fixed]['final_permutation'][:2] = [1, 0]
    elif fault == 'negative_time': s['training_costs'][name]['fit_wall_seconds'] = -1
    elif fault == 'bool_time': s['training_costs'][name]['proposal_wall_seconds'] = True
    elif fault == 'double_count_time': s['training_costs'][name]['proposal_wall_seconds'] = 500.
    elif fault == 'aggregate_wall': s['wall_seconds'] = 10.
    elif fault == 'missing_checkpoint_hash': s['training_costs'][name]['checkpoint_sha256'] = ''
    elif fault == 'changed_fit_source': s['training_costs'][name]['source_sha256'] = {'wrong.py': 'b' * 64}
    elif fault == 'empty_fit_source': s['training_costs'][name]['source_sha256'] = {}
    elif fault == 'missing_eval': s['metrics'].pop('direct-97')
    elif fault == 'missing_split': s['metrics'][name].pop('shift')
    elif fault == 'missing_engine': s['engine_metrics'].pop(name)
    elif fault == 'wrong_positions': s['engine_metrics'][name]['dev']['positions'] = 64
    elif fault == 'bounded_disagreement': s['engine_metrics'][name]['dev']['mean_signed_bounded_loss'] += .1
    elif fault == 'bad_cp_tail': s['engine_metrics'][name]['dev']['p95_cp_loss'] = 1e6
    elif fault == 'missing_latency': s['latency'].pop(name)
    elif fault == 'wrong_latency': s['latency'][name]['mean_ms'] = 20.
    elif fault == 'changed_gate': s['mapping_comparison']['continuation_passed'] = True
    elif fault == 'missing_check': s['mapping_comparison']['checks'].pop()
    elif fault == 'reordered_checks': s['mapping_comparison']['checks'].reverse()
    elif fault == 'boolean_gate': s['mapping_comparison']['checks'][0]['passed'] = 1
    elif fault == 'false_mean': s['mapping_comparison']['means']['direct']['dev'] += .01
    elif fault == 'bool_engine_loss':
        s['engine_loss'][name]['dev'] = True
        s['engine_metrics'][name]['dev']['mean_signed_bounded_loss'] = 1.
        s['mapping_comparison'] = plot.recompute_comparison(s['engine_loss'], plan['protocol']['scope'])
    elif fault == 'wrong_grading': s['grading_costs']['requested_nodes'] += 1
    elif fault == 'wrong_scope': s['limits'][-1] = 'Real gameplay result'
    elif fault == 'new_calls': s['report_new_model_or_engine_calls'] = 1
    elif fault == 'wrong_protocol': plan['protocol']['relative_loss_reduction'] = .01
    with pytest.raises(ValueError):
        plot.validate(s, plan, expected_plan_sha256='a' * 64, synthetic_fixture=True)


def test_negative_losses_and_raw_cp_tails_are_not_clipped_or_pooled():
    plan, s = fixture_summary()
    for seed in plot.SEEDS:
        name = f'biological-fixed-{seed}'
        s['engine_loss'][name]['dev'] = -.1
        s['engine_metrics'][name]['dev'].update(mean_signed_bounded_loss=-.1, mean_cp_loss=-400.,
                                               p95_cp_loss=-20., max_cp_loss=-1)
    s['mapping_comparison'] = plot.recompute_comparison(s['engine_loss'], plan['protocol']['scope'])
    plot.validate(s, plan, expected_plan_sha256='a' * 64, synthetic_fixture=True)
    assert s['mapping_comparison']['checks'][0]['reduction'] is None
    assert s['mapping_comparison']['checks'][0]['passed'] is False
    values = plot.chart_values(s)
    preserved = [r for r in values['engine_tail_rows'] if r['configuration'].startswith('biological-fixed') and r['split'] == 'dev']
    assert len(preserved) == 3
    assert all(r['mean_signed_bounded_loss'] == -.1 and r['p95_cp_loss'] == -20 for r in preserved)


@pytest.mark.parametrize('raw', ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e309}'])
def test_ambiguous_nonfinite_json_is_rejected(raw):
    with pytest.raises(ValueError): plot.parse(raw)


@pytest.mark.parametrize('fault', ['summary_bytes', 'receipt_bytes', 'wrong_plan_hash', 'wrong_receipt_hash',
                                  'failed', 'extra', 'symlink', 'changed_source', 'source_symlink',
                                  'new_calls', 'partial_receipt', 'receipt_schema', 'member_manifest'])
def test_completed_audit_authentication_refuses_tampering(tmp_path, fault):
    audit, plan, options = write_fixture(tmp_path)
    if fault == 'summary_bytes': (audit / 'summary.json').write_bytes((audit / 'summary.json').read_bytes() + b' ')
    elif fault == 'receipt_bytes': (audit / 'receipt.json').write_bytes((audit / 'receipt.json').read_bytes() + b' ')
    elif fault == 'wrong_plan_hash': options['expected_plan_sha256'] = 'c' * 64
    elif fault == 'wrong_receipt_hash': options['expected_receipt_sha256'] = 'c' * 64
    elif fault == 'failed': (audit / 'failed.json').write_text('{}')
    elif fault == 'extra': (audit / 'extra.json').write_text('{}')
    elif fault == 'symlink':
        (audit / 'summary.json').rename(tmp_path / 'saved.json')
        (audit / 'summary.json').symlink_to(tmp_path / 'saved.json')
    elif fault in ('changed_source', 'source_symlink'):
        file = options['source_root'] / min(plot.CORE_SOURCES)
        if fault == 'changed_source': file.write_text('changed')
        else:
            file.rename(tmp_path / 'saved.py')
            file.symlink_to(tmp_path / 'saved.py')
    else:
        file = audit / 'receipt.json'
        receipt = json.loads(file.read_text())
        if fault == 'new_calls': receipt['new_model_calls'] = 1
        elif fault == 'partial_receipt': receipt['status'] = 'started'
        elif fault == 'receipt_schema': receipt['extra'] = 'undeclared'
        elif fault == 'member_manifest': receipt['files']['summary.json'] = 'f' * 64
        file.write_text(json.dumps(receipt))
        options['expected_receipt_sha256'] = plot.sha256(file.read_bytes())
    with pytest.raises(ValueError): plot.load_completed(audit, plan, **options)


def test_synthetic_flag_and_tag_must_agree(tmp_path):
    audit, plan, options = write_fixture(tmp_path)
    options['synthetic_fixture'] = False
    with pytest.raises(ValueError, match='Synthetic'): plot.load_completed(audit, plan, **options)
    p, s = fixture_summary()
    s.pop('synthetic_fixture')
    with pytest.raises(ValueError, match='Synthetic'):
        plot.validate(s, p, expected_plan_sha256='a' * 64, synthetic_fixture=True)


def test_render_binds_every_output_and_preserves_all_cp_rows(tmp_path):
    audit, plan, options = write_fixture(tmp_path)
    out = tmp_path / 'figure'
    result = plot.render(audit, plan, out, **options)
    assert result['synthetic_fixture'] is True and result['new_model_or_engine_calls'] == 0
    assert result['fit_count'] == 30 and result['evaluated_models'] == 33 and result['criteria_count'] == 40
    assert {p.name for p in out.iterdir()} == {'figure.png', 'figure.pdf', 'engine_cp_tails.csv', 'provenance.json'}
    assert (out / 'figure.png').read_bytes().startswith(b'\x89PNG')
    assert (out / 'figure.pdf').read_bytes().startswith(b'%PDF')
    for name, digest in result['files'].items(): assert plot.sha256((out / name).read_bytes()) == digest
    with (out / 'engine_cp_tails.csv').open() as stream: tails = list(csv.DictReader(stream))
    assert len(tails) == 66
    assert len({(r['configuration'], r['split']) for r in tails}) == 66
    assert json.loads((out / 'provenance.json').read_text()) == result
    with pytest.raises(FileExistsError): plot.render(audit, plan, out, **options)
    with pytest.raises(ValueError, match='outside'): plot.render(audit, plan, audit / 'figure', **options)


def test_figure_has_four_panels_all_forty_status_cells_and_visible_titles():
    _, s = fixture_summary()
    fig = plot.figure(s, synthetic_fixture=True)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        assert len(fig.axes) == 4
        assert len(fig.axes[3].patches) == 40
        assert len([t for t in fig.axes[3].texts if t.get_text() in ('P', 'F')]) == 40
        texts = list(fig.texts) + [a.title for a in fig.axes]
        for text in texts:
            if not text.get_text(): continue
            box = text.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0
            assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
    finally:
        plot.plt.close(fig)
