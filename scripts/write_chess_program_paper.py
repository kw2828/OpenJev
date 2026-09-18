"""Generate draft tables from authenticated existing reports, without inference."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'paper/chess-iclr2027'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def scientific(value):
    mantissa, exponent = f'{value:.2e}'.split('e')
    return mantissa + r'\times10^{' + str(int(exponent)) + '}'


def authenticated(directory, publication=False):
    directory = ROOT/directory
    summary, receipt = read(directory/'summary.json'), read(directory/'completed.json')
    if summary['status'] != 'completed' or receipt['status'] != 'completed':
        raise ValueError('Incomplete evidence')
    if publication:
        expected = receipt.get('summary_sha256', receipt.get('official_summary_sha256'))
        if sha(directory/'summary.json') != expected or sha(directory/'manifest.json') != receipt['manifest_sha256']:
            raise ValueError('Published summary/manifest changed')
    else:
        for name, digest in receipt['files'].items():
            if sha(directory/name) != digest:
                raise ValueError('Report member changed: '+name)
    expected_plan = receipt.get('plan_sha256')
    if expected_plan is None and not publication:
        if 'started.json' not in receipt['files']:
            raise ValueError('Missing authenticated plan identity')
        expected_plan = read(directory/'started.json')['plan_sha256']
    if summary['plan_sha256'] != expected_plan:
        raise ValueError('Plan identity mismatch')
    return summary


def table(path, caption, columns, header, rows):
    lines = [r'\begin{table}[t]', r'\centering\small', r'\caption{'+caption+'}',
             r'\begin{tabular}{'+columns+'}', r'\toprule', header+r' \\', r'\midrule']
    lines += [' & '.join(row)+r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    path.write_text('\n'.join(lines)+'\n')


def main():
    sources = {
        'candidate': 'evidence/chess-candidate-v2/results',
        'transfer': 'evidence/chessbench-transfer-v1/results',
        'biology': 'runs/chess-connectome-v1/report',
        'locality': 'runs/chess-locality-v1/execution',
        'batch': 'runs/chess-locality-batch-v1/execution',
        'transport': 'runs/chess-transport-v1/execution',
        'transport_regret': 'runs/chess-transport-regret-v1/execution',
        'relation_baselines': 'runs/chess-relation-baselines-v1/execution',
        'baseline_regret': 'runs/chess-relation-baselines-regret-v1/execution',
        'counterfactual': 'runs/chess-counterfactual-transport-v1/execution',
        'joint': 'runs/chess-transport-joint-v1/execution',
        'graph_contrast': 'runs/chess-graph-contrast-v2/execution',
        'graph_contrast_shared': 'runs/chess-graph-contrast-v3/execution',
        'graph_cache': 'runs/chess-child-graph-cache-v1/execution',
        'wldn_preflight': 'runs/chess-wldn-preflight-v1/execution',
        'wldn_quality': 'runs/chess-wldn-quality-v1/execution',
        'wldn_interventions': 'runs/chess-wldn-interventions-v1/execution',
        'union_quality': 'runs/chess-union-difference-quality-v1/execution',
        'union_cost': 'runs/chess-union-difference-cost-v1/execution',
        'union_preflight': 'runs/chess-union-difference-preflight-v1/execution',
        'union_input': 'runs/chess-union-edit-input-v1/execution',
        'union_consistent': 'runs/chess-union-consistent-preflight-v1/execution',
        'graph_quality': 'runs/chess-graph-contrast-quality-v1/execution',
        'cgr_quality': 'runs/chess-cgr-quality-v1/execution',
        'graph_baselines': 'runs/chess-graph-baseline-comparison-v1/execution',
        'normalization': 'runs/chess-edit-normalization-v1/execution',
        'cgr_preflight': 'runs/chess-cgr-preflight-v1/execution',
        'path_baseline': 'runs/chess-path-baseline-v1/execution',
        'attack_edits': 'runs/chess-attack-edit-v1/execution',
        'spatial': 'evidence/chess-spatial-v1/results',
        'capacity': 'evidence/chess-capacity-v1/results',
        'anchor': 'evidence/chess-anchor-v1/results',
    }
    data = {k: authenticated(v, publication=k not in ('biology', 'locality', 'batch', 'transport', 'transport_regret', 'relation_baselines', 'baseline_regret', 'counterfactual', 'joint', 'graph_contrast', 'graph_contrast_shared', 'graph_cache', 'wldn_preflight', 'wldn_quality', 'wldn_interventions', 'union_preflight', 'union_input', 'union_consistent', 'union_quality', 'union_cost', 'graph_quality', 'cgr_quality', 'graph_baselines', 'normalization', 'cgr_preflight', 'path_baseline', 'attack_edits')) for k, v in sources.items()}
    basis_path = 'evidence/chess-union-difference-basis-v1/receipt.json'
    basis = read(ROOT/basis_path)
    if (basis['status'] != 'completed' or basis['pytest'] != '7 passed'
            or any(sha(ROOT/p) != h for p,h in basis['sources'].items())
            or basis['edit_from_presence_matrix'] != [[-1,1,1],[1,-1,0],[1,0,-1]]
            or basis['new_training_updates'] != 0 or basis['new_quality_predictions'] != 0):
        raise ValueError('Unverified edge-feature basis checks')
    c, t, b, l = (data[k] for k in ('candidate', 'transfer', 'biology', 'locality'))
    joint = data['joint']
    joint_audit_path = ROOT/'evidence/chess-transport-joint-v1/audit/receipt.json'
    joint_audit = read(joint_audit_path)
    if (joint_audit['status'] != 'completed'
            or joint_audit['summary_sha256'] != sha(ROOT/sources['joint']/'summary.json')
            or joint_audit['execution_receipt_sha256'] != sha(ROOT/sources['joint']/'completed.json')
            or joint_audit['evaluation_predictions_replayed'] != 98304
            or joint_audit['gate_recomputed'] != joint['gate_checks']):
        raise ValueError('Joint-representation audit does not bind all predictions')
    joint_means={arm:{split:{metric:statistics.mean(joint['metrics'][f'{arm}-{seed}'][split][metric] for seed in (97,109,127))
                             for metric in ('agreement','target_nll')} for split in ('dev','shift')}
                 for arm in ('base','direct','transport','no_overlap','uniform','static','rewired')}
    joint_rows=[]
    for arm,label in [('base','Frozen direct'),('direct','Joint direct'),('transport','Joint transport'),
                      ('no_overlap','Joint no overlap'),('uniform','Joint uniform'),('static','Joint static'),('rewired','Joint rewired')]:
        joint_rows.append([label,*(f"{100*joint_means[arm][split]['agreement']:.2f}" for split in ('dev','shift')),
                           *(f"{joint_means[arm][split]['target_nll']:.3f}" for split in ('dev','shift'))])
    table(OUT/'joint-transport-table.tex','Joint-training follow-up. All three seeds retained. Agreement is percent; NLL is lower-is-better. '
          'The direct comparator receives the same additional cross-entropy training as all transport variants.',
          'lrrrr','Policy & Agree., ordinary & Agree., shifted & NLL, ordinary & NLL, shifted',joint_rows)
    followup_audits = {}
    for key, path, field, count in [
        ('baseline_regret','evidence/chess-relation-baselines-regret-v1/audit/receipt.json','policy_position_records',3840),
        ('counterfactual','evidence/chess-counterfactual-transport-v1/audit/receipt.json','root_seed_replays',96),
        ('graph_contrast','evidence/chess-graph-contrast-v2/audit/receipt.json','numerical_replays',96),
        ('graph_contrast_shared','evidence/chess-graph-contrast-v3/audit/receipt.json','numerical_replays',96),
        ('wldn_preflight','evidence/chess-wldn-preflight-v1/audit/receipt.json','numerical_replays',24),
        ('wldn_quality','evidence/chess-wldn-quality-v1/audit/receipt.json','evaluation_predictions_replayed',24576),
        ('wldn_interventions','evidence/chess-wldn-interventions-v1/audit/receipt.json','evaluation_predictions_replayed',86016),
        ('union_quality','evidence/chess-union-difference-quality-v1/audit/receipt.json','evaluation_predictions_replayed',86016),
        ('union_cost','evidence/chess-union-difference-cost-v1/audit/receipt.json','native_choice_replays',288),
        ('union_preflight','evidence/chess-union-difference-preflight-v1/audit/receipt.json','numerical_replays',96),
        ('union_input','evidence/chess-union-edit-input-v1/audit/receipt.json','candidate_replays',1087523),
        ('union_consistent','evidence/chess-union-consistent-preflight-v1/audit/receipt.json','candidate_score_checks',627),
        ('graph_quality','evidence/chess-graph-contrast-quality-v1/audit/receipt.json','evaluation_predictions_replayed',73728),
        ('cgr_quality','evidence/chess-cgr-quality-v1/audit/receipt.json','evaluation_predictions_replayed',24576),
        ('graph_baselines','evidence/chess-graph-baseline-comparison-v1/audit/receipt.json','distinct_decision_records_joined',73728),
        ('normalization','evidence/chess-edit-normalization-v1/audit/receipt.json','candidate_replays',3920),
        ('cgr_preflight','evidence/chess-cgr-preflight-v1/audit/receipt.json','numerical_replays',24),
        ('path_baseline','evidence/chess-path-baseline-v1/audit/receipt.json','evaluation_predictions_replayed',36864),
        ('attack_edits','evidence/chess-attack-edit-v1/audit/receipt.json','candidate_replays',1087523)]:
        audit = read(ROOT/path)
        if (audit['status'] != 'completed' or audit[field] != count
                or audit['summary_sha256'] != sha(ROOT/sources[key]/'summary.json')
                or audit['execution_receipt_sha256'] != sha(ROOT/sources[key]/'completed.json')):
            raise ValueError('Incomplete follow-up audit: '+key)
        if key.startswith('graph_contrast'):
            if (audit['additional_artificial_update_replays'] != 12
                    or any(audit['final_checkpoint_max_errors'].values())
                    or audit['numerical_passed'] != data[key]['numerical_passed']):
                raise ValueError('Incomplete graph contrast replay')
        if key in ('wldn_preflight', 'cgr_preflight'):
            if (audit['additional_artificial_update_replays'] != 3
                    or audit['final_checkpoint_max_error'] != 0
                    or not audit['numerical_passed'] or not data[key]['numerical_passed']
                    or audit['candidate_checks'] != data[key]['candidate_checks']
                    or data[key]['new_development_evaluations'] != 0):
                raise ValueError('Incomplete graph-difference baseline preflight replay')
        if key == 'path_baseline':
            if audit['training_updates_checked'] != 4608 or audit['gate_recomputed'] != data[key]['gate_checks']:
                raise ValueError('Incomplete path baseline replay')
        if key in ('wldn_quality', 'cgr_quality'):
            if (audit['training_updates_checked'] != 4608 or audit['max_replay_nll_error'] != 0
                    or audit['new_head_predictions_replayed'] != 12288
                    or audit['copied_reference_predictions_replayed'] != 12288
                    or not audit['comparison_pending']):
                raise ValueError('Incomplete graph-difference quality replay')
        if key == 'graph_quality':
            if (audit['training_updates_checked'] != 18432 or audit['max_replay_nll_error'] != 0
                    or audit['gate_recomputed'] != data[key]['gate_checks']
                    or audit['bootstrap_intervals_recomputed'] != 10):
                raise ValueError('Incomplete scalar graph-contrast quality replay')
        if key == 'wldn_interventions':
            if (audit['new_predictions_replayed'] != 73728 or audit['copied_base_predictions_replayed'] != 12288
                    or audit['original_native_predictions_checked'] != 12288
                    or audit['max_replay_nll_error'] > 1e-5 or audit['native_max_nll_error'] > 1e-5
                    or audit['bootstrap_intervals_recomputed'] != 10 or not audit['head_states_unchanged']
                    or not audit['temporary_hooks_removed']
                    or audit['auditor_sha256'] != sha(ROOT/'scripts/chess_wldn_interventions.py')):
                raise ValueError('Incomplete WLDN fixed-weight intervention replay')
        if key in ('union_quality', 'union_cost'):
            study_name = 'quality' if key == 'union_quality' else 'cost'
            plan_path = ROOT/f'evidence/chess-union-difference-{study_name}-v1/protocol/plan.json'
            plan = read(plan_path)
            if (audit['plan_sha256'] != sha(plan_path) or data[key]['plan_sha256'] != sha(plan_path)
                    or any(sha(ROOT/p) != h for p,h in plan['sources'].items())):
                raise ValueError('Union quality/cost source identity changed')
        if key == 'union_quality':
            if (audit['training_updates_checked'] != 18432 or not audit['fresh_initial_states_exact']
                    or audit['new_head_predictions_replayed'] != 61440
                    or audit['copied_reference_predictions_replayed'] != 24576
                    or audit['max_replay_nll_error'] > 1e-5 or audit['bootstrap_intervals_recomputed'] != 10
                    or audit['gate_recomputed'] != data[key]['gate_checks']
                    or audit['auditor_sha256'] != sha(ROOT/'scripts/chess_union_difference_study.py')
                    or not 0 < data[key]['wall_seconds'] <= 14400 or not 0 < audit['wall_seconds'] <= 1800):
                raise ValueError('Incomplete union quality replay')
        if key == 'union_cost':
            if (audit['timing_records_checked'] != 1728
                    or audit['auditor_sha256'] != sha(ROOT/'scripts/chess_union_difference_cost.py')
                    or not 0 < data[key]['wall_seconds'] <= 900 or not 0 < audit['wall_seconds'] <= 900
                    or data[key]['inputs']['evidence/chess-union-difference-quality-v1/audit/receipt.json']
                       != sha(ROOT/'evidence/chess-union-difference-quality-v1/audit/receipt.json')):
                raise ValueError('Incomplete union native-cost audit')
        if key == 'union_preflight':
            if (audit['candidate_checks'] != 2508 or audit['additional_artificial_update_replays'] != 12
                    or audit['final_checkpoint_max_error'] != 0 or not audit['independent_python_packing']
                    or not audit['numerical_passed'] or data[key]['new_development_evaluations'] != 0
                    or audit['auditor_sha256'] != sha(ROOT/'scripts/chess_union_difference_preflight.py')):
                raise ValueError('Incomplete union difference-layer engineering replay')
        if key == 'union_input':
            if (audit['roots_replayed'] != 36864 or not audit['all_role_digests_exact']
                    or not audit['all_control_counts_exact'] or data[key]['quality_predictions'] != 0
                    or audit['auditor_sha256'] != sha(ROOT/'scripts/chess_union_edit_input_audit.py')):
                raise ValueError('Incomplete full edit-control input replay')
        if key == 'union_consistent':
            plan_path = ROOT/'evidence/chess-union-consistent-preflight-v1/protocol/plan.json'
            plan = read(plan_path)
            if (audit['native_roots_reconstructed'] != 128 or audit['input_candidates'] != 3920
                    or audit['artificial_updates_replayed'] != 3 or audit['final_checkpoint_max_error'] != 0
                    or not audit['all_probe_digests_exact'] or not audit['input_counts_and_digests_exact']
                    or audit['reverse_inconsistent_candidates'] != 0 or audit['max_score_error'] > 1e-5
                    or data[key]['new_quality_predictions'] != 0
                    or audit['plan_sha256'] != sha(plan_path) or data[key]['plan_sha256'] != sha(plan_path)
                    or any(sha(ROOT/p) != h for p,h in plan['sources'].items())
                    or audit['auditor_sha256'] != sha(ROOT/'scripts/chess_union_consistent_preflight.py')):
                raise ValueError('Incomplete reverse-consistent control engineering replay')
        if key == 'graph_baselines':
            if (audit['combined_gate_checks'] != 14 or audit['bootstrap_intervals_recomputed'] != 6
                    or audit['timing_records_checked'] != 576
                    or audit['combined_continuation_passed'] != data[key]['combined_continuation_passed']):
                raise ValueError('Incomplete combined graph-baseline comparison')
        if key == 'normalization':
            if audit['max_numeric_replay_error'] != 0 or any(data[key][k] != v for k, v in audit['recomputed'].items()):
                raise ValueError('Incomplete normalization input replay')
        if key == 'attack_edits':
            if audit['roots_replayed'] != 36864 or sum(r['counts']['candidates'] for r in data[key]['splits'].values()) != audit['candidate_replays']:
                raise ValueError('Incomplete attack edit input replay')
        followup_audits[key] = {'path':path,'sha256':sha(ROOT/path)}
    cache_audit_path = 'evidence/chess-child-graph-cache-v1/audit/receipt.json'
    cache_audit = read(ROOT/cache_audit_path)
    if (cache_audit['status'] != 'completed'
            or cache_audit['summary_sha256'] != sha(ROOT/sources['graph_cache']/'summary.json')
            or cache_audit['execution_receipt_sha256'] != sha(ROOT/sources['graph_cache']/'completed.json')
            or cache_audit['reconstructed'] != {s:{k:v[k] for k in ('roots','children')} for s,v in data['graph_cache']['splits'].items()}):
        raise ValueError('Native graph-cache audit does not cover the complete cache')
    followup_audits['graph_cache'] = {'path':cache_audit_path,'sha256':sha(ROOT/cache_audit_path)}
    dependency_path = 'evidence/chess-panel-dependencies-v1/audit/receipt.json'
    dependency = read(ROOT/dependency_path)
    if dependency['status'] != 'completed' or not dependency['original_criteria_unchanged']:
        raise ValueError('Incomplete dependency audit')
    for path,digest in {**dependency['inputs_sha256'],**dependency['sources_sha256']}.items():
        if sha(ROOT/path) != digest: raise ValueError('Dependency audit member changed')
    followup_audits['dependencies']={'path':dependency_path,'sha256':sha(ROOT/dependency_path)}
    engine_baselines = data['baseline_regret']
    counterfactual = data['counterfactual']
    graph_contrast = data['graph_contrast']
    shared_contrast = data['graph_contrast_shared']
    path_means = {arm: {split: {metric: statistics.mean(data['path_baseline']['metrics'][f'{arm}-{seed}'][split][metric]
                   for seed in (97, 109, 127)) for metric in ('agreement', 'target_nll')}
                   for split in ('dev', 'shift')} for arm in ('base', 'transport', 'path_pna')}
    path_rows = [[label, *(f"{100*path_means[arm][split]['agreement']:.2f}" for split in ('dev', 'shift')),
                  *(f"{path_means[arm][split]['target_nll']:.3f}" for split in ('dev', 'shift'))]
                 for arm, label in [('base', 'Frozen direct (copied)'), ('transport', 'Transport (copied)'), ('path_pna', 'NBFNet-style path head')]]
    table(OUT/'path-baseline-table.tex', 'Separately frozen path-baseline comparison. Agreement is percent; NLL is lower-is-better. '
          'All three paired seeds retained; original references copied with authenticated provenance.\\label{tab:path-baseline}',
          'lrrrr', 'Head & Agree., ordinary & Agree., shifted & NLL, ordinary & NLL, shifted', path_rows)
    path_table = OUT/'path-baseline-table.tex'
    path_table.write_text(path_table.read_text().replace('[t]', '[ht]', 1))
    path_latency = read(ROOT/followup_audits['path_baseline']['path'])['median_complete_latency_ms']
    wldn_means = {arm: {split: {metric: statistics.mean(data['wldn_quality']['metrics'][f'{arm}-{seed}'][split][metric]
                   for seed in (97, 109, 127)) for metric in ('agreement', 'target_nll')}
                   for split in ('dev', 'shift')} for arm in ('base', 'wldn')}
    wldn_latency = read(ROOT/followup_audits['wldn_quality']['path'])['median_complete_latency_ms']
    graph_means = {arm: {split: {metric: statistics.mean(data['graph_quality']['metrics'][f'{arm}-{seed}'][split][metric]
                   for seed in (97, 109, 127)) for metric in ('agreement', 'target_nll')}
                   for split in ('dev', 'shift')} for arm in ('base', 'root', 'child', 'contrast', 'permuted_contrast', 'contrast_corrupted')}
    table(OUT/'graph-contrast-table.tex', 'Completed scalar-contrast study on ordinary (O) and shifted (S) panels. Agreement is percent; NLL is lower-is-better. '
          'All three seeds retained. Corrupted contrast reuses fitted contrast weights; it is a diagnostic, not a new fit.\\label{tab:graph-contrast}',
          'lrrrr', 'Head & Agree. O & Agree. S & NLL O & NLL S',
          [[label, *(f"{100*graph_means[arm][s]['agreement']:.2f}" for s in ('dev', 'shift')),
            *(f"{graph_means[arm][s]['target_nll']:.3f}" for s in ('dev', 'shift'))]
           for arm, label in [('base','Frozen base'), ('root','Root only'), ('child','Child only'), ('contrast','Scalar contrast'),
                              ('permuted_contrast','Trained shuffled contrast'), ('contrast_corrupted','Test-time shuffled contrast')]])
    combined = data['graph_baselines']; combined_means = combined['means']
    table(OUT/'graph-baselines-table.tex', 'Identity-joined graph-baseline comparison on the same ordinary (O) and shifted (S) development panels. '
          'Agreement is percent; NLL is lower-is-better. Milliseconds are median complete native decisions in one balanced six-method process. '
          'Three paired seeds; similar parameter counts do not match inference compute.\\label{tab:graph-baselines}',
          'lrrrrr', 'Head & Agree. O & Agree. S & NLL O & NLL S & ms',
          [[label, *(f"{100*combined_means[arm][s]['agreement']:.2f}" for s in ('dev','shift')),
            *(f"{combined_means[arm][s]['target_nll']:.3f}" for s in ('dev','shift')),
            f"{combined['median_complete_latency_ms'][arm]:.3f}"]
           for arm,label in [('base','Frozen base'),('transport','Root transport'),('path_pna','NBFNet-style'),
                             ('contrast','Scalar contrast'),('wldn','Nodewise WLDN'),('cgr','Condensed graph')]])
    interventions = data['wldn_interventions']; intervention_means = interventions['panel_means']
    table(OUT/'wldn-interventions-table.tex', 'Fixed-weight WLDN pathway interventions on ordinary (O) and shifted (S) development panels. '
          'All three seeds retained. Agreement is percent; NLL is lower-is-better. These are interventions on trained heads, not separately trained architectures.\\label{tab:wldn-interventions}',
          'lrrrr', 'Intervention & Agree. O & Agree. S & NLL O & NLL S',
          [[label, *(f"{100*intervention_means[s][mode]['agreement']:.2f}" for s in ('dev','shift')),
            *(f"{intervention_means[s][mode]['target_nll']:.3f}" for s in ('dev','shift'))]
           for mode,label in [('base','Frozen base reference'),('native','Native WLDN'),('zero_delta','Zero node difference'),
                              ('root_diff_graph','Root graph at difference layer'),('zero_edge_flags','Zero difference-layer edge flags'),
                              ('zero_pooled','Zero pooled graph channels'),('permuted_delta','Shuffle node differences')]])
    for name in ('graph-contrast-table.tex', 'graph-baselines-table.tex', 'wldn-interventions-table.tex'):
        p = OUT/name; p.write_text(p.read_text().replace('[t]', '[ht]', 1))
    engine_rows = []
    for arm,label in [('base','Frozen direct'),('transport','Full transport'),('generic','Generic MLP'),
                      ('common_neighbor','NCN adaptation'),('typed_common_neighbor','Typed overlap')]:
        engine_rows.append([label,*(f"{engine_baselines['means'][arm][split]['bounded_regret']:.5f}" for split in ('dev','shift')),
                            *(f"{engine_baselines['means'][arm][split]['cp_loss']:.1f}" for split in ('dev','shift'))])
    table(OUT/'baseline-regret-table.tex', 'Post-hoc engine assessment of all fifteen simpler-head policies on the same fixed 128 positions per panel. '
          'All calls are fresh and request 20,000 nodes. Both loss measures are lower-is-better; raw centipawn loss includes an artificial mate-score encoding.',
          'lrrrr', 'Head & Bounded, ordinary & Bounded, shifted & CP, ordinary & CP, shifted', engine_rows)
    appendix_table = OUT/'baseline-regret-table.tex'
    appendix_table.write_text(appendix_table.read_text().replace('[t]','[ht]',1))
    baselines = data['relation_baselines']
    baseline_audit_path = ROOT/'evidence/chess-relation-baselines-v1/audit/receipt.json'
    baseline_audit = read(baseline_audit_path)
    if (baseline_audit['status'] != 'completed'
            or baseline_audit['summary_sha256'] != sha(ROOT/sources['relation_baselines']/'summary.json')
            or baseline_audit['execution_receipt_sha256'] != sha(ROOT/sources['relation_baselines']/'completed.json')
            or baseline_audit['evaluation_predictions_replayed'] != 61440
            or baseline_audit['gate_recomputed'] != baselines['gate_checks']):
        raise ValueError('Baseline audit does not bind all comparisons')
    baseline_means = {arm:{split:statistics.mean(baselines['metrics'][f'{arm}-{seed}'][split]['agreement']
                                               for seed in (97,109,127)) for split in ('dev','shift')}
                      for arm in ('base','transport','generic','common_neighbor','typed_common_neighbor')}
    baseline_rows = []
    for arm,label,count in [('base','Frozen direct',0),('transport','Full transport',16744),
                            ('generic','Generic MLP',16240),('common_neighbor','NCN adaptation',16380),
                            ('typed_common_neighbor','Typed overlap',16144)]:
        baseline_rows.append([label,f'{count:,}',*(f"{100*baseline_means[arm][split]:.2f}" for split in ('dev','shift'))])
    table(OUT/'relation-baselines-table.tex', 'Additional fixed-backbone head comparison, using all three seeds and the same old development panels. '
          'Agreement is percent. The first two rows are authenticated earlier results; three new heads train with the same examples and update budget. '
          'Head counts exclude the shared backbone.',
          'lrrr', 'Head & Head parameters & Agree., ordinary & Agree., shifted', baseline_rows)
    batch = data['batch']
    transport = data['transport']
    transport_regret = data['transport_regret']
    regret_audit_path = ROOT/'evidence/chess-transport-regret-v1/audit/receipt.json'
    regret_audit = read(regret_audit_path)
    if (regret_audit['status'] != 'completed'
            or regret_audit['summary_sha256'] != sha(ROOT/sources['transport_regret']/'summary.json')
            or regret_audit['policy_position_records'] != 4608
            or regret_audit['primary_gate_changed']):
        raise ValueError('Transport engine audit does not bind the full result')
    transport_audit_path = ROOT/'evidence/chess-transport-v1/audit/receipt.json'
    transport_audit = read(transport_audit_path)
    if (transport_audit['status'] != 'completed'
            or transport_audit['summary_sha256'] != sha(ROOT/sources['transport']/'summary.json')
            or transport_audit['execution_receipt_sha256'] != sha(ROOT/sources['transport']/'completed.json')
            or transport_audit['evaluation_predictions_replayed'] != 86016
            or transport_audit['gate_recomputed'] != transport['gate_checks']):
        raise ValueError('Transport prediction audit does not bind this completed result')
    transport_means = {arm:{split:{metric:statistics.mean(transport['metrics'][f'{arm}-{seed}'][split][metric]
                                                        for seed in (97,109,127))
                                  for metric in ('agreement','target_nll')}
                            for split in ('dev','shift')}
                       for arm in ('base','transport','no_overlap','uniform','static','rewired')}
    transport_rows = []
    for arm,label in [('base','Frozen direct'),('transport','Full transport'),('no_overlap','No overlap'),
                      ('uniform','Uniform routing'),('static','Static endpoints'),('rewired','Rewired relations')]:
        transport_rows.append([label,*(f"{100*transport_means[arm][s]['agreement']:.2f}" for s in ('dev','shift')),
                               *(f"{transport_means[arm][s]['target_nll']:.3f}" for s in ('dev','shift'))])
    table(OUT/'transport-table.tex', 'Transport-head experiment on the existing candidate panels. '
          'All three paired backbones retained. Agreement is percent; target negative log likelihood is lower-is-better. '
          'These are development teacher-imitation measurements, not engine grading or game scores.',
          'lrrrr', 'Head & Agree., ordinary & Agree., shifted & NLL, ordinary & NLL, shifted', transport_rows)
    regret_rows=[]
    for arm,label in [('base','Frozen direct'),('transport','Full transport'),('no_overlap','No overlap'),
                      ('uniform','Uniform routing'),('static','Static endpoints'),('rewired','Rewired relations')]:
        regret_rows.append([label,*(f"{transport_regret['means'][arm][s]['bounded_regret']:.5f}" for s in ('dev','shift')),
                            *(f"{transport_regret['means'][arm][s]['cp_loss']:.1f}" for s in ('dev','shift'))])
    table(OUT/'transport-regret-table.tex', 'Post-hoc engine characterization, frozen after the primary transport gate failed. '
          'All eighteen policies use the original 128-position subsets per panel and fresh 20,000-node calls. '
          'Both signed bounded loss and raw centipawn loss are lower-is-better; finite-search differences can be negative.',
          'lrrrr', 'Head & Bounded, ordinary & Bounded, shifted & CP, ordinary & CP, shifted', regret_rows)
    candidate_rows = []
    for arm, label in [('direct', 'Direct'), ('action_only', 'Action only'), ('delta', 'Exact input delta'),
                       ('full_afterstate', 'Full afterstate')]:
        losses = [statistics.mean(c['regret'][f'{arm}-{seed}'][s] for seed in (97,109,127)) for s in ('dev','shift')]
        candidate_rows.append([label, *(f"{100*c['means'][arm][s]['agreement']:.2f}" for s in ('dev','shift')),
                               *(f'{x:.5f}' for x in losses)])
    table(OUT/'candidate-table.tex',
          'Candidate experiment. All three seeds averaged within each arm. Agreement is percent; '
          'bounded engine loss is lower-is-better. These panels differ from Table~2.',
          'lrrrr', 'Arm & Agree., ordinary & Agree., shifted & Loss, ordinary & Loss, shifted', candidate_rows)
    bio_rows = []
    for arm,label in [('direct','Frozen direct'),('biological','Biological'),('rewire151','Rewire 151'),
                      ('rewire163','Rewire 163'),('rewire179','Rewire 179'),('dense','Dense'),('node_local','Node-local')]:
        bio_rows.append([label, *(f"{b['topology_comparison']['means'][arm][s]:.5f}" for s in ('dev','shift')),
                        *(f"{100*statistics.mean(b['metrics'][f'{arm}-{seed}'][s]['agreement'] for seed in (97,109,127)):.2f}" for s in ('dev','shift'))])
    table(OUT/'biology-table.tex', 'Biological-topology experiment. Every arm and all three seeds retained. '
          'Loss is a signed bounded finite-engine score difference; agreement is percent.',
          'lrrrr', 'Arm & Loss, ordinary & Loss, shifted & Agree., ordinary & Agree., shifted', bio_rows)
    local_rows = []
    for arm in ('factorized','broadcast'):
        for depth in (2,4,8):
            r=[r for r in l['aggregates'] if r['arm']==arm and r['depth']==depth]
            local_rows.append([arm.capitalize(),str(depth),f"{100*statistics.mean(x['node_update_ratio'] for x in r):.2f}",
                               f"{max(x['max_hidden_error'] for x in r):.2e}"])
    table(OUT/'locality-table.tex', 'Untrained locality screen. Logical recurrent-node updates include '
          'root caching, as a percentage of full recomputation. Errors are maxima over all candidates '
          'and three seeds, not measures of chess quality.',
          'llrr', 'Architecture & Depth & Node updates (\\%) & Max. hidden error', local_rows)
    bio_pass = b['topology_comparison']['continuation_passed']
    checks = b['topology_comparison']['checks']
    bio_reductions = [x['reduction']*100 for x in checks if x['criterion']=='mean_relative_reduction']
    union_quality, union_cost = data['union_quality'], data['union_cost']
    union_means = {arm: {split: {metric: statistics.mean(union_quality['metrics'][f'{arm}-{seed}'][split][metric]
                    for seed in (97,109,127)) for metric in ('agreement', 'target_nll')}
                    for split in ('dev','shift')} for arm in ('base','wldn','child','union','edits','rotated','edits_corrupted')}
    union_rows = []
    for arm,label in [('base','Frozen direct'),('wldn','Original WLDN'),('child','Matched child'),
                      ('union','Union'),('edits','Edit labels'),('rotated','Rotated labels'),
                      ('edits_corrupted','Edits, corrupted at test')]:
        ratio = '--' if arm == 'edits_corrupted' else f"{union_cost['median_paired_ratio_to_wldn'][arm]:.3f}"
        union_rows.append([label, *(f"{100*union_means[arm][split]['agreement']:.2f}" for split in ('dev','shift')),
                           *(f"{union_means[arm][split]['target_nll']:.3f}" for split in ('dev','shift')), ratio])
    table(OUT/'union-difference-table.tex', 'Union difference-layer development comparison. All three paired seeds retained. '
          'Agreement is percent; NLL is lower-is-better. Time is the median paired complete-native decision ratio to WLDN '
          'on sixteen fixed ordinary roots, with six rotating repeats. The test-time corruption reuses edits weights and has no separate timing. '
          r'\label{tab:union-difference}', 'lrrrrr',
          'Policy & Agree., ord. & Agree., shift & NLL, ord. & NLL, shift & Time / WLDN', union_rows)
    followup_union_error = read(ROOT/followup_audits['union_quality']['path'])['max_replay_nll_error']
    union_intervals = {r['split']: [100*x for x in r['percentile95']]
                       for r in union_quality['conditional_game_bootstrap'] if r['comparator'] == 'wldn'}
    union_replay_error_text = 'zero' if followup_union_error == 0 else '$'+scientific(followup_union_error)+'$'
    macro = {
        'UnionDifferenceQuality':
            f"All twelve fits and 18,432 updates complete; {sum(r['passed'] for r in union_quality['gate_checks'])} of ten frozen checks pass, "
            f"so the all-check criterion {'passes' if union_quality['continuation_passed'] else 'fails'} (Table~\\ref{{tab:union-difference}}). "
            f"Edits minus original WLDN agreement is {100*(union_means['edits']['dev']['agreement']-union_means['wldn']['dev']['agreement']):+.2f} "
            f"points ordinary and {100*(union_means['edits']['shift']['agreement']-union_means['wldn']['shift']['agreement']):+.2f} shifted, against a required one-point mean advantage on each panel. "
            'All 86,016 new/copied predictions replay, all minibatch-index receipts are checked, and all ten conditional source-game bootstrap intervals are recomputed. '
            f"The maximum replay NLL error is {union_replay_error_text}. "
            f"The descriptive 95\\% intervals for edits minus WLDN are [{union_intervals['dev'][0]:+.2f}, {union_intervals['dev'][1]:+.2f}] and "
            f"[{union_intervals['shift'][0]:+.2f}, {union_intervals['shift'][1]:+.2f}] points; both include zero. "
            'These are adaptive teacher-imitation results on already exposed panels, not independent confirmation or measured game strength.',
        'UnionDifferenceCost':
            f"Complete-native timing retains 1,728 decisions; all 288 distinct seed/root/method choices replay against the authenticated panel predictions. "
            f"The edits/WLDN median paired ratio is {union_cost['median_paired_ratio_to_wldn']['edits']:.3f}. "
            'This descriptive same-process, shared-host measurement includes board parsing, legal menus, backbone evaluation, native child graphs and the head. '
            'It does not establish a hardware-general speed advantage or equal training/inference compute.',

        'UnionEditControlInput':
            'A separate input-only audit and independent NumPy replay cover all 36,864 cached roots and 1,087,523 candidates; the earlier native cache audit is authenticated, not rerun. '
            f"The rotation changes every candidate and relabels {100*data['union_input']['totals']['true_edits_relabelled']/data['union_input']['totals']['true_edit_instances']:.2f}\\% of added/removed direction/color instances. "
            f"It also breaks reverse-direction class consistency in {100*data['union_input']['totals']['reverse_inconsistent_candidates']/data['union_input']['totals']['candidates']:.2f}\\% of candidates. "
            'Thus this control is a broader corruption test: a performance difference cannot be attributed uniquely to correct edit placement. A reverse-consistent control would be required for that narrower claim.',
        'UnionDifferenceEngineering':
            'The four-arm engineering screen and audit cover 96 root/seed/arm checks and 2,508 candidate scores, including independent Python graph packing. '
            f"The maximum batched-versus-individual score discrepancy is ${scientific(data['union_preflight']['max_score_error'])}$ under a frozen $10^{{-5}}$ tolerance. "
            'All twelve artificial updates and their final states replay exactly. These use first-legal-action targets and establish no chess-quality result.',
        'UnionConsistentEngineering':
            'A separate implemented control rotates raw directed/color classes, then derives incoming flags by transposition. '
            'It preserves union support, per-candidate/channel class counts and reverse consistency. '
            'A bounded screen reconstructs 128 native training roots and independently checks all 3,920 candidates: every candidate changes, with zero reverse inconsistencies. '
            f"There are 627 numerical score checks with maximum discrepancy ${scientific(max(r['max_score_error'] for r in data['union_consistent']['probes']))}$ against a frozen $10^{{-5}}$ tolerance. "
            'The original head with independently packed flags replays all three artificial updates and the final checkpoint exactly. '
            'This control has no trained quality result; its artificial labels need not describe a legal chess successor. The original twelve-fit study retains its declared corrupted-label control.',
        'GraphQualityResult':
            f"The completed twelve-fit study retains 18,432 updates and fails its original criterion: {sum(r['passed'] for r in data['graph_quality']['gate_checks'])} of eight checks pass (Table~\\ref{{tab:graph-contrast}}). "
            f"Contrast minus child-only agreement is {100*(graph_means['contrast']['dev']['agreement']-graph_means['child']['dev']['agreement']):+.2f} points ordinary and {100*(graph_means['contrast']['shift']['agreement']-graph_means['child']['shift']['agreement']):+.2f} shifted. "
            'It improves over the base and trained shuffled-child control but does not establish the required advantage over root-only or child-only scoring. '
            'All 73,728 saved predictions replay with zero NLL discrepancy; all ten descriptive bootstrap intervals are recomputed. Root-only checkpoints exactly reproduce original transport for all three seeds.',
        'GraphBaselineResult':
            f"The combined fourteen-check criterion also fails: {sum(r['passed'] for r in combined['gate_checks'])} checks pass. "
            f"WLDN exceeds scalar contrast by {100*(combined_means['wldn']['dev']['agreement']-combined_means['contrast']['dev']['agreement']):.2f} and {100*(combined_means['wldn']['shift']['agreement']-combined_means['contrast']['shift']['agreement']):.2f} agreement points on the two panels. "
            'The audit authenticates all four model replay audits, joins all 73,728 distinct method/seed/position records, retains all earlier failed checks, and recomputes six source-game intervals. '
            'These descriptive development comparisons favor an established nodewise-difference adaptation over the proposed scalar contrast. They provide no independent confirmation, engine-strength result or new algorithmic contribution.',
        'NormalizationInputResult':
            f"A separate input-only screen over {data['normalization']['roots']} old training roots finds nonzero inverse-channel mixed differences from separate normalization on {data['normalization']['separate_inverse_nonzero']:,} of {data['normalization']['candidates']:,} legal candidates; a fixed untrained linear projection detects {data['normalization']['separate_linear_probe_nonzero']:,}. "
            'Independent native-graph and NumPy reconstruction replays every value exactly. Thus an interaction response can arise without learned nonlinear scoring. '
            'A common union denominator with diagonal slack cancels the single-step mixed difference to numerical precision, but changes the information path and computational cost and is not a demonstrated policy improvement.',
        'WLDNInterventionResult':
            'All 73,728 intervention/native predictions and 12,288 copied base references replay through the original head with temporary hooks; the native predictions reproduce the original study. '
            f"Zeroing encoded node differences reduces mean agreement by {100*(intervention_means['dev']['native']['agreement']-intervention_means['dev']['zero_delta']['agreement']):.2f} / {100*(intervention_means['shift']['native']['agreement']-intervention_means['shift']['zero_delta']['agreement']):.2f} points on ordinary/shifted panels. "
            f"Shuffling those differences within each legal menu reduces it by {100*(intervention_means['dev']['native']['agreement']-intervention_means['dev']['permuted_delta']['agreement']):.2f} / {100*(intervention_means['shift']['native']['agreement']-intervention_means['shift']['permuted_delta']['agreement']):.2f} points. "
            'Every native-minus-intervention agreement interval is positive under the declared source-game bootstrap; these ten descriptive intervals do not correct for adaptive development or multiple comparisons.',
        'WLDNQualityResult':
            'The WLDN study completes all three final fits and 4,608 updates; all 24,576 new/copied predictions replay with zero NLL discrepancy. '
            f"Mean teacher agreement is {100*wldn_means['wldn']['dev']['agreement']:.2f}\\% ordinary and {100*wldn_means['wldn']['shift']['agreement']:.2f}\\% shifted, versus {100*wldn_means['base']['dev']['agreement']:.2f}\\% and {100*wldn_means['base']['shift']['agreement']:.2f}\\% for the frozen base. "
            f"Mean WLDN NLL is {wldn_means['wldn']['dev']['target_nll']:.3f} and {wldn_means['wldn']['shift']['target_nll']:.3f}. "
            'These are descriptive development results for an established-method adaptation; they do not establish a new architecture, engine strength or equal inference compute.',
        'PathBaselineResult':
            f"A separately frozen NBFNet-style adaptation exceeds transport by {100*(path_means['path_pna']['dev']['agreement']-path_means['transport']['dev']['agreement']):.2f} and {100*(path_means['path_pna']['shift']['agreement']-path_means['transport']['shift']['agreement']):.2f} agreement points on ordinary and shifted panels (Table~\\ref{{tab:path-baseline}}). "
            f"Transport passes {sum(c['passed'] for c in data['path_baseline']['gate_checks'])} of its two additional advantage checks; all 36,864 baseline/reference predictions replay.",
        'PathBaselineLatency':
            f"Paired complete-decision medians are {path_latency['base']:.3f}, {path_latency['transport']:.3f} and {path_latency['path_pna']:.3f} ms for the backbone, transport and path head, respectively. "
            f"The path head takes {path_latency['path_pna']/path_latency['transport']:.2f} times transport's median in this rotating shared-host measurement; the quality comparison is not a matched-FLOP or matched-latency result.",
        'AttackEditInputResult':
            'An input-only audit and independent replay cover all 1,087,523 cached legal candidates. Every candidate changes the graph and vacates a nonempty own-attack source row; no candidates within a root have identical child graphs. '
            f"Stationary pieces change attack lines for {100*data['attack_edits']['splits']['train']['stationary_candidate_fraction']:.2f}\\%, {100*data['attack_edits']['splits']['dev']['stationary_candidate_fraction']:.2f}\\% and {100*data['attack_edits']['splits']['shift']['stationary_candidate_fraction']:.2f}\\% of training, ordinary and shifted candidates, respectively. "
            'Thus the exact-zero-change invariant is synthetic on these panels, and a child-assignment shuffle can disrupt simple move/source consistency as well as tactical relations. No fitted-model reliance on either signal is established.',
        'DifferenceBaselineEngineering':
            'Each baseline passes a separate native-input screen over 24 root/seed pairs and 627 candidates. '
            f"Maximum separate-candidate versus batched score errors are ${scientific(data['wldn_preflight']['max_score_error'])}$ for WLDN and ${scientific(data['cgr_preflight']['max_score_error'])}$ for the condensed-graph adapter, "
            'below their predeclared $10^{-5}$ tolerance. Each preserves the initial backbone exactly and replays three artificial-target updates to an identical checkpoint. '
            'These tolerances do not change earlier failed screens. No teacher-agreement, engine or gameplay result is established by these preflights.',
        'GraphContrastSharedResults':
            f"A separately frozen implementation shares root propagation and retains compact legal-child operators. Its {shared_contrast['root_seed_checks']} root/seed and {shared_contrast['candidate_checks']:,} candidate checks pass with maximum discrepancy ${scientific(max(shared_contrast['max_errors'].values()))}$. "
            'Exact zero-change behavior and twelve additional artificial-update checkpoint replays also pass. '
            f"A lossless bit cache covers {sum(v['roots'] for v in data['graph_cache']['splits'].values()):,} existing roots and {sum(v['children'] for v in data['graph_cache']['splits'].values()):,} legal children; every stored graph is independently reconstructed. "
            'These engineering checks establish neither trained quality nor an inference-speed advantage.',
        'GraphContrastResults': f"The numerical screen {'passes' if graph_contrast['numerical_passed'] else 'fails'} over {graph_contrast['root_seed_checks']} root/seed checks and {graph_contrast['candidate_checks']:,} candidate checks. "
                                f"Maximum root, child and score-difference discrepancies are each at most ${scientific(max(graph_contrast['max_errors'].values()))}$ against a fixed $2\\times10^{{-6}}$ tolerance. "
                                'Identical operators produce exactly zero correction and parameter gradient. '
                                'Twelve artificial-target optimizer updates check gradient flow and cost; their complete checkpoint replay is exact. '
                                'Those twelve audit updates are additional computation. No development teacher agreement, engine score or game result is measured for this head.',
        'JointResults': f"The joint-training criterion {'passes' if joint['continuation_passed'] else 'fails'}: "
                        f"{sum(c['passed'] for c in joint['gate_checks'])} of twelve checks pass. "
                        f"Joint transport minus jointly trained direct agreement is "
                        f"{100*(joint_means['transport']['dev']['agreement']-joint_means['direct']['dev']['agreement']):+.2f} "
                        f"and {100*(joint_means['transport']['shift']['agreement']-joint_means['direct']['shift']['agreement']):+.2f} percentage points on ordinary and shifted panels. "
                        f"Relative to the preceding frozen-representation transport-minus-base margin, this changes the margin by "
                        f"{100*joint['representation_interaction'][0]['difference_of_differences']:+.2f} "
                        f"and {100*joint['representation_interaction'][1]['difference_of_differences']:+.2f} points. "
                        'The audit checks all 27,648 training records and replays all 98,304 evaluation predictions, including graph corruption. '
                        'Encoder and policy weights change while unused auxiliary/value/projection parameters remain fixed. '
                        'This matched-update optimization comparison does not establish a new transport mechanism, fresh confirmation or gameplay improvement.',
        'BaselineEngineResults': f"The new assessment uses {engine_baselines['cost']['calls']} fresh calls and {engine_baselines['cost']['requested_nodes']:,} requested nodes. "
                                 'Full transport has lower mean bounded loss than each simpler head on both panels. '
                                 'Raw centipawn loss favors typed overlap on both panels and untyped overlap on the ordinary panel. '
                                 'Thus the mean bounded-loss ranking survives these simpler controls, but there is no uniform ranking across endpoints. '
                                 'The audit checks all 3,840 policy-position records, and fresh backbone/transport means exactly match the preceding engine assessment. '
                                 'No new neural inference or gameplay is performed; this post-hoc extension cannot replace either failed primary criterion.',
        'CounterfactualResults': f"The untrained screen covers {counterfactual['candidate_step_comparisons']:,} candidate/step comparisons over 32 old training roots and three seeds. "
                                 f"Maximum distribution error is ${scientific(counterfactual['max_distribution_error'])}$, but maximum readout error is ${scientific(counterfactual['max_score_error'])}$, "
                                 'above the fixed absolute tolerance of $2\\times10^{-6}$. '
                                 f"Median sparse/dense operator-storage ratio is {counterfactual['median_sparse_dense_storage_ratio']:.4f}, with lower storage on every root; this is not peak memory. "
                                 f"Median paired complete-pipeline latency ratio is {counterfactual['median_paired_sparse_dense_latency_ratio']:.3f}, failing the required ratio of at most 0.90. "
                                 'The original numerical and combined engineering criteria fail. All output/storage records replay; timings are authenticated rather than repeated.',
        'RelationBaselineResults': f"The additional transport-advantage rule {'passes' if baselines['transport_advantage_passed'] else 'fails'}: "
                                   f"{sum(c['passed'] for c in baselines['gate_checks'])} of six comparisons pass. "
                                   f"Transport minus generic-head agreement is "
                                   f"{100*(baseline_means['transport']['dev']-baseline_means['generic']['dev']):+.2f} "
                                   f"and {100*(baseline_means['transport']['shift']-baseline_means['generic']['shift']):+.2f} percentage points on the two panels. "
                                   'The audit replays all 36,864 new baseline outputs and 24,576 copied reference outputs. '
                                   'Every paired-seed agreement difference favors transport, but the fixed minimum effect is not reached. This comparison does not change the earlier failed criterion and is not fresh confirmation.',
        'LocalSavings': f"{100*(1-max(x['node_update_ratio'] for x in l['aggregates'] if x['arm']=='factorized')):.1f}--"
                        f"{100*(1-min(x['node_update_ratio'] for x in l['aggregates'] if x['arm']=='factorized')):.1f}",
        'BioAbstract': 'The biological topology '+('passes' if bio_pass else 'fails')+' its frozen continuation criterion.',
        'LocalAbstract': f"The untrained screen agrees with full recomputation within $2\\times10^{{-6}}$ while reducing logical recurrent-node updates by "
                         f"{100*(1-max(x['node_update_ratio'] for x in l['aggregates'] if x['arm']=='factorized')):.1f}--"
                         f"{100*(1-min(x['node_update_ratio'] for x in l['aggregates'] if x['arm']=='factorized')):.1f}\\%. "
                         f"Its median paired complete-latency ratio is {l['median_paired_latency_ratio']['factorized']:.3f} relative to full recomputation, so the combined engineering criterion "
                         +('passes.' if l['engineering_gate_passed'] else 'fails.'),
        'BatchAbstract': f"A separately frozen batched baseline is numerically equivalent and makes the original incremental path "
                         f"{batch['median_paired_latency_ratios']['factorized']['incremental_over_batched']:.2f} times slower in a paired complete-loop comparison.",
        'DeltaDevReduction': f"{100*c['checks'][0]['observed']:.2f}",
        'DeltaShiftWorsening': f"{-100*c['checks'][1]['observed']:.2f}",
        'DeltaDirectPoints': f"{100*c['arena']['by_opponent']['direct']['delta']['score_lower_bound']:.2f}",
        'DeltaActionPoints': f"{100*c['arena']['by_opponent']['action_only']['delta']['score_lower_bound']:.2f}",
        'TransferMin': f"{100*min(v['agreement'] for v in t['means'].values()):.2f}",
        'TransferMax': f"{100*max(v['agreement'] for v in t['means'].values()):.2f}",
        'BioResults': f"The original criterion {'passes' if bio_pass else 'fails'}: {sum(x['passed'] for x in checks)} of {len(checks)} constituent checks pass. "
                      f"Biological mean loss reductions against individual rewires range from {min(bio_reductions):+.2f}\\% to {max(bio_reductions):+.2f}\\% across the two panels. "
                      'These are descriptive paired comparisons, not a population-level significance test.',
        'LocalResults': f"The median paired incremental/full latency ratio is {l['median_paired_latency_ratio']['factorized']:.3f} for the factorized network "
                        f"and {l['median_paired_latency_ratio']['broadcast']:.3f} for the broadcast control. "
                        f"The combined engineering criterion {'passes' if l['engineering_gate_passed'] else 'fails'}. "
                        'No training or game-quality evaluation is performed. Logical savings and numerical equality do not establish a useful learned policy.',
        'BatchResults': f"All 36,594 batched/single numerical comparisons pass the same tolerance; the largest hidden discrepancy is "
                        f"${scientific(max(g['max_hidden_error'] for g in batch['groups']))}$ and the largest score discrepancy is "
                        f"${scientific(max(g['max_score_error'] for g in batch['groups']))}$. "
                        f"Median paired batched/full latency ratios are {batch['median_paired_latency_ratios']['factorized']['batched_over_full']:.3f} "
                        f"(factorized) and {batch['median_paired_latency_ratios']['broadcast']['batched_over_full']:.3f} (broadcast). "
                        f"The original incremental implementation takes {batch['median_paired_latency_ratios']['factorized']['incremental_over_batched']:.3f} "
                        f"and {batch['median_paired_latency_ratios']['broadcast']['incremental_over_batched']:.3f} times the batched baseline, respectively. "
                        'Its separately fixed requirement of at least 10\\% lower latency fails. These shared-host development measurements establish no hardware-general speed ratio.',
        'TransportAbstract': 'A fifteen-fit comparison of two-endpoint relation transport against routing, overlap and topology controls '
                             +('passes' if transport['continuation_passed'] else 'fails')+' its development teacher-agreement criterion.',
        'TransportResults': f"The fixed continuation rule {'passes' if transport['continuation_passed'] else 'fails'}: "
                            f"{sum(c['passed'] for c in transport['gate_checks'])} of {len(transport['gate_checks'])} panel/comparator checks pass. "
                            f"Transport's mean agreement changes relative to the unchanged backbone are "
                            f"{100*(transport_means['transport']['dev']['agreement']-transport_means['base']['dev']['agreement']):+.2f} "
                            f"and {100*(transport_means['transport']['shift']['agreement']-transport_means['base']['shift']['agreement']):+.2f} percentage points on ordinary and shifted panels. "
                            f"Relative to the overlap ablation, they are "
                            f"{100*(transport_means['transport']['dev']['agreement']-transport_means['no_overlap']['dev']['agreement']):+.2f} "
                            f"and {100*(transport_means['transport']['shift']['agreement']-transport_means['no_overlap']['shift']['agreement']):+.2f} points. "
                            f"Local median complete-decision latencies are {transport_audit['median_complete_latency_ms']['base']:.3f} ms for the backbone "
                            f"and {transport_audit['median_complete_latency_ms']['transport']:.3f} ms for full transport. "
                            'The saved-record audit replays all 86,016 evaluation predictions, including test-time topology corruptions, and independently rebuilds the metric arithmetic and original criterion. '
                            'No training replay, game evaluation or untouched confirmation is performed; a separate exploratory engine assessment follows.',
        'TransportRegretResults': f"The separate assessment uses {transport_regret['cost']['calls']} fresh engine calls and "
                                  f"{transport_regret['cost']['requested_nodes']:,} requested nodes, with no new neural inference. "
                                  f"Full transport reduces mean bounded loss from the backbone by "
                                  f"{100*(1-transport_regret['means']['transport']['dev']['bounded_regret']/transport_regret['means']['base']['dev']['bounded_regret']):.2f}\\% on ordinary positions and "
                                  f"{100*(1-transport_regret['means']['transport']['shift']['bounded_regret']/transport_regret['means']['base']['shift']['bounded_regret']):.2f}\\% on shifted positions. "
                                  f"Raw centipawn loss, however, increases by "
                                  f"{100*(transport_regret['means']['transport']['shift']['cp_loss']/transport_regret['means']['base']['shift']['cp_loss']-1):.2f}\\% on the shifted panel. "
                                  'The rewired control has lower shifted bounded loss than full transport. The bounded transform saturates at extreme scores, while raw centipawn loss depends on the arbitrary mate-score encoding; these conflicting endpoints prevent a uniform quality-improvement claim. '
                                  'This post-hoc characterization does not alter the failed primary criterion and is not independent confirmation.',
    }
    (OUT/'results.tex').write_text('% Generated from verified saved reports; no inference.\n'+
                                  '\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macro.items())+'\n')
    receipt = {'status':'completed','scope':'Saved-summary authentication and table generation, not a raw experiment re-audit.',
               'script_sha256':sha(__file__),
               'sources':{k:{'path':v+'/summary.json','sha256':sha(ROOT/v/'summary.json')} for k,v in sources.items()},
               'followup_audits':followup_audits,
               'edge_basis':{'path':basis_path,'sha256':sha(ROOT/basis_path)},
               'joint_audit':{'path':str(joint_audit_path.relative_to(ROOT)),'sha256':sha(joint_audit_path)},
               'relation_baselines_audit':{'path':str(baseline_audit_path.relative_to(ROOT)),'sha256':sha(baseline_audit_path)},
               'transport_audit':{'path':str(transport_audit_path.relative_to(ROOT)),'sha256':sha(transport_audit_path)},
               'transport_regret_audit':{'path':str(regret_audit_path.relative_to(ROOT)),'sha256':sha(regret_audit_path)},
               'outputs':{name:sha(OUT/name) for name in ['results.tex','candidate-table.tex','biology-table.tex','locality-table.tex','transport-table.tex','transport-regret-table.tex','relation-baselines-table.tex','baseline-regret-table.tex','joint-transport-table.tex','path-baseline-table.tex','graph-contrast-table.tex','graph-baselines-table.tex','wldn-interventions-table.tex','union-difference-table.tex']},
               'model_calls':0,'engine_calls':0}
    (OUT/'table-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':'completed','biology_passed':bio_pass,'locality_passed':l['engineering_gate_passed']}))


if __name__ == '__main__':
    main()
