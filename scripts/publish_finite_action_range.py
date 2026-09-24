"""Publish an originally closed cohort study from saved audited JSON only.

No model, generator, optimizer, scientific-array decode or audit replay occurs.
Source/process admission is delegated to the pinned metadata-only orchestrator.
Archives contain opaque bytes and their complete portable member inventories.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'output/finite-action-range-study-v1'
PRIMITIVE = ROOT / 'output/finite-action-range-qualification-v1'
OUTPUT = ROOT / 'research/finite-action-range-results'
OVERVIEW = ROOT / 'research/finite-action-range-results.md'
PLAN_SHA = 'b6648cd272f51c12fffbc4ea041b7587f59325be2d26b085c82dbc150395abd6'
ENGINEERING_SHA = 'e620a83ab56933463723e18bb507585b35f56472c3b861487bcd1a4b0280bae8'
PRIMITIVE_SHA = '4730b3c017be7a360cad2d6fb6e60510d29e69f82fea41774236061c9b92649c'
VERSION = 'finite-action-range-publication-v1'
ARMS = ('rounded_mse','rounded_double','rounded_range','free_mse','free_double','free_range')
LABELS = ('Rounded\nMSE','Rounded\n2 × MSE','Rounded\nRange','Matched free\nMSE','Matched free\n2 × MSE','Matched free\nRange')
CRITERIA = ('SHORT_HORIZON_LEARNING','BLIND_EXTRAPOLATION','OBSERVED_FILTERING_EXTRAPOLATION')
REGIMES, HORIZONS = ('base','shift'), (1,2,4,8)
COHORTS = tuple((437260924+i,437261001+i) for i in range(5))
ARCHIVE_LIMIT = 95_000_000
RAW_PART_LIMIT = 85_000_000


def require(value,message):
    if not value:
        raise ValueError(message)


def descriptor(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve() == path.absolute(), 'canonical ordinary evidence file')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            digest.update(block)
    return {'sha256':digest.hexdigest(),'bytes':path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def compare(left,right,path='saved scalar join'):
    """Match the independent audit's documented scalar comparison tolerance."""
    if isinstance(left,dict) and isinstance(right,dict):
        require(set(left) == set(right),path+' keys')
        for key in left:
            compare(left[key],right[key],path+'/'+str(key))
    elif isinstance(left,list) and isinstance(right,list):
        require(len(left) == len(right),path+' length')
        for i,(a,b) in enumerate(zip(left,right,strict=True)):
            compare(a,b,path+'/'+str(i))
    elif type(left) in (int,float) and type(right) in (int,float):
        require(math.isfinite(left) and math.isfinite(right) and math.isclose(left,right,rel_tol=1e-10,abs_tol=1e-12),path)
    else:
        require(type(left) is type(right) and left == right,path)


def native_files(launch):
    prefix = str(launch).removesuffix('.launch.json')
    return [Path(prefix+suffix) for suffix in ('.launch.json','.terminal.json','.log')]


def test_completion(path):
    rows = re.findall(r'(?m)^\s*(\d+) passed(?:, (\d+) warnings?)? in [^\n]+$',Path(path).read_text())
    require(len(rows) == 1 and int(rows[0][0]) > 0,'one complete passing original test command')
    return {'tests_passed':int(rows[0][0]),'warnings':int(rows[0][1] or 0)}


def authenticate():
    """No current metrics are opened until all original phases are closed."""
    require(Path.cwd() == ROOT,'publication uses the registered checkout and runtime')
    plans = {}
    for mode,pin in (('engineering',ENGINEERING_SHA),('study',PLAN_SHA)):
        path = STUDY/(mode+'-registration.json')
        require(descriptor(path)['sha256'] == pin,'immutable '+mode+' registration')
        plans[mode] = read(path)
        require(len(plans[mode]['sources']) == 161 and all(descriptor(ROOT/name) == value
            for name,value in plans[mode]['sources'].items()),'unchanged full source closure before imports')
    from finite_action_range_study import admit_qualification, closed_phase, files, validate
    from qualify_finite_action_range import authenticate_closed
    engineering,plan = plans['engineering'],plans['study']
    require(engineering['mode'] == 'engineering' and plan['mode'] == 'study'
        and engineering['sources'] == plan['sources'],'same prospective choices and scientific sources')
    validate(engineering,ENGINEERING_SHA)
    validate(plan,PLAN_SHA)
    primitive = authenticate_closed(PRIMITIVE_SHA)
    require(primitive == plan['primitive_qualification'] and len(primitive['sources']) == 152
        and primitive['scientific_admission'] is False,'original separate fabricated primitive qualification')
    require(admit_qualification(plan) == plan['qualification'],'original smoke/exposure closes before scientific admission')
    phases = {}
    for phase in ('qualify','fit','audit'):
        selected,sha = (engineering,ENGINEERING_SHA) if phase == 'qualify' else (plan,PLAN_SHA)
        receipt,terminal_path = closed_phase(selected,sha,phase)
        spec = selected['phases'][phase]
        phases[phase] = {'receipt':receipt,'receipt_path':Path(spec['output']+'.receipt.json'),
            'terminal':read(terminal_path),'terminal_path':Path(terminal_path),'launch_path':Path(spec['supervision'])}
    for before,after in (('qualify','fit'),('fit','audit')):
        left,right = phases[before]['terminal'],phases[after]['terminal']
        require(left['clock_backend'] == right['clock_backend'] and left['finished_ns'] <= right['started_ns'], 'original phase ordering')
    audit_receipt = phases['audit']['receipt']
    require(audit_receipt['producer_receipt'] == descriptor(phases['fit']['receipt_path'])
        and audit_receipt['producer_terminal'] == descriptor(phases['fit']['terminal_path']), 'audit binds original closed scientific producer')
    require(phases['fit']['receipt']['result'] == {'fits':30,'rows':240},'complete original thirty-fit producer')
    # Build exact original inventories, not an unrestricted directory tar.
    study_expected = {}
    for mode in plans:
        path = STUDY/(mode+'-registration.json')
        study_expected[path.name] = descriptor(path)
        snapshot = STUDY/('source-snapshot-'+mode)
        study_expected.update({str(path.relative_to(STUDY)):pin for name,pin in files(snapshot).items() for path in [snapshot/name]})
    for phase,row in phases.items():
        directory = Path(row['receipt']['output'])
        study_expected.update({str((directory/name).relative_to(STUDY)):pin for name,pin in row['receipt']['files'].items()})
        for path in [row['receipt_path'],*native_files(row['launch_path'])]:
            study_expected[str(path.relative_to(STUDY))] = descriptor(path)
    require(files(STUDY) == study_expected,'every original study file belongs to a closed phase or source snapshot')
    primitive_plan = read(PRIMITIVE/'registration.json')
    primitive_expected = {'registration.json':descriptor(PRIMITIVE/'registration.json')}
    primitive_snapshot = PRIMITIVE/'source-snapshot'
    primitive_expected.update({'source-snapshot/'+name:pin for name,pin in files(primitive_snapshot).items()})
    primitive_receipt_path = Path(primitive_plan['output']+'.receipt.json')
    primitive_receipt = read(primitive_receipt_path)
    primitive_expected.update({str((Path(primitive_plan['output'])/name).relative_to(PRIMITIVE)):pin
        for name,pin in primitive_receipt['files'].items()})
    for path in [primitive_receipt_path,*native_files(primitive_plan['supervision'])]:
        primitive_expected[str(path.relative_to(PRIMITIVE))] = descriptor(path)
    require(files(PRIMITIVE) == primitive_expected,'complete original primitive proof and152-source snapshot')
    # These small parent receipt descriptors remain external evidence. Their
    # complete original archives are reauthenticated by the primitive admission.
    diagnostic_dir = ROOT/'research/finite-decision-error-results'
    diagnostic_receipt = read(diagnostic_dir/'receipt.json')
    require(descriptor(diagnostic_dir/'receipt.json') == primitive_plan['prerequisite']['receipt'], 'original diagnostic publication binding')
    parent = diagnostic_receipt['external_parent']
    require(parent['passing_conditions'] == 14 and parent['conditions'] == 15
        and parent['parent_status'] == 'HEAD_INDEPENDENT_ADVANCE_FAIL','historical failed parent remains failed')
    external = {'diagnostic_receipt':{'path':str(diagnostic_dir/'receipt.json'),**descriptor(diagnostic_dir/'receipt.json')},
        'diagnostic_archive':{'path':str(diagnostic_dir/'evidence.tar.gz'),**descriptor(diagnostic_dir/'evidence.tar.gz')},
        'original_learning_parent':parent,'included_archives':False}
    require(external['diagnostic_archive']['sha256'] == diagnostic_receipt['files']['evidence.tar.gz']['sha256'], 'external diagnostic archive unchanged')
    # All scientific scalar result reads occur after successful original closure.
    audit = read(Path(plan['phases']['audit']['output'])/'audit.json')
    require(audit['version'] == 'finite-action-range-learning-audit-v1' and audit['profile'] == 'science'
        and audit['agreement'] is True and audit['exact_oracle_agreement'] is True
        and audit['technical_complete'] is False and audit['requires_original_supervisor_closure'] is True,
        'independent scientific audit agreement requires now-supplied closure')
    require(audit['counts'] == {'array_decodes':220,'checkpoint_decodes':90,'optimizer_json_decodes':90,
        'initializer_reconstructions':5,'model_calls':0,'optimizer_calls':0,'world_or_generator_calls':0,'native_calls':0}, 'complete independent saved-output audit roster')
    require(audit['metadata'] == {'payloads':377,'fits':30,'model_boundary_files':90,'optimizer_boundary_files':90,
        'global_checkpoint_barrier_verified':True}, 'all retained scientific artifacts and global barrier')
    require(audit['rule'] == plan['rule'] and len(audit['cohorts']) == 5,'fixed rule and all independent cohorts')
    fits = audit['fits']
    fit_keys = {(r['cohort_index'],r['arm'],r['seed'],r['seed_namespace']) for r in fits}
    expected_fits = {(i,arm,seed,namespace) for i,(namespace,seed) in enumerate(COHORTS) for arm in ARMS}
    require(len(fits) == len(fit_keys) == 30 and fit_keys == expected_fits,'every declared model retained without selection')
    expected_rows = {(i,arm,seed,namespace,regime,horizon) for i,arm,seed,namespace in expected_fits for regime in REGIMES for horizon in HORIZONS}
    require(len(audit['rows']) == 240 and {(r['cohort_index'],r['arm'],r['seed'],r['seed_namespace'],r['regime'],r['horizon']) for r in audit['rows']} == expected_rows,'all 240 endpoint rows')
    require(len(audit['prefix_rows']) == 60 and {(r['cohort_index'],r['arm'],r['seed'],r['seed_namespace'],r['regime']) for r in audit['prefix_rows']}
        == {(i,a,s,n,g) for i,a,s,n in expected_fits for g in REGIMES},'all 60 all-attempt prefix rows')
    require(len(audit['baseline_rows']) == 40 and {(r['cohort_index'],r['regime'],r['horizon']) for r in audit['baseline_rows']}
        == {(i,g,h) for i in range(5) for g in REGIMES for h in HORIZONS},'every known-dynamics uniform-state reference')
    require(audit['logical_counts']['fit_count'] == 30 and audit['logical_counts']['accepted_prefix_steps'] == 30720
        and audit['logical_counts']['accepted_joint_steps'] == 92160
        and audit['logical_counts']['accepted_optimizer_steps'] == 122880
        and audit['logical_counts']['dev_generation_count'] == 10,'complete fixed updates and independent regimes')
    verify_rule(audit)
    require(audit_receipt['result'] == {'advance':audit['advance']},'published decision equals original closed audit decision')
    qualification = test_completion(Path(engineering['phases']['qualify']['output'])/'command-1.log')
    primitive_tests = test_completion(Path(primitive_plan['output'])/'command-1.log')
    return {'plan':plan,'phases':phases,'audit':audit,'study_files':study_expected,'primitive_files':primitive_expected,
        'primitive':primitive,'primitive_tests':primitive_tests,'qualification':qualification,'external':external,
        'phase_seconds':{name:row['terminal']['wall_seconds'] for name,row in phases.items()},
        'primitive_seconds':read(native_files(primitive_plan['supervision'])[1])['wall_seconds']}


def verify_rule(audit):
    """Check saved scalar gate/contrast consistency without replaying the audit."""
    require(set(audit['gates']) == set(REGIMES),'both regimes retained')
    conditions = {}
    for regime in REGIMES:
        require(set(audit['gates'][regime]) == set(ARMS),'all six absolute gate groups')
        for arm in ARMS:
            require(set(audit['gates'][regime][arm]) == set(CRITERIA),'all three absolute criteria')
            for name,gate in audit['gates'][regime][arm].items():
                require(set(gate['cohorts']) == {str(i) for i in range(5)},'all five cohort gates')
                for item in gate['cohorts'].values():
                    require(item['conditions'] and all(type(v) is bool for v in item['conditions'].values())
                        and item['passed'] == all(item['conditions'].values()),'individual absolute conjunction')
                require(type(gate['passed']) is bool and gate['passed'] == all(x['passed'] for x in gate['cohorts'].values()),'every cohort must pass')
                if arm == 'rounded_range':
                    conditions[regime+'_'+name] = gate['passed']
    indexed = {(x['cohort_index'],x['arm'],x['regime'],x['horizon']):x for x in audit['rows']}
    pair_keys,mean_keys = set(),set()
    require(len(audit['paired_loss_contrasts']) == 40 and len(audit['equal_cohort_means']) == 8,'complete40paired and8mean contrasts')
    for row in audit['paired_loss_contrasts']:
        c,control,regime,horizon = (row[k] for k in ('cohort_index','control','regime','horizon'))
        key = c,control,regime,horizon
        require(key not in pair_keys and row['candidate'] == 'rounded_range','fixed candidate and unique paired contrast')
        pair_keys.add(key)
        left,right = indexed[c,'rounded_range',regime,horizon],indexed[c,control,regime,horizon]
        compare([row['candidate_regret'],row['control_regret'],row['difference']],
            [left['blind_regret'],right['blind_regret'],left['blind_regret']-right['blind_regret']])
        require(row['seed'] == left['seed'] == right['seed'] and row['seed_namespace'] == left['seed_namespace'] == right['seed_namespace'],'paired independent-cohort identity')
        conditions[f'{control}_{regime}_h{horizon}_cohort{c}_strict'] = row['difference'] < 0
    wanted_pairs = {(c,a,g,h) for c in range(5) for a in ('rounded_mse','rounded_double') for g in REGIMES for h in (4,8)}
    require(pair_keys == wanted_pairs,'exact primary control and long-horizon roster')
    for row in audit['equal_cohort_means']:
        control,regime,horizon = (row[k] for k in ('control','regime','horizon'))
        key = control,regime,horizon
        require(key not in mean_keys and row['candidate'] == 'rounded_range' and row['cohorts'] == 5,'equal independent-cohort mean')
        mean_keys.add(key)
        left = math.fsum(indexed[c,'rounded_range',regime,horizon]['blind_regret'] for c in range(5))/5
        right = math.fsum(indexed[c,control,regime,horizon]['blind_regret'] for c in range(5))/5
        compare([row['candidate_regret'],row['control_regret'],row['difference'],row['reduction_fraction']],
            [left,right,left-right,None if right <= 0 else 1-left/right])
        conditions[f'{control}_{regime}_h{horizon}_mean_reduction'] = right > 0 and left <= .9*right
    require(mean_keys == {(a,g,h) for a in ('rounded_mse','rounded_double') for g in REGIMES for h in (4,8)}, 'all8mean conditions')
    advance = audit['advance']
    require(len(conditions) == 54 and advance['name'] == 'ACTION_RANGE_ADVANCE'
        and advance['conditions'] == conditions and advance['passed'] == all(conditions.values())
        and advance['status'] == 'ACTION_RANGE_ADVANCE_'+('PASS' if advance['passed'] else 'FAIL'),'unmodified54-condition rule')
    require(len(audit['architecture_contrasts']) == 120 and audit['architecture_contrasts_descriptive_only'] is True
        and audit['architecture_claim'] is False and audit['latent_identification_claim'] is False,'architecture contrasts cannot rescue the loss rule')


def chart(summary,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    audit = summary['audit']
    indexed = {(r['cohort_index'],r['arm'],r['regime'],r['horizon']):r for r in audit['rows']}
    fits = {(r['cohort_index'],r['arm']):r for r in audit['fits']}
    colors = ('#3066a8','#bc7526','#298572','#955f9e','#bd465b')
    fig = plt.figure(figsize=(14,11),layout='constrained')
    grid = fig.add_gridspec(3,2,height_ratios=(1,1,.88))
    axes = []
    for ri,regime in enumerate(REGIMES):
        for hi,horizon in enumerate((4,8)):
            axis = fig.add_subplot(grid[ri,hi]); axes.append(axis)
            for c in range(5):
                axis.scatter([i+(c-2)*.055 for i in range(6)],
                    [indexed[c,arm,regime,horizon]['blind_regret'] for arm in ARMS],
                    color=colors[c],s=36,alpha=.95,label=f'Cohort {c+1}',zorder=3)
            means = [math.fsum(indexed[c,arm,regime,horizon]['blind_regret'] for c in range(5))/5 for arm in ARMS]
            axis.scatter(range(6),means,color='#172431',marker='_',s=230,linewidths=2.3,label='Equal-cohort mean',zorder=4)
            axis.set_title(f'{regime.upper()} (observation noise {0.12 if regime == "base" else 0.30:.2f}), H{horizon}',fontsize=11)
            axis.set_ylabel('Blind decision regret (lower is better)',fontsize=9)
    runtime = fig.add_subplot(grid[2,:]); axes.append(runtime)
    for c in range(5):
        runtime.scatter([i+(c-2)*.055 for i in range(6)],[fits[c,arm]['seconds'] for arm in ARMS],color=colors[c],s=36,zorder=3)
    runtime.scatter(range(6),[math.fsum(fits[c,arm]['seconds'] for c in range(5))/5 for arm in ARMS],
        color='#172431',marker='_',s=230,linewidths=2.3,zorder=4)
    runtime.set_title('Recorded fitting time for all 30 models (updates, snapshots and checkpoints)',fontsize=11)
    runtime.set_ylabel('Seconds',fontsize=9)
    for axis in axes:
        axis.set_yscale('linear'); axis.set_ylim(bottom=0)
        axis.set_xlim(-.4,5.4); axis.set_xticks(range(6),LABELS,fontsize=8)
        axis.tick_params(axis='y',labelsize=8)
        axis.grid(axis='y',color='#e0e6ec',zorder=0)
        axis.spines[['top','right']].set_visible(False)
        axis.axvline(2.5,color='#b7c0cb',linestyle=':',linewidth=1)
    passed = sum(audit['advance']['conditions'].values())
    fig.suptitle(f'Action-range loss: {audit["advance"]["status"].removeprefix("ACTION_RANGE_ADVANCE_")} ({passed}/54 conditions)\n'
        'Five independent cohorts; six fixed arms; every cohort retained',fontsize=15,weight='bold')
    handles,labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=6,frameon=False,fontsize=9)
    fig.savefig(path,dpi=180,metadata={'Description':'All five cohorts, all six arms, BASE and SHIFT H4/H8 blind regret, and every recorded fit runtime. Linear axes start at zero. Saved audited JSON only; no model or scientific array decoding.'})
    plt.close(fig)


def fmt(value):
    return 'undefined' if value is None else f'{value:.9g}'


def document(summary):
    audit = summary['audit']; decision = audit['advance']
    passing,total = sum(decision['conditions'].values()),len(decision['conditions'])
    verdict = 'PASS' if decision['passed'] else 'FAIL'
    lines = ['# Action-range loss across independent cohorts','',
        f'**Technical audit complete. Prospective rule: {verdict}, {passing}/{total} conditions.**','',
        'The fixed rounded-range candidate is compared with rounded MSE and rounded double-MSE. All five cohorts, both observation regimes and every predeclared condition count. The matched-free arms are descriptive architecture controls and cannot rescue this decision.','',
        '![Every cohort, both regimes and complete fitting cost](benchmark.png)','',
        'Each cohort independently generates 512 TRAIN, 512 BASE and 512 SHIFT attempts. Its six models share data, initial task-independent random heads and paired joint batches. Loss variants within each architecture also share identical initial and prefix-boundary model/Adam states. All 30 final checkpoints precede either DEV regime.','',
        'Training uses 1024 full-prefix updates followed by 3072 joint updates on H1/H2 targets. Only the blind cost term changes: mean squared error, twice mean squared error, or one quarter of squared four-action error range. Observed cost, survival, event and public-prefix losses are unchanged. The same update count does not imply equal computation, gradients or elapsed time.','',
        'BASE has observation noise 0.12; SHIFT increases it to 0.30 without retraining. Both retain the same transition, hazard and true-cost structure. The known transition is doubly stochastic, so this remains a structurally favorable synthetic setting for rounded transport.','',
        'The 54 conditions consist of 40 strictly lower cohort-specific H4/H8 regret comparisons, 8 reductions of at least 10% in equal-cohort mean regret with positive control means, and 6 unchanged absolute criteria across both regimes. No average rescues a failed individual condition.','',
        'The earlier independent-head comparison remains FAIL at 14/15. Its retrospective decision-error decomposition motivated this separate comparison; neither that diagnostic nor this fresh result changes the earlier outcome. No statistical-significance, loss-novelty, architectural-superiority, latent-identification or native-transfer claim is made.','',
        '## Prospective outcome','',
        '| Condition | Outcome |','|---|---|']
    lines += [f'| `{key}` | {"PASS" if value else "FAIL"} |' for key,value in decision['conditions'].items()]
    lines += ['','## Absolute criteria for every arm','',
        '| Regime | Arm | Short-horizon learning | Blind extrapolation | Observed filtering extrapolation |','|---|---|---|---|---|']
    for regime in REGIMES:
        for arm in ARMS:
            results = ['PASS' if audit['gates'][regime][arm][name]['passed'] else 'FAIL' for name in CRITERIA]
            lines.append(f'| {regime} | {arm} | '+' | '.join(results)+' |')
    lines += ['','Observed filtering receives intervening observations. Its criterion is separate from blind forecasts. Each group above requires every cohort; complete per-cohort condition records remain in summary.json and the archived original audit.','',
        '## Equal-cohort regret means','',
        '| Control | Regime | H | Rounded range | Control | Difference | Reduction |','|---|---|---:|---:|---:|---:|---:|']
    for row in audit['equal_cohort_means']:
        reduction = 'undefined' if row['reduction_fraction'] is None else f'{100*row["reduction_fraction"]:.3f}%'
        lines.append(f'| {row["control"]} | {row["regime"]} | {row["horizon"]} | {fmt(row["candidate_regret"])} | {fmt(row["control_regret"])} | {fmt(row["difference"])} | {reduction} |')
    lines += ['','## Every paired loss contrast','',
        '| Cohort | Seed | Control | Regime | H | Candidate regret | Control regret | Candidate minus control |','|---:|---:|---|---|---:|---:|---:|---:|']
    for row in audit['paired_loss_contrasts']:
        lines.append(f'| {row["cohort_index"]+1} | {row["seed"]} | {row["control"]} | {row["regime"]} | {row["horizon"]} | {fmt(row["candidate_regret"])} | {fmt(row["control_regret"])} | {fmt(row["difference"])} |')
    lines += ['','## Population and cost','',
        '| Cohort | Data namespace | Fit seed | TRAIN retained | BASE retained | SHIFT retained |','|---:|---:|---:|---:|---:|---:|']
    for row in audit['cohorts']:
        counts = row['data_cases']
        lines.append(f'| {row["cohort_index"]+1} | {row["seed_namespace"]} | {row["fit_seed"]} | {counts["train"]} | {counts["base"]} | {counts["shift"]} |')
    lines += ['','| Original closed phase | Native wall seconds |','|---|---:|',
        f'| Fabricated primitive qualification | {fmt(summary["primitive_seconds"])} |']
    lines += [f'| {phase} | {fmt(seconds)} |' for phase,seconds in summary['phase_seconds'].items()]
    lines += ['',f'The primitive qualification passed {summary["primitive_tests"]["tests_passed"]} tests. Integration passed {summary["qualification"]["tests_passed"]} tests and retained the original smoke, TRAIN-only exposure and all cost projections. These engineering costs are separate from the scientific fit phase.','',
        '| Cohort | Arm | Train-call seconds | Safety-clock seconds | Constructor seconds | Nested final-summary seconds |','|---:|---|---:|---:|---:|---:|']
    for row in audit['fits']:
        lines.append(f'| {row["cohort_index"]+1} | {row["arm"]} | {fmt(row["seconds"])} | {fmt(row["timed_seconds"])} | {fmt(row["construction_seconds"])} | {fmt(row["final_summary_seconds"])} |')
    lines += ['','The train-call measurement includes model/optimizer setup, accepted updates, snapshots, checkpoint files and durable allocation trace. It ends before final fit-row publication; that work is charged in the original complete scientific phase. Safety-clock, constructor and final-summary measurements overlap and must not be added to train-call time. All 30 fits retain 352 stored float64 parameters; parameter count does not equalize transition degrees of freedom.','',
        'The independent audit decoded 220 NPZ files, including 90 parameter boundaries, and 90 Adam JSON boundaries. It reconstructed 5 initial head streams and called no models, optimizers or world generators. Publishing uses only saved audited scalars plus opaque file hashing. Intermediate training losses, gradients and historical clock readings are not independently replayed.','',
        '## All endpoint results','',
        '| Cohort | Arm | Regime | H | Blind regret | Blind MSE | Blind survival MAE | Observed MSE | Observed survival MAE | Observed KL | Shuffled regret |',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    metrics = ('blind_regret','blind_cost_mse','blind_survival_mae','observed_cost_mse','observed_survival_mae','observed_kl','shuffled_blind_regret')
    for row in sorted(audit['rows'],key=lambda x:(x['cohort_index'],x['regime'],ARMS.index(x['arm']),x['horizon'])):
        lines.append(f'| {row["cohort_index"]+1} | {row["arm"]} | {row["regime"]} | {row["horizon"]} | '+' | '.join(fmt(row[name]) for name in metrics)+' |')
    lines += ['','## All public-prefix diagnostics','',
        '| Cohort | Arm | Regime | Attempts | Valid events | Event-weighted NLL |','|---:|---|---|---:|---:|---:|']
    for row in audit['prefix_rows']:
        lines.append(f'| {row["cohort_index"]+1} | {row["arm"]} | {row["regime"]} | {row["attempts"]} | {row["valid_events"]} | {fmt(row["mean_nll"])} |')
    lines += ['','All 120 same-loss rounded-minus-free comparisons, complete loss-operation counts, physical forward-operation receipts, logical exposures, and every per-cohort absolute condition remain in the saved summary. Those architecture comparisons are descriptive; the primary gate tests the rounded model’s training loss.','',
        '[Protocol](../finite-action-range-study-protocol.md) · [Complete summary](summary.json) · [Archive parts and hashes](archive-index.json) · [Every member](manifest.json) · [Publication receipt](receipt.json)','',
        'The archive contains every original current-study payload, all 30 traces, 90 parameter and 90 Adam boundaries, both 161-source registration snapshots, and the original primitive qualification with its 152-source snapshot. Portable archive names retain original path descriptors. Older learning and diagnostic archives stay external with their original hashes; their failed outcomes remain unchanged. No evidence is dropped to meet the archive-size limit.','']
    return '\n'.join(lines)


def build_archives(auth):
    paths = {'study/'+name:(STUDY/name,pin) for name,pin in auth['study_files'].items()}
    paths.update({'primitive-qualification/'+name:(PRIMITIVE/name,pin) for name,pin in auth['primitive_files'].items()})
    for name in ('summary.json','report.md','benchmark.png'):
        paths['publication/'+name] = (OUTPUT/name,descriptor(OUTPUT/name))
    paths['publication/overview.md'] = (OVERVIEW,descriptor(OVERVIEW))
    paths['publication/publisher.py'] = (Path(__file__).resolve(),descriptor(Path(__file__).resolve()))
    manifest = {'version':VERSION,'registration':{'path':str(STUDY/'study-registration.json'),**descriptor(STUDY/'study-registration.json')},
        'files':{name:{'original_path':str(path),**pin} for name,(path,pin) in sorted(paths.items())},
        'external_references':auth['external'],
        'scope':'Complete current study, both 161-source snapshots, primitive152-source snapshot and all original closures; older archives remain externally hash-pinned.',
        'self_exclusion':'Manifest excludes itself; it is archived. Archive index and final publication receipt remain outside archives to avoid circular hashes.',
        'size_policy':{'maximum_archive_bytes':ARCHIVE_LIMIT,'conservative_raw_part_bytes':RAW_PART_LIMIT,
            'split':'If a single conservative raw-size bound exceeds the part limit, group by scientific cohort plus shared evidence. Oversized groups split deterministically by sorted whole-file members.'}}
    write(OUTPUT/'manifest.json',manifest)
    paths['publication/manifest.json'] = (OUTPUT/'manifest.json',descriptor(OUTPUT/'manifest.json'))
    total_raw = sum(pin['bytes']+2048 for _,pin in paths.values())
    if total_raw <= RAW_PART_LIMIT:
        groups = {'evidence':sorted(paths)}
    else:
        groups = {'evidence-shared':[]}
        for name in sorted(paths):
            found = re.match(r'study/run-01/(cohort-\d\d)/',name)
            key = 'evidence-'+found[1] if found else 'evidence-shared'
            groups.setdefault(key,[]).append(name)
    parts = []
    for group,members in sorted(groups.items()):
        batches,current,size = [],[],0
        for member in members:
            estimated = paths[member][1]['bytes']+2048
            require(estimated <= RAW_PART_LIMIT,'one whole evidence member exceeds the explicitly supported archive part bound')
            if current and size+estimated > RAW_PART_LIMIT:
                batches.append(current); current=[]; size=0
            current.append(member); size+=estimated
        if current:
            batches.append(current)
        for i,batch in enumerate(batches):
            filename = group+(f'-part-{i+1:02d}' if len(batches)>1 else '')+'.tar.gz'
            destination = OUTPUT/filename
            with destination.open('xb') as raw, gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as zipped, tarfile.open(fileobj=zipped,mode='w',format=tarfile.PAX_FORMAT) as archive:
                for member in batch:
                    source,pin = paths[member]
                    require(descriptor(source) == pin,'unchanged opaque archive member')
                    info = tarfile.TarInfo(member)
                    info.size,info.mode,info.mtime = pin['bytes'],0o644,0
                    with source.open('rb') as stream:
                        archive.addfile(info,stream)
            pin = descriptor(destination)
            require(pin['bytes'] <= ARCHIVE_LIMIT,'archive remains below95MB without omitted evidence')
            with tarfile.open(destination,'r:gz') as archive:
                require(archive.getnames() == batch,'exact deterministic archive member roster')
                for member in archive.getmembers():
                    require(member.isfile(),'ordinary archive member')
                    digest = hashlib.sha256(); length=0
                    with archive.extractfile(member) as stream:
                        for block in iter(lambda:stream.read(1048576),b''):
                            digest.update(block); length+=len(block)
                    require({'sha256':digest.hexdigest(),'bytes':length} == paths[member.name][1]
                        and descriptor(paths[member.name][0]) == paths[member.name][1], 'complete opaque byte-hash archive roundtrip')
            parts.append({'path':filename,**pin,'members':batch,'member_count':len(batch),
                'uncompressed_member_bytes':sum(paths[member][1]['bytes'] for member in batch)})
    flattened = [name for part in parts for name in part['members']]
    require(len(flattened) == len(set(flattened)) == len(paths) and set(flattened) == set(paths),'every original byte inventory archived exactly once')
    index = {'version':VERSION,'archives':parts,'archive_members':len(paths),'manifest':descriptor(OUTPUT/'manifest.json'),
        'whole_file_members':True,'opaque_hash_roundtrip':True,'maximum_archive_bytes':ARCHIVE_LIMIT,
        'members':{name:part['path'] for part in parts for name in part['members']}}
    write(OUTPUT/'archive-index.json',index)
    return index


def main():
    require(not OUTPUT.exists() and not OVERVIEW.exists(),'exclusive new publication outputs')
    publisher = descriptor(Path(__file__).resolve())
    auth = authenticate()
    summary = {'version':VERSION,'status':auth['audit']['advance']['status'],'technical_complete':True,
        'registration':{'path':str(STUDY/'study-registration.json'),**descriptor(STUDY/'study-registration.json')},
        'engineering_registration':{'path':str(STUDY/'engineering-registration.json'),**descriptor(STUDY/'engineering-registration.json')},
        'audit':auth['audit'],'config':auth['plan']['config'],'runtime':auth['plan']['runtime'],
        'phase_seconds':auth['phase_seconds'],'primitive_seconds':auth['primitive_seconds'],
        'qualification':auth['qualification'],'primitive_tests':auth['primitive_tests'],'primitive_qualification':auth['primitive'],
        'external_references':auth['external'],'scientific_metrics_source':'Original closed independent audit JSON; no predictive replay or reselection.',
        'model_selection':False,'architecture_claim':False,'statistical_significance_claim':False,
        'loss_novelty_claim':False,'prior_failure_rescued':False}
    OUTPUT.mkdir()
    write(OUTPUT/'summary.json',summary)
    chart(summary,OUTPUT/'benchmark.png')
    report = document(summary)
    with (OUTPUT/'report.md').open('x') as stream:
        stream.write(report)
    overview = report.replace('(benchmark.png)','(finite-action-range-results/benchmark.png)')
    overview = overview.replace('(../finite-action-range-study-protocol.md)','(finite-action-range-study-protocol.md)')
    for name in ('summary.json','archive-index.json','manifest.json','receipt.json'):
        overview = overview.replace('('+name+')','(finite-action-range-results/'+name+')')
    with OVERVIEW.open('x') as stream:
        stream.write(overview)
    archives = build_archives(auth)
    after = authenticate()
    require(after['study_files'] == auth['study_files'] and after['primitive_files'] == auth['primitive_files']
        and after['external'] == auth['external'] and descriptor(Path(__file__).resolve()) == publisher,
        'all original evidence and publisher unchanged after publication')
    from finite_action_range_study import files
    write(OUTPUT/'receipt.json',{'version':VERSION,'status':'PASS','registration':summary['registration'],
        'engineering_registration':summary['engineering_registration'],'publisher':publisher,
        'overview':{'path':str(OVERVIEW),**descriptor(OVERVIEW)},'files':files(OUTPUT),
        'sources':auth['plan']['sources'],'study_files':auth['study_files'],'primitive_files':auth['primitive_files'],
        'phase_closures':{name:{'receipt':descriptor(row['receipt_path']),'terminal':descriptor(row['terminal_path']),
            'launch':descriptor(row['launch_path'])} for name,row in auth['phases'].items()},
        'primitive_qualification':auth['primitive'],'external_references':auth['external'],
        'archive_members':archives['archive_members'],'archive_parts':len(archives['archives']),
        'publication_counts':dict.fromkeys(('scientific_array_decodes','checkpoint_decodes','model_calls','optimizer_calls',
            'world_or_generator_calls','audit_reruns'),0),'all_inputs_unchanged':True,'opaque_hash_roundtrip':True,
        'result_reads_after_original_closure':True,'plots_saved_scalar_data_only':True,'visual_review_required':True})
    print(json.dumps({'output':str(OUTPUT),'status':summary['status'],'archive_members':archives['archive_members'],
        'archive_parts':len(archives['archives'])}))


if __name__ == '__main__':
    main()
