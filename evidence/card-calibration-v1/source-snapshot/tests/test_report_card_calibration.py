"""Handwritten arrays and fake byte trees only; no models/native/random calls."""
import copy
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from openjev.research.card_memory_picker import CardPolicyTracker
from openjev.research.card_probability_calibration import fit_temperature, score_episodes

PATH = Path(__file__).resolve().parents[1] / 'scripts/report_card_calibration.py'
SPEC = importlib.util.spec_from_file_location('card_calibration_report_test', PATH)
report = importlib.util.module_from_spec(SPEC)
with patch.object(sys, 'path', [str(PATH.parent), *sys.path]):
    SPEC.loader.exec_module(report)


def score_episode(prob=.8, length=4):
    raw = np.full((length, 52, 13), (1-prob)/12, np.float64)
    raw[..., 0] = prob
    y = np.full((length, 52), -1, np.int64)
    mask = np.zeros(y.shape, bool); ages = np.full(y.shape, -1, np.int32)
    mask[1:, :2] = True; y[mask] = 0
    ages[1:, :2] = np.arange(1, length)[:, None]
    return {'raw_probabilities': raw, 'targets': y, 'target_mask': mask, 'ages': ages}


def test_causal_targets_exclude_current_visibility_and_future():
    obs = np.full((5, 52), 13, np.int64)
    obs[1, 0], obs[2, :2], obs[3, 2], obs[4, 3] = 4, [4, 7], 9, 10
    y, mask, ages = report.public_targets(obs)
    assert not mask[:3].any()
    assert y[3, :3].tolist() == [4, 7, -1] and ages[3, :3].tolist() == [1, 1, -1]
    assert not mask[:, 3].any()
    obs[4, 0] = 2
    with pytest.raises(ValueError, match='rank changed'): report.public_targets(obs)


@pytest.mark.parametrize('policy,beta', [('baseline', None), ('temperature', .05), ('temperature', 1.), ('temperature', 20.), ('hard', None)])
def test_independent_proper_scores_match_pure_helper(policy, beta):
    episodes = [score_episode(.8), score_episode(.4, 40)]
    episodes[1]['targets'][3, 0] = 1
    actual = report.proper_scores(episodes, policy, beta)
    saved = score_episodes(episodes, mode=policy, beta=beta if beta is not None else 1.)
    report.compare_scores(actual, saved)
    assert actual['all']['eligible_episodes'] == 2


def test_equal_episode_boundary_query_weights_not_pooled():
    a, b = score_episode(.8), score_episode(.2, 10)
    b['target_mask'][1:, :] = True; b['targets'][1:, :] = 0; b['ages'][1:, :] = 1
    value = report.proper_scores([a, b], 'baseline')['all']['nll']
    assert value == pytest.approx((-np.log(.8)-np.log(.2))/2)
    assert value != pytest.approx((-np.log(.8)*6-np.log(.2)*468)/474)


def test_infinite_hard_nll_and_temperature_underflow_are_distinct():
    ep = score_episode(.8)
    ep['targets'][1, 0] = 1
    hard = report.proper_scores([ep], 'hard')['all']
    assert hard['nll'] is None and hard['nll_is_infinite'] and hard['infinite_nll_queries'] == 1
    ep['raw_probabilities'][1, 0] = 0
    ep['raw_probabilities'][1, 0, :2] = [1., 1e-100]
    temp = report.proper_scores([ep], 'temperature', 20.)['all']
    assert np.isfinite(temp['nll']) and not temp['nll_is_infinite']
    assert temp['probability_underflow_queries'] == 1
    ep['raw_probabilities'][1, 0, 1] = 0
    truezero = report.proper_scores([ep], 'temperature', 20.)['all']
    assert truezero['nll'] is None and truezero['infinite_nll_queries'] == 1


def test_scalar_trace_all_passes_and_resealed_mutation():
    episodes = [score_episode(.8), score_episode(.4)]
    episodes[1]['targets'][2, 0] = 1
    fitted = fit_temperature(episodes)
    result = report.validate_scalar_fit(episodes, fitted)
    assert result['oracle_evaluations'] == 68
    for key, bad in [('beta', fitted['beta']+.1), ('chosen_derivative', .01), ('weighting', 'pooled')]:
        altered = copy.deepcopy(fitted); altered[key] = bad
        with pytest.raises(ValueError): report.validate_scalar_fit(episodes, altered)
    altered = copy.deepcopy(fitted); altered['trace'][4]['nll'] += .01
    with pytest.raises(ValueError, match='objective'): report.validate_scalar_fit(episodes, altered)


def test_scalar_endpoint_and_identity_exact_tie():
    for ep in (score_episode(.9), score_episode(1/13)):
        fitted = fit_temperature([ep])
        assert report.validate_scalar_fit([ep], fitted)['beta'] == fitted['beta']
    assert fitted['beta'] == 1.


def rows_scores():
    rows = {p: {} for p in report.POLICIES}
    for p in report.POLICIES:
        for name in (*report.NAMES, *report.REFERENCES) if p == 'baseline' else report.NAMES:
            value = .8 if name in report.REFERENCES else .1 if p == 'baseline' else .13
            rows[p][name] = {'mean_return': value, 'per_seed_returns': [value]*64,
                'per_seed_successes': [name in report.REFERENCES]*64, 'native_steps': 52*64, 'wall_seconds': 1.}
    scores = {n: {p: {'all': {'eligible_episodes': 64, 'query_cards': 100,
        'nll': 1. if p == 'baseline' else .95, 'nll_is_infinite': False,
        'infinite_nll_queries': 0, 'brier': .1, 'accuracy': .9}} for p in report.POLICIES} for n in report.NAMES}
    return rows, scores


def set_return(rows, policy, name, value):
    rows[policy][name]['mean_return'] = value
    rows[policy][name]['per_seed_returns'] = [value]*64


def test_twelve_inclusive_checks_and_references_once():
    rows, scores = rows_scores()
    families, contrasts, transfer, gate = report.aggregate(rows, scores)
    assert gate['passed'] and gate['checks_passed'] == gate['total_checks'] == 12
    assert len(gate['grouped_requirements']) == 5 and all(gate['grouped_requirements'].values())
    assert contrasts['temperature_minus_baseline']['mean_difference'] == .03
    assert transfer['temperature']['nll'] == .95
    assert all(set(v) == set(report.MODES) for v in families.values())


@pytest.mark.parametrize('bad', ['mean', 'pair', 'family', 'nll', 'brier', 'infinite', 'zero_baseline'])
def test_any_primary_failure_not_rescued_by_hard_or_references(bad):
    rows, scores = rows_scores()
    if bad == 'mean':
        for n in report.NAMES: set_return(rows, 'temperature', n, .12999999999)
    elif bad == 'pair':
        for m in report.MODES:
            set_return(rows, 'temperature', f'{m}-pair0', .1)
            set_return(rows, 'temperature', f'{m}-pair1', .2)
    elif bad == 'family':
        for n in report.NAMES: set_return(rows, 'temperature', n, .2)
        for i in range(3): set_return(rows, 'temperature', f'gru-pair{i}', .089999999)
    elif bad == 'zero_baseline':
        for n in report.NAMES:
            scores[n]['baseline']['all']['nll'] = scores[n]['temperature']['all']['nll'] = 0.
    else:
        for n in report.NAMES:
            scores[n]['temperature']['all'][{'nll':'nll','brier':'brier','infinite':'nll_is_infinite'}[bad]] = (
                .950000000001 if bad == 'nll' else .100000000001 if bad == 'brier' else True)
            if bad == 'infinite': scores[n]['temperature']['all']['nll'] = None
    assert not report.aggregate(rows, scores)[3]['passed']


def test_missing_rows_and_fake_replicated_reference_rejected():
    rows, scores = rows_scores(); del rows['hard']['gru-pair2']
    with pytest.raises(ValueError): report.aggregate(rows, scores)
    rows, scores = rows_scores(); rows['hard']['exact'] = rows['baseline']['exact']
    with pytest.raises(ValueError): report.aggregate(rows, scores)


def test_exact_manifest56rows3584games7226payloads():
    files = report.expected_members()
    assert len(files) == 7226
    assert sum(n.endswith('.npz') for n in files) == 3584
    assert sum(n.endswith('/completed.json') for n in files) == 56
    assert 'controllers/exact/hard/episodes/000.npz' not in files


def fake_episode(tmp_path, policy='temperature', beta=2.):
    tracker = CardPolicyTracker('C'); frame = np.full(52, 13, np.int64); tracker.reset(frame)
    raw = np.full((52, 13), 1/13, np.float64)
    arrays = {k: [] for k in ('observations','actions','ranks','rewards','terminated','truncated','raw_probabilities','picker_probabilities')}
    arrays['observations'].append(frame.copy()); diagnostics = []
    for step in range(52):
        action, picker, diagnostic = tracker.decision(report.transform(raw, policy, beta))
        assert action == step
        after = frame.copy(); after[action] = action//4
        reward = 0. if step%2 == 0 else 2/52
        event = tracker.observe(action, reward, after)
        for k,v in [('observations',after.copy()),('actions',action),('ranks',event.rank),('rewards',reward),
            ('terminated',step==51),('truncated',False),('raw_probabilities',raw.copy()),('picker_probabilities',picker)]: arrays[k].append(v)
        frame=after; diagnostics.append(diagnostic)
    arrays={k:np.array(v) for k,v in arrays.items()}; path=tmp_path/'episode.npz'; np.savez(path,**arrays)
    counters = ('reset_attempted','reset_returned','identity_read_attempted','identity_read_returned',
        'init_attempted','init_returned','decision_attempted','decision_returned','transform_attempted','transform_returned',
        'native_attempted','native_returned','predict_attempted','predict_returned','write_attempted','write_returned')
    receipt={'status':'complete','controller':'kalman-pair0','policy':policy,'tracker_policy':'C','beta':beta,
        'npz_sha256':report.sha(path),'npz_bytes':path.stat().st_size,'native_steps':52,
        'array_bytes':sum(a.nbytes for a in arrays.values()),'decisions':diagnostics,'return':1.,'success':True,'matched_pairs':26,
        'counts':{k:1 if 'reset' in k or 'identity' in k or 'init' in k else 52 for k in counters},
        'timings':{'native_seconds':.1,'transform_seconds':.01},'wall_seconds':.2}
    return path,receipt,arrays


@pytest.mark.parametrize('policy,beta',[('baseline',None),('temperature',2.),('hard',None)])
def test_actual_public_picker_replay_with_original_raw_probabilities(tmp_path,policy,beta):
    path,receipt,_=fake_episode(tmp_path,policy,beta)
    result=report.audit_episode(path,receipt,policy=policy,controller='kalman-pair0',beta=beta)
    assert result['success'] and result['unique_positions']==52


@pytest.mark.parametrize('kind',['beta','diagnostic','picker','transformcount'])
def test_resealed_episode_mutations_rejected(tmp_path,kind):
    path,receipt,arrays=fake_episode(tmp_path)
    if kind=='beta': receipt['beta']=3.
    elif kind=='diagnostic': receipt['decisions'][0]['rowmass_max_abs']=.001
    elif kind=='transformcount': receipt['counts']['transform_returned']=51
    else:
        arrays['picker_probabilities'][0,0,0]=.2; np.savez(path,**arrays)
        receipt.update(npz_sha256=report.sha(path),npz_bytes=path.stat().st_size)
    with pytest.raises(ValueError): report.audit_episode(path,receipt,policy='temperature',controller='kalman-pair0',beta=2.)


def test_authenticated_tree_rejects_extra_or_changed_bytes(tmp_path):
    folder=tmp_path/'sealed';folder.mkdir();(folder/'payload').write_bytes(b'fixed')
    report.write(folder/'completed.json',{'status':'complete','files':{'payload':{'sha256':report.sha(folder/'payload'),'bytes':5}}})
    digest=report.sha(folder/'completed.json')
    assert report.authenticate_tree(folder,digest,{'payload'})['status']=='complete'
    (folder/'extra').write_bytes(b'bad')
    with pytest.raises(ValueError,match='membership'): report.authenticate_tree(folder,digest,{'payload'})


@pytest.mark.parametrize('corrupt_work', [False, True])
def test_outer56rows_and_terminal_receipts_with_failed_scientific_gate(tmp_path, monkeypatch, corrupt_work):
    """Real7,226 fake payload files; numeric/native helpers exercised separately."""
    root=tmp_path; evaluation=root/'evaluation';evaluation.mkdir(); calibration=root/'calibration';calibration.mkdir()
    paths={k:root/(k+'.json') for k in ('protocol','inputs','bindings','checkpoint_map')}
    for p in paths.values(): p.write_text('{}')
    expected={k:report.sha(p) for k,p in paths.items()}
    recipe={'evaluation':{'wall_cap_seconds':1800,'output_cap_bytes':6_000_000_000},'lineage':{}}
    cases={'evaluation':[{'seed':i,'policy_order':list(report.POLICIES[i%3:]+report.POLICIES[:i%3])} for i in range(64)]}
    maps,restored,entries={}, {}, {}
    for name in report.NAMES:
        p=root/(name+'.json');report.write(p,{'final_weights_sha256':'a'*64})
        maps[name]={'checkpoint_sha256':'b'*64,'completed_path':p.name}
        restored[name]={'checkpoint_sha256':'b'*64,'weights_sha256':'a'*64}
        entries[name]={'beta':2.,'temperature':.5,'receipt_sha256':'c'*64}
    cal_done={'status':'complete','fits':entries,'wall_seconds':1.,'oracle_evaluations':54,'files':{}}
    report.write(calibration/'completed.json',cal_done);cal_sha=report.sha(calibration/'completed.json')
    monkeypatch.setattr(report,'authenticate_inputs',lambda *_:(recipe,cases,maps,{'files':{}},root,cases))
    monkeypatch.setattr(report,'audit_calibration',lambda *_:(cal_done,{n:{'beta':2.} for n in report.NAMES}))
    layout=np.arange(52,dtype=np.int64)//4;layout_sha=report.prior.layout_digest(layout)
    counters=('reset_attempted','reset_returned','identity_read_attempted','identity_read_returned',
        'init_attempted','init_returned','predict_attempted','predict_returned','write_attempted','write_returned',
        'decision_attempted','decision_returned','transform_attempted','transform_returned','native_attempted','native_returned')
    totals=dict.fromkeys(counters,0)
    def fake_audit(_path,receipt,**_kwargs):
        return {'return':receipt['return'],'success':True,'native_steps':52,'unique_positions':52,
            'repeat_mismatching_pairs':0,'layout':layout.copy(),'counts':receipt['counts'],
            'timings':{'native_seconds':.0001},'picker_counts':{'max_rowmass_abs':0.,'first_phase':26},'score_episode':{}}
    monkeypatch.setattr(report,'audit_episode',fake_audit)
    def fake_scores(episodes,policy,beta=None):
        assert len(episodes)==64
        return {'all':{'eligible_episodes':64,'query_cards':100,'nll':1. if policy=='baseline' else .96,
            'nll_is_infinite':False,'infinite_nll_queries':0,'brier':.1,'accuracy':.9}}
    monkeypatch.setattr(report,'proper_scores',fake_scores)
    report.write(evaluation/'started.json',{'status':'started','automatic_retry':False,'evaluation':recipe['evaluation'],
        'tracker_policy':'C','calibration_completed_sha256':cal_sha,**{k+'_sha256':v for k,v in expected.items()}})
    report.write(evaluation/'all-fits-ready.json',{'status':'complete','fits':maps,'calibration_fits':entries,
        'calibration_completed_sha256':cal_sha})
    for name in (*report.NAMES,*report.REFERENCES):
        neural=name in report.NAMES
        counts={k:1 if 'reset' in k or 'identity' in k else int(neural) if 'init' in k
                else 52*int(neural) if 'predict' in k or 'write' in k else 52 for k in counters}
        for policy in report.POLICIES if neural else ('baseline',):
            folder=evaluation/'controllers'/name/policy;(folder/'episodes').mkdir(parents=True)
            value=.1 if policy=='baseline' else .13
            for index in range(64):
                stem=folder/'episodes'/f'{index:03d}';stem.with_suffix('.npz').write_bytes(b'fake numerical payload')
                report.write(stem.with_suffix('.json'),{'index':index,'seed':index,'return':value,
                    'protocol_sha256':expected['protocol'],'inputs_sha256':expected['inputs'],
                    'checkpoint_sha256':'b'*64 if neural else None,'layout_sha256':layout_sha,
                    'calibration_completed_sha256':cal_sha,'calibration_receipt_sha256':'c'*64 if neural else None,
                    'array_bytes':1,'whole_episode_seconds':.001,'wall_seconds':.0001,
                    'construction_seconds':.0001,'close_seconds':.0001,'serialization_seconds':.0001,'counts':counts})
                for k,v in counts.items(): totals[k]+=v
            report.write(folder/'completed.json',{'status':'complete','controller':name,'policy':policy,'episodes':64,
                'mean_return':value,'successes':64,'native_steps':64*52,'whole_episode_seconds':.064,
                'controller_shared_restore_seconds':.01,'tracker_policy':'C','beta':2. if policy=='temperature' else None,
                'calibration_completed_sha256':cal_sha,'calibration_receipt_sha256':'c'*64 if neural else None})
    files={str(p.relative_to(evaluation)):{'sha256':report.sha(p),'bytes':p.stat().st_size} for p in evaluation.rglob('*') if p.is_file()}
    assert set(files)==report.expected_members()
    if corrupt_work: totals['write_returned']-=1
    complete={'status':'complete','version':'card-calibration-v1','controllers':20,'episodes':3584,
        'policies':list(report.POLICIES),'tracker_policy':'C','calibration_completed_sha256':cal_sha,
        'model_restores':18,'new_fits':0,'new_optimizer_steps':0,'native_replay_calls':0,'lineage':{},
        **{k+'_sha256':v for k,v in expected.items()},'started_sha256':report.sha(evaluation/'started.json'),
        'restored_models':restored,'counts':totals,'uncompressed_array_bytes':3584,
        'layout_sha256_by_case':{str(i):layout_sha for i in range(64)},'distinct_layouts':1,
        'payload_bytes':sum(v['bytes'] for v in files.values()),'files':files,'wall_seconds':6.,
        'controller_wall_seconds':dict.fromkeys((*report.NAMES,*report.REFERENCES),.25),'final_hashing_seconds':.1}
    report.write(evaluation/'completed.json',complete)
    kwargs={'root':root,**paths,**{'expected_'+k+'_sha256':v for k,v in expected.items()},
        'calibration':calibration,'expected_calibration_completed_sha256':cal_sha,
        'evaluation':evaluation,'expected_evaluation_completed_sha256':report.sha(evaluation/'completed.json'),'out':root/'report'}
    if corrupt_work:
        with pytest.raises(ValueError,match='Aggregate'):report.report(**kwargs)
        assert (root/'report'/'failed.json').exists() and not (root/'report'/'receipt.json').exists()
    else:
        result=report.report(**kwargs)
        assert result['status']=='complete' and result['continuation_passed'] is False
        summary=report.read(root/'report'/'summary.json')
        assert summary['coverage']['native_games']==3584 and len(summary['references'])==2
        assert summary['continuation_gate']['checks_passed']==11
        with pytest.raises(FileExistsError):report.report(**kwargs)
