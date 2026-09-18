# SPDX-License-Identifier: GPL-3.0-only
import copy
import json
from pathlib import Path
import sys

import chess
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import chess_pin_trained_cost as cost
from openjev.research.chess_candidate import CandidateChess
from openjev.research.chess_pin_cache import pack
from openjev.research.chess_pin_factors import candidate_factors, reference_witnesses

FENS = (chess.STARTING_FEN, 'k3r3/8/8/8/8/8/4N3/4K3 w - - 0 1',
        'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
        '7k/P7/8/8/8/8/8/7K w - - 0 1', '7k/8/8/3pP3/8/8/8/7K w - d6 0 1')


def models():
    cost.study.configure()
    model = CandidateChess('direct',seed=97,width=32,root_depth=1).eval().requires_grad_(False)
    heads = {'base':None, **{m:cost.study.mechanics.make_head(m,97).eval() for m in cost.METHODS[1:]}}
    with torch.no_grad():
        for head in heads.values():
            if head is not None:
                head.output.weight.copy_(torch.randn(head.output.weight.shape,generator=torch.Generator().manual_seed(881))*.1)
    return model,heads


@pytest.mark.parametrize('fen',FENS)
def test_complete_optimized_paths_match_full_native_inputs_with_nonzero_heads(fen):
    model,heads = models()
    # Real updates verify that equivalence is not an artifact of initial zero residuals.
    # Train on a pinned fixture so every factor parameter has a gradient; then
    # evaluate the resulting policy on pinned, empty-factor and special-move roots.
    with torch.no_grad(): train_names,train_args,train_factors = cost.study.native_arguments(model,FENS[1])
    target=torch.tensor([len(train_names)-1]); opt=cost.study.mechanics.optimizer(heads['joint'])
    for _ in range(3): cost.study.mechanics.update(heads['joint'],'joint',opt,train_args,train_factors,target)
    with torch.no_grad(): names,args,factors = cost.study.native_arguments(model,fen)
    with torch.no_grad():
        for method,head in heads.items():
            scores = args[4] if head is None else cost.study.mechanics.logits(head,method,args,factors)
            expected = cost.original.result(names,scores)
            observed = cost.decision(model,head,method,fen)
            check=cost.comparison(observed,expected)
            assert check['score_tolerance_passed'] and check['choice_matches'], (method,check)


def test_unused_input_work_is_skipped_and_bad_identity_rejected(monkeypatch):
    model,heads=models()
    def forbidden(*args,**kwargs): raise AssertionError('Unused input work was executed')
    monkeypatch.setattr(cost.original,'candidate_factors',forbidden)
    for m in ('base','wldn','union:edits','graph_mlp','root_only'):
        cost.decision(model,heads[m],m,FENS[1])
    monkeypatch.setattr(cost.source,'candidate_graphs',forbidden)
    monkeypatch.setattr(cost.source.prior,'candidate_features',forbidden)
    cost.decision(model,None,'base',FENS[1])
    with pytest.raises(ValueError):cost.decision(model,None,'joint',FENS[1])
    with pytest.raises(ValueError):cost.decision(model,heads['joint'],'unknown',FENS[1])
    with pytest.raises(ValueError):cost.decision(model,None,'base','8/8/8/8/8/8/8/8 w - - 0 1')


def test_protocol_membership_rotates_every_method_into_every_slot():
    keys=cost.order();assert len(keys)==len(set(keys))==31104
    for seed in cost.SEEDS:
        for i in range(128):
            group=[k for k in keys if k[:2]==(seed,i)]
            for method in cost.METHODS:
                assert sorted(j%9 for j,k in enumerate(group) if k[3]==method)==list(range(9))
    assert cost.PROTOCOL['native_audit_decisions']==3456
    assert cost.PROTOCOL['time_cap_seconds']==cost.PROTOCOL['audit_time_cap_seconds']==1800


def test_pin_coverage_matches_independent_detector_and_transition_records():
    boards=[chess.Board(f) for f in FENS]
    native=[candidate_factors(b) for b in boards]
    reference=[candidate_factors(b,reference=True) for b in boards]
    assert native==reference
    cache=pack(native,FENS); result=cost.coverage(cache)
    assert result['roots_with_root_pins']>0 and result['roots_with_candidate_pin_change']>0
    for i,b in enumerate(boards):
        row=result['roots'][i]
        assert row['root_pin_present']==bool(reference_witnesses(b,b.turn))
        changed=0
        for move in b.legal_moves:
            child=b.copy();child.push(move)
            changed += reference_witnesses(b,b.turn)!=reference_witnesses(child,b.turn)
        assert row['candidates_with_pin_change']==changed
        assert row['pin_change_present']==bool(changed)


def tiny(monkeypatch):
    p=copy.deepcopy(cost.PROTOCOL);p.update(roots=2,timed_records=162,native_audit_decisions=18)
    monkeypatch.setattr(cost,'PROTOCOL',p);monkeypatch.setattr(cost,'SEEDS',(97,))
    panel=[{'index':i,'id':f'fixture-{i}','fen':FENS[i]} for i in range(2)]
    covered=cost.coverage(pack([candidate_factors(chess.Board(r['fen'])) for r in panel],tuple(r['fen'] for r in panel)))
    return {'panel':panel,'pin_input_coverage':covered,'quality_plan_sha256':'quality-plan'}


def toy_records(plan):
    prediction={'menus':['a2a3','a2a4'],'scores':[1.,0.],'choice':'a2a3'}
    expected={(97,i,m):copy.deepcopy(prediction) for i in range(2) for m in cost.METHODS}
    def record(key):
        s,i,r,m=key
        item={'seed':s,'root_index':i,'repeat':r,'method':m,'id':plan['panel'][i]['id'],
              'prediction':copy.deepcopy(prediction),'comparison':cost.comparison(prediction,prediction)}
        item['milliseconds']=(2 if m=='wldn' else 6)*(1+i+r)
        return item
    return [record(k) for k in cost.order()], [record((97,0,r,m)) for m in cost.METHODS for r in range(2)],expected


def test_complete_summary_preserves_failures_and_checks_raw_vectors_membership(monkeypatch):
    plan=tiny(monkeypatch);records,warmups,expected=toy_records(plan)
    stats=cost.summarize(records,warmups,expected,plan)
    assert stats['timings']['median_paired_ratio_to_wldn']['joint']==3
    assert stats['cost_path_numerical_gate_passed'] and stats['timed_checks']['records']==162
    for bad in (records[:-1],records+[records[0]],records[::-1]):
        with pytest.raises(AssertionError):cost.summarize(bad,warmups,expected,plan)
    for key,value in (('milliseconds',float('nan')),('milliseconds',0),('id','different')):
        bad=copy.deepcopy(records);bad[0][key]=value
        with pytest.raises(AssertionError):cost.summarize(bad,warmups,expected,plan)
    records[0]['prediction']['scores'][0]+=2e-5
    records[0]['comparison']=cost.comparison(records[0]['prediction'],expected[97,0,records[0]['method']])
    stats=cost.summarize(records,warmups,expected,plan)
    assert not stats['cost_path_numerical_gate_passed'] and stats['timed_checks']['failed_records']==1
    warmups[0]['prediction'].update(scores=[0.,1.],choice='a2a4')
    warmups[0]['comparison']=cost.comparison(warmups[0]['prediction'],expected[97,0,warmups[0]['method']])
    assert cost.summarize(records,warmups,expected,plan)['warmup_checks']['choice_changes']==1
    records[0]['comparison']['max_score_error']=0
    with pytest.raises(AssertionError):cost.summarize(records,warmups,expected,plan)


def test_ratio_uses_matched_pairs_and_empty_strata_remain_null(monkeypatch):
    plan=tiny(monkeypatch);records,warmups,expected=toy_records(plan)
    for row in plan['pin_input_coverage']['roots']: row['root_pin_present']=False
    for r in records:
        k=r['root_index']*9+r['repeat']
        r['milliseconds']=float(1 if k<9 else 100)
        if r['method']=='joint':r['milliseconds']=float(2 if k<9 else 100)
    value=cost.summarize(records,warmups,expected,plan)
    stats=value['timings']
    assert stats['median_paired_ratio_to_wldn']['joint']==1.5
    assert stats['median_complete_ms']['joint']/stats['median_complete_ms']['wldn']!=1.5
    assert value['strata']['root_pin_present=true'] is None
    bad=copy.deepcopy(records[0]['prediction']);bad['scores'][0]=float('nan')
    with pytest.raises(AssertionError):cost.comparison(bad,expected[97,0,records[0]['method']])
    bad=copy.deepcopy(records[0]['prediction']);bad['choice']='a2a4'
    with pytest.raises(AssertionError):cost.comparison(bad,expected[97,0,records[0]['method']])


def test_preoutcome_freeze_rejects_evaluation_or_terminal_markers(tmp_path,monkeypatch):
    monkeypatch.setattr(cost,'ROOT',tmp_path)
    plan=tmp_path/cost.PLAN;plan.parent.mkdir(parents=True);plan.write_text('{}')
    execution=tmp_path/cost.EXECUTION;(execution/'fits').mkdir(parents=True)
    cost.study.write(execution/'started.json',{'unix':1.,'plan_sha256':cost.source.file_hash(plan)})
    assert cost.before_evaluation()['completed_fits']==0
    for name in ('all-training-complete.json','evaluation-started.json','predictions','native','summary.json','completed.json','failed.json'):
        (execution/name).write_text('{}')
        with pytest.raises(ValueError):cost.before_evaluation()
        (execution/name).unlink()


def test_no_quality_outputs_are_read_before_complete_audit(tmp_path,monkeypatch):
    monkeypatch.setattr(cost,'ROOT',tmp_path)
    def forbidden(*args,**kwargs):raise AssertionError('Premature quality result read')
    monkeypatch.setattr(cost.study,'read',forbidden)
    with pytest.raises(ValueError,match='full replay audit required'):cost.authenticate({})


def quality_fixture(tmp_path,monkeypatch):
    plan=tiny(monkeypatch);monkeypatch.setattr(cost,'ROOT',tmp_path)
    p=cost.study.protocol(cost.METHODS[1:],20000.);p['evaluation_roots_per_panel']=2
    plan.update(quality_signature={'protocol':p},freeze_state={'quality_started_unix':1.},prepared_unix=2.)
    directory=tmp_path/cost.EXECUTION;(directory/'native').mkdir(parents=True)
    audit_path=tmp_path/cost.AUDIT;audit_path.parent.mkdir(parents=True)
    (audit_path.parent/'native').mkdir()
    auditor=tmp_path/'scripts/chess_pin_quality_study_v2.py';auditor.parent.mkdir();auditor.write_text('fixture')
    numerical={'numerical_gate_passed':False,'native_choice_changes':1}
    saved={'status':'completed','plan_sha256':'quality-plan','fits':p['fits'],'training_updates':p['training_updates'],
           'prediction_records':p['prediction_records'],'gate_checks':[False]*16,'quality_gate_passed':False,
           **numerical,'continuation_passed':False,'new_engine_calls':0,'external_model_calls':0,
           'copied_reference_predictions':0,'original_failed_criteria_unchanged':True,'wall_seconds':100.}
    cost.study.write(directory/'summary.json',saved)
    cost.study.write(directory/'started.json',{'unix':1.})
    cost.study.write(directory/'all-training-complete.json',{'unix':3.})
    cost.study.write(directory/'evaluation-started.json',{'unix':4.,'training_stamp_sha256':cost.source.file_hash(directory/'all-training-complete.json')})
    files={}
    for split in cost.study.SPLITS:
        values=[{**r,'split':split,'seed':97,'menus':['a2a3','a2a4'],
                 'scores':{m:[1.,0.] for m in cost.METHODS}} for r in plan['panel']]
        path=f'native/97-{split}.jsonl'
        text=''.join(json.dumps(v)+'\n' for v in values)
        (directory/path).write_text(text);(audit_path.parent/path).write_text(text)
        files[path]=cost.source.file_hash(directory/path)
    cost.study.write(directory/'completed.json',{'status':'completed','plan_sha256':'quality-plan',
                     'files':{str(f.relative_to(directory)):cost.source.file_hash(f) for f in directory.rglob('*') if f.is_file()}})
    audit={'status':'completed','plan_sha256':'quality-plan','summary_sha256':cost.source.file_hash(directory/'summary.json'),
           'execution_receipt_sha256':cost.source.file_hash(directory/'completed.json'),'auditor_sha256':cost.source.file_hash(auditor),
           'training_updates_checked':p['training_updates'],'fresh_initial_states_exact':p['fits'],
           'final_checkpoint_identities_checked':p['fits'],'prediction_records_replayed':p['prediction_records'],
           'candidate_scores_replayed':p['candidate_scores'],'cached_scores_and_nll_exact':True,'native_vectors_exact':True,
           'native_root_backbone_reconstructions':p['native_root_backbone_reconstructions'],'gate_recomputed':saved['gate_checks'],
           'bootstrap_intervals_recomputed':p['quality_checks'],'continuation_passed':False,'scope':p['audit'],
           'numerical_gate_recomputed':numerical,'replay_native_files_sha256':files,'wall_seconds':20.}
    cost.study.write(audit_path,audit)
    return plan,audit_path,audit


def test_quality_failure_is_allowed_but_incomplete_audit_or_late_freeze_is_not(tmp_path,monkeypatch):
    plan,path,audit=quality_fixture(tmp_path,monkeypatch)
    expected,evidence=cost.authenticate(plan)
    assert len(expected)==18 and evidence['quality_gate_passed'] is False
    assert evidence['quality_numerical_gate_passed'] is False
    for field,value in (('candidate_scores_replayed',1),('cached_scores_and_nll_exact',False),
                        ('fresh_initial_states_exact',0),('bootstrap_intervals_recomputed',1),
                        ('replay_native_files_sha256',{}),('summary_sha256','wrong')):
        bad=copy.deepcopy(audit);bad[field]=value;path.write_text(json.dumps(bad))
        with pytest.raises(AssertionError):cost.authenticate(plan)
    path.write_text(json.dumps(audit));plan['prepared_unix']=3.
    with pytest.raises(AssertionError,match='freeze ordering'):cost.authenticate(plan)
    plan['prepared_unix']=2.;(path.parent/'native/97-dev.jsonl').write_text('corrupt')
    with pytest.raises(AssertionError,match='Native audit vectors'):cost.authenticate(plan)


def test_primary_and_audit_execute_complete_native_fixture_and_detect_saved_corruption(tmp_path,monkeypatch):
    plan=tiny(monkeypatch);model,heads=models();expected={}
    for row in plan['panel']:
        with torch.no_grad():
            names,args,factors=cost.study.native_arguments(model,row['fen'])
            for m,h in heads.items():
                scores=args[4] if h is None else cost.study.mechanics.logits(h,m,args,factors)
                expected[97,row['index'],m]=cost.original.result(names,scores)
    monkeypatch.setattr(cost,'ROOT',tmp_path);monkeypatch.setattr(cost.source,'PARENT','parent.json')
    (tmp_path/'parent.json').write_text('{}')
    monkeypatch.setattr(cost,'validate_plan',lambda path:(plan,'fixture-plan'))
    monkeypatch.setattr(cost,'authenticate',lambda plan:(expected,{'quality_gate_passed':False}))
    monkeypatch.setattr(cost.source.prior,'backbone',lambda *a:copy.deepcopy(model))
    monkeypatch.setattr(cost.study,'load_head',lambda execution,seed,method,hash:copy.deepcopy(heads[method]))
    execution=tmp_path/'execution';audit=tmp_path/'audit'
    cost.execute(tmp_path/'plan',execution)
    cost.execute(tmp_path/'plan',audit,execution)
    saved=cost.study.read(execution/'summary.json');receipt=cost.study.read(audit/'receipt.json')
    assert saved['timed_checks']['records']==162 and saved['warmup_checks']['records']==18
    assert saved['cost_path_numerical_gate_passed'] and receipt['native_audit']['records']==18
    assert receipt['cost_path_numerical_gate_passed'] and receipt['quality_evidence']['quality_gate_passed'] is False
    records=cost.source.prior.rows(execution/'timings.jsonl');records[0]['milliseconds']=0
    (execution/'timings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    # Even a self-consistent manifest cannot hide invalid duration arithmetic.
    complete=cost.study.read(execution/'completed.json')
    complete['files']['timings.jsonl']=cost.source.file_hash(execution/'timings.jsonl')
    (execution/'completed.json').write_text(json.dumps(complete))
    with pytest.raises(AssertionError,match='Invalid timing'):cost.execute(tmp_path/'plan',tmp_path/'bad-audit',execution)
    assert cost.study.read(tmp_path/'bad-audit/failed.json')['status']=='failed'


def test_deadlines_remain_bounded(monkeypatch):
    monkeypatch.setattr(cost.time,'monotonic',lambda:1802.)
    for audit in (False,True):
        with pytest.raises(TimeoutError):cost.deadline(1.,audit)
        cost.deadline(3.,audit)
