# SPDX-License-Identifier: GPL-3.0-only
import copy
import json
from pathlib import Path
import sys
import time

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import chess_pin_quality_study_v2 as study
from openjev.research.chess_child_graph_cache import ChildGraphCache, write_cache
from openjev.research.chess_training_inputs import TrainingInputs
from test_chess_training_inputs import fixture

ARMS = (*study.mechanics.CORE_ARMS,'union:edits')


def inputs(tmp_path):
    boards, model, features, pins = fixture(); data = TrainingInputs(features,pins)
    rows = [{'id':f'fixture-{i}','game_id':i,'fen':b.fen(en_passant='fen'),
             'target_uci':data.features['menus'][i][-1]} for i,b in enumerate(boards)]
    write_cache(rows,tmp_path/'graphs',provenance={'constructed_fixture':True})
    graphs = ChildGraphCache(tmp_path/'graphs')
    return rows,model,data,graphs,study.targets_for(rows,data,graphs)


def tiny_protocol():
    p = study.protocol(ARMS,20000.)
    p.update(training_roots=3,epochs=3,batch_size=3,updates_per_fit=3,
             arms=['joint'],methods=['base','joint'],seeds=[97],fits=1,training_updates=3)
    return p


def test_complete_protocol_preserves_selected_scope():
    p = study.protocol(ARMS,20573.874974548817)
    assert p['fits']==24 and p['training_updates']==36864
    assert p['prediction_records']==110592 and p['candidate_scores']==3226149
    assert p['native_root_backbone_reconstructions']==12288 and p['quality_checks']==16
    assert p['score_tolerance']==1e-5 and p['time_cap_seconds']==43200 and p['audit_time_cap_seconds']==7200
    assert p['parameters']['wldn']==16638 and p['parameters']['counts']==16656
    assert p['parameters']['joint']==16658 and p['parameters']['union:edits']==16740
    with pytest.raises(ValueError): study.protocol(ARMS,float('nan'))
    with pytest.raises(ValueError): study.protocol(ARMS[1:],100.)


def test_teacher_targets_bind_to_native_complete_menus(tmp_path):
    rows,_,data,graphs,targets = inputs(tmp_path)
    assert targets.tolist()==[len(m)-1 for m in data.features['menus']]
    for mutation in ('illegal','duplicate','fen','order'):
        bad = copy.deepcopy(rows)
        if mutation=='illegal': bad[0]['target_uci']='a1a8'
        if mutation=='duplicate': bad[0]['id']=bad[1]['id']
        if mutation=='fen': bad[0]['fen']=bad[1]['fen']
        if mutation=='order': bad=bad[::-1]
        with pytest.raises(AssertionError): study.targets_for(bad,data,graphs)


def trained_fixture(tmp_path):
    rows,model,data,graphs,targets = inputs(tmp_path)
    execution = tmp_path/'execution'; (execution/'fits').mkdir(parents=True)
    p = tiny_protocol(); directory = execution/'fits'/'joint-97'
    study.train_fit(97,'joint',data,graphs,targets,directory,'frozen-test-plan',p,time.monotonic())
    return execution,p,rows,model,data,graphs,targets


def test_fit_checkpoint_identity_and_full_batch_receipts(tmp_path):
    execution,p,*_ = trained_fixture(tmp_path)
    directory = execution/'fits'/'joint-97'
    records = study.source.prior.rows(directory/'learning.jsonl')
    study.check_learning(records,97,p)
    study.training_inventory(execution,p,'frozen-test-plan',replay_states=True)
    loaded = study.load_head(execution,97,'joint','frozen-test-plan')
    assert loaded.output.weight.count_nonzero() > 0
    for bad in (records[:-1],records[::-1]):
        with pytest.raises(AssertionError): study.check_learning(bad,97,p)
    bad = copy.deepcopy(records); bad[0]['loss']=float('nan')
    with pytest.raises(AssertionError): study.check_learning(bad,97,p)
    with pytest.raises(AssertionError): study.load_head(execution,97,'joint','different-plan')
    with pytest.raises(AssertionError): study.check_learning(records,109,p)


def test_every_fit_must_finish_before_evaluation_and_stamp_binds_receipts(tmp_path):
    execution,p,*_ = trained_fixture(tmp_path)
    missing = copy.deepcopy(p); missing['arms'].append('wldn'); missing['fits']=2
    with pytest.raises(AssertionError): study.complete_training(execution,missing,'frozen-test-plan')
    assert not (execution/'all-training-complete.json').exists()
    (execution/'predictions').mkdir()
    with pytest.raises(AssertionError): study.complete_training(execution,p,'frozen-test-plan')
    (execution/'predictions').rmdir()
    stamp = study.complete_training(execution,p,'frozen-test-plan')
    assert stamp['fits']==1 and stamp['updates']==3 and stamp['before_any_evaluation']
    assert study.evaluation_barrier(execution,p,'frozen-test-plan')==stamp
    meta=execution/'fits'/'joint-97'/'training.json';value=study.read(meta);value['seconds']+=1
    meta.write_text(json.dumps(value))
    with pytest.raises(AssertionError): study.evaluation_barrier(execution,p,'frozen-test-plan')


def test_complete_cached_native_capture_replays_all_methods_and_detects_corruption(tmp_path):
    rows,model,data,graphs,targets = inputs(tmp_path);p=study.protocol(ARMS,20000.)
    p['batch_size']=3
    heads={'base':None,**{a:study.mechanics.make_head(a,97).eval() for a in ARMS}}
    # Nonzero projections exercise each branch; the joint head also gets real optimizer updates on fixture labels.
    with torch.no_grad():
        for arm,head in heads.items():
            if head is not None:
                head.output.weight.copy_(torch.randn(head.output.weight.shape,generator=torch.Generator().manual_seed(193))*.1)
    args,factors=data.batch(torch.arange(3),graphs.batch(torch.arange(3)))
    opt=study.mechanics.optimizer(heads['joint'])
    for _ in range(3): study.mechanics.update(heads['joint'],'joint',opt,args,factors,targets)
    execution=tmp_path/'execution';(execution/'predictions').mkdir(parents=True);(execution/'native').mkdir()
    checks=study.evaluate_loaded(data,graphs,targets,rows,heads,model,97,'dev',execution,execution,p,time.monotonic())
    assert len(checks)==3 and all(set(r['comparisons'])==set(p['methods']) for r in checks)
    assert all(c['score_tolerance_passed'] and c['choice_matches'] for r in checks for c in r['comparisons'].values())
    for arm in p['methods']:
        saved=study.source.prior.rows(execution/'predictions'/f'{study.name(arm,97)}-dev.jsonl')
        assert len(saved)==3 and all(len(r['scores'])==len(r['menus']) for r in saved)
    audit=tmp_path/'audit';(audit/'native').mkdir(parents=True)
    replay=study.evaluate_loaded(data,graphs,targets,rows,heads,model,97,'dev',execution,audit,p,time.monotonic(),audit=True)
    assert replay==checks
    native_file=execution/'native'/'97-dev.jsonl';original=native_file.read_text()
    values=[json.loads(line) for line in original.splitlines()];values[0]['scores']['joint'][0]+=.01
    native_file.write_text(''.join(json.dumps(r)+'\n' for r in values))
    failed=tmp_path/'audit-corrupt-native';(failed/'native').mkdir(parents=True)
    with pytest.raises(AssertionError,match='Native score/comparison'):
        study.evaluate_loaded(data,graphs,targets,rows,heads,model,97,'dev',execution,failed,p,time.monotonic(),audit=True)
    native_file.write_text(original)
    pred=execution/'predictions'/'joint-97-dev.jsonl';values=study.source.prior.rows(pred);values[0]['scores'][0]+=.01
    pred.write_text(''.join(json.dumps(r)+'\n' for r in values))
    failed=tmp_path/'audit-corrupt-cached';(failed/'native').mkdir(parents=True)
    with pytest.raises(AssertionError,match='Cached score/NLL'):
        study.evaluate_loaded(data,graphs,targets,rows,heads,model,97,'dev',execution,failed,p,time.monotonic(),audit=True)


def test_finite_numerical_failures_are_retained_and_fail_complete_gate():
    p=study.protocol(ARMS,20000.);p.update(evaluation_roots_per_panel=2,native_root_backbone_reconstructions=12,
                                        prediction_records=108,candidate_scores=216)
    def case(): return study.vector_comparison([1.,0.],[1.,0.],['a2a3','a2a4'],p['score_tolerance'])
    records=[{'split':split,'seed':seed,'index':i,'candidates':2,
              'comparisons':{a:case() for a in p['methods']}} for split in study.SPLITS for seed in study.SEEDS for i in range(2)]
    assert study.numerical_summary(records,p)['numerical_gate_passed']
    records[0]['comparisons']['joint']=study.vector_comparison([1.,0.],[1.00002,0.],['a2a3','a2a4'],1e-5)
    result=study.numerical_summary(records,p)
    assert not result['numerical_gate_passed'] and result['failed_native_method_cases']==1 and result['native_choice_changes']==0
    records[1]['comparisons']['joint']=study.vector_comparison([1.,0.],[0.,1.],['a2a3','a2a4'],1e-5)
    result=study.numerical_summary(records,p)
    assert result['failed_native_method_cases']==2 and result['native_choice_changes']==1
    assert result['candidate_score_comparisons']==216 and result['native_method_comparisons']==108
    with pytest.raises(AssertionError): study.numerical_summary(records[:-1],p)
    with pytest.raises(AssertionError): study.vector_comparison([float('nan'),0.],[1.,0.],['a2a3','a2a4'],1e-5)
    with pytest.raises(AssertionError): study.vector_comparison([1.],[1.,0.],['a2a3','a2a4'],1e-5)


def test_deadlines_stop_execution_and_audit_without_extension(monkeypatch):
    p=study.protocol(ARMS,20000.)
    monkeypatch.setattr(study.time,'monotonic',lambda:50000.)
    with pytest.raises(TimeoutError): study.deadline(1.,p)
    study.deadline(7000.,p)
    with pytest.raises(TimeoutError): study.deadline(7000.,p,audit=True)


def test_evaluation_membership_rejects_missing_and_extra_artifacts(tmp_path):
    p=study.protocol(ARMS,20000.)
    (tmp_path/'predictions').mkdir();(tmp_path/'native').mkdir()
    for split in study.SPLITS:
        for seed in p['seeds']:
            (tmp_path/'native'/f'{seed}-{split}.jsonl').write_text('')
            for arm in p['methods']:
                (tmp_path/'predictions'/f'{study.name(arm,seed)}-{split}.jsonl').write_text('')
    study.evaluation_membership(tmp_path,p)
    extra=tmp_path/'predictions'/'joint-999-dev.jsonl';extra.write_text('')
    with pytest.raises(AssertionError):study.evaluation_membership(tmp_path,p)
    extra.unlink()
    (tmp_path/'native'/'97-shift.jsonl').unlink()
    with pytest.raises(AssertionError):study.evaluation_membership(tmp_path,p)


def test_historical_metadata_is_checked_before_training_and_keeps_integer_zero(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'ROOT',tmp_path)
    directory=tmp_path/study.selection.RUN;directory.mkdir(parents=True)
    for split in study.SPLITS:
        values=[{'game_id':i//128} for i in range(2048)]
        (directory/f'base-97-{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values))
    metadata=study.historical_game_metadata()
    assert all(r['game_id_type']=='int' and r['games']==16 and r['records']==2048 for r in metadata.values())
    values[0]['game_id']='0'
    (directory/'base-97-dev.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values))
    with pytest.raises(ValueError):study.historical_game_metadata()
