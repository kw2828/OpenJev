# SPDX-License-Identifier: GPL-3.0-only
"""Matched twelve-fit development study of difference-layer graph information."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch.nn import functional as F

import chess_union_difference_preflight as preflight
import chess_wldn_study as wldn
import chess_wldn_interventions as diagnostic
from openjev.research.chess_union_difference import ARMS, ChessUnionDifferenceHead

source=preflight.source
ROOT=source.ROOT
SEEDS=(97,109,127)
SPLITS=('dev','shift')
MODES=(*ARMS,'edits_corrupted')
PREFLIGHT_PLAN='evidence/chess-union-difference-preflight-v1/protocol/plan.json'
PREFLIGHT_AUDIT='evidence/chess-union-difference-preflight-v1/audit/receipt.json'
VERSION='wldn-union-edit-matched-quality-v1'
CODE=['scripts/chess_union_difference_study.py','tests/test_chess_union_difference_study.py']
PROTOCOL={
    'version':VERSION,'arms':list(ARMS),'seeds':list(SEEDS),'parameters':16740,
    'hypothesis':'With the shared WLDN nodewise encoder retained, labeling retained/added/removed relations on the union graph improves legal-move agreement beyond child-only, union-only and corrupted-edit controls.',
    'prior_knowledge':'All earlier development outcomes and fixed-weight WLDN intervention results known. No trained union-difference quality outcomes inspected before freeze. This is adaptive development, not independent confirmation.',
    'architecture':preflight.PROTOCOL['architecture'],
    'graph_arms':{a:preflight.PROTOCOL[a] for a in ARMS},
    'information':'Same frozen root features in root and child branches, native complete legal-move child graphs, original120 action features, no child backbone recomputation or teacher shortlist.',
    'capacity_limit':preflight.PROTOCOL['capacity_limit'],
    'training_examples':32768,'epochs':6,'batch_size':128,'updates_per_fit':1536,
    'fits':12,'total_updates':18432,'head_seed':'backbone_seed+1100',
    'fit_order':'Seed97,109,127; within each seed child,union,edits,rotated. Every final fit completes before any quality evaluation.',
    'batch_order_seed':'700000+100*backbone_seed+epoch','optimizer':'Adam','learning_rate':.001,
    'betas':[.9,.999],'eps':1e-8,'weight_decay':0.,'gradient_clip':1.,'loss':'Legal-move cross entropy only',
    'runtime':'CPU2 threads, deterministic algorithms',
    'evaluation':'Both complete old2048-position panels, all3 seeds, final weights only; no replacement, tuning, early quality peek or best-checkpoint selection.',
    'corrupted_diagnostic':'Reuse trained edits weights but apply rotated graph labels at evaluation; no extra fit. Keep native node differences and union connectivity.',
    'references':'Copy authenticated original WLDN and base predictions with provenance; both freshly replayed in audit.',
    'new_predictions':61440,'copied_predictions':24576,
    'gate':'All10 checks required: edits-minus-base mean>=0 and edits-minus-child/union/rotated/originalWLDN mean>=.01 on EACH panel; every paired seed>=-.005 in every check. No earlier failed criteria changed.',
    'bootstrap':'2000 source-game draws per comparison, paired correctness averaged across3 seeds then position weighted. Seed995101+shift_indicator; same draws for five comparators within each panel. Conditional descriptive95% intervals, no adaptive/multiple-testing correction.',
    'audit':'Reauthenticate frozen inputs and full preflight replay; check every minibatch index receipt, fresh initial states, training-before-evaluation stamp and trained checkpoint; native root/menu/label checks; replay all86016 new/copied predictions and recompute gates/intervals. Full neural replay uses production graph packing; independent Python graph-packing checks are bounded to the prior engineering screen. No full retraining or complete native child-cache reconstruction.',
    'time_cap_seconds':14400,'audit_time_cap_seconds':1800,
    'budget_rationale':'Preflight median update costs project about3 hours for18432 updates; freeze4-hour execution ceiling with evaluation/setup margin. No extension on timeout.',
    'limits':'Established WLDN and condensed-reaction-graph ingredients; this extension is not by itself an algorithmic novelty claim. Stored parameters do not match active capacity, FLOPs or prior tuning. No new engine calls, gameplay, independent confirmation or complete-native latency study.'}


def signature():
    p=json.loads((ROOT/PREFLIGHT_PLAN).read_text())
    assert all(p[k]==v for k,v in preflight.signature().items())
    execution=ROOT/'runs/chess-union-difference-preflight-v1/execution'
    diagnostic.manifest(execution,source.file_hash(ROOT/PREFLIGHT_PLAN))
    audit=json.loads((ROOT/PREFLIGHT_AUDIT).read_text())
    assert audit['status']=='completed' and audit['numerical_passed'] and audit['independent_python_packing']
    assert audit['numerical_replays']==96 and audit['candidate_checks']==2508
    assert audit['additional_artificial_update_replays']==12 and audit['final_checkpoint_max_error']==0
    assert audit['plan_sha256']==source.file_hash(ROOT/PREFLIGHT_PLAN)
    assert audit['summary_sha256']==source.file_hash(execution/'summary.json')
    assert audit['execution_receipt_sha256']==source.file_hash(execution/'completed.json')
    assert audit['auditor_sha256']==source.file_hash(ROOT/'scripts/chess_union_difference_preflight.py')
    return {'protocol':PROTOCOL,'sources':{p:source.file_hash(ROOT/p) for p in CODE},
            'preflight_signature':preflight.signature(),'preflight_plan_sha256':source.file_hash(ROOT/PREFLIGHT_PLAN),
            'preflight_audit_sha256':source.file_hash(ROOT/PREFLIGHT_AUDIT),
            'original_wldn_plan_sha256':source.file_hash(ROOT/diagnostic.PLAN)}


def prepare(out):
    p=signature();p['prepared_unix']=time.time();out.mkdir(parents=True,exist_ok=False)
    source.prior.write(out/'plan.json',p)
    print(json.dumps({'plan_sha256':source.file_hash(out/'plan.json'),'fits':12,'updates':18432}),flush=True)


def deadline(begin,audit=False):
    if time.time()-begin>PROTOCOL['audit_time_cap_seconds' if audit else 'time_cap_seconds']:
        raise TimeoutError('Frozen union-difference time ceiling exceeded')


def load(directory,seed,arm,plan_hash):
    fitted='edits' if arm=='edits_corrupted' else arm
    payload=torch.load(directory/f'{fitted}-{seed}/weights.pt',weights_only=True,map_location='cpu')
    assert payload['version']==VERSION and payload['seed']==seed and payload['arm']==fitted and payload['plan_sha256']==plan_hash
    head=ChessUnionDifferenceHead('rotated' if arm=='edits_corrupted' else arm,seed=seed+1100)
    head.load_state_dict(payload['state_dict']);return head.eval()


def train(seed,arm,model,data,cache,graphs,out,plan_hash,begin):
    out.mkdir();head=ChessUnionDifferenceHead(arm,seed=seed+1100)
    source.prior.save(out/'initial.pt',head.state_dict());start=time.perf_counter()
    optimizer=torch.optim.Adam(head.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    updates=0
    with (out/'learning.jsonl').open('x') as stream:
        for epoch in range(6):
            order=np.random.default_rng(700000+100*seed+epoch).permutation(32768)
            for offset in range(0,32768,128):
                deadline(begin);index=torch.from_numpy(order[offset:offset+128]);graph=graphs.batch(index)
                optimizer.zero_grad(set_to_none=True)
                logits=head(*source.arguments(model,data,cache,index,graph))
                loss=F.cross_entropy(logits,data['targets'][index]);assert torch.isfinite(loss)
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();updates+=1
                stream.write(json.dumps({'update':updates,'epoch':epoch,'examples':128,'loss':float(loss.detach()),
                    'gradient_norm':float(norm),'indices_sha256':hashlib.sha256(index.numpy().tobytes()).hexdigest()},allow_nan=False)+'\n');stream.flush()
            print(json.dumps({'fit':f'{arm}-{seed}','epoch':epoch+1,'updates':updates,'wall_seconds':time.time()-begin}),flush=True)
    deadline(begin)
    source.prior.save(out/'weights.pt',{'version':VERSION,'seed':seed,'arm':arm,'plan_sha256':plan_hash,'state_dict':head.state_dict()})
    source.prior.write(out/'training.json',{'status':'completed','updates':updates,'examples_seen':updates*128,
        'seconds':time.perf_counter()-start,'parameters':16740,'weights_sha256':source.file_hash(out/'weights.pt'),
        'initial_sha256':source.file_hash(out/'initial.pt'),'learning_sha256':source.file_hash(out/'learning.jsonl')})


@torch.no_grad()
def evaluate(head,model,data,cache,graphs,rows,begin,audit=False):
    records=[]
    for start in range(0,len(rows),128):
        deadline(begin,audit);index=torch.arange(start,min(start+128,len(rows)));graph=graphs.batch(index)
        assert graph['fens']==[r['fen'] for r in rows[start:start+128]]
        args=source.arguments(model,data,cache,index,graph)
        logits=args[4] if head is None else head(*args)
        records.extend(diagnostic.records_for(logits,data['targets'][index],graph,rows,start))
    return records


def gate(metrics):
    checks=[]
    for split in SPLITS:
        for arm in ('base','child','union','rotated','wldn'):
            gains=[metrics[f'edits-{seed}'][split]['agreement']-metrics[f'{arm}-{seed}'][split]['agreement'] for seed in SEEDS]
            threshold=0. if arm=='base' else .01
            checks.append({'split':split,'comparator':arm,'mean_gain':statistics.mean(gains),'minimum_paired_seed_gain':min(gains),
                           'required_mean_gain':threshold,'passed':statistics.mean(gains)>=threshold and min(gains)>=-.005})
    return checks


def analyze(directory):
    metrics={};intervals=[]
    for split in SPLITS:
        rows=source.prior.rows(ROOT/source.prior.DATA[split]);assert len(rows)==2048
        records={}
        for seed in SEEDS:
            for arm in (*MODES,'base','wldn'):
                r=diagnostic.aligned(source.prior.rows(directory/f'{arm}-{seed}-{split}.jsonl'),rows)
                records[arm,seed]=r;metrics.setdefault(f'{arm}-{seed}',{})[split]=diagnostic.metrics(r)
        ids=np.array([r['game_id'] for r in rows]);groups=[np.flatnonzero(ids==g) for g in sorted(set(ids))]
        sizes=np.array([len(g) for g in groups]);draw=np.random.default_rng(995101+(split=='shift')).integers(0,len(groups),(2000,len(groups)))
        for comparator in ('base','child','union','rotated','wldn'):
            delta=np.mean([[int(a['correct'])-int(b['correct']) for a,b in zip(records['edits',seed],records[comparator,seed])] for seed in SEEDS],axis=0)
            sums=np.array([delta[g].sum() for g in groups])
            intervals.append({'split':split,'comparator':comparator,'games':len(groups),'point_gain':float(delta.mean()),
                'percentile95':np.quantile(sums[draw].sum(1)/sizes[draw].sum(1),[.025,.975]).tolist()})
    checks=gate(metrics)
    return {'metrics':metrics,'gate_checks':checks,'continuation_passed':all(c['passed'] for c in checks),'conditional_game_bootstrap':intervals}


def run(plan_path,out):
    plan=json.loads(plan_path.read_text());assert all(plan[k]==v for k,v in signature().items())
    plan_hash=source.file_hash(plan_path);out.mkdir(parents=True,exist_ok=False);begin=time.time()
    source.prior.write(out/'started.json',{'unix':begin,'pid':os.getpid(),'plan_sha256':plan_hash})
    try:
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        old=json.loads((ROOT/source.PARENT).read_text())
        data=torch.load(ROOT/source.PREVIOUS/'packed-train.pt',weights_only=True,map_location='cpu')
        graphs=source.ChildGraphCache(ROOT/source.CACHE/'train');cache_checks=[]
        for seed in SEEDS:
            deadline(begin);model=source.prior.backbone(seed,old);cache=source.prior.root_cache(model,data)
            saved=torch.load(ROOT/source.PREVIOUS/f'root-cache-train-{seed}.pt',weights_only=True,map_location='cpu')
            assert torch.equal(torch.isneginf(cache['base_logits']),torch.isneginf(saved['base_logits']))
            error=max(float((cache['nodes']-saved['nodes']).abs().max()),float((cache['base_logits'][data['mask']]-saved['base_logits'][data['mask']]).abs().max()))
            assert error<=2e-6;cache_checks.append({'seed':seed,'roots':32768,'max_error':error});del saved
            for arm in ARMS:train(seed,arm,model,data,cache,graphs,out/f'{arm}-{seed}',plan_hash,begin)
            del model,cache;gc.collect()
        del data,graphs;gc.collect()
        source.prior.write(out/'training-cache-check.json',cache_checks)
        source.prior.write(out/'all-training-complete.json',{'fits':12,'updates':18432,'before_any_evaluation':True,'unix':time.time(),'plan_sha256':plan_hash})
        references=[]
        for split in SPLITS:
            rows=source.prior.rows(ROOT/source.prior.DATA[split]);assert len(rows)==2048
            data=torch.load(ROOT/source.PREVIOUS/f'packed-{split}.pt',weights_only=True,map_location='cpu');graphs=source.ChildGraphCache(ROOT/source.CACHE/split)
            for seed in SEEDS:
                model=source.prior.backbone(seed,old);cache=source.prior.root_cache(model,data)
                for arm in MODES:
                    head=load(out,seed,arm,plan_hash);records=evaluate(head,model,data,cache,graphs,rows,begin)
                    with (out/f'{arm}-{seed}-{split}.jsonl').open('x') as stream:
                        for r in records:stream.write(json.dumps(r,allow_nan=False)+'\n')
                for arm in ('base','wldn'):
                    name=f'{arm}-{seed}-{split}.jsonl';origin=ROOT/diagnostic.EXECUTION/name
                    with (out/name).open('xb') as stream:stream.write(origin.read_bytes())
                    references.append({'file':name,'copied_from':str(origin.relative_to(ROOT)),'sha256':source.file_hash(origin),'fresh_inference':False})
                del model,cache,head;gc.collect()
            del data,graphs;gc.collect()
        result=analyze(out);deadline(begin)
        result.update({'status':'completed','plan_sha256':plan_hash,'fits':12,'training_updates':18432,
            'new_prediction_records':61440,'copied_reference_records':24576,'new_engine_calls':0,'external_model_calls':0,
            'original_failed_criteria_unchanged':True,'wall_seconds':time.time()-begin,'limits':PROTOCOL['limits']})
        source.prior.write(out/'reference-provenance.json',references);source.prior.write(out/'summary.json',result)
        source.prior.write(out/'completed.json',{'status':'completed','plan_sha256':plan_hash,
            'files':{str(p.relative_to(out)):source.file_hash(p) for p in out.rglob('*') if p.is_file()}})
        print(json.dumps({'status':'completed','wall_seconds':result['wall_seconds'],'continuation_passed':result['continuation_passed']}),flush=True)
    except BaseException as error:
        source.prior.write(out/'failed.json',{'status':'failed','error':repr(error),'wall_seconds':time.time()-begin});raise


def check_learning(records,seed):
    assert len(records)==1536
    for epoch in range(6):
        order=np.random.default_rng(700000+100*seed+epoch).permutation(32768)
        for j in range(256):
            r=records[epoch*256+j];index=order[j*128:(j+1)*128]
            assert r['update']==epoch*256+j+1 and r['epoch']==epoch and r['examples']==128
            assert r['indices_sha256']==hashlib.sha256(index.tobytes()).hexdigest()
            assert math.isfinite(r['loss']) and r['loss']>=0 and math.isfinite(r['gradient_norm']) and r['gradient_norm']>=0


def audit(plan_path,execution,out):
    plan=json.loads(plan_path.read_text());assert all(plan[k]==v for k,v in signature().items())
    plan_hash=source.file_hash(plan_path);diagnostic.manifest(execution,plan_hash)
    summary=json.loads((execution/'summary.json').read_text())
    assert summary['status']=='completed' and summary['plan_sha256']==plan_hash
    assert summary['fits']==12 and summary['training_updates']==18432
    assert summary['new_prediction_records']==61440 and summary['copied_reference_records']==24576
    assert summary['new_engine_calls']==summary['external_model_calls']==0 and summary['original_failed_criteria_unchanged']
    assert summary['limits']==PROTOCOL['limits'] and 0<summary['wall_seconds']<=14400
    out.mkdir(parents=True,exist_ok=False);begin=time.time()
    source.prior.write(out/'started.json',{'unix':begin,'pid':os.getpid(),'plan_sha256':plan_hash})
    try:
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        stamp=json.loads((execution/'all-training-complete.json').read_text())
        assert stamp['fits']==12 and stamp['updates']==18432 and stamp['before_any_evaluation'] and stamp['plan_sha256']==plan_hash
        checks=json.loads((execution/'training-cache-check.json').read_text())
        assert [r['seed'] for r in checks]==list(SEEDS) and all(r['roots']==32768 and 0<=r['max_error']<=2e-6 for r in checks)
        for seed in SEEDS:
            for arm in ARMS:
                deadline(begin,True);directory=execution/f'{arm}-{seed}';meta=json.loads((directory/'training.json').read_text())
                assert meta['status']=='completed' and meta['updates']==1536 and meta['examples_seen']==196608 and meta['parameters']==16740
                assert math.isfinite(meta['seconds']) and meta['seconds']>0
                for key,name in [('initial_sha256','initial.pt'),('weights_sha256','weights.pt'),('learning_sha256','learning.jsonl')]:
                    assert meta[key]==source.file_hash(directory/name)
                assert (directory/'weights.pt').stat().st_mtime<=stamp['unix']
                initial=torch.load(directory/'initial.pt',weights_only=True,map_location='cpu')
                fresh=ChessUnionDifferenceHead(arm,seed=seed+1100).state_dict()
                assert initial.keys()==fresh.keys() and all(torch.equal(v,initial[k]) for k,v in fresh.items())
                assert not initial['output.weight'].count_nonzero()
                head=load(execution,seed,arm,plan_hash)
                assert sum(p.numel() for p in head.parameters())==16740 and all(torch.isfinite(p).all() for p in head.parameters())
                assert (head.output.weight!=initial['output.weight']).any()
                check_learning(source.prior.rows(directory/'learning.jsonl'),seed)
        references=json.loads((execution/'reference-provenance.json').read_text())
        expected={f'{a}-{s}-{p}.jsonl' for a in ('base','wldn') for s in SEEDS for p in SPLITS}
        assert len(references)==12 and {r['file'] for r in references}==expected
        for r in references:
            assert r['copied_from']==f"{diagnostic.EXECUTION}/{r['file']}" and not r['fresh_inference']
            assert r['sha256']==source.file_hash(ROOT/r['copied_from'])==source.file_hash(execution/r['file'])
        old=json.loads((ROOT/source.PARENT).read_text());count=0;error=0.;metrics={}
        for split in SPLITS:
            rows=source.prior.rows(ROOT/source.prior.DATA[split]);assert len(rows)==2048
            data=torch.load(ROOT/source.PREVIOUS/f'packed-{split}.pt',weights_only=True,map_location='cpu')
            diagnostic.native_inputs(rows,data);graphs=source.ChildGraphCache(ROOT/source.CACHE/split)
            for seed in SEEDS:
                model=source.prior.backbone(seed,old);cache=source.prior.root_cache(model,data)
                for arm in (*MODES,'base','wldn'):
                    head=None if arm=='base' else wldn.load(ROOT/diagnostic.EXECUTION,seed,plan['original_wldn_plan_sha256']) if arm=='wldn' else load(execution,seed,arm,plan_hash)
                    actual=evaluate(head,model,data,cache,graphs,rows,begin,True)
                    path=execution/f'{arm}-{seed}-{split}.jsonl';assert path.stat().st_mtime>=stamp['unix']
                    expected=diagnostic.aligned(source.prior.rows(path),rows)
                    error=max(error,diagnostic.compare_records(actual,expected));count+=len(actual)
                    metrics.setdefault(f'{arm}-{seed}',{})[split]=diagnostic.metrics(expected)
                del head,model,cache;gc.collect()
            del data,graphs;gc.collect()
            print(json.dumps({'panel':split,'replayed_predictions':count}),flush=True)
        assert count==86016 and metrics==summary['metrics']
        recomputed=analyze(execution);assert all(summary[k]==v for k,v in recomputed.items());deadline(begin,True)
        result={'status':'completed','plan_sha256':plan_hash,'summary_sha256':source.file_hash(execution/'summary.json'),
            'execution_receipt_sha256':source.file_hash(execution/'completed.json'),'auditor_sha256':source.file_hash(__file__),
            'evaluation_predictions_replayed':count,'new_head_predictions_replayed':61440,'copied_reference_predictions_replayed':24576,
            'training_updates_checked':18432,'fresh_initial_states_exact':True,'max_replay_nll_error':error,
            'gate_recomputed':summary['gate_checks'],'bootstrap_intervals_recomputed':10,'wall_seconds':time.time()-begin,
            'scope':PROTOCOL['audit']}
        source.prior.write(out/'receipt.json',result);print(json.dumps(result),flush=True)
    except BaseException as error:
        source.prior.write(out/'failed.json',{'status':'failed','error':repr(error),'wall_seconds':time.time()-begin});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','audit'])
    p.add_argument('--plan',type=Path);p.add_argument('--execution',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.command=='prepare':prepare(a.out)
    elif a.command=='run':run(a.plan,a.out)
    else:audit(a.plan,a.execution,a.out)
