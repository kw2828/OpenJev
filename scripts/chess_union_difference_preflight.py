# SPDX-License-Identifier: GPL-3.0-only
"""Frozen engineering screen for union/edit propagation of WLDN node changes."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time

import chess
import torch
from torch.nn import functional as F
import chess_wldn_preflight as parent
import chess_wldn_interventions as diagnostic
from openjev.research.chess_union_difference import (
    ARMS, VERSION, ChessUnionDifferenceHead, difference_edges, reference_edges)

source = parent.source
ROOT = source.ROOT
CODE = ['src/openjev/research/chess_union_difference.py', 'tests/test_chess_union_difference.py',
        'scripts/chess_union_difference_preflight.py']
DIAGNOSTIC_PLAN = 'evidence/chess-wldn-interventions-v1/protocol/plan.json'
DIAGNOSTIC_AUDIT = 'evidence/chess-wldn-interventions-v1/audit/receipt.json'
PROTOCOL = {
    'version': VERSION, 'scope': 'Engineering only; no teacher agreement or architecture-quality outcome',
    'arms': list(ARMS), 'parameters': 16740,
    'architecture': 'Original shared width32 three-step WL encoder on root and child, nodewise child-minus-root; one width32 difference step with12 edge inputs;sum64;120 action channels;58-unit readout;fresh zero output.',
    'child': 'Child adjacency and4 direction/color flags in first4 slots,8 zero slots.',
    'union': 'Union adjacency and4 union direction/color flags in first4 slots,8 zero slots.',
    'edits': 'Union adjacency;12 flags grouped retained,added,removed, each with4 direction/color channels.',
    'rotated': 'Same union support. Within every compact candidate and direction/color channel, rotate edit classes over nonzero ordered square pairs by ceil(n/2). Preserve union flags and all class counts. Fixed square order, no teacher dependence; not globally invertible by renaming classes. Reverse consistency may be broken; homogeneous/singleton groups can remain unchanged.',
    'capacity_limit': 'Same stored parameters and initialized tensors across arms; child/union leave8 input channels unused, so active capacity and graph-dependent compute are not identical. Original WLDN has16638 parameters and59 readout units, not the same initialized head.',
    'backbone_seed': 97, 'probe_seeds': [211,223,227], 'probe_roots': 8,
    'probe_order': 'First8 old training roots, all legal candidates. No label selection.',
    'probe_projection': 'Same normal(0,.1) projection, generator seed900000+head_seed, all arms.',
    'probe_comparison': 'Shared batch versus individual candidate calls; fresh native graph reconstruction; vectorized edit packing versus independent Python extraction/rotation. Audit uses independent packing inside single-candidate head forwards.',
    'absolute_score_tolerance': 1e-5,
    'training_roots': 128, 'head_seed': 1197,
    'artificial_updates_per_arm': 3, 'total_artificial_updates': 12,
    'target': 'First legal move for every root, no teacher target use.',
    'optimizer': 'Adam lr.001,betas(.9,.999),eps1e-8,weight_decay0;clip norm1.',
    'runtime': 'CPU2 threads, deterministic algorithms', 'time_cap_seconds': 900,
    'cost': 'Artificial update includes cached native child decoding, union/edit packing, all model forwards,loss,backward,clip,Adam. Setup timed separately; no full native latency advantage claim.',
    'limits': 'Known WLDN/contrast/CGR/NBF outcomes and fixed-weight diagnostic motivate this adaptive extension. WLDN and condensed reaction graphs are established prior methods. Engineering parity and trainability do not establish quality, biological relevance or algorithmic novelty.'}


def signature():
    prepared = json.loads((ROOT/DIAGNOSTIC_PLAN).read_text())
    assert all(prepared[k] == v for k,v in diagnostic.signature().items())
    execution = ROOT/'runs/chess-wldn-interventions-v1/execution'
    diagnostic.manifest(execution,source.file_hash(ROOT/DIAGNOSTIC_PLAN))
    audit = json.loads((ROOT/DIAGNOSTIC_AUDIT).read_text())
    assert audit['status'] == 'completed' and audit['evaluation_predictions_replayed'] == 86016
    assert audit['max_replay_nll_error'] == 0 and audit['bootstrap_intervals_recomputed'] == 10
    assert audit['plan_sha256'] == source.file_hash(ROOT/DIAGNOSTIC_PLAN)
    assert audit['summary_sha256'] == source.file_hash(execution/'summary.json')
    assert audit['execution_receipt_sha256'] == source.file_hash(execution/'completed.json')
    assert audit['auditor_sha256'] == source.file_hash(ROOT/'scripts/chess_wldn_interventions.py')
    return {'protocol':PROTOCOL,'sources':{p:source.file_hash(ROOT/p) for p in CODE},
            'diagnostic_signature':diagnostic.signature(),
            'diagnostic_plan_sha256':source.file_hash(ROOT/DIAGNOSTIC_PLAN),
            'diagnostic_audit_sha256':source.file_hash(ROOT/DIAGNOSTIC_AUDIT)}


def deadline(begin):
    if time.time()-begin > 900: raise TimeoutError('Frozen15-minute engineering ceiling exceeded')


def prepare(out):
    plan = signature();plan['prepared_unix']=time.time()
    out.mkdir(parents=True,exist_ok=False);source.prior.write(out/'plan.json',plan)
    print(json.dumps({'plan_sha256':source.file_hash(out/'plan.json'),'artificial_updates':12}),flush=True)


@torch.no_grad()
def probes(model,data,features,graphs,rows,begin,auditing):
    batch = graphs.batch(torch.arange(8));args=source.arguments(model,data,features,torch.arange(8),batch)
    native=source.candidate_graphs([chess.Board(r['fen']) for r in rows[:8]])
    assert torch.equal(batch['root'],native['root']) and torch.equal(batch['children'],native['children'][native['mask']])
    assert batch['menus']==list(map(tuple,native['menus']))
    owner,_=args[3].nonzero(as_tuple=True);expanded=args[5][owner]
    for arm in ARMS:
        a=difference_edges(expanded,args[6],arm,torch.float32)
        b=reference_edges(expanded,args[6],arm,torch.float32)
        assert all(torch.equal(x,y) for x,y in zip(a,b))
    records=[]
    for seed in PROTOCOL['probe_seeds']:
        for arm in ARMS:
            deadline(begin);head=ChessUnionDifferenceHead(arm,seed=seed)
            assert torch.equal(head(*args),args[4])
            g=torch.Generator().manual_seed(900000+seed)
            head.output.weight.copy_(torch.randn(head.output.weight.shape,generator=g)*.1)
            batched=head(*args);offset=0
            for b,names in enumerate(batch['menus']):
                deadline(begin);error=0.
                for m in range(len(names)):
                    single=[args[0][b:b+1],args[1][b:b+1,m:m+1],args[2][b:b+1,m:m+1],
                        torch.ones(1,1,dtype=torch.bool),args[4][b:b+1,m:m+1],args[5][b:b+1],args[6][offset+m:offset+m+1]]
                    value=head(*single,reference=auditing)[0,0]
                    assert torch.isfinite(value)
                    error=max(error,float((value-batched[b,m]).abs()))
                offset+=len(names)
                records.append({'seed':seed,'arm':arm,'root_index':b,'id':rows[b]['id'],'candidates':len(names),
                                'max_score_error':error,'passed':error<=1e-5})
    return records


def updates(model,data,features,graphs,begin):
    states={};records={};common=None
    for arm in ARMS:
        head=ChessUnionDifferenceHead(arm,seed=1197)
        initial={k:v.clone() for k,v in head.state_dict().items()}
        assert sum(p.numel() for p in head.parameters())==16740
        if common is None:common=initial
        assert all(torch.equal(v,common[k]) for k,v in initial.items())
        opt=torch.optim.Adam(head.parameters(),lr=.001)
        logs=[]
        for update in range(3):
            deadline(begin);start=time.perf_counter();index=torch.arange(128);graph=graphs.batch(index)
            opt.zero_grad(set_to_none=True)
            logits=head(*source.arguments(model,data,features,index,graph))
            loss=F.cross_entropy(logits,torch.zeros(128,dtype=torch.long));assert torch.isfinite(loss)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            opt.step();logs.append({'update':update+1,'loss':float(loss.detach()),'gradient_norm':float(norm),'seconds':time.perf_counter()-start})
        final=head.state_dict();assert (final['output.weight']!=initial['output.weight']).any()
        states[arm]=(initial,final);records[arm]=logs
    return states,records


def execute(plan_path,out,execution=None):
    plan=json.loads(plan_path.read_text());assert all(plan[k]==v for k,v in signature().items())
    plan_hash=source.file_hash(plan_path);auditing=execution is not None
    if auditing:diagnostic.manifest(execution,plan_hash)
    out.mkdir(parents=True,exist_ok=False);begin=time.time()
    source.prior.write(out/'started.json',{'unix':begin,'pid':os.getpid(),'plan_sha256':plan_hash,'audit':auditing})
    try:
        model,data,features,graphs,rows,setup=parent.setup()
        records=probes(model,data,features,graphs,rows,begin,auditing)
        assert len(records)==96 and sum(r['candidates'] for r in records)==2508
        assert all(r['passed'] for r in records),'Frozen score comparison failed'
        states,training=updates(model,data,features,graphs,begin)
        deadline(begin)
        changes={arm:{k:int((v!=states[arm][0][k]).sum()) for k,v in states[arm][1].items()} for arm in ARMS}
        if not auditing:
            for arm,(initial,final) in states.items():
                source.prior.save(out/f'{arm}-initial.pt',initial);source.prior.save(out/f'{arm}-weights.pt',final)
            with (out/'probes.jsonl').open('x') as f:
                for r in records:f.write(json.dumps(r)+'\n')
            summary={'status':'completed','plan_sha256':plan_hash,'numerical_passed':True,'initial_policy_exact':True,
                'root_seed_arm_checks':96,'candidate_checks':2508,'max_score_error':max(r['max_score_error'] for r in records),
                'parameters':16740,'artificial_training_updates':12,'training':training,'setup':setup,
                'median_update_seconds':{a:statistics.median(r['seconds'] for r in training[a]) for a in ARMS},
                'changed_parameter_coordinates':changes,'new_engine_calls':0,'new_development_evaluations':0,
                'wall_seconds':time.time()-begin,'limits':PROTOCOL['limits']}
            source.prior.write(out/'summary.json',summary)
            source.prior.write(out/'completed.json',{'status':'completed','plan_sha256':plan_hash,
                'files':{p.name:source.file_hash(p) for p in out.iterdir() if p.is_file()}})
            print(json.dumps(summary),flush=True)
        else:
            old=json.loads((execution/'summary.json').read_text());saved=source.prior.rows(execution/'probes.jsonl')
            assert records==saved
            assert old['status']=='completed' and old['plan_sha256']==plan_hash and old['numerical_passed'] and old['initial_policy_exact']
            assert old['parameters']==16740 and old['artificial_training_updates']==12
            assert old['root_seed_arm_checks']==96 and old['candidate_checks']==2508
            assert old['max_score_error']==max(r['max_score_error'] for r in records)
            assert old['new_engine_calls']==old['new_development_evaluations']==0 and 0<old['wall_seconds']<=900
            assert old['changed_parameter_coordinates']==changes and old['setup']['cache_max_error']==setup['cache_max_error']
            for arm,(initial,final) in states.items():
                for name,state in [('initial',initial),('weights',final)]:
                    expected=torch.load(execution/f'{arm}-{name}.pt',weights_only=True)
                    assert expected.keys()==state.keys() and all(torch.equal(v,expected[k]) for k,v in state.items())
                for a,b in zip(training[arm],old['training'][arm]):
                    assert {k:v for k,v in a.items() if k!='seconds'}=={k:v for k,v in b.items() if k!='seconds'}
                    assert math.isfinite(b['seconds']) and b['seconds']>0
                assert old['median_update_seconds'][arm]==statistics.median(r['seconds'] for r in old['training'][arm])
            result={'status':'completed','plan_sha256':plan_hash,'summary_sha256':source.file_hash(execution/'summary.json'),
                'execution_receipt_sha256':source.file_hash(execution/'completed.json'),'auditor_sha256':source.file_hash(__file__),
                'numerical_replays':96,'candidate_checks':2508,'numerical_passed':True,'additional_artificial_update_replays':12,
                'final_checkpoint_max_error':0.,'independent_python_packing':True,'wall_seconds':time.time()-begin,
                'scope':'All source/artifact hashes; native first8 training roots; all4 graph packings via Python reference;2508 single-candidate score checks; all12 artificial updates and final states replayed. Cached128-root setup authenticated against original. No trained quality or latency advantage claim.'}
            source.prior.write(out/'receipt.json',result);print(json.dumps(result),flush=True)
    except BaseException as error:
        source.prior.write(out/'failed.json',{'status':'failed','error':repr(error),'wall_seconds':time.time()-begin});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','audit'])
    p.add_argument('--plan',type=Path);p.add_argument('--execution',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.out)
    else:execute(a.plan,a.out,a.execution if a.command=='audit' else None)
