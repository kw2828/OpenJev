"""Audit and summarize saved card-pilot artifacts without native/model calls."""
from __future__ import annotations

import argparse
import json
import math
from decimal import Decimal
from pathlib import Path

import numpy as np
from card_memory_common import check, sha, write_json

from openjev.research.card_memory_task import PublicCardTracker

MODES=('delta','gated_delta','kalman','innovation_local','innovation_matched','gru')


def read(path):
    return json.loads(Path(path).read_text())


def mean(values):
    values=list(values)
    check(bool(values) and all(math.isfinite(v) for v in values), 'Missing/nonfinite metric')
    return math.fsum(values)/len(values)


def margin_pass(left, right, margin):
    """Inclusive frozen decimal margin, without a binary-subtraction edge."""
    if left is None or right is None:
        return False
    check(all(math.isfinite(x) for x in (left, right, margin)), 'Nonfinite criterion')
    return Decimal(str(left)) - Decimal(str(right)) >= Decimal(str(margin))


def authenticate_training(root, training, train):
    check(train['fits'] == 18 and train['updates'] == 2304, 'Incomplete fit coverage')
    for receipt_key, filename in (('started_sha256', 'started.json'),
                                 ('training_completed_sha256', 'training-completed.json'),
                                 ('checkpoint_map_sha256', 'checkpoint-map.json')):
        check(sha(training / filename) == train[receipt_key], 'Training boundary changed')
    boundary = read(training / 'training-completed.json')
    checkpoints = read(training / 'checkpoint-map.json')
    names = {f'{mode}-pair{i}' for mode in MODES for i in range(3)}
    check(boundary['status'] == 'complete' and boundary['development_or_native_evaluations'] == 0,
          'Incomplete pre-evaluation boundary')
    check(boundary['checkpoint_map_sha256'] == train['checkpoint_map_sha256'], 'Checkpoint map changed')
    check(set(checkpoints) == set(train['development']) == names and len(boundary['fits']) == 18
          and {f['id'] for f in boundary['fits']} == names, 'Missing or duplicate fits')
    for fitted in boundary['fits']:
        name = fitted['id']
        check(read(training / (name + '-receipt.json')) == fitted, 'Unbound fit summary')
        check(sha(training / (name + '-initial.pt')) == fitted['initial_sha256'], 'Initial changed')
        entry = checkpoints[name]
        check(all(fitted[k] == v for k, v in entry.items()), 'Fit/checkpoint identity mismatch')
        check(name == f"{entry['mode']}-pair{entry['pair']}", 'Mode/pair mismatch')
        checkpoint = (root / entry['checkpoint_path']).resolve()
        receipt = (root / entry['completed_path']).resolve()
        check(checkpoint.is_relative_to(training.resolve()) and receipt.is_relative_to(training.resolve()),
              'Fit outside training output')
        check(sha(checkpoint) == entry['checkpoint_sha256'] and sha(receipt) == entry['completed_sha256'],
              'Fit payload changed')
        done = read(receipt)
        check(done['status'] == 'complete' and done['counts']['successful_updates'] == 128,
              'Incomplete fit')
        check({x.name for x in receipt.parent.iterdir() if x.is_file()} == set(done['files']) | {'completed.json'},
              'Fit membership mismatch')
        for path, identity in done['files'].items():
            artifact = receipt.parent / path
            check(artifact.resolve().is_relative_to(receipt.parent.resolve())
                  and sha(artifact) == identity['sha256'] and artifact.stat().st_size == identity['bytes'],
                  'Fit member changed')
        dev = read(training / (name + '-development.json'))
        check({k: v for k, v in dev.items() if k != 'per_episode'} == train['development'][name],
              'Unbound development summary')
    return checkpoints


def audit_episode(npz, receipt):
    check(sha(npz)==receipt['npz_sha256'], 'Episode hash mismatch')
    with np.load(npz,allow_pickle=False) as saved:
        arrays={name:saved[name] for name in saved.files}
    check(receipt['status'] == 'complete', 'Incomplete episode')
    n=len(arrays['actions'])
    schemas = {'actions': ((n,), np.int64), 'ranks': ((n,), np.int64),
               'rewards': ((n,), np.float64), 'terminated': ((n,), np.bool_),
               'truncated': ((n,), np.bool_), 'observations': ((n+1,52), np.int64),
               'raw_probabilities': ((n,52,13), np.float64),
               'picker_probabilities': ((n,52,13), np.float64)}
    check(set(arrays) == set(schemas), 'Episode array membership mismatch')
    for name, (shape, dtype) in schemas.items():
        check(arrays[name].shape == shape and arrays[name].dtype == dtype
              and np.isfinite(arrays[name]).all(), 'Invalid episode array: ' + name)
    check(n==receipt['native_steps'] and 1<=n<=104 and len(arrays['observations'])==n+1, 'Episode length mismatch')
    tracker=PublicCardTracker()
    tracker.reset(arrays['observations'][0])
    for t in range(n):
        p=arrays['raw_probabilities'][t]
        check(np.array_equal(tracker.probabilities(p), arrays['picker_probabilities'][t]), 'Saved picker mismatch')
        action=tracker.choose(p)
        check(action==int(arrays['actions'][t]), 'Saved action differs from shared picker')
        reveal=tracker.observe(action,float(arrays['rewards'][t]),arrays['observations'][t+1])
        check(reveal.rank==int(arrays['ranks'][t]), 'Saved selected rank mismatch')
        check(bool(arrays['terminated'][t])==bool(tracker.matched.all()), 'Saved terminal flag mismatch')
        check(bool(arrays['truncated'][t])==(t==103), 'Saved truncation flag mismatch')
        check(bool(arrays['terminated'][t] or arrays['truncated'][t])==(t==n-1), 'Saved horizon mismatch')
    check(math.isclose(math.fsum(arrays['rewards']),receipt['return'],rel_tol=0,abs_tol=1e-12), 'Saved return mismatch')
    check(receipt['success']==bool(arrays['terminated'][-1]), 'Saved success mismatch')
    check(receipt['matched_pairs'] == int(tracker.matched.sum()) // 2, 'Matched count mismatch')
    counts=receipt['counts']
    check(counts['reset_attempted'] == counts['reset_returned'] == 1, 'Missing native reset')
    check(counts['native_attempted']==counts['native_returned']==n, 'Missing native action')
    if receipt['controller'] not in ('exact','last32'):
        check(counts['predict_attempted']==counts['predict_returned']==counts['write_attempted']==counts['write_returned']==n,
              'Missing read/write event')
    else:
        check(all(counts[k] == 0 for k in ('predict_attempted', 'predict_returned', 'write_attempted', 'write_returned')),
              'Reference contains learned calls')
    return n


def report(root,protocol,training,evaluation,out):
    out.mkdir(parents=True,exist_ok=False)
    recipe=read(protocol)
    train=read(training/'completed.json')
    result=read(evaluation/'completed.json')
    check(train['status']==result['status']=='complete', 'Incomplete experiment')
    check(not (training/'failed.json').exists() and not (evaluation/'failed.json').exists(), 'Failed experiment')
    check(result['protocol_sha256']==sha(protocol), 'Evaluation protocol changed')
    bindings=read(root/recipe['bindings_file'])
    check(result['bindings_sha256']==sha(root/recipe['bindings_file']), 'Source binding changed')
    for path,digest in bindings['files'].items():
        check(sha(root/path)==digest, 'Bound source changed: '+path)
    check({str(p.relative_to(evaluation)) for p in evaluation.rglob('*') if p.is_file()}==set(result['files'])|{'completed.json'},
          'Evaluation membership mismatch')
    for path,identity in result['files'].items():
        check(sha(evaluation/path)==identity['sha256'] and (evaluation/path).stat().st_size==identity['bytes'], 'Evaluation artifact changed')
    checkpoints = authenticate_training(root, training, train)
    started = read(training / 'started.json')
    check(started['protocol_sha256'] == result['protocol_sha256']
          and started['inputs_sha256'] == result['inputs_sha256']
          and started['bindings_sha256'] == result['bindings_sha256'], 'Training provenance mismatch')
    input_paths = [root / p for p, digest in bindings['files'].items() if digest == result['inputs_sha256']]
    check(len(input_paths) == 1, 'Missing bound evaluation inputs')
    inputs = read(input_paths[0])
    check(len(inputs['evaluation']) == 64, 'Wrong evaluation seed count')
    check(read(evaluation / 'all-fits-ready.json')['fits'] == checkpoints, 'Evaluation checkpoint identities differ')
    check(result['controllers'] == 20 and result['episodes'] == 1280 and result['new_fits'] == 0,
          'Incomplete evaluation coverage')
    rows,fit_rows,native_steps=[],{},0
    for mode in (*MODES,'exact','last32'):
        names=[f'{mode}-pair{i}' for i in range(3)] if mode in MODES else [mode]
        returns,successes,wall,counts=[],0,0.,0
        for name in names:
            directory=evaluation/'controllers'/name
            item=read(directory/'completed.json')
            receipts=[]
            for i in range(64):
                rec=read(directory/'episodes'/f'{i:03d}.json')
                native_steps+=audit_episode(directory/'episodes'/f'{i:03d}.npz',rec)
                check(rec['controller']==name and rec['index']==i, 'Episode identity mismatch')
                check(rec['seed'] == inputs['evaluation'][i]['seed']
                      and rec['protocol_sha256'] == result['protocol_sha256']
                      and rec['inputs_sha256'] == result['inputs_sha256']
                      and rec['checkpoint_sha256'] == (checkpoints[name]['checkpoint_sha256'] if name in checkpoints else None),
                      'Episode seed/provenance mismatch')
                receipts.append(rec)
            check(item['status'] == 'complete' and item['controller'] == name and item['episodes'] == 64
                  and item['native_steps'] == sum(r['native_steps'] for r in receipts)
                  and item['successes'] == sum(r['success'] for r in receipts), 'Controller coverage mismatch')
            individual=[r['return'] for r in receipts]
            fit_rows[name]={'mean_return':mean(individual),'successes':sum(r['success'] for r in receipts),
                            'per_seed_returns':individual,'native_steps':sum(r['native_steps'] for r in receipts)}
            check(math.isclose(mean(individual),item['mean_return'],abs_tol=1e-12),'Controller aggregation mismatch')
            returns+=individual
            successes+=sum(r['success'] for r in receipts)
            wall+=item['wall_seconds']
            counts+=sum(r['native_steps'] for r in receipts)
        row={'mode':mode,'mean_return':mean(returns),'successes':successes,'episodes':len(returns),
             'whole_controller_seconds':wall,'native_steps':counts,'whole_controller_ms_per_action':1000*wall/counts}
        if mode in MODES:
            metrics=[read(training/(name+'-development.json')) for name in names]
            row['development_accuracy']=mean(m['all']['accuracy'] for m in metrics)
            row['development_age_gt32_accuracy']=(mean(m['age_gt32']['accuracy'] for m in metrics)
                if all(m['age_gt32']['accuracy'] is not None for m in metrics) else None)
            row['development_age_gt32_eligible_episodes']=[m['age_gt32']['eligible_episodes'] for m in metrics]
            fitted=[read(training/(name+'-receipt.json')) for name in names]
            row['registered_parameters']=fitted[0]['registered_parameters']
            row['state_bytes_per_case']=fitted[0]['state_bytes_per_case']
            row['training_seconds']=math.fsum(f['wall_seconds'] for f in fitted)
        rows.append(row)
    by={r['mode']:r for r in rows}
    criteria=recipe['continuation']
    best=max(criteria['conventional_modes'],key=lambda mode:by[mode]['mean_return'])
    local=by['innovation_local'];matched=by['innovation_matched']
    per_pair_base=[fit_rows[f'innovation_local-pair{i}']['mean_return']-fit_rows[f'{best}-pair{i}']['mean_return'] for i in range(3)]
    per_pair_matched=[fit_rows[f'innovation_local-pair{i}']['mean_return']-fit_rows[f'innovation_matched-pair{i}']['mean_return'] for i in range(3)]
    checks={
      'utility_over_strongest_conventional':margin_pass(local['mean_return'], by[best]['mean_return'], criteria['utility_margin_over_strongest_conventional_mean']),
      'utility_positive_all_pairs':all(x>0 for x in per_pair_base),
      'local_over_matched_utility':margin_pass(local['mean_return'], matched['mean_return'], criteria['local_minus_matched_mean_return_margin']),
      'local_over_matched_positive_all_pairs':all(x>0 for x in per_pair_matched),
      'long_age_accuracy':margin_pass(local['development_age_gt32_accuracy'], matched['development_age_gt32_accuracy'], criteria['development_age_gt32_accuracy_margin_local_minus_matched']),
      'long_age_coverage':min(local['development_age_gt32_eligible_episodes']+matched['development_age_gt32_eligible_episodes'])>=criteria['development_min_eligible_age_gt32_episodes_per_fit']}
    summary={'status':'complete','continuation_passed':all(checks.values()),'criteria':checks,'rows':rows,'per_fit':fit_rows,
             'strongest_conventional':best,'local_minus_strongest_conventional_by_pair':per_pair_base,
             'local_minus_matched_by_pair':per_pair_matched,'native_steps_audited':native_steps,
             'native_calls_during_reporting':0,'model_calls_during_reporting':0,
             'training_receipt_sha256':sha(training/'completed.json'),'evaluation_receipt_sha256':sha(evaluation/'completed.json'),
             'claim_limit':'small supervised single-task pilot, no novelty, biological, calibration, RL, or SOTA claim'}
    write_json(out/'summary.json',summary)
    lines=['# Learned card-memory pilot','',f"Continuation rule: **{'PASS' if summary['continuation_passed'] else 'FAIL'}** ({sum(checks.values())}/{len(checks)} checks).",'',
           'All 18 fits completed before evaluation: six modes, three paired initializations, 128 training episodes and 128 updates per fit. Each final policy played the same 64 fresh ConcentrationHard seeds. Public exact-table and last-32 references use the same decision rule.',
           '', '| Memory | Mean native return | Successes | Old hidden-card accuracy | Parameters | End-to-end ms/action |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        accuracy=f"{100*r['development_age_gt32_accuracy']:.2f}%" if r.get('development_age_gt32_accuracy') is not None else 'n/a'
        lines.append(f"| {r['mode']} | {r['mean_return']:.4f} | {r['successes']}/{r['episodes']} | {accuracy} | {r.get('registered_parameters','symbolic')} | {r['whole_controller_ms_per_action']:.3f} |")
    lines+=['','Old-card accuracy measures development queries more than 32 actions since last public visibility. It is a separate metric from native return. Timings include model restore, public bookkeeping, game execution and episode storage on this host; they are not model-only latency or matched compute.',
            '',f'Strongest conventional learned arm: **{best}**. Local minus this baseline, by paired fit: '+', '.join(f'{x:+.4f}' for x in per_pair_base)+'.',
            '', 'Local minus gain-matched diffuse control, by fit: '+', '.join(f'{x:+.4f}' for x in per_pair_matched)+'.', '',
            '| Continuation criterion | Result |','| --- | --- |']
    lines += [f"| {name} | {'PASS' if passed else 'FAIL'} |" for name,passed in checks.items()]
    lines += ['', 'The proposed innovation rule is a heuristic extension of existing delta/Kalman memory. Its covariance is not calibrated uncertainty. The GRU has the same 512 mean-state scalars but many more parameters; Kalman variants add 32 covariance scalars. Every arm also receives public seen/matched/phase bookkeeping. This is not a connectome experiment, end-to-end RL, or a POPGym leaderboard reproduction.', '',
              'No checkpoint selection, replacement seeds, or outcome-driven tuning occurred in this attempt. The next experiment, if warranted, must be separately specified on fresh data.', '']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps({'continuation_passed':summary['continuation_passed'],'criteria':checks,'rows':rows},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('root','protocol','training','evaluation','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    report(**vars(parser.parse_args()))
