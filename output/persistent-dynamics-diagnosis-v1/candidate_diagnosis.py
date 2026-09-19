"""Saved candidate/control arithmetic only. No simulation, model, search or RNG."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'output/persistent-dynamics-qualification-v1/control-engineering-01'
BASE = Path(__file__).resolve().parent
COMPLETED = '064a0e7157088a6625bfc53afd1022d6d16232dc85dbf6cbde3238c1314bc050'
ARMS = ('nominal','adaptive','frozen','public_gain','true_state')
WINDOWS = {'all':(0,200),'goal0':(0,50),'goal1_before_switch':(50,80),'goal1_after_switch':(80,100),'goal2':(100,150),'goal3':(150,200)}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def tip(q):
    return np.stack((.10*np.cos(q[...,0])+.11*np.cos(q.sum(-1)),
                     .10*np.sin(q[...,0])+.11*np.sin(q.sum(-1))),-1)


def main():
    assert sha(RUN/'completed.json') == COMPLETED
    done = read(RUN/'completed.json')
    assert done['status']=='completed' and not (RUN/'failed.json').exists()
    rows = {r['arm']:r for r in done['rows']}
    sources = {name:sha(ROOT/name) for name in ('src/openjev/research/reacher_adaptive_search.py',
        'src/openjev/research/reacher_geometry_physics.py','src/openjev/research/reacher_geometry_reward.py')}
    result={'status':'completed','scope':'one engineering case, saved arithmetic only','outer_completed_sha256':COMPLETED,
            'windows':WINDOWS,'source_sha256':sources,'rows':{},'new_simulator_calls':0,'new_rng_draws':0,'new_model_calls':0}
    raw={}
    bindings={}
    for arm in ARMS:
        folder=RUN/'rows'/arm
        assert sha(folder/'completed.json')==rows[arm]['completed_sha256']
        receipt=read(folder/'completed.json')
        def checked(name, folder=folder, receipt=receipt):
            path=folder/name
            assert sha(path)==receipt['files'][name]
            bindings[str(path.relative_to(ROOT))]=receipt['files'][name]
            return path
        episode=read(checked('episode.json'))
        observed=episode['audit']['transitions']
        metrics=[]
        for t in range(200):
            with np.load(checked(f'decisions/{t:03d}.npz'),allow_pickle=False) as v:
                scores=v['scores']; selected=int(v['selected_id']); rewards=v['raw_rewards']; u=v['action']
                angles=v['predicted_angles']; q=np.arctan2(angles[...,2:],angles[...,:2]).astype(np.float64)
                target=v['public_packet'][4:6].astype(np.float64)
                predicted_tip=tip(q[selected,0]); predicted_distance=np.linalg.norm(predicted_tip-target)
                q_actual=np.array(observed[t]['native_after']['qpos'][:2]); native_fk_distance=np.linalg.norm(tip(q_actual)-target)
                native_distance=-observed[t]['reward_dist']; native_cost=-observed[t]['reward']
                first_expected_cost=-float(v['selected_reward'])
                angle_error=np.arctan2(np.sin(q[selected,0]-q_actual),np.cos(q[selected,0]-q_actual))
                seq=v['sequences'][selected]
                stage_best=[float(scores[i*64:(i+1)*64].max()) for i in range(4)]
                root_tip=tip(v['root_qpos'][:2]); root_distance=np.linalg.norm(root_tip-target)
                metrics.append({
                    'step':t,'selected_id':selected,'selected_stage':selected//64,
                    'selected_is_final_mean':int(selected==255),'selected_is_initial_anchor':int(selected<7),
                    'selected_is_zero_anchor':int(selected==0),
                    'best_score':float(scores[selected]),'initial_best_score':stage_best[0],
                    'best_minus_initial_per_step':float((scores[selected]-scores[:64].max())/len(seq)),
                    'best_minus_zero_per_step':float((scores[selected]-scores[0])/len(seq)),
                    'final_generation_worse_than_initial':int(stage_best[-1]<stage_best[0]),
                    'first_command_sqnorm':float(np.sum(u.astype(np.float64)**2)),
                    'later_command_sqnorm_mean':float(np.mean(np.sum(seq[1:].astype(np.float64)**2,-1))) if len(seq)>1 else 0.,
                    'candidate_clip_low_fraction':float(np.mean(rewards < -2.5)),
                    'candidate_clip_high_fraction':float(np.mean(rewards > 0)),
                    'first_predicted_distance':float(predicted_distance),'native_distance':native_distance,
                    'first_distance_error':float(predicted_distance-native_distance),
                    'native_fk_minus_cached_distance':float(native_fk_distance-native_distance),
                    'first_total_cost_error':float(first_expected_cost-native_cost),
                    'first_expected_control_cost':float(first_expected_cost-predicted_distance),
                    'native_control_cost':-observed[t]['reward_ctrl'],
                    'first_tip_error_m':float(np.linalg.norm(predicted_tip-tip(q_actual))),
                    'first_angle_error_rmse':float(np.sqrt(np.mean(angle_error**2))),
                    'root_distance':float(root_distance),
                    'selected_terminal_distance':float(np.linalg.norm(tip(q[selected,-1])-target)),
                    'selected_terminal_distance_change':float(np.linalg.norm(tip(q[selected,-1])-target)-root_distance),
                })
        raw[arm]=metrics
        summary={}
        for name,(a,b) in WINDOWS.items():
            values=metrics[a:b]
            row={'actions':b-a,'selected_stage_counts':[sum(v['selected_stage']==i for v in values) for i in range(4)]}
            for key in metrics[0]:
                if key in ('step','selected_id','selected_stage'):continue
                x=np.array([v[key] for v in values])
                row[key]={'mean':float(x.mean()),'median':float(np.median(x)),'max_abs':float(np.abs(x).max()),'rmse':float(np.sqrt(np.mean(x*x)))}
            summary[name]=row
        result['rows'][arm]=summary
    result['authenticated_payloads']=bindings
    out=BASE/'candidate-results-01'
    out.mkdir(exist_ok=False)
    for name,value in [('results.json',result),('per-step.json',raw)]:
        with (out/name).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    receipt={'status':'completed','source_sha256':sha(Path(__file__)),
             'files':{name:sha(out/name) for name in ('results.json','per-step.json')},'new_native_model_rng_calls':0}
    with (out/'receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    for arm in ARMS:
        s=result['rows'][arm]['all']
        print(json.dumps({'arm':arm,'stages':s['selected_stage_counts'],
            'final_mean_fraction':s['selected_is_final_mean']['mean'],
            'candidate_clipping':s['candidate_clip_low_fraction']['mean'],
            'one_step_distance_error_rmse':s['first_distance_error']['rmse'],
            'native_fk_cache_error_rmse':s['native_fk_minus_cached_distance']['rmse'],
            'first_command_sqnorm':s['first_command_sqnorm']['mean'],
            'tip_error':s['first_tip_error_m']['mean'],
            'first_angle_rmse':s['first_angle_error_rmse']['rmse']}))


if __name__=='__main__':
    main()
