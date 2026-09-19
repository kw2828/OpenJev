"""All-nine-row saved-result review; no simulation, inference or RNG."""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
RUN=BASE/'attempt-01'
OUT=BASE/'review-01'
ARMS=('nominal','public_gain','true_state')
MODES=('cold','repeat_last','shift_plan')
WINDOWS={'full':(0,200),'pre':(0,80),'post':(80,200),'move':(100,150)}


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(path.read_text())


def main():
    done=read(RUN/'completed.json');started=read(RUN/'started.json')
    assert done['status']=='completed' and done['engineering'] and done['scientific_gate'] is None
    assert not (RUN/'failed.json').exists() and not done['scientific_data_allocated']
    assert len(done['rows'])==len(done['audits'])==9
    assert {r['name'] for r in done['rows']}=={a+'--'+m for a in ARMS for m in MODES}
    assert [(r['arm'],r['proposal_mode']) for r in done['rows']]==[(r['arm'],r['proposal_mode']) for r in started['protocol']['matrix']]
    assert sha(RUN/'execution-completed.json')==done['execution_completed_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in done['source_sha256'].items())
    assert done['source_sha256']==started['source_sha256']
    for name,digest in done['source_sha256'].items():assert sha(RUN/'source-snapshot'/name)==digest
    assert {p['arm'] for p in done['cold_parity']}==set(ARMS)
    assert all(p['status']=='identical' and p['episode_equal'] for p in done['cold_parity'])
    audits={r['name']:r for r in done['audits']}
    rows={};bindings={};reward_arrays={};native_work=selected_work=candidate_work=0
    for row in done['rows']:
        name=row['name'];folder=RUN/'rows'/name
        assert sha(folder/'completed.json')==row['completed_sha256']
        receipt=read(folder/'completed.json')
        assert receipt['status']=='completed' and receipt['engineering'] is True
        assert receipt['arm']==row['arm'] and receipt['proposal_mode']==row['proposal_mode']
        assert receipt['steps']==receipt['native_decisions']==receipt['policy_decisions']==receipt['observation_updates']==200
        assert {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}==set(receipt['files'])|{'completed.json'}
        for member,digest in receipt['files'].items():
            assert sha(folder/member)==digest
            bindings[str((folder/member).relative_to(ROOT))]=digest
        audit_path=RUN/('audit-'+name+'.json')
        assert sha(audit_path)==audits[name]['sha256']
        audit=read(audit_path)
        assert audit['status']=='completed' and audit['row_completed_sha256']==row['completed_sha256']
        assert audit['candidate_transitions_checked']==597504 and audit['selected_transitions_checked']==200
        assert audit['identifier_transitions_checked']==0 and audit['native_control']['transitions']==200
        assert audit['max_candidate_state_abs_error']==audit['native_control']['max_abs_error']==0.
        episode=read(folder/'episode.json');transitions=episode['audit']['transitions']
        rewards=np.array([t['reward'] for t in transitions],np.float64)
        assert np.array_equal(rewards,audit['native_control']['native_rewards'])
        assert -float(rewards.sum())==receipt['native_cost']
        d=-np.array([t['reward_dist'] for t in transitions]);c=-np.array([t['reward_ctrl'] for t in transitions])
        costs={w:{'actions':b-a,'native_sum':float(-rewards[a:b].sum()),'native_mean':float(-rewards[a:b].mean()),
                    'distance_mean':float(d[a:b].mean()),'control_mean':float(c[a:b].mean())} for w,(a,b) in WINDOWS.items()}
        rows[name]={**row,'costs':costs,'memory_seconds':receipt['proposal_memory_seconds'],
                    'memory_costs':read(folder/'proposal-memory.json')['costs'],
                    'audit_sha256':sha(audit_path)}
        reward_arrays[name]=rewards
        native_work+=200;selected_work+=200;candidate_work+=597504
    checks=[]
    for arm in ARMS:
        chosen=rows[arm+'--shift_plan']['costs']['full']['native_sum']
        for control in ('cold','repeat_last'):
            baseline=rows[arm+'--'+control]['costs']['full']['native_sum']
            checks.append({'arm':arm,'control':control,'shift_plan_native_sum':chosen,'baseline_native_sum':baseline,
                           'improvement_percent':100*(baseline-chosen)/baseline,'pass':baseline>0 and chosen<=.97*baseline})
    summary={'status':'completed','scope':'single reused engineering case; no scientific gate or architecture claim',
             'outer_completed_sha256':sha(RUN/'completed.json'),'protocol_sha256':sha(BASE/'protocol.json'),
             'rows':rows,'mechanism_support_checks':checks,'mechanism_support_pass':all(c['pass'] for c in checks),
             'cold_parity':done['cold_parity'],'work':{'native_decisions':native_work,'selected_native_transitions':selected_work,
             'candidate_native_transitions':candidate_work,'identifier_native_transitions':0},
             'execution_wall_seconds':read(RUN/'execution-completed.json')['wall_seconds'],
             'audit_wall_seconds':done['audit_wall_seconds'],'authenticated_payload_sha256':bindings,
             'new_simulator_calls':0,'new_model_calls':0,'new_rng_draws':0}
    assert summary['work']==started['protocol']['planned_work']
    OUT.mkdir(exist_ok=False)
    with (OUT/'summary.json').open('x') as f:json.dump(summary,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    lines=['# Proposal-memory engineering comparison','',
        '**One reused seed410 case, all nine rows. No scientific qualification or architectural novelty is established.**','',
        'Native cost per action, lower is better. The full episode is the prospectively specified comparison; other windows cannot rescue it.','',
        '| State/dynamics role | Proposal mode | Full | PRE | POST | MOVE | Whole row seconds |',
        '|---|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for mode in MODES:
            r=rows[arm+'--'+mode];x=r['costs']
            lines.append(f"| {arm} | {mode} | "+' | '.join(f"{x[w]['native_mean']:.8f}" for w in WINDOWS)+f" | {r['wall_seconds']:.3f} |")
    lines += ['',f"Descriptive engineering support check: **{'PASS' if summary['mechanism_support_pass'] else 'FAIL'}**, {sum(c['pass'] for c in checks)}/6 comparisons meet the fixed 3% margin.",
              'This is a practical engineering check on an exposed case, not statistical confirmation or permission to start neural training.','',
              '| Role | Shifted plan versus | Full-cost improvement | Meets 3% |','|---|---|---:|---|']
    lines += [f"| {c['arm']} | {c['control']} | {c['improvement_percent']:.3f}% | {c['pass']} |" for c in checks]
    lines += ['', 'All three cold-mode native episodes and saved candidate arrays exactly reproduce their corresponding earlier rows.',
              f"Execution {summary['execution_wall_seconds']:.3f}s; independent replay {summary['audit_wall_seconds']:.3f}s. Shared-host timing includes setup and evidence recording.",
              f"Work: {native_work:,} real transitions, {candidate_work:,} candidate transitions and {selected_work:,} selected advances. No identification updates or training.",
              '', '![All nine native costs and execution times](figure.png)', '',
              'All methods use the same physics, score, horizon, candidate budget and original innovations. They follow different closed-loop trajectories. The inherited zero-command row is not a new measurement.',
              'The prospective fresh-case six-controller qualification remains unlaunched; no failed criterion was relaxed.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    labels=('Public state, nominal gain','Public state, known gain','Known state and gain')
    colors=('#64748b','#d97706','#2563eb')
    x=np.arange(3);width=.24
    for i,(mode,color) in enumerate(zip(MODES,colors,strict=True)):
        axes[0].bar(x+(i-1)*width,[rows[a+'--'+mode]['costs']['full']['native_mean'] for a in ARMS],width,color=color,label=mode)
        axes[1].bar(x+(i-1)*width,[rows[a+'--'+mode]['wall_seconds'] for a in ARMS],width,color=color,label=mode)
    axes[0].set(title='Whole-episode native cost',ylabel='Cost per action (lower is better)')
    axes[1].set(title='Complete instrumented row time',ylabel='Seconds / 200 actions')
    for ax in axes:
        ax.set_xticks(x,labels,fontsize=8);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8,frameon=False)
    fig.suptitle('Remembering the previous plan: all nine engineering rows\nSame H12 / CEM256; one reused case; no architecture claim',fontsize=12)
    fig.savefig(OUT/'figure.png',dpi=180);plt.close(fig)
    receipt={'status':'completed','source_sha256':sha(Path(__file__)),'outer_completed_sha256':sha(RUN/'completed.json'),
             'files':{n:sha(OUT/n) for n in ('summary.json','report.md','figure.png')}}
    with (OUT/'receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps({'status':'completed','mechanism_support_pass':summary['mechanism_support_pass'],'checks':checks}))


if __name__=='__main__':main()
