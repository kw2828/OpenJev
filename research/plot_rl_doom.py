"""Plot preserved RL results; never train or select a controller."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS={'ppo_current':'PPO current','ppo_history':'PPO history','dqn_history':'DQN history',
        'rules':'Rules','rule_event':'Rules + event memory','always_fire':'Always fire',
        'alternating_fire':'Alternate fire/wait','visible_fire':'Visible-target fire'}
COLORS={'ppo_current':'#346c9c','ppo_history':'#3a8b65','dqn_history':'#9d6539',
        'rules':'#777777','rule_event':'#ad4144','always_fire':'#8860a6',
        'alternating_fire':'#9974b0','visible_fire':'#ae866e'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('analysis',type=Path)
    ap.add_argument('--prefix',type=Path,required=True)
    ap.add_argument('--training',type=Path)
    ap.add_argument('--controls',type=Path)
    args=ap.parse_args()
    data=json.loads(args.analysis.read_text())
    if args.controls:
        controls=json.loads(args.controls.read_text())
        for scenario,values in controls.items():
            for arm,row in values.items():
                data['summary'][scenario][arm]={**row,'per_fit_kills':[row['kills']]}
    stage='Confirmation' if 'confirmation_passed' in data else 'Development screen'
    fig,axes=plt.subplots(2,2,figsize=(11,7.8),layout='constrained')
    for row,(scenario,values) in enumerate(data['summary'].items()):
        arms=list(values)
        for col,metric in enumerate(['kills','game_seconds']):
            ax=axes[row,col]
            for i,arm in enumerate(arms):
                ax.scatter(values[arm][metric],i,color=COLORS[arm],s=60,zorder=3)
                if metric=='kills':
                    fit=values[arm]['per_fit_kills']
                    if len(fit)>1:
                        ax.scatter(fit,[i]*len(fit),color=COLORS[arm],s=15,alpha=.65,marker='x',zorder=4)
            ax.set_yticks(range(len(arms)),[LABELS[a] for a in arms])
            ax.invert_yaxis()
            ax.set_xlabel('Mean kills' if metric=='kills' else 'Mean duration (game seconds)')
            ax.set_title(('Center' if scenario.endswith('center') else 'Line')+': '+('kills' if metric=='kills' else 'duration'))
            ax.grid(axis='x',alpha=.2)
            for spine in ('top','right'):
                ax.spines[spine].set_visible(False)
    chosen=data['selection']['selected']
    status=(('primary gate vs event rules passed' if data['confirmation_passed'] else 'primary gate vs event rules failed') if 'confirmation_passed' in data
            else (f'selected {LABELS[chosen]}' if chosen else 'no family qualified'))
    fig.suptitle(f'OpenJev RL | {stage}: {status}',fontsize=14)
    footer = ('Fresh confirmation seeds; all selected training fits retained.' if 'confirmation_passed' in data
              else 'Development is not held-out confirmation.')
    fig.supxlabel('Dots are means; crosses are training-fit mean kills, not confidence intervals.\nEqual interactions, rule steering, finite 180-window horizon. '+footer,fontsize=9)
    args.prefix.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg','pdf'):
        fig.savefig(args.prefix.with_suffix('.'+ext),dpi=180)
    plt.close(fig)
    if args.training:
        fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True,layout='constrained')
        for ax,arm in zip(axes,('ppo_current','ppo_history','dqn_history'),strict=True):
            for rep in range(3):
                rows=json.loads((args.training/f'{arm}-{rep}-training-summary.json').read_text())['completed_episodes']
                steps=np.cumsum([r['steps'] for r in rows])
                kills=np.array([r['kills'] for r in rows])
                if len(kills)>=20:
                    rolling=np.convolve(kills,np.ones(20)/20,mode='valid')
                    ax.plot(steps[19:],rolling,label=f'fit {rep+1}',alpha=.8,lw=1)
            ax.set_title(LABELS[arm]);ax.set_xlabel('Training interactions');ax.grid(alpha=.2)
        axes[0].set_ylabel('Kills per episode: trailing 20 mean')
        axes[-1].legend(fontsize=8)
        fig.suptitle('Center training only | training behavior differs across algorithms')
        for ext in ('png','svg','pdf'):
            fig.savefig(args.prefix.parent/f'learning-curves.{ext}',dpi=180)


if __name__=='__main__':
    main()
