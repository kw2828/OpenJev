"""Plot preserved cadence confirmation contrasts; never rerun Doom."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis', type=Path)
    parser.add_argument('--prefix', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    fig, axes = plt.subplots(2,2,figsize=(11,6.4))
    for row, (scenario, name) in enumerate([('defend_the_center','Center'),('defend_the_line','Line')]):
        for col, metric in enumerate(['net_utility','kills']):
            ax = axes[row,col]
            for y, control in enumerate(['rules','history_map']):
                item = data['primary'][scenario][control][metric]
                mean = item['mean_difference']
                lo,hi = item['interval']
                qualified = lo > (0 if metric == 'net_utility' else -1)
                if metric == 'net_utility':
                    qualified = qualified and mean >= .5
                color = '#3674a5' if qualified else '#ad5348'
                ax.plot([lo,hi],[y,y],color=color,lw=2)
                ax.plot([lo,hi],[y,y],linestyle='',marker='|',color=color,markersize=8)
                ax.scatter([mean],[y],color=color,s=45,zorder=3)
            ax.axvline(0,color='gray',ls='--',lw=1)
            ax.axvline(.5 if metric=='net_utility' else -1,color='gray',ls=':',alpha=.7)
            ax.set(yticks=[0,1],yticklabels=['vs rules','vs history MAP'],ylim=(1.5,-.5),
                   title=f'{name}: '+('net utility difference' if metric=='net_utility' else 'kill difference'))
            ax.spines[['top','right']].set_visible(False)
            ax.grid(axis='x',alpha=.15)
    result = 'passed' if data['confirmation_passed'] else 'not met'
    labels = {'ammo_rest1':'Ammo-triggered rest','combined_rest1':'Combined event memory',
              'portable_history_event':'Portable history + events','portable_current_event':'Portable current state + events',
              'original_event':'Original history ensemble + events','original_ensemble':'Original history ensemble',
              'portable_history':'Portable history ensemble','portable_current':'Portable current-state ensemble',
              'rule_event':'Rules + event memory'}
    fig.suptitle(f"{labels.get(data['selected'], data['selected'])}: confirmation gate {result}",fontsize=14)
    fig.text(.5,.025,'96 fresh paired seeds per scenario; five preserved history fits. 99.375% paired bootstrap intervals.\n'
             'Require utility gain and kill noninferiority in both scenarios. Decision compute excludes engine and I/O.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.10,1,.95])
    for ext in ('png','svg','pdf'):
        fig.savefig(args.prefix.with_suffix('.'+ext),dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
