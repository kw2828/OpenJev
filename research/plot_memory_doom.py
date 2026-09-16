"""Render the predeclared memory contrasts and compute/utility tradeoff."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evidence/memory-doom-v1'
LABELS = {'current_map':'Current MAP','expanded_map':'Expanded MAP','history_map':'History MAP',
          'current_bayes':'Current Bayes','expanded_bayes':'Expanded Bayes','history_bayes':'History Bayes',
          'rules':'Rules','tiny_imitation':'Imitation'}


def main():
    data = json.loads((OUT/'analysis-001.json').read_text())
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    colors = ['#3674a5','#ad5348']
    for ax, scenario, title in zip(axes,data['primary'],['Center','Shifted line']):
        for i,control in enumerate(['current_map','expanded_map']):
            row=data['primary'][scenario]['history_map_minus_'+control]
            mean=row['mean_net_utility_gain']; low,high=row['bonferroni_98_75_interval']
            ax.errorbar(mean,i,xerr=[[mean-low],[high-mean]],fmt='o',color=colors[i],capsize=5,markersize=8)
        ax.axvline(0,color='gray',ls='--'); ax.axvline(.5,color='gray',ls=':',alpha=.5)
        ax.set(yticks=[0,1],yticklabels=['vs current MAP','vs expanded MAP'],ylim=(-.5,1.5),title=title,
               xlabel='History MAP net utility difference; higher is better')
        ax.invert_yaxis();ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
    fig.suptitle('Memory ablation: '+('continuation gate passed' if data['continuation']['passed'] else 'continuation gate not met'))
    fig.text(.5,.015,'1,980 new episodes; shared historical training data. Four primary contrasts, Bonferroni 98.75% bootstrap intervals.\nNet utility = command-window utility minus decision-compute seconds. All four contrasts and both timing gates must pass.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.14,1,.94])
    for ext in ('png','svg','pdf'):fig.savefig(OUT/f'memory-contrasts.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,scenario,title in zip(axes,data['summary'],['Center','Shifted line']):
        for kind,color in zip(['current','expanded','history'],['#77808e','#3674a5','#ad5348']):
            for mode,marker in [('map','o'),('bayes','D')]:
                row=data['summary'][scenario][kind+'_'+mode]
                ax.scatter(row['compute_ms_per_decision'],row['utility'],color=color,marker=marker,s=55,
                           label=LABELS[kind+'_'+mode])
        ax.set(title=title,xlabel='Mean decision compute (ms); lower is better',ylabel='Mean raw utility; higher is better',xscale='log')
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[1].legend(fontsize=8,loc='best')
    fig.suptitle('Utility versus measured computation: six matched learned policies')
    fig.text(.5,.015,'Same training episodes, five fits and 30 paired evaluation seeds per scenario. One workstation; engine and I/O excluded.\nMAP uses a direct dot product; Bayes includes quadrature. This is not a browser-model or end-to-end benchmark.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.14,1,.94])
    for ext in ('png','svg','pdf'):fig.savefig(OUT/f'memory-tradeoff.{ext}',dpi=180)
    plt.close(fig)
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Policy & Center $U$ & Line $U$ & Center ms & Line ms \\',r'\midrule']
    for policy,label in LABELS.items():
        c=data['summary']['defend_the_center'][policy];l=data['summary']['defend_the_line'][policy]
        lines.append(f"{label} & {c['utility']:.3f} & {l['utility']:.3f} & {c['compute_ms_per_decision']:.3f} & {l['compute_ms_per_decision']:.3f} "+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (ROOT/'paper/generated/memory-table.tex').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
