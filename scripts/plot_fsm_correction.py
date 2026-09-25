"""Presentation-only FSM development figures from two saved scalar JSON files.

The caller must wait for the original run and independent audit to close before
invoking this helper. It does not admit scientific evidence, load arrays, import
model code, refit, replay, or recompute any continuation condition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

NEURAL = ('dense','selective','rewired','fixed0','fixed1','fixed2','autoregressive')
SEEDS = (9101,9102,9103)
VARX = tuple(f'varx{order}-ridge{alpha}' for order in (8,16,32) for alpha in ('1e-06','0.001','0.1'))
FAMILIES = NEURAL+VARX
RECORDS = {f'{amp}-realization-{r}-period-{p}' for amp in ('100mV','200mV')
           for r in (3,4,5) for p in (0,1)}
CONDITIONS = ('all_declared_fits_complete','all_declared_evaluations_complete','candidate_eligible',
              'five_percent_below_strongest_control','latency_within_ten_percent_dense',
              'storage_within_ten_percent_dense','no_record_over_two_percent_dense',
              'every_seed_beats_dense','two_blocks_at_least_five_percent_each_seed')
LABELS = {'dense':'Dense correction','selective':'Selective correction','rewired':'Cyclic rewiring',
          'fixed0':'Fixed block 0','fixed1':'Fixed block 1','fixed2':'Fixed block 2',
          'autoregressive':'Autoregressive GRU'}
for _order in (8,16,32):
    for _alpha,_display in (('1e-06','0.000001'),('0.001','0.001'),('0.1','0.1')):
        LABELS[f'varx{_order}-ridge{_alpha}']=f'VARX {_order}, ridge {_display}'


def require(condition,message):
    if not condition:raise ValueError(message)


def nonnegative(value):
    return type(value) in (int,float) and math.isfinite(value) and value>=0


def positive(value):
    return nonnegative(value) and value>0


def descriptor(path):
    data=Path(path).read_bytes()
    return {'path':str(Path(path).resolve()),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def read_json(path):
    def invalid(value):raise ValueError('nonfinite JSON token '+value)
    return json.loads(Path(path).read_text(),parse_constant=invalid)


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def extract(summary,evaluations):
    """Keep every declared family/seed slot, including absent or failed values."""
    require(isinstance(summary,dict) and summary.get('status') in ('DEVELOPMENT_PASS','DEVELOPMENT_FAIL'),
            'saved completed development summary required')
    require(set(summary.get('families',{}))==set(FAMILIES),'exact sixteen-family summary required')
    conditions=summary.get('conditions')
    require(isinstance(conditions,dict) and set(conditions)==set(CONDITIONS)
            and all(type(v) is bool for v in conditions.values()),'exact nine saved conditions required')
    require(summary.get('total')==9 and summary.get('passed')==sum(conditions.values()),'saved condition counts mismatch')
    require((summary['status']=='DEVELOPMENT_PASS')==all(conditions.values()),'saved status/conditions mismatch')
    require(isinstance(evaluations,list),'evaluation list required')
    indexed={}
    for evaluation in evaluations:
        require(isinstance(evaluation,dict),'evaluation object required')
        family,seed=evaluation.get('family'),evaluation.get('seed')
        require(family in FAMILIES and (seed in SEEDS if family in NEURAL else seed is None),
                'unknown evaluation family/seed')
        key=(family,seed)
        require(key not in indexed,'duplicate evaluation identity')
        rows=evaluation.get('rows')
        require(isinstance(rows,list) and len(rows)==12 and {r.get('record_id') for r in rows}==RECORDS,
                'exact twelve-record attempted evaluation required')
        for row in rows:
            require(row.get('status') in ('complete','failed'),'unknown record status')
            if row['status']=='complete':
                require(nonnegative(row.get('rmse')) and row.get('requests')==32 and row.get('horizon')==128,
                        'finite complete record scalar and declared geometry required')
            else:require(isinstance(row.get('error'),str) and bool(row['error']),'failed record error required')
        indexed[key]=evaluation
    families=[]
    for family in FAMILIES:
        entry=summary['families'][family]
        require(type(entry.get('eligible')) is bool,'saved family eligibility required')
        members=[]
        for seed in (SEEDS if family in NEURAL else (None,)):
            row=indexed.get((family,seed))
            member={'seed':seed,'rmse':None,'latency_ms':None,'complete_records':0,
                    'status':'absent','timing_error':None,'routing_counts':None}
            if row is not None:
                good=[r for r in row['rows'] if r['status']=='complete']
                member['complete_records']=len(good)
                member['status']='complete' if len(good)==12 else 'incomplete'
                if len(good)==12:member['rmse']=fmean(r['rmse'] for r in good)
                member['timing_error']=row.get('timing_error')
                if row.get('timing_error') is None:
                    samples=row.get('request_ms')
                    require(isinstance(samples,list) and len(samples)==24 and all(positive(v) for v in samples)
                            and positive(row.get('median_request_ms')),'complete saved timing needs 24 finite positive samples')
                    member['latency_ms']=row['median_request_ms']
                else:require(isinstance(row['timing_error'],str) and bool(row['timing_error']),'timing failure error required')
                counts=row.get('routing_nonzero_counts')
                if family=='selective':
                    require(isinstance(counts,list) and len(counts)==3 and all(type(v) is int and v>=0 for v in counts),
                            'three nonnegative routing counts required')
                    member['routing_counts']=counts
            members.append(member)
        if entry['eligible']:
            require(all(m['rmse'] is not None and m['latency_ms'] is not None for m in members),
                    'eligible family must have complete saved scores and timings')
            for key,values in (('mean_rmse',[m['rmse'] for m in members]),
                               ('mean_seed_median_ms',[m['latency_ms'] for m in members])):
                require(nonnegative(entry.get(key)) and math.isclose(entry[key],fmean(values),rel_tol=1e-10,abs_tol=1e-12),
                        'summary/evaluation scalar disagreement: '+family+'/'+key)
        families.append({'family':family,'label':LABELS[family],'kind':'neural float32' if family in NEURAL else 'VARX float64',
                         'eligible':entry['eligible'],'members':members,
                         'mean_rmse':entry.get('mean_rmse') if entry['eligible'] else None,
                         'mean_seed_median_ms':entry.get('mean_seed_median_ms') if entry['eligible'] else None})
    return {'status':summary['status'],'passed':summary['passed'],'total':summary['total'],
            'conditions':conditions,'strongest_control':summary.get('strongest_control'),'families':families,
            'scope':'Custom C100/H128 development; 12 records include correlated periods. No official benchmark or SOTA claim.',
            'point_semantics':'Neural dots are three seed summaries, not confidence intervals. Diamonds use saved complete-evaluation family means.',
            'cost_scope':'Warm CPU full request includes normalization, casting, conditioning and 128-step forecast; excludes loading/disk I/O.',
            'no_replay':True}


def axis_spec(values):
    """Every error axis includes zero; large outliers use explicitly labeled symlog."""
    require(all(nonnegative(v) for v in values),'finite nonnegative chart values required')
    positives=[v for v in values if v>0]
    if not positives:return {'scale':'linear','limits':(0.,1.),'linthresh':None}
    largest,smallest=max(positives),min(positives)
    if largest/smallest>100:
        return {'scale':'symlog','limits':(0.,largest*1.25),'linthresh':smallest*.5}
    return {'scale':'linear','limits':(0.,largest*1.15),'linthresh':None}


def render(values,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold','svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(17,10),sharey=True)
    seed_markers={9101:'o',9102:'s',9103:'^',None:'o'}
    for ax,metric,mean_key,title in zip(axes,('rmse','latency_ms'),('mean_rmse','mean_seed_median_ms'),
                                      ('Forecast error','Complete request latency'),strict=True):
        numbers=[]
        for family in values['families']:
            numbers.extend(m[metric] for m in family['members'] if m[metric] is not None)
            if family[mean_key] is not None:numbers.append(family[mean_key])
        spec=axis_spec(numbers)
        ax.set_xscale(spec['scale'],**({'linthresh':spec['linthresh']} if spec['scale']=='symlog' else {}))
        ax.set_xlim(*spec['limits'])
        for index,family in enumerate(values['families']):
            color='#c76b16' if family['family']=='selective' else '#28739d' if family['family'] in NEURAL else '#717579'
            members=family['members'];positions=(-.18,0.,.18) if len(members)==3 else (0.,)
            absent=[]
            for member,offset in zip(members,positions,strict=True):
                if member[metric] is None:
                    absent.append(str(member['seed']) if member['seed'] is not None else 'reference')
                else:
                    ax.scatter(member[metric],index+offset,s=36,color=color,marker=seed_markers[member['seed']],
                               edgecolors='white',linewidths=.45,zorder=4)
            if family[mean_key] is not None:
                ax.scatter(family[mean_key],index,s=66,marker='D',facecolors='none',edgecolors='black',linewidths=1.2,zorder=5)
            elif any(m[metric] is not None for m in members):
                ax.text(.985,index+.36,'incomplete family',transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=8,color='#8a3f29')
            if absent:
                ax.text(.985,index-.25,'unavailable: '+', '.join(absent),transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=7.4,color='#8a3f29')
        ax.axhline(6.5,color='#b4b7ba',linewidth=.8)
        ax.set_yticks(range(len(FAMILIES)),[LABELS[name] for name in FAMILIES])
        ax.set_ylim(len(FAMILIES)-.45,-.65)
        ax.grid(axis='x',color='#dddddd',linewidth=.6,alpha=.8)
        ax.set_axisbelow(True)
        suffix='; symlog scale, zero included' if spec['scale']=='symlog' else '; zero included'
        ax.set_title(title,loc='left',pad=12)
        ax.set_xlabel(('Mean record RMSE, FIT-standardized units' if metric=='rmse' else 'Mean seed-median latency, ms')+'\nLower is better'+suffix)
    axes[1].tick_params(axis='y',labelleft=False)
    fig.suptitle(f"FSM correction: {values['status']} ({values['passed']}/{values['total']} continuation checks)",x=.08,ha='left',fontsize=17,fontweight='bold')
    fig.text(.08,.933,'Custom 100-observation context / 128-step conditional forecast on 12 development records with correlated periods.',fontsize=11)
    legend=[Line2D([0],[0],marker=seed_markers[s],linestyle='none',color='#28739d',label=f'Seed {s}') for s in SEEDS]
    legend += [Line2D([0],[0],marker='D',linestyle='none',markerfacecolor='none',color='black',label='Complete evaluation mean'),
               Line2D([0],[0],marker='o',linestyle='none',color='#717579',label='VARX reference')]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.54,.074),ncol=5,frameon=False,fontsize=9)
    fig.text(.08,.052,'Neural models: float32. VARX controls: float64. Three neural markers are individual seeds, not confidence intervals.',fontsize=9)
    fig.text(.08,.033,'Cost: each seed median covers 24 full normalized requests; family diamonds average those medians. Failed slots remain labeled.',fontsize=9)
    fig.text(.08,.014,'Warm in-memory CPU inference excludes model loading/disk I/O. Development only; no official benchmark, SOTA or generalization claim.',fontsize=9)
    fig.subplots_adjust(left=.18,right=.98,top=.875,bottom=.165,wspace=.15)
    benchmark=output/'benchmark.png';fig.savefig(benchmark,dpi=180,facecolor='white');plt.close(fig)

    candidate=next(f for f in values['families'] if f['family']=='selective')
    fig,ax=plt.subplots(figsize=(10,4.4))
    colors=('#28739d','#d08c32','#7d5da3')
    for i,member in enumerate(candidate['members']):
        counts=member['routing_counts'];total=sum(counts) if counts is not None else 0
        if total:
            left=0.
            for block,count in enumerate(counts):
                share=count/total
                ax.barh(i,share,left=left,height=.56,color=colors[block],label=f'Block {block}' if i==0 else None)
                if share>=.05:ax.text(left+share/2,i,f'{100*share:.1f}%',ha='center',va='center',color='white',fontsize=10)
                left+=share
            ax.text(1.015,i,f'n={total:,}'+(' (partial evaluation)' if member['status']!='complete' else ''),va='center',fontsize=9)
        else:
            message='Diagnostic unavailable' if counts is None else 'No nonzero-gradient corrections'
            ax.text(.02,i,message,va='center',color='#8a3f29',fontsize=10)
    ax.set_xlim(0,1);ax.set_ylim(2.6,-.6);ax.set_yticks(range(3),[f'Seed {s}' for s in SEEDS])
    ax.set_xticks((0,.25,.5,.75,1),('0%','25%','50%','75%','100%'))
    ax.set_xlabel('Share of nonzero-gradient context corrections')
    ax.set_title('Selective routing by seed',loc='left',pad=13)
    handles=[Line2D([0],[0],color=color,linewidth=8,label=f'Block {block}') for block,color in enumerate(colors)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.44,.065),ncol=3,frameon=False)
    fig.text(.12,.025,'A collapsed bar indicates a fixed route in these contexts. Counts are correlated events, not independent samples.',fontsize=9)
    fig.subplots_adjust(left=.12,right=.77,top=.84,bottom=.30)
    routing=output/'routing.png';fig.savefig(routing,dpi=180,facecolor='white');plt.close(fig)
    return (benchmark,routing)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    paths=(args.study/'summary.json',args.study/'evaluations.json')
    inputs={p.name:descriptor(p) for p in paths}
    values=extract(read_json(paths[0]),read_json(paths[1]))
    args.output.mkdir(parents=True,exist_ok=False)
    figures=render(values,args.output)
    data_path=args.output/'plotted-values.json';write_json(data_path,values)
    require(inputs=={p.name:descriptor(p) for p in paths},'saved scalar inputs changed during rendering')
    write_json(args.output/'plot-receipt.json',{'scope':'Presentation-only scalar rendering after caller-admitted original run/audit; no model, array or scientific-rule replay.',
               'study':str(args.study.resolve()),'inputs':inputs,'renderer':descriptor(__file__),
               'outputs':{p.name:descriptor(p) for p in (*figures,data_path)}})


if __name__=='__main__':main()
