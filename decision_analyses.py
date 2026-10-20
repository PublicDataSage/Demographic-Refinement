"""Additional FAccT analyses: paired-model audit rankings, tolerance verdicts,
and population-weighting/minority-group sensitivity.

New analyses are exploratory unless otherwise labeled. No tolerance is claimed
as a universally accepted fairness standard. Model ranking uses exactly the
same refined group labels for LR/RF/HGB for each repetition.
"""
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from refinement_study import (get_dataset, build_model, random_refine_labels,
                              select_refinement_targets, audit_metrics)

METRICS = ['max_min', 'pairwise', 'weighted_pairwise', 'mad', 'weighted_mad',
           'cvar_50_conserved', 'cvar_50_uniform']
BASES = ['ppr','tpr','fpr']


def make_paired_data(datasets, models, levels, reps, seed, max_rows, output):
    output.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(seed)
    result=[]
    for dataset in datasets:
        d=get_dataset(dataset)
        X,y,demo=d.X.copy(),d.y.copy(),d.demographics.copy()
        if max_rows and len(X)>max_rows:
            inds,_=train_test_split(np.arange(len(X)),train_size=max_rows,
                                    stratify=np.asarray(y),random_state=seed)
            X=X.iloc[inds];y=y.iloc[inds];demo=demo.iloc[inds]
        train,test=train_test_split(np.arange(len(X)),test_size=.25,
                                    random_state=seed,stratify=np.asarray(y))
        yt=np.asarray(y.iloc[test]).astype(int)
        groups=demo.iloc[test][d.primary_group].astype(str).to_numpy()
        targets=select_refinement_targets(groups,min_parent_n=max(levels))
        predictions={}
        for model in models:
            print('Training',dataset,model,flush=True)
            estimator=build_model(model,X.iloc[train],seed=seed)
            estimator.fit(X.iloc[train],np.asarray(y.iloc[train]).astype(int))
            predictions[model]=estimator.predict(X.iloc[test]).astype(int)
        for role,target,n in targets:
            for level in levels:
                for rep in range(1 if level==1 else reps):
                    refined=random_refine_labels(groups,target,level,rng)
                    # Same exact grouping applied to every model.
                    for model,pred in predictions.items():
                        for base in BASES:
                            values,_=audit_metrics(yt,pred,refined,base)
                            for metric in METRICS:
                                if metric in values and np.isfinite(values[metric]):
                                    result.append(dict(dataset=dataset,model=model,
                                        base_quantity=base,target_size_class=role,
                                        refined_group=target,m=level,rep=rep,
                                        metric=metric,value=values[metric]))
                print('Completed',dataset,target,level,flush=True)
    data=pd.DataFrame(result)
    data.to_csv(output/'paired_audits.csv',index=False)
    return data


def decision_results(df, output, tolerances):
    output.mkdir(parents=True,exist_ok=True)
    keys=['dataset','model','base_quantity','target_size_class','refined_group','metric']
    baseline=df[df.m==1][keys+['value']].rename(columns={'value':'coarse_value'})
    refined=df[df.m>1].merge(baseline,on=keys,validate='many_to_one')
    verdict=[]
    for t in tolerances:
        frame=refined.copy()
        frame['tolerance']=float(t)
        frame['coarse_pass']=frame.coarse_value.le(t)
        frame['refined_pass']=frame.value.le(t)
        frame['reversal']=frame.coarse_pass.ne(frame.refined_pass)
        frame['direction']=np.where(frame.reversal,
                np.where(frame.coarse_pass,'pass_to_fail','fail_to_pass'),'unchanged')
        verdict.append(frame)
    detail=pd.concat(verdict,ignore_index=True)
    detail.to_csv(output/'verdict_detail.csv',index=False)
    summary=detail.groupby(['dataset','base_quantity','metric','m','tolerance'],as_index=False).agg(
        reversals=('reversal','sum'),comparisons=('reversal','size'),
        reversal_share=('reversal','mean'))
    summary.to_csv(output/'verdict_summary.csv',index=False)
    # Sort by value: smaller disparity = fairer. Ties handled explicitly.
    wide=df.pivot_table(index=['dataset','base_quantity','target_size_class',
          'refined_group','m','rep','metric'],columns='model',values='value').reset_index()
    model_cols=[c for c in wide.columns if c in ['lr','rf','hgb','mlp']]
    ranks=[]
    groupkeys=['dataset','base_quantity','target_size_class','refined_group','metric']
    coarse=wide[wide.m==1].set_index(groupkeys)
    for _,row in wide[wide.m>1].iterrows():
        k=tuple(row[x] for x in groupkeys)
        if k not in coarse.index:continue
        b=coarse.loc[k]
        if isinstance(b,pd.DataFrame):b=b.iloc[0]
        for a,c in combinations(model_cols,2):
            if pd.isna(row[a]) or pd.isna(row[c]) or pd.isna(b[a]) or pd.isna(b[c]):continue
            original=float(b[a])-float(b[c]);new=float(row[a])-float(row[c])
            tol=1e-12
            before=0 if abs(original)<=tol else np.sign(original)
            after=0 if abs(new)<=tol else np.sign(new)
            ranks.append({**{g:row[g] for g in groupkeys},'m':row.m,'rep':row.rep,
                'model_a':a,'model_b':c,'coarse_difference':original,
                'refined_difference':new,'ranking_reversal':bool(before*after==-1),
                'tie_involved':bool(before==0 or after==0)})
    rank=pd.DataFrame(ranks)
    rank.to_csv(output/'ranking_detail.csv',index=False)
    if len(rank):
        rs=rank.groupby(['dataset','base_quantity','metric','m','model_a','model_b'],
                        as_index=False).agg(reversals=('ranking_reversal','sum'),
                                comparisons=('ranking_reversal','size'),
                                reversal_share=('ranking_reversal','mean'),
                                ties=('tie_involved','sum'))
        rs.to_csv(output/'ranking_summary.csv',index=False)
    print('Verdict comparisons:',len(detail),'ranking comparisons:',len(rank))


def minority_sensitivity(output):
    """Illustrative population profiles, NOT observations from benchmark data."""
    from refinement_study import meta_metrics
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for minority_share in [0.001,0.005,0.01,0.05,0.1,0.2]:
        for disadvantaged_rate in [0.0,0.1,0.3,0.5]:
            # Majority rate 0.8 and minority rate below it.
            profile=np.array([0.8,disadvantaged_rate]);weights=np.array([1-minority_share,minority_share])
            scores=meta_metrics(profile,weights)
            for metric in METRICS:
                rows.append(dict(minority_share=minority_share,
                  majority_rate=0.8,minority_rate=disadvantaged_rate,
                  metric=metric,disparity=scores[metric],
                  note='synthetic illustration; not an estimate of social harm'))
    pd.DataFrame(rows).to_csv(output/'minority_sensitivity.csv',index=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path('results_extensions'))
    p.add_argument('--datasets',nargs='+',default=['acs','adult','compas','german'])
    p.add_argument('--models',nargs='+',default=['lr','rf','hgb'])
    p.add_argument('--levels',nargs='+',type=int,default=[1,2,4,8,16,32])
    p.add_argument('--reps',type=int,default=100)
    p.add_argument('--seed',type=int,default=20260916)
    p.add_argument('--max-rows',type=int,default=None)
    p.add_argument('--tolerances',nargs='+',type=float,
                   default=[0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.40])
    p.add_argument('--from-paired-csv',type=Path,default=None)
    a=p.parse_args()
    df=pd.read_csv(a.from_paired_csv) if a.from_paired_csv else make_paired_data(
        a.datasets,a.models,a.levels,a.reps,a.seed,a.max_rows,a.out)
    decision_results(df,a.out,a.tolerances)
    minority_sensitivity(a.out)

if __name__=='__main__':main()
