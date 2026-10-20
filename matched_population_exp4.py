"""Matched-population sensitivity for Experiment 4.

Uses ACS 2018 California local person CSV and the original experiment's model
builder. Requires pandas, numpy, matplotlib, scikit-learn. Folktables not required
for the local reconstruction, but compare reconstruction against original results.

The retained rows are determined once from the observed fine partition, for
*each* transition, audit rate, and minimum denominator. Coarse, observed fine,
and all null permutations use precisely this same retained set. Null labels are
permuted within coarse parent groups, conserving the fine category sizes.

Outputs are new: original results are not overwritten.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from refinement_study import build_model, age_band, audit_metrics, audit_group_table, meta_metrics

FEATURES = ['AGEP','COW','SCHL','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P']
READ_COLS = FEATURES + ['PINCP','PWGTP']
SEED = 20260916

def load_local_acs(csv_path: Path):
    raw = pd.read_csv(csv_path, usecols=READ_COLS)
    # Folktables ACSIncome feature set, target and standard 2018 filters.
    raw = raw.dropna(subset=READ_COLS)
    raw = raw.loc[(raw.AGEP > 16) & (raw.PINCP > 100) & (raw.PWGTP >= 1)].reset_index(drop=True)
    X = raw[FEATURES].copy()
    y = (raw.PINCP > 50000).astype(int)
    demo = pd.DataFrame({'sex': X.SEX.astype(str), 'race': X.RAC1P.astype(str), 'age': X.AGEP})
    print(
        f"ACS reconstruction: {len(X):,} eligible rows; "
        f"positive target rate={y.mean():.6f}",
        flush=True,
    )
    return X,y,demo

def denom_mask(y, base):
    if base == 'ppr': return np.ones(len(y), dtype=bool)
    if base == 'tpr': return y == 1
    if base == 'fpr': return y == 0
    raise ValueError(base)

def metric_for_subset(y, pred, groups, base):
    return audit_metrics(y, pred, groups, base)[0]

def permute(fine, coarse, rng):
    result = fine.copy()
    for parent in np.unique(coarse):
        inds = np.where(coarse == parent)[0]
        result[inds] = rng.permutation(result[inds])
    return result

def analyse_one(
    y, pred, coarse, fine, base, min_n,
    permutations, rng,
    enforce_null_min_n=False,
):
    eligible = denom_mask(y, base)
    counts = pd.Series(fine[eligible]).value_counts()
    retained_categories = counts.index[counts.to_numpy() >= max(min_n,1)]
    keep = np.isin(fine, retained_categories)
    yc, pc, cc, fc = y[keep], pred[keep], coarse[keep], fine[keep]
    if (
        not np.any(keep)
        or len(np.unique(cc)) < 2
        or len(np.unique(fc)) < 2
    ):
        return None, []
    coarse_mm = metric_for_subset(yc, pc, cc, base)
    fine_mm = metric_for_subset(yc, pc, fc, base)
    common = sorted(set(coarse_mm).intersection(fine_mm))
    if not common: return None, []
    nulls = {m: [] for m in common}
    perm_rows = []
    invalid = 0
    fine_codes, fine_labels = pd.factorize(fc, sort=True)
    coarse_codes, coarse_labels = pd.factorize(cc, sort=True)
    parent_indices = [np.flatnonzero(coarse_codes==i) for i in range(len(coarse_labels))]
    nr = len(fine_labels)
    eligible_rows = denom_mask(yc, base)
    positive = (pc == 1) & eligible_rows
    for b in range(permutations):
        shuffled = fine_codes.copy()
        for inds in parent_indices:
            shuffled[inds] = rng.permutation(shuffled[inds])
        counts = np.bincount(shuffled[eligible_rows], minlength=nr)
        # Observed fine-group eligibility always uses min_n. By default,
        # null permutations only require positive denominators.
        required_n = max(min_n, 1) if enforce_null_min_n else 1
        if np.any(counts < required_n):
            invalid += 1
            continue
        positives = np.bincount(shuffled[positive], minlength=nr)
        rates = positives/counts
        weights = counts/counts.sum()
        mm = meta_metrics(rates, weights)
        for m in common:
            v = mm[m] - coarse_mm[m]
            nulls[m].append(v)
            perm_rows.append({'base_quantity':base,'min_group_n':min_n,'perm':b,'metric':m,'delta_null':v})
    out=[]
    for m in common:
        arr=np.asarray(nulls[m]);
        if not len(arr): continue
        real=fine_mm[m]-coarse_mm[m]
        lo,hi=np.quantile(arr,[.025,.975]); mean=float(np.mean(arr))
        # Direction-agnostic extremeness relative to permutation center.
        p_two=(1+np.sum(np.abs(arr-mean)>=abs(real-mean)))/(len(arr)+1)
        out.append(dict(base_quantity=base,min_group_n=min_n,metric=m,
                        coarse_value=coarse_mm[m],fine_value=fine_mm[m],
                        delta_real=real,null_mean=mean,null_q025=lo,null_q975=hi,
                        ARE=real-mean,p_two_sided_mc=p_two,
                        n_test_total=len(y),n_retained=len(yc),
                        fraction_retained=len(yc)/len(y),
                        coarse_groups=len(np.unique(cc)),fine_groups=len(np.unique(fc)),
                        valid_permutations=len(arr),invalid_permutations=invalid))
    return out,perm_rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--acs-file', type=Path, default=ROOT/'data/2018/1-Year/psam_p06.csv')
    ap.add_argument('--output',type=Path,default=ROOT/'facct_matched_population_results')
    ap.add_argument('--models',nargs='+',default=['lr','rf','hgb'])
    ap.add_argument('--permutations',type=int,default=500)
    ap.add_argument('--bases',nargs='+',default=['ppr','tpr','fpr'])
    ap.add_argument('--thresholds',nargs='+',type=int,default=[0,30,50])
    ap.add_argument(
        '--enforce-null-min-n',
        action='store_true',
        help=(
            'Also enforce the minimum denominator on every '
            'permuted subgroup. Changes the null reference '
            'distribution and is intended for sensitivity analysis.'
        ),
    )
    ap.add_argument('--max-rows',type=int,default=None)
    args=ap.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    X,y,demo=load_local_acs(args.acs_file)
    if args.max_rows is not None and len(X)>args.max_rows:
        idx,_=train_test_split(np.arange(len(X)),train_size=args.max_rows,random_state=SEED,stratify=y.to_numpy())
        X=X.iloc[idx];y=y.iloc[idx];demo=demo.iloc[idx]
    train_idx,test_idx=train_test_split(np.arange(len(X)),test_size=.25,random_state=SEED,stratify=y.to_numpy())
    Xtrain,Xtest=X.iloc[train_idx],X.iloc[test_idx]
    yt=y.iloc[test_idx].to_numpy(dtype=int)
    dtest=demo.iloc[test_idx].reset_index(drop=True)
    sex=dtest.sex.astype(str);race=dtest.race.astype(str);age=age_band(dtest.age)
    parts={'sex':sex.to_numpy(),'race_sex':(race+'|'+sex).to_numpy(),'race_sex_age':(race+'|'+sex+'|'+age).to_numpy()}
    summaries=[];null_rows=[]
    for model_name in args.models:
        print('Training',model_name,flush=True)
        model=build_model(model_name,Xtrain,seed=SEED)
        model.fit(Xtrain,y.iloc[train_idx].to_numpy(dtype=int))
        pred=model.predict(Xtest).astype(int)
        print(model_name,'test',len(yt),'prediction rate',pred.mean(),flush=True)
        for ci,fi in [('sex','race_sex'),('race_sex','race_sex_age')]:
            for base in args.bases:
                for threshold in args.thresholds:
                    # Independent reproducible streams for each configuration.
                    rng=np.random.default_rng(SEED+sum(map(ord,model_name+ci+fi+base))+threshold)
                    result, rows = analyse_one(
                        yt, pred, parts[ci], parts[fi], base, threshold,
                        args.permutations, rng,
                        enforce_null_min_n=args.enforce_null_min_n,
                    )
                    if result is None:continue
                    for r in result:r.update(model=model_name,transition=ci+'->'+fi)
                    for r in rows:r.update(model=model_name,transition=ci+'->'+fi)
                    summaries.extend(result);null_rows.extend(rows)
                    print(' completed',model_name,ci,fi,base,threshold,flush=True)
        pd.DataFrame(summaries).to_csv(args.output/'experiment4_matched_population_summary.csv',index=False)
        pd.DataFrame(null_rows).to_csv(args.output/'experiment4_matched_population_nulls.csv',index=False)
    (args.output/'run_config.json').write_text(json.dumps({
        'models': args.models,
        'permutations': args.permutations,
        'max_rows': args.max_rows,
        'seed': SEED,
        'population_rule': 'observed fine groups eligible by base denominator; common retained rows for all metrics and null permutations',
        'input_csv': str(args.acs_file),
        'enforce_null_min_n': args.enforce_null_min_n,
        'null_denominator_policy': (
            'enforce threshold on every permutation'
            if args.enforce_null_min_n
            else 'observed eligibility only; null denominators must be positive'
        ),
    }, indent=2))
    print('Saved to',args.output)

if __name__=='__main__':main()
