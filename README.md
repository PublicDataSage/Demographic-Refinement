# Demographic Refinement Invariance in Algorithmic Fairness Audits

Research code for **“The Granularity Paradox in Algorithmic Fairness Audits: Demographic Refinement Invariance and Measurement Sensitivity”** (ACM FAccT submission).

This repository studies whether a fairness assessment changes when a demographic group is subdivided without adding new fairness-relevant information. It combines exact structural analysis, finite-sample simulations, benchmark audits, matched-null intersectional analysis, and downstream fairness-verdict/model-ranking sensitivity.

## Study components

| Component | Script | Purpose |
| --- | --- | --- |
| Experiment 1 | `refinement_study.py --experiment 1` | Exact null refinement; structural invariance and CVaR taxonomy steering |
| Experiment 2 | `refinement_study.py --experiment 2` | Sampling-related refinement sensitivity (SRI/ASRI) |
| Experiment 3 | `refinement_study.py --experiment 3` | Controlled refinements on four datasets and three classifiers |
| Experiment 4 | `refinement_study.py --experiment 4` | Observed ACS intersectional refinement versus matched-null permutations |
| Paired audit decisions | `decision_analyses.py` | Threshold-verdict reversals, paired model-ranking reversals, and illustrative minority-group profiles |
| Matched-population robustness | `matched_population_exp4.py` | Recomputes coarse/refined/null results on the same retained ACS observations |
| Decision figures / margins | `generate_decision_figures.py` | Figures and ranking-margin reconciliation (if present in your working tree) |
| Output validation | `validate_results.py` | Cross-file, arithmetic, and numerical-convention checks (if present in your working tree) |

The first four experiments use the original `refinement_study.py` workflow. The remaining scripts are **separately identified extensions**; do not combine their results by matching repetition numbers across independently generated refinements.

## Setup (Windows PowerShell)

Use Python **3.10 or newer** with compatible versions of the declared packages. From the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

The core dependencies are `numpy`, `pandas`, `matplotlib`, `scikit-learn`, and `folktables`. Use a dependency lock file, if supplied, to recreate the original package versions more closely. The experimental runs reported in the manuscript use seed **20260916**.

### Datasets

- **ACSIncome:** ACS/Folktables, **2018, 1-year, California (CA)** in the reported runs. The original script obtains ACS data through Folktables. For the separate *local* matched-population script, place the 2018 California ACS person CSV at `data/2018/1-Year/psam_p06.csv` or pass its path with `--acs-file`.
- **Adult Income** and **German Credit:** fetched via OpenML by the original script.
- **COMPAS:** retrieved from ProPublica's public COMPAS analysis data by the original script.

An internet connection may be required on first download. Review the original datasets' licenses and terms before redistribution; raw data, caches, virtual environments, and large generated files should not normally be committed to Git.

**ACS preprocessing caution:** `matched_population_exp4.py` reconstructs ACSIncome directly from the local CSV, whereas `refinement_study.py` uses Folktables. The two preprocessing pipelines should be checked for equivalence before treating their estimates as directly interchangeable.

## Reproduce the experiments

The following commands assume the repository root as the current directory and keep all outputs beneath `results/`.

### 1. Original Experiments 1–4

Run individually so that a failure in one experiment does not prevent the others:

```powershell
python refinement_study.py --experiment 1 --out results --seed 20260916
python refinement_study.py --experiment 2 --out results --reps2 500 --seed 20260916
python refinement_study.py --experiment 3 --out results --datasets acs adult compas german --models lr rf hgb --acs-states CA --reps3 100 --seed 20260916
python refinement_study.py --experiment 4 --out results --models lr rf hgb --acs-states CA --perms4 500 --seed 20260916
```

Alternatively, run all four with `python refinement_study.py --experiment all --out results --reps2 500 --reps3 100 --perms4 500 --datasets acs adult compas german --models lr rf hgb --acs-states CA --seed 20260916`.

Use `--max-rows` and smaller `--reps2`, `--reps3`, and `--perms4` values for **smoke tests only**; such outputs are not the full manuscript experiments. The `results/run_config.json` file is overwritten on successive invocations of the main script, so retain per-run command logs/configuration copies if an audit trail is required.

### 2. Paired verdict, ranking, and minority-group analyses

```powershell
python decision_analyses.py --out results/decisions --datasets acs adult compas german --models lr rf hgb --levels 1 2 4 8 16 32 --reps 100 --seed 20260916
```

This script applies **the same randomized refinement to all evaluated classifiers** for each paired comparison. The exploratory fairness-tolerance grid is `0.01 0.02 0.05 0.10 0.15 0.20 0.25 0.30 0.40`. Scores at or below a tolerance pass; different meta-metrics have different scales, so pooled reversal frequencies are descriptive. For ranking reversals, score differences with absolute value at most **1e-12** are numerical ties, not strict reversals.

To recompute decision summaries from saved paired scores *without retraining*:

```powershell
python decision_analyses.py --from-paired-csv results/decisions/paired_audits.csv --out results/decisions
```

### 3. Matched-population Experiment 4 robustness

```powershell
python matched_population_exp4.py --acs-file data/2018/1-Year/psam_p06.csv --output results/matched_population --models lr rf hgb --permutations 500 --bases ppr tpr fpr --thresholds 0 30 50
```

This analysis selects eligible **observed refined groups** using each quantity's denominator, retains the same observations for coarse and refined assessments, and permutes child labels within their parents while preserving child-group sizes. For PPR, denominators count all group observations; for TPR and FPR, they count observations with `Y=1` and `Y=0`, respectively.

**Permutation validity matters:** Observed-group denominator filtering does *not* automatically guarantee that every permuted TPR/FPR subgroup meets the same threshold. Null permutations with undefined outcome-conditional rates may be discarded. Always report the valid/invalid permutation counts. Some newer local versions of this script support an optional `--enforce-null-min-n` flag; use it **only if your installed script lists the flag in `--help`**, and treat that stricter conditional reference distribution as a separate sensitivity analysis.

### 4. Figures, margin analysis, and validation

If the corresponding scripts are included in your working tree:

```powershell
python generate_decision_figures.py --input results/decisions --figures figures/decisions --validation results/validation --margins 0 0.001 0.005 0.01 --dpi 300
python validate_results.py --results results --output results/validation
```

`validate_results.py --strict` treats warnings more strictly; the documented manuscript validation has numerical-boundary/tie-convention warnings that do not imply failed nonboundary checks. The figure-generation command assumes the CLI version developed for this study; run `python generate_decision_figures.py --help` if your local script has different arguments.

## Output structure

```text
results/
  experiment1/
    experiment1_structural.csv
    experiment1_cvar_steering.csv
    fig_exp1_*.png
  experiment2/
    experiment2_finite_sample.csv
    experiment2_summary.csv
    fig_exp2_*.png
  experiment3/
    experiment3_benchmark_refinement.csv
    experiment3_predictive_performance.csv
    experiment3_summary.csv
    fig_exp3_*.png
  experiment4/
    experiment4_null_permutations.csv
    experiment4_summary.csv
    fig_exp4_*.png
  decisions/
    paired_audits.csv
    verdict_detail.csv
    verdict_summary.csv
    ranking_detail.csv
    ranking_summary.csv
    minority_sensitivity.csv
  matched_population/
    experiment4_matched_population_summary.csv
    experiment4_matched_population_nulls.csv
    run_config.json
  validation/
    validation_report.csv
    validation_summary.json
    validation_summary.txt
    ranking_margin_summary.csv
    ranking_margin_by_metric.csv
    ranking_validation_summary.csv
    ranking_summary_reconciliation.csv
    verdict_disagreements.csv
figures/
  decisions/
    fig_decision_reversals.png
    fig_decision_reversals.pdf
```

Some optional diagnostic files are produced only by the corresponding analysis/validation script. Published figure assets may be copied from the experiment output folders into a top-level `figures/` directory for LaTeX use.

## Results and validation snapshot

The following are the **previously validated manuscript-run outputs**, not guarantees for a new environment or a reduced smoke test:

| Analysis | Reversals | Comparisons | Frequency |
| --- | ---: | ---: | ---: |
| Fairness-verdict decisions | 263,630 | 2,835,000 | 9.30% |
| Pairwise model rankings (tie-aware) | 36,852 | 315,000 | 11.70% |

The validation run reported **54 passes, 0 failures, and 2 warnings**:

1. **56 serialized fairness-threshold boundary comparisons:** recomputed verdicts differ from stored flags only at numeric boundaries (within `1e-12`); there were no nonboundary discrepancies. Keep the original stored decision flags for reported counts and disclose the serialization sensitivity.
2. **43 ranking tie-convention differences:** strict sign comparison yields 36,895 reversals versus 36,852 with the original `1e-12` numerical-tie convention. The original tie-aware logic matches the stored ranking flags.

The dataset/model/refinement-grid observations are correlated, and pooled rates should **not** be interpreted as independent-event probabilities, confidence intervals, or general deployment risks. Similarly, matched-null **95% reference intervals are not confidence intervals for the adjusted refinement effect (ARE)**. ARE measures a departure from the selected random-subdivision reference; it is not a causal effect.

The matched-population PPR analyses had fully valid permutation sets in the reported primary filtered configurations. Certain TPR/FPR configurations had substantially fewer valid permutations, in some cases too few for stable reference-interval interpretation; treat these analyses as exploratory.

## Measurement conventions

- Group audit quantities: positive prediction rate (PPR), true-positive rate (TPR), and false-positive rate (FPR).
- Meta-metrics: `max_min`, `pairwise`, `mad`, `variance`, `weighted_mad`, `weighted_variance`, `weighted_pairwise`, `normalized_weighted_pairwise`, `cvar_50_conserved`, and `cvar_50_uniform`.
- Group weights in empirical audits are the shares of **eligible audit denominators** (all group observations for PPR, positive outcomes for TPR, negative outcomes for FPR).
- `weighted_pairwise` is **unnormalized**; `normalized_weighted_pairwise` is a separate metric and generally need not satisfy DRI under null refinement.
- Exact null refinement concerns the generating distribution. Independently randomized child labels generally yield nonidentical *sample* estimates.
- Structural invariance does not establish normative fairness: population-weighted scores may underemphasize substantial disparities in small groups. Synthetic minority-group configurations are illustrations, not population harm estimates.

## Reproducibility and citation

Use the manuscript's Experimental Design and Results sections for definitions, dataset splits, settings, and interpretation. Record Python/package versions, random seeds, source-data versions, full command lines, and run configs when reproducing the study. Do not describe independently generated subdivisions as matched across models; use `decision_analyses.py` for paired model comparisons.

If you use or extend this code, please cite the associated manuscript once a public citation or DOI is available. A stable repository release/commit and an archival code DOI can be added here after publication.
