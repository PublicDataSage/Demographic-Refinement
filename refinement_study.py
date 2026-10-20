from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_SEED = 20260916
COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_onehot():
    """Compatibility across scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    s = w.sum()
    return float(np.dot(x, w) / s) if s > 0 else np.nan


def weighted_cvar(losses: Sequence[float], weights: Sequence[float], alpha: float = 0.5) -> float:
    """
    CVaR_alpha(L) = min_t [t + E[(L-t)_+]/(1-alpha)] for a discrete weighted loss.
    The optimum occurs at a support point, so evaluating unique losses is exact.
    """
    losses = np.asarray(losses, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(losses) & np.isfinite(weights) & (weights >= 0)
    losses, weights = losses[mask], weights[mask]
    if len(losses) == 0 or weights.sum() <= 0:
        return np.nan
    weights = weights / weights.sum()
    if alpha >= 1.0:
        return float(np.max(losses))
    candidates = np.unique(losses)
    vals = [
        t + np.sum(weights * np.maximum(losses - t, 0.0)) / (1.0 - alpha)
        for t in candidates
    ]
    return float(np.min(vals))


def meta_metrics(
    values: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    cvar_alpha: float = 0.5,
) -> Dict[str, float]:
    """Compute category-weighted and refinement-consistent meta-metrics."""
    x = np.asarray(values, dtype=float)
    if weights is None:
        w = np.ones(len(x), dtype=float) / len(x)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()

    valid = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x, w = x[valid], w[valid]
    if len(x) == 0:
        return {}
    w = w / w.sum()

    k = len(x)
    mean_u = float(np.mean(x))
    mean_w = weighted_mean(x, w)

    max_min = float(np.max(x) - np.min(x)) if k > 1 else 0.0

    pair_sum = 0.0
    wpair_sum = 0.0
    wpair_den = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            d = abs(x[i] - x[j])
            pair_sum += d
            wij = w[i] * w[j]
            wpair_sum += wij * d
            wpair_den += wij

    pair = pair_sum / math.comb(k, 2) if k > 1 else 0.0
    mad = float(np.mean(np.abs(x - mean_u)))
    var = float(np.mean((x - mean_u) ** 2))
    wmad = float(np.sum(w * np.abs(x - mean_w)))
    wvar = float(np.sum(w * (x - mean_w) ** 2))
    wpair = float(wpair_sum)
    nwpair = float(wpair_sum / wpair_den) if wpair_den > 0 else 0.0

    # Fairness loss for CVaR: absolute deviation from the refinement-consistent
    # weighted reference point. Only the audit weighting policy changes.
    losses = np.abs(x - mean_w)
    cvar_conserved = weighted_cvar(losses, w, alpha=cvar_alpha)
    cvar_uniform = weighted_cvar(losses, np.ones(k) / k, alpha=cvar_alpha)

    return {
        "max_min": max_min,
        "pairwise": float(pair),
        "mad": mad,
        "variance": var,
        "weighted_mad": wmad,
        "weighted_variance": wvar,
        "weighted_pairwise": wpair,
        "normalized_weighted_pairwise": nwpair,
        f"cvar_{int(cvar_alpha*100)}_conserved": cvar_conserved,
        f"cvar_{int(cvar_alpha*100)}_uniform": cvar_uniform,
    }


def refine_profile(
    values: Sequence[float],
    weights: Sequence[float],
    target_idx: int,
    m: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact null refinement of one category into m identical descendants."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    a = values[target_idx]
    p = weights[target_idx]
    new_values = np.delete(values, target_idx).tolist() + [a] * m
    new_weights = np.delete(weights, target_idx).tolist() + [p / m] * m
    return np.asarray(new_values), np.asarray(new_weights)


def age_band(age: pd.Series) -> pd.Series:
    bins = [-np.inf, 24, 34, 44, 54, 64, np.inf]
    labels = ["<=24", "25-34", "35-44", "45-54", "55-64", "65+"]
    return pd.cut(pd.to_numeric(age, errors="coerce"), bins=bins, labels=labels).astype(str)


# -----------------------------------------------------------------------------
# Experiment 1: deterministic structural refinement
# -----------------------------------------------------------------------------


def experiment1(outdir: Path) -> None:
    outdir = ensure_dir(outdir / "experiment1")

    profiles = {
        "P1_balanced": (np.array([0.40, 0.60, 0.80]), np.array([1/3, 1/3, 1/3])),
        "P2_asymmetric": (np.array([0.10, 0.40, 0.90]), np.array([1/3, 1/3, 1/3])),
        "P3_imbalanced": (np.array([0.20, 0.50, 0.80]), np.array([0.60, 0.25, 0.15])),
    }
    levels = [1, 2, 4, 8, 16, 32, 64]
    names = ["A", "B", "C"]

    rows = []
    for profile_name, (vals, weights) in profiles.items():
        for target_idx, target_name in enumerate(names):
            base = meta_metrics(vals, weights)
            for m in levels:
                rv, rw = refine_profile(vals, weights, target_idx, m)
                mm = meta_metrics(rv, rw)
                for metric, value in mm.items():
                    b = base[metric]
                    rows.append({
                        "profile": profile_name,
                        "refined_group": target_name,
                        "m": m,
                        "metric": metric,
                        "value": value,
                        "baseline": b,
                        "delta": value - b,
                        "RI": abs(value - b),
                        "ratio": value / b if abs(b) > 1e-15 else np.nan,
                    })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "experiment1_structural.csv", index=False)

    # Main normalized structural curve: P1, refine A.
    p = df[(df.profile == "P1_balanced") & (df.refined_group == "A")]
    metrics = [
        "max_min", "pairwise", "mad", "variance",
        "weighted_mad", "weighted_variance", "weighted_pairwise",
    ]
    plt.figure(figsize=(8, 5))
    for metric in metrics:
        q = p[p.metric == metric].sort_values("m")
        plt.plot(q.m, q.ratio, marker="o", label=metric)
    plt.xscale("log", base=2)
    plt.xlabel("Refinement level m")
    plt.ylabel(r"Normalized score $D^{(m)}/D^{(1)}$")
    plt.title("Experiment 1: structural refinement")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "fig_exp1_structural_ratio.png", dpi=220)
    plt.close()

    # Dedicated CVaR taxonomy-steering demonstration.
    losses = np.array([0.10, 0.50, 0.90])
    original_w = np.array([1/3, 1/3, 1/3])
    cvar_rows = []
    for alpha in [0.50, 0.80, 0.90]:
        baseline = weighted_cvar(losses, original_w, alpha)
        for target_idx, target_name in enumerate(names):
            for m in levels:
                # Conserved audit mass.
                lv, cw = refine_profile(losses, original_w, target_idx, m)
                conserved = weighted_cvar(lv, cw, alpha)
                # Re-uniformized named categories.
                uw = np.ones(len(lv)) / len(lv)
                uniform = weighted_cvar(lv, uw, alpha)
                cvar_rows.extend([
                    {
                        "alpha": alpha, "refined_group": target_name, "m": m,
                        "weighting": "conserved", "cvar": conserved,
                        "baseline": baseline,
                    },
                    {
                        "alpha": alpha, "refined_group": target_name, "m": m,
                        "weighting": "uniform_categories", "cvar": uniform,
                        "baseline": baseline,
                    },
                ])
    cdf = pd.DataFrame(cvar_rows)
    cdf.to_csv(outdir / "experiment1_cvar_steering.csv", index=False)

    plt.figure(figsize=(8, 5))
    q = cdf[(cdf.alpha == 0.50) & (cdf.weighting == "uniform_categories")]
    for group in names:
        z = q[q.refined_group == group].sort_values("m")
        plt.plot(z.m, z.cvar, marker="o", label=f"refine {group}")
    plt.xscale("log", base=2)
    plt.xlabel("Refinement level m")
    plt.ylabel(r"CVaR$_{0.50}$")
    plt.title("Experiment 1: CVaR taxonomy steering")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fig_exp1_cvar_steering.png", dpi=220)
    plt.close()

    print(f"Experiment 1 complete -> {outdir}")


# -----------------------------------------------------------------------------
# Experiment 2: finite-sample refinement
# -----------------------------------------------------------------------------


def simulate_group_sample(
    rng: np.random.Generator,
    N: int,
    pi: np.ndarray,
    rates: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n = rng.multinomial(N, pi)
    pos = np.array([rng.binomial(n_i, r_i) for n_i, r_i in zip(n, rates)])
    return n, pos


def refine_counts(
    rng: np.random.Generator,
    n_parent: int,
    pos_parent: int,
    m: int,
) -> Tuple[np.ndarray, np.ndarray]:
    probs = np.ones(m) / m
    child_pos = rng.multinomial(pos_parent, probs)
    child_neg = rng.multinomial(n_parent - pos_parent, probs)
    child_n = child_pos + child_neg
    return child_n, child_pos


def count_metrics(n: np.ndarray, pos: np.ndarray) -> Optional[Dict[str, float]]:
    if np.any(n <= 0):
        return None
    rates = pos / n
    weights = n / n.sum()
    return meta_metrics(rates, weights)


def experiment2(outdir: Path, reps: int = 500, seed: int = RANDOM_SEED) -> None:
    outdir = ensure_dir(outdir / "experiment2")
    rng = np.random.default_rng(seed)

    rates = np.array([0.40, 0.60, 0.80])
    levels = [1, 2, 4, 8, 16, 32]
    Ns = [1_000, 5_000, 25_000, 100_000]
    prevalences = {
        "balanced": np.array([1/3, 1/3, 1/3]),
        "minority10": np.array([0.10, 0.45, 0.45]),
    }

    rows = []
    for prev_name, pi in prevalences.items():
        for N in Ns:
            for rep in range(reps):
                n, pos = simulate_group_sample(rng, N, pi, rates)
                base_metrics = count_metrics(n, pos)
                if base_metrics is None:
                    continue

                for m in levels:
                    # Exact structural reference from true population quantities.
                    rv, rw = refine_profile(rates, pi, target_idx=0, m=m)
                    structural_metrics = meta_metrics(rv, rw)
                    base_pop = meta_metrics(rates, pi)

                    if m == 1:
                        n_ref, pos_ref = n.copy(), pos.copy()
                        empty_children = 0
                    else:
                        child_n, child_pos = refine_counts(rng, int(n[0]), int(pos[0]), m)
                        empty_children = int(np.sum(child_n == 0))
                        n_ref = np.concatenate([n[1:], child_n])
                        pos_ref = np.concatenate([pos[1:], child_pos])

                    empirical = count_metrics(n_ref, pos_ref)
                    if empirical is None:
                        # Record audit failure caused by empty empirical categories.
                        rows.append({
                            "prevalence": prev_name, "N": N, "rep": rep, "m": m,
                            "metric": "AUDIT_UNDEFINED", "value": np.nan,
                            "baseline": np.nan, "delta_emp": np.nan,
                            "delta_struct": np.nan, "SRI": np.nan, "ASRI": np.nan, "RI": np.nan,
                            "expected_child_n": N * pi[0] / m,
                            "empty_children": empty_children,
                        })
                        continue

                    for metric, value in empirical.items():
                        b = base_metrics[metric]
                        d_emp = value - b
                        d_struct = structural_metrics[metric] - base_pop[metric]
                        rows.append({
                            "prevalence": prev_name,
                            "N": N,
                            "rep": rep,
                            "m": m,
                            "metric": metric,
                            "value": value,
                            "baseline": b,
                            "delta_emp": d_emp,
                            "delta_struct": d_struct,
                            "SRI": d_emp - d_struct,
                            "ASRI": abs(d_emp - d_struct),
                            "RI": abs(d_emp),
                            "expected_child_n": N * pi[0] / m,
                            "empty_children": empty_children,
                        })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "experiment2_finite_sample.csv", index=False)

    valid = df[df.metric != "AUDIT_UNDEFINED"].copy()
    summary = (
        valid.groupby(["prevalence", "N", "m", "metric"], as_index=False)
        .agg(
            mean_RI=("RI", "mean"),
            sd_RI=("RI", "std"),
            mean_SRI=("SRI", "mean"),
            sd_SRI=("SRI", "std"),
            mean_ASRI=("ASRI", "mean"),
            sd_ASRI=("ASRI", "std"),
            mean_delta=("delta_emp", "mean"),
            expected_child_n=("expected_child_n", "first"),
            runs=("rep", "count"),
        )
    )
    summary["ci95_RI"] = 1.96 * summary.sd_RI / np.sqrt(summary.runs)
    summary["ci95_ASRI"] = 1.96 * summary.sd_ASRI / np.sqrt(summary.runs)
    summary.to_csv(outdir / "experiment2_summary.csv", index=False)

    # Main finite-sample figure for selected metrics, balanced case.
    selected = ["max_min", "pairwise", "weighted_pairwise"]
    q = summary[(summary.prevalence == "balanced") & (summary.metric.isin(selected))]
    plt.figure(figsize=(8, 5))
    for (metric, N), z in q.groupby(["metric", "N"]):
        z = z.sort_values("m")
        plt.plot(z.m, z.mean_ASRI, marker="o", label=f"{metric}, N={N:,}")
    plt.xscale("log", base=2)
    plt.xlabel("Refinement level m")
    plt.ylabel("Mean absolute sampling refinement instability")
    plt.title("Experiment 2: finite-sample refinement")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(outdir / "fig_exp2_refinement_instability.png", dpi=220)
    plt.close()

    # Collapse against expected child group size.
    q = summary[summary.metric.isin(selected)].copy()
    plt.figure(figsize=(8, 5))
    for metric, z in q.groupby("metric"):
        plt.scatter(z.expected_child_n, z.mean_ASRI, label=metric, alpha=0.7)
    plt.xscale("log")
    plt.xlabel("Expected observations per refined child group")
    plt.ylabel("Mean absolute sampling refinement instability")
    plt.title("Experiment 2: sampling instability vs. effective subgroup size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fig_exp2_child_size.png", dpi=220)
    plt.close()

    print(f"Experiment 2 complete -> {outdir}")


# -----------------------------------------------------------------------------
# Public dataset loaders for Experiments 3 and 4
# -----------------------------------------------------------------------------


@dataclass
class BenchmarkData:
    name: str
    X: pd.DataFrame
    y: pd.Series
    demographics: pd.DataFrame
    primary_group: str


def load_adult() -> BenchmarkData:
    ds = fetch_openml("adult", version=2, as_frame=True)
    X = ds.data.copy()
    target = ds.target.astype(str).str.replace(".", "", regex=False).str.strip()
    y = (target == ">50K").astype(int)
    demo = pd.DataFrame(index=X.index)
    demo["sex"] = X["sex"].astype(str)
    demo["race"] = X["race"].astype(str)
    demo["age"] = pd.to_numeric(X["age"], errors="coerce")
    return BenchmarkData("adult", X, y, demo, "sex")


def load_compas() -> BenchmarkData:
    df = pd.read_csv(COMPAS_URL)
    # Standard ProPublica-style filtering.
    df = df[
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].copy()

    features = [
        "age", "sex", "race", "priors_count", "c_charge_degree",
        "juv_fel_count", "juv_misd_count", "juv_other_count",
    ]
    X = df[features].copy()
    y = df["two_year_recid"].astype(int)
    demo = pd.DataFrame(index=df.index)
    demo["sex"] = df["sex"].astype(str)
    demo["race"] = df["race"].astype(str)
    demo["age"] = pd.to_numeric(df["age"], errors="coerce")
    return BenchmarkData("compas", X, y, demo, "race")


def load_german() -> BenchmarkData:
    ds = fetch_openml("credit-g", version=1, as_frame=True)
    X = ds.data.copy()
    y = (ds.target.astype(str).str.lower() == "good").astype(int)
    demo = pd.DataFrame(index=X.index)
    ps = X["personal_status"].astype(str).str.lower()
    demo["sex"] = np.where(ps.str.contains("female"), "Female", "Male")
    demo["age"] = pd.to_numeric(X["age"], errors="coerce")
    return BenchmarkData("german_credit", X, y, demo, "sex")


def load_acs(states: Sequence[str] = ("CA",), year: int = 2018) -> BenchmarkData:
    try:
        from folktables import ACSDataSource, ACSIncome
    except ImportError as exc:
        raise ImportError("Install folktables with: pip install folktables") from exc

    source = ACSDataSource(survey_year=str(year), horizon="1-Year", survey="person")
    raw = source.get_data(states=list(states), download=True)
    X, y, acs_group = ACSIncome.df_to_pandas(raw)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    if isinstance(acs_group, pd.DataFrame):
        acs_group = acs_group.iloc[:, 0]
    y = pd.Series(np.asarray(y).ravel().astype(int), index=X.index)
    acs_group = pd.Series(np.asarray(acs_group).ravel(), index=X.index)

    demo = pd.DataFrame(index=X.index)
    demo["sex"] = X["SEX"].astype(str)
    demo["race"] = acs_group.astype(str)
    demo["age"] = pd.to_numeric(X["AGEP"], errors="coerce")
    return BenchmarkData("acs_income", X, y, demo, "race")


def get_dataset(name: str, acs_states: Sequence[str] = ("CA",)) -> BenchmarkData:
    name = name.lower()
    if name == "adult":
        return load_adult()
    if name == "compas":
        return load_compas()
    if name in {"german", "german_credit", "credit-g"}:
        return load_german()
    if name in {"acs", "folktables", "acs_income"}:
        return load_acs(states=acs_states)
    raise ValueError(f"Unknown dataset: {name}")


# -----------------------------------------------------------------------------
# Model training
# -----------------------------------------------------------------------------


def column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split columns robustly into numeric and categorical features.

    Pandas may represent text with object, category, StringDtype, or the newer
    str dtype.  Testing a short list of dtype names can therefore misclassify
    categorical columns (e.g., COMPAS sex/race) as numeric.  Treat only truly
    numeric dtypes as numeric and send everything else through categorical
    imputation/encoding.
    """
    numeric = [
        c for c in X.columns
        if pd.api.types.is_numeric_dtype(X[c].dtype)
    ]
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def build_model(name: str, X: pd.DataFrame, seed: int = RANDOM_SEED):
    numeric, categorical = column_types(X)

    numeric_scaled = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    numeric_plain = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])
    categorical_onehot = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", make_onehot()),
    ])

    if name == "lr":
        prep = ColumnTransformer([
            ("num", numeric_scaled, numeric),
            ("cat", categorical_onehot, categorical),
        ])
        clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=seed)
        return Pipeline([("prep", prep), ("clf", clf)])

    if name == "rf":
        prep = ColumnTransformer([
            ("num", numeric_plain, numeric),
            ("cat", categorical_onehot, categorical),
        ])
        clf = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1,
            class_weight="balanced_subsample", random_state=seed,
        )
        return Pipeline([("prep", prep), ("clf", clf)])

    if name == "mlp":
        prep = ColumnTransformer([
            ("num", numeric_scaled, numeric),
            ("cat", categorical_onehot, categorical),
        ])
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=150, early_stopping=True,
            random_state=seed,
        )
        return Pipeline([("prep", prep), ("clf", clf)])

    if name == "hgb":
        # Dense ordinal representation avoids enormous one-hot matrices on ACS.
        cat_ord = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        prep = ColumnTransformer([
            ("num", numeric_plain, numeric),
            ("cat", cat_ord, categorical),
        ], sparse_threshold=0.0)
        clf = HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=200, max_leaf_nodes=31,
            random_state=seed,
        )
        return Pipeline([("prep", prep), ("clf", clf)])

    raise ValueError(f"Unknown model: {name}")


def predictive_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray]) -> Dict[str, float]:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        out["auc"] = roc_auc_score(y_true, y_prob)
    else:
        out["auc"] = np.nan
    return out


# -----------------------------------------------------------------------------
# Group audit helpers
# -----------------------------------------------------------------------------


def audit_group_table(
    y: np.ndarray,
    yhat: np.ndarray,
    groups: Sequence,
    base: str,
) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    yhat = np.asarray(yhat).astype(int)
    groups = np.asarray(groups).astype(str)

    rows = []
    for g in pd.unique(groups):
        gm = groups == g
        if base == "ppr":
            denom = gm
        elif base == "tpr":
            denom = gm & (y == 1)
        elif base == "fpr":
            denom = gm & (y == 0)
        else:
            raise ValueError(base)

        n = int(np.sum(denom))
        if n == 0:
            continue
        value = float(np.mean(yhat[denom] == 1))
        rows.append({"group": g, "value": value, "n": n})

    tab = pd.DataFrame(rows)
    if len(tab) == 0:
        return tab
    tab["weight"] = tab["n"] / tab["n"].sum()
    return tab


def audit_metrics(y, yhat, groups, base: str) -> Tuple[Dict[str, float], pd.DataFrame]:
    tab = audit_group_table(y, yhat, groups, base)
    if len(tab) == 0:
        return {}, tab
    return meta_metrics(tab.value.to_numpy(), tab.weight.to_numpy()), tab


def random_refine_labels(
    groups: Sequence,
    target_group: str,
    m: int,
    rng: np.random.Generator,
) -> np.ndarray:
    g = np.asarray(groups, dtype=object).copy()
    g = np.array([str(v) for v in g], dtype=object)
    idx = np.where(g == str(target_group))[0]
    if m == 1 or len(idx) == 0:
        return g
    if len(idx) < m:
        raise ValueError(f"Target group has {len(idx)} observations but m={m}.")
    idx = rng.permutation(idx)
    child = np.arange(len(idx)) % m
    for k in range(m):
        g[idx[child == k]] = f"{target_group}::r{k:02d}"
    return g


def select_refinement_targets(groups: Sequence, min_parent_n: int = 32) -> List[Tuple[str, str, int]]:
    """Select large, medium, and small feasible parent groups without duplicates."""
    counts = pd.Series(np.asarray(groups).astype(str)).value_counts()
    feasible = counts[counts >= min_parent_n]
    if len(feasible) == 0:
        return []
    ordered = list(feasible.items())  # value_counts is descending
    if len(ordered) == 1:
        picks = [("large", ordered[0])]
    elif len(ordered) == 2:
        picks = [("large", ordered[0]), ("small", ordered[-1])]
    else:
        picks = [
            ("large", ordered[0]),
            ("medium", ordered[len(ordered) // 2]),
            ("small", ordered[-1]),
        ]
    return [(role, str(group), int(n)) for role, (group, n) in picks]


def denominator_counts(y: np.ndarray, groups: Sequence, base: str) -> Dict[str, int]:
    """Return audit denominators for every named group, including zero denominators."""
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups).astype(str)
    out: Dict[str, int] = {}
    for g in pd.unique(groups):
        gm = groups == g
        if base == "ppr":
            denom = gm
        elif base == "tpr":
            denom = gm & (y == 1)
        elif base == "fpr":
            denom = gm & (y == 0)
        else:
            raise ValueError(base)
        out[str(g)] = int(np.sum(denom))
    return out


# -----------------------------------------------------------------------------
# Experiment 3: controlled refinement on real benchmarks
# -----------------------------------------------------------------------------


def experiment3(
    outdir: Path,
    datasets: Sequence[str],
    models: Sequence[str],
    repeats: int = 100,
    seed: int = RANDOM_SEED,
    acs_states: Sequence[str] = ("CA",),
    max_rows: Optional[int] = None,
) -> None:
    outdir = ensure_dir(outdir / "experiment3")
    rng = np.random.default_rng(seed)
    levels = [1, 2, 4, 8, 16, 32]
    bases = ["ppr", "tpr", "fpr"]

    result_rows = []
    perf_rows = []

    for dname in datasets:
        data = get_dataset(dname, acs_states=acs_states)
        X, y, demo = data.X.copy(), data.y.copy(), data.demographics.copy()

        if max_rows is not None and len(X) > max_rows:
            idx, _ = train_test_split(
                np.arange(len(X)), train_size=max_rows,
                random_state=seed, stratify=np.asarray(y),
            )
            X = X.iloc[idx].copy()
            y = y.iloc[idx].copy()
            demo = demo.iloc[idx].copy()

        idx = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            idx, test_size=0.25, random_state=seed, stratify=np.asarray(y)
        )
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = np.asarray(y.iloc[train_idx]).astype(int)
        y_test = np.asarray(y.iloc[test_idx]).astype(int)
        demo_test = demo.iloc[test_idx].reset_index(drop=True)

        primary = data.primary_group
        original_groups = demo_test[primary].astype(str).to_numpy()
        targets = select_refinement_targets(original_groups, min_parent_n=max(levels))
        if not targets:
            print(f"[Experiment 3] {data.name}: no group has at least {max(levels)} test observations; skipping.")
            continue

        for model_name in models:
            print(f"[Experiment 3] {data.name} / {model_name}")
            model = build_model(model_name, X_train, seed=seed)
            model.fit(X_train, y_train)
            yhat = model.predict(X_test).astype(int)
            if hasattr(model, "predict_proba"):
                try:
                    yprob = model.predict_proba(X_test)[:, 1]
                except Exception:
                    yprob = None
            else:
                yprob = None

            pm = predictive_metrics(y_test, yhat, yprob)
            perf_rows.append({"dataset": data.name, "model": model_name, **pm})

            for base in bases:
                base_metrics, _ = audit_metrics(y_test, yhat, original_groups, base)
                parent_denoms = denominator_counts(y_test, original_groups, base)

                for target_role, target_group, parent_group_n in targets:
                    parent_denom = int(parent_denoms.get(target_group, 0))
                    if parent_denom == 0:
                        continue

                    for m in levels:
                        if parent_group_n < m:
                            continue
                        nrep = 1 if m == 1 else repeats
                        for rep in range(nrep):
                            refined = random_refine_labels(
                                original_groups, target_group, m, rng
                            )
                            mm, tab = audit_metrics(y_test, yhat, refined, base)
                            denom_map = denominator_counts(y_test, refined, base)
                            if m == 1:
                                child_names = [target_group]
                            else:
                                prefix = f"{target_group}::r"
                                child_names = [g for g in pd.unique(refined) if str(g).startswith(prefix)]
                            child_denoms = np.asarray([denom_map.get(str(g), 0) for g in child_names], dtype=float)
                            child_min = int(np.min(child_denoms)) if len(child_denoms) else 0
                            child_mean = float(np.mean(child_denoms)) if len(child_denoms) else np.nan
                            child_zero = int(np.sum(child_denoms == 0)) if len(child_denoms) else 0

                            for metric, value in mm.items():
                                b = base_metrics[metric]
                                result_rows.append({
                                    "dataset": data.name,
                                    "model": model_name,
                                    "base_quantity": base,
                                    "protected_attribute": primary,
                                    "target_size_class": target_role,
                                    "refined_group": target_group,
                                    "parent_group_n": parent_group_n,
                                    "parent_denominator": parent_denom,
                                    "m": m,
                                    "rep": rep,
                                    "metric": metric,
                                    "value": value,
                                    "baseline": b,
                                    "delta": value - b,
                                    "RI": abs(value - b),
                                    "ratio": value / b if abs(b) > 1e-15 else np.nan,
                                    "num_audited_groups": len(tab),
                                    "child_min_denominator": child_min,
                                    "child_mean_denominator": child_mean,
                                    "child_zero_denominators": child_zero,
                                })

    rdf = pd.DataFrame(result_rows)
    pdf = pd.DataFrame(perf_rows)
    rdf.to_csv(outdir / "experiment3_benchmark_refinement.csv", index=False)
    pdf.to_csv(outdir / "experiment3_predictive_performance.csv", index=False)

    if len(rdf):
        summary = (
            rdf.groupby([
                "dataset", "model", "base_quantity", "target_size_class",
                "refined_group", "m", "metric"
            ], as_index=False)
            .agg(
                mean_value=("value", "mean"),
                mean_RI=("RI", "mean"),
                sd_RI=("RI", "std"),
                mean_ratio=("ratio", "mean"),
                parent_group_n=("parent_group_n", "first"),
                parent_denominator=("parent_denominator", "first"),
                mean_child_min_denominator=("child_min_denominator", "mean"),
                mean_child_denominator=("child_mean_denominator", "mean"),
                mean_zero_children=("child_zero_denominators", "mean"),
                runs=("rep", "count"),
            )
        )
        summary["ci95_RI"] = 1.96 * summary.sd_RI / np.sqrt(summary.runs)
        summary.to_csv(outdir / "experiment3_summary.csv", index=False)

        # Main figure uses the large parent group for cross-dataset comparability.
        q = summary[
            (summary.model == "lr")
            & (summary.base_quantity == "ppr")
            & (summary.target_size_class == "large")
            & (summary.metric.isin(["max_min", "pairwise", "weighted_pairwise"]))
        ]
        if len(q):
            plt.figure(figsize=(8, 5))
            for (dataset, metric), z in q.groupby(["dataset", "metric"]):
                z = z.sort_values("m")
                plt.plot(z.m, z.mean_RI, marker="o", label=f"{dataset}: {metric}")
            plt.xscale("log", base=2)
            plt.xlabel("Refinement level m")
            plt.ylabel("Mean refinement instability")
            plt.title("Experiment 3: public benchmark refinement (large parent group)")
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(outdir / "fig_exp3_benchmark_RI.png", dpi=220)
            plt.close()

    print(f"Experiment 3 complete -> {outdir}")


# -----------------------------------------------------------------------------
# Experiment 4: natural intersectional refinement with matched nulls
# -----------------------------------------------------------------------------


def permute_fine_within_parent(
    fine: np.ndarray,
    parent: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    fine = np.asarray(fine).astype(str)
    parent = np.asarray(parent).astype(str)
    out = fine.copy()
    for p in pd.unique(parent):
        idx = np.where(parent == p)[0]
        out[idx] = rng.permutation(out[idx])
    return out


def audit_metrics_min_n(
    y: np.ndarray,
    yhat: np.ndarray,
    groups: Sequence,
    base: str,
    min_n: int = 0,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Compute audit metrics after excluding groups below a denominator threshold."""
    tab = audit_group_table(y, yhat, groups, base)
    if min_n > 0 and len(tab):
        tab = tab[tab["n"] >= min_n].copy()
        if len(tab):
            tab["weight"] = tab["n"] / tab["n"].sum()
    if len(tab) < 2:
        return {}, tab
    return meta_metrics(tab.value.to_numpy(), tab.weight.to_numpy()), tab


def experiment4(
    outdir: Path,
    models: Sequence[str] = ("lr",),
    permutations: int = 500,
    seed: int = RANDOM_SEED,
    acs_states: Sequence[str] = ("CA",),
    max_rows: Optional[int] = None,
    min_group_ns: Sequence[int] = (0, 30, 50),
) -> None:
    outdir = ensure_dir(outdir / "experiment4")
    rng = np.random.default_rng(seed)
    data = load_acs(states=acs_states)
    X, y, demo = data.X.copy(), data.y.copy(), data.demographics.copy()

    if max_rows is not None and len(X) > max_rows:
        idx, _ = train_test_split(
            np.arange(len(X)), train_size=max_rows,
            random_state=seed, stratify=np.asarray(y),
        )
        X = X.iloc[idx].copy()
        y = y.iloc[idx].copy()
        demo = demo.iloc[idx].copy()

    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.25, random_state=seed, stratify=np.asarray(y)
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = np.asarray(y.iloc[train_idx]).astype(int)
    y_test = np.asarray(y.iloc[test_idx]).astype(int)
    dtest = demo.iloc[test_idx].reset_index(drop=True)

    sex = dtest["sex"].astype(str)
    race = dtest["race"].astype(str)
    ageg = age_band(dtest["age"])

    partitions = {
        "sex": sex.to_numpy(),
        "race_sex": (race + "|" + sex).to_numpy(),
        "race_sex_age": (race + "|" + sex + "|" + ageg).to_numpy(),
    }
    transitions = [
        ("sex", "race_sex"),
        ("race_sex", "race_sex_age"),
    ]
    bases = ["ppr", "tpr", "fpr"]

    raw_rows = []
    summary_rows = []

    for model_name in models:
        print(f"[Experiment 4] ACS / {model_name}")
        model = build_model(model_name, X_train, seed=seed)
        model.fit(X_train, y_train)
        yhat = model.predict(X_test).astype(int)

        for coarse_name, fine_name in transitions:
            coarse = partitions[coarse_name]
            fine = partitions[fine_name]

            for base in bases:
                for min_group_n in min_group_ns:
                    coarse_mm, coarse_tab = audit_metrics_min_n(
                        y_test, yhat, coarse, base, min_n=min_group_n
                    )
                    fine_mm, fine_tab = audit_metrics_min_n(
                        y_test, yhat, fine, base, min_n=min_group_n
                    )
                    common_metrics = sorted(set(coarse_mm).intersection(fine_mm))
                    if not common_metrics:
                        continue

                    null_by_metric: Dict[str, List[float]] = {k: [] for k in common_metrics}
                    for b in range(permutations):
                        null_fine = permute_fine_within_parent(fine, coarse, rng)
                        null_mm, null_tab = audit_metrics_min_n(
                            y_test, yhat, null_fine, base, min_n=min_group_n
                        )
                        for metric in common_metrics:
                            if metric not in null_mm:
                                continue
                            delta_null = null_mm[metric] - coarse_mm[metric]
                            null_by_metric[metric].append(delta_null)
                            raw_rows.append({
                                "model": model_name,
                                "base_quantity": base,
                                "transition": f"{coarse_name}->{fine_name}",
                                "min_group_n": min_group_n,
                                "perm": b,
                                "metric": metric,
                                "delta_null": delta_null,
                                "null_groups_retained": len(null_tab),
                            })

                    for metric in common_metrics:
                        arr = np.asarray(null_by_metric[metric], dtype=float)
                        if len(arr) == 0:
                            continue
                        delta_real = fine_mm[metric] - coarse_mm[metric]
                        null_mean = float(np.mean(arr))
                        null_sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan
                        are = delta_real - null_mean
                        z = are / null_sd if null_sd > 0 else np.nan
                        lo, hi = np.quantile(arr, [0.025, 0.975])
                        summary_rows.append({
                            "model": model_name,
                            "base_quantity": base,
                            "transition": f"{coarse_name}->{fine_name}",
                            "min_group_n": min_group_n,
                            "metric": metric,
                            "coarse_value": coarse_mm[metric],
                            "fine_value": fine_mm[metric],
                            "delta_real": delta_real,
                            "null_mean": null_mean,
                            "null_sd": null_sd,
                            "null_q025": lo,
                            "null_q975": hi,
                            "ARE": are,
                            "z_from_null": z,
                            "coarse_groups": len(coarse_tab),
                            "fine_groups": len(fine_tab),
                            "fine_min_denominator": int(fine_tab.n.min()) if len(fine_tab) else 0,
                            "fine_median_denominator": float(fine_tab.n.median()) if len(fine_tab) else np.nan,
                        })

    raw = pd.DataFrame(raw_rows)
    summary = pd.DataFrame(summary_rows)
    raw.to_csv(outdir / "experiment4_null_permutations.csv", index=False)
    summary.to_csv(outdir / "experiment4_summary.csv", index=False)

    if len(summary):
        # Primary visualization remains unfiltered; n>=30 and n>=50 are sensitivity analyses.
        q = summary[
            (summary.model == models[0])
            & (summary.base_quantity == "ppr")
            & (summary.min_group_n == 0)
            & (summary.metric.isin(["max_min", "pairwise", "weighted_pairwise"]))
        ].copy()
        if len(q):
            x = np.arange(len(q))
            yv = q.delta_real.to_numpy()
            plt.figure(figsize=(9, 5))
            plt.scatter(x, yv, label="real refinement")
            for i, row in q.reset_index(drop=True).iterrows():
                plt.vlines(i, row.null_q025, row.null_q975, alpha=0.6)
            labels = [f"{r.transition}\n{r.metric}" for r in q.itertuples()]
            plt.xticks(x, labels, rotation=30, ha="right")
            plt.ylabel("Change from coarse partition")
            plt.title("Experiment 4: real refinement vs matched-null interval")
            plt.tight_layout()
            plt.savefig(outdir / "fig_exp4_real_vs_null.png", dpi=220)
            plt.close()

        # Sensitivity figure for minimum denominator thresholds.
        q2 = summary[
            (summary.model == models[0])
            & (summary.base_quantity == "ppr")
            & (summary.metric.isin(["max_min", "pairwise", "weighted_pairwise"]))
        ].copy()
        if len(q2):
            plt.figure(figsize=(9, 5))
            for (transition, metric), z in q2.groupby(["transition", "metric"]):
                z = z.sort_values("min_group_n")
                plt.plot(
                    z.min_group_n, z.ARE, marker="o",
                    label=f"{transition}: {metric}"
                )
            plt.xlabel("Minimum subgroup denominator threshold")
            plt.ylabel("Adjusted refinement effect (ARE)")
            plt.title("Experiment 4: denominator-threshold sensitivity")
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(outdir / "fig_exp4_threshold_sensitivity.png", dpi=220)
            plt.close()

    print(f"Experiment 4 complete -> {outdir}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Demographic Refinement Invariance experiments")
    p.add_argument("--experiment", choices=["1", "2", "3", "4", "all"], default="all")
    p.add_argument("--out", type=Path, default=Path("results_refinement"))
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--reps2", type=int, default=500, help="Experiment 2 simulation replicates")
    p.add_argument("--reps3", type=int, default=100, help="Experiment 3 random refinements per condition")
    p.add_argument("--perms4", type=int, default=500, help="Experiment 4 matched-null permutations")
    p.add_argument(
        "--datasets", nargs="+", default=["acs", "adult", "compas", "german"],
        help="Experiment 3 datasets"
    )
    p.add_argument(
        "--models", nargs="+", default=["lr", "rf", "hgb"],
        choices=["lr", "rf", "hgb", "mlp"],
    )
    p.add_argument("--acs-states", nargs="+", default=["CA"])
    p.add_argument(
        "--max-rows", type=int, default=None,
        help="Optional cap for fast/debug runs; leave unset for full data."
    )
    return p.parse_args()


def main():
    args = parse_args()
    out = ensure_dir(args.out)
    with open(out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args) | {"out": str(args.out)}, f, indent=2, default=str)

    if args.experiment in {"1", "all"}:
        experiment1(out)
    if args.experiment in {"2", "all"}:
        experiment2(out, reps=args.reps2, seed=args.seed)
    if args.experiment in {"3", "all"}:
        experiment3(
            out, datasets=args.datasets, models=args.models,
            repeats=args.reps3, seed=args.seed,
            acs_states=args.acs_states, max_rows=args.max_rows,
        )
    if args.experiment in {"4", "all"}:
        experiment4(
            out, models=args.models, permutations=args.perms4,
            seed=args.seed, acs_states=args.acs_states,
            max_rows=args.max_rows,
        )


if __name__ == "__main__":
    main()
