
#!/usr/bin/env python3
"""
validate_results.py

Validation of the FAccT Demographic Refinement Invariance study.

Checks:
    1. Experiment 1: structural invariance and baseline consistency
    2. Experiment 2: sampling-refinement quantities
    3. Experiment 3: benchmark refinement outputs
    4. Experiment 4: matched-null effects and permutation diagnostics
    5. Audit-verdict and model-ranking reconciliation
    6. Ranking-margin robustness and reproducibility metadata

Outputs:
    results/validation/validation_report.csv
    results/validation/validation_summary.txt
    results/validation/validation_summary.json

Usage:
    python validate_results.py

    python validate_results.py --results results

    python validate_results.py --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


FLOAT_TOL = 1e-9
TIE_TOL = 1e-12

ROOT = Path(__file__).resolve().parent

EXPECTED_LEVELS = {1, 2, 4, 8, 16, 32}
EXPECTED_DATASETS = {
    "acs", "acs_income",
    "adult", "compas",
    "german", "german_credit",
}
EXPECTED_MODELS = {"lr", "rf", "hgb"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate FAccT experiment outputs."
    )

    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "validation",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return failure status for missing optional files.",
    )

    return parser.parse_args()


class Validator:
    def __init__(self):
        self.records = []

    def add(
        self,
        component,
        check,
        passed,
        details="",
        severity="error",
    ):
        status = "PASS" if passed else (
            "WARN" if severity == "warning" else "FAIL"
        )

        record = {
            "component": component,
            "check": check,
            "status": status,
            "details": str(details),
        }

        self.records.append(record)

        print(
            f"[{status}] "
            f"{component}: {check}"
            + (f" | {details}" if details else "")
        )

    def required_columns(
        self,
        df,
        columns,
        component,
    ):
        missing = sorted(set(columns) - set(df.columns))

        self.add(
            component,
            "Required columns",
            len(missing) == 0,
            f"Missing: {missing}" if missing else "All present",
        )

        return not missing

    def report(self, output):
        output.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(
            self.records,
            columns=[
                "component", "check", "status", "details"
            ],
        )

        df.to_csv(
            output / "validation_report.csv",
            index=False,
        )

        counts = df.status.value_counts().to_dict()

        summary = {
            "total_checks": len(df),
            "passed": counts.get("PASS", 0),
            "failed": counts.get("FAIL", 0),
            "warnings": counts.get("WARN", 0),
            "checks": self.records,
        }

        with open(
            output / "validation_summary.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=2)

        lines = [
            "FAccT Demographic Refinement Validation",
            "=" * 45,
            f"Total checks: {summary['total_checks']}",
            f"Passed:       {summary['passed']}",
            f"Failed:       {summary['failed']}",
            f"Warnings:     {summary['warnings']}",
            "",
        ]

        for record in self.records:
            lines.append(
                f"[{record['status']}] "
                f"{record['component']} - "
                f"{record['check']}: "
                f"{record['details']}"
            )

        (output / "validation_summary.txt").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        return summary


def find_csv(root, filename, directories):
    """
    Check documented output directories.
    Never silently use a CSV with an unrelated name.
    """
    for directory in directories:
        path = root / directory / filename
        if path.exists():
            return path

    return None


def load_optional(root, filename, directories, validator):
    path = find_csv(root, filename, directories)

    if path is None:
        validator.add(
            filename,
            "File availability",
            False,
            "Not found",
            severity="warning",
        )
        return None

    try:
        df = pd.read_csv(path, low_memory=False)

        validator.add(
            filename,
            "File availability",
            True,
            f"{len(df):,} rows; {path}",
        )

        return df

    except Exception as exc:
        validator.add(
            filename,
            "CSV readability",
            False,
            str(exc),
        )
        return None


def numeric_close(a, b, atol=FLOAT_TOL):
    return bool(
        np.allclose(
            np.asarray(a, dtype=float),
            np.asarray(b, dtype=float),
            atol=atol,
            rtol=0,
            equal_nan=True,
        )
    )


def check_finite(df, columns, validator, component):
    for column in columns:
        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column], errors="coerce"
        )

        valid = np.isfinite(values)

        validator.add(
            component,
            f"Finite values: {column}",
            bool(valid.all()),
            f"{valid.sum():,}/{len(valid):,} finite",
            severity="warning",
        )


def validate_experiment1(df, v):
    component = "Experiment 1"

    required = [
        "profile", "refined_group", "m",
        "metric", "value", "baseline",
        "delta", "RI",
    ]

    if not v.required_columns(df, required, component):
        return

    v.add(
        component,
        "Refinement levels",
        set(df.m.unique()).issubset(
            {1, 2, 4, 8, 16, 32, 64}
        ),
        f"Observed: {sorted(df.m.unique())}",
    )

    v.add(
        component,
        "Signed refinement shift",
        numeric_close(
            df["delta"],
            df["value"] - df["baseline"],
        ),
    )

    v.add(
        component,
        "Absolute refinement index",
        numeric_close(
            df["RI"],
            df["delta"].abs(),
        ),
    )

    baseline = df[df.m == 1]

    v.add(
        component,
        "Baseline refinement identity",
        numeric_close(
            baseline["value"],
            baseline["baseline"],
        ),
    )

    # These metrics must remain invariant under exact
    # null refinement with conserved audit weights.
    invariant = {
        "max_min",
        "weighted_mad",
        "weighted_variance",
        "weighted_pairwise",
        "cvar_50_conserved",
    }

    subset = df[df.metric.isin(invariant)]

    v.add(
        component,
        "Exact structural DRI",
        bool(
            np.all(
                np.abs(subset["delta"]) <= FLOAT_TOL
            )
        ) if len(subset) else False,
        f"Checked {len(subset):,} invariant metric rows",
    )


def validate_experiment2(df, v):
    component = "Experiment 2"

    required = ["m", "metric"]

    if not v.required_columns(df, required, component):
        return

    v.add(
        component,
        "Valid refinement levels",
        set(df.m.unique()).issubset(EXPECTED_LEVELS),
        f"Observed: {sorted(df.m.unique())}",
    )

    v.add(
        component,
        "Nonempty output",
        len(df) > 0,
        f"{len(df):,} rows",
    )

    for column in [
        "mean_ASRI",
        "ASRI",
        "expected_child_n",
    ]:
        if column in df.columns:
            values = pd.to_numeric(
                df[column], errors="coerce"
            )

            v.add(
                component,
                f"Nonnegative {column}",
                bool((values.dropna() >= 0).all()),
            )


def validate_experiment3(df, v):
    component = "Experiment 3"

    required = [
        "dataset", "model", "metric", "m"
    ]

    if not v.required_columns(df, required, component):
        return

    v.add(
        component,
        "Valid refinement levels",
        set(df.m.unique()).issubset(EXPECTED_LEVELS),
    )

    v.add(
        component,
        "Expected datasets",
        set(df.dataset.unique()).issubset(
            EXPECTED_DATASETS
        ),
        f"Observed: {sorted(df.dataset.unique())}",
    )

    v.add(
        component,
        "Expected model families",
        set(df.model.unique()).issubset(
            EXPECTED_MODELS
        ),
        f"Observed: {sorted(df.model.unique())}",
    )

    v.add(
        component,
        "Nonempty output",
        len(df) > 0,
        f"{len(df):,} rows",
    )


def validate_experiment4(df, v):
    component = "Experiment 4 matched population"

    required = [
        "base_quantity",
        "min_group_n",
        "metric",
        "coarse_value",
        "fine_value",
        "delta_real",
        "null_mean",
        "null_q025",
        "null_q975",
        "ARE",
        "n_test_total",
        "n_retained",
        "fraction_retained",
        "valid_permutations",
        "invalid_permutations",
    ]

    if not v.required_columns(df, required, component):
        return

    v.add(
        component,
        "Observed refinement shift",
        numeric_close(
            df["delta_real"],
            df["fine_value"] - df["coarse_value"],
        ),
    )

    v.add(
        component,
        "Adjusted refinement effect",
        numeric_close(
            df["ARE"],
            df["delta_real"] - df["null_mean"],
        ),
    )

    v.add(
        component,
        "Ordered null reference intervals",
        bool(
            (df["null_q025"] <= df["null_q975"]).all()
        ),
    )

    valid = (
        (df["n_retained"] >= 0)
        & (df["n_retained"] <= df["n_test_total"])
    )

    v.add(
        component,
        "Population retention bounds",
        bool(valid.all()),
    )

    v.add(
        component,
        "Population fraction consistency",
        numeric_close(
            df["fraction_retained"],
            df["n_retained"] / df["n_test_total"],
        ),
    )

    v.add(
        component,
        "Positive valid permutation count",
        bool((df.valid_permutations > 0).all()),
    )

    total_perms = (
        df.valid_permutations
        + df.invalid_permutations
    )

    v.add(
        component,
        "Permutation totals consistent",
        bool(total_perms.nunique() == 1),
        f"Totals: {sorted(total_perms.unique())}",
    )

    v.add(
        component,
        "Expected denominators",
        set(df.min_group_n.unique()).issubset(
            {0, 30, 50}
        ),
    )

    v.add(
        component,
        "Expected audit quantities",
        set(df.base_quantity.unique()).issubset(
            {"ppr", "tpr", "fpr"}
        ),
    )


def parse_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    values = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    accepted = {"true", "false", "1", "0"}

    bad = values.dropna()[
        ~values.dropna().isin(accepted)
    ]

    if not bad.empty:
        raise ValueError(
            f"Unexpected Boolean values: {bad.unique()}"
        )

    return values.isin({"true", "1"}).fillna(False)


def validate_verdicts(detail, summary, v):
    component = "Fairness verdict reversals"
    boundary_tolerance = 1e-12

    required_detail = [
        "dataset", "base_quantity", "metric",
        "m", "tolerance", "coarse_value",
        "value", "reversal",
    ]

    required_summary = [
        "dataset", "base_quantity", "metric",
        "m", "tolerance", "reversals",
        "comparisons", "reversal_share",
    ]

    if not v.required_columns(detail, required_detail, component):
        return

    if not v.required_columns(summary, required_summary, component):
        return

    # ---------------------------------------------------------
    # 1. Validate numerical inputs
    # ---------------------------------------------------------
    coarse = pd.to_numeric(detail["coarse_value"], errors="coerce")
    refined = pd.to_numeric(detail["value"], errors="coerce")
    tolerance = pd.to_numeric(detail["tolerance"], errors="coerce")

    valid = (
        np.isfinite(coarse)
        & np.isfinite(refined)
        & np.isfinite(tolerance)
    )

    v.add(
        component,
        "Finite verdict inputs",
        bool(valid.all()),
        f"Valid: {int(valid.sum()):,}/{len(detail):,}",
    )

    if not valid.all():
        return

    # ---------------------------------------------------------
    # 2. Recompute verdicts using the ORIGINAL decision rule
    # ---------------------------------------------------------
    coarse_pass = coarse <= tolerance
    refined_pass = refined <= tolerance

    computed = coarse_pass != refined_pass
    stored = parse_bool(detail["reversal"])

    mismatch = computed != stored

    # ---------------------------------------------------------
    # 3. Identify numerical-boundary disagreements
    # ---------------------------------------------------------
    coarse_gap = (coarse - tolerance).abs()
    refined_gap = (refined - tolerance).abs()

    minimum_gap = pd.concat(
        [coarse_gap, refined_gap],
        axis=1
    ).min(axis=1)

    near_boundary = minimum_gap <= boundary_tolerance

    boundary_mismatches = mismatch & near_boundary
    genuine_mismatches = mismatch & ~near_boundary

    total_mismatches = int(mismatch.sum())
    boundary_count = int(boundary_mismatches.sum())
    genuine_count = int(genuine_mismatches.sum())

    # ---------------------------------------------------------
    # 4. Save diagnostic records
    # ---------------------------------------------------------
    diagnostic_path = (
        Path("results")
        / "validation"
        / "verdict_disagreements.csv"
    )

    diagnostic_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if total_mismatches > 0:
        diagnostics = detail.loc[mismatch].copy()

        diagnostics["coarse_threshold_gap"] = (
            coarse_gap.loc[mismatch]
        )

        diagnostics["refined_threshold_gap"] = (
            refined_gap.loc[mismatch]
        )

        diagnostics["minimum_threshold_gap"] = (
            minimum_gap.loc[mismatch]
        )

        diagnostics["boundary_sensitive"] = (
            near_boundary.loc[mismatch]
        )

        diagnostics["recomputed_reversal"] = (
            computed.loc[mismatch]
        )

        diagnostics.to_csv(
            diagnostic_path,
            index=False
        )

        print("\nVerdict disagreement diagnostics:")
        print(f"Total disagreements: {total_mismatches}")
        print(f"Boundary disagreements: {boundary_count}")
        print(f"Genuine disagreements: {genuine_count}")

        print(
            "Maximum minimum threshold gap:",
            diagnostics["minimum_threshold_gap"].max()
        )

        print(
            "\nSample disagreement records:"
        )

        print(
            diagnostics[
                [
                    "tolerance",
                    "coarse_value",
                    "value",
                    "minimum_threshold_gap",
                    "reversal",
                    "recomputed_reversal",
                    "boundary_sensitive",
                ]
            ].head(20).to_string(index=False)
        )

    else:
        # Avoid leaving a stale diagnostic file from a prior run.
        if diagnostic_path.exists():
            diagnostic_path.unlink()

    # ---------------------------------------------------------
    # 5. Validate verdict logic outside numerical boundaries
    # ---------------------------------------------------------
    v.add(
        component,
        "Verdict logic away from numerical boundaries",
        genuine_count == 0,
        (
            f"Genuine disagreements: {genuine_count}; "
            f"boundary disagreements: {boundary_count}"
        ),
    )

    # Boundary differences are reported, not hidden.
    v.add(
        component,
        "Numerical-boundary disagreements",
        boundary_count == 0,
        (
            f"{boundary_count} boundary-sensitive comparisons "
            f"within tolerance {boundary_tolerance:.0e}"
        ),
        severity="warning",
    )

    # ---------------------------------------------------------
    # 6. Validate detail/summary counts
    # ---------------------------------------------------------
    detail_count = int(stored.sum())
    summary_count = int(summary["reversals"].sum())

    v.add(
        component,
        "Detail/summary reversal counts",
        detail_count == summary_count,
        (
            f"Detail={detail_count:,}, "
            f"summary={summary_count:,}"
        ),
    )

    detail_comparisons = len(detail)
    summary_comparisons = int(summary["comparisons"].sum())

    v.add(
        component,
        "Detail/summary comparison counts",
        detail_comparisons == summary_comparisons,
        (
            f"Detail={detail_comparisons:,}, "
            f"summary={summary_comparisons:,}"
        ),
    )

    # ---------------------------------------------------------
    # 7. Validate summary reversal rates
    # ---------------------------------------------------------
    comparisons = pd.to_numeric(
        summary["comparisons"],
        errors="coerce"
    )

    reversals = pd.to_numeric(
        summary["reversals"],
        errors="coerce"
    )

    valid_summary = (
        np.isfinite(comparisons)
        & np.isfinite(reversals)
        & (comparisons > 0)
        & (reversals >= 0)
        & (reversals <= comparisons)
    )

    v.add(
        component,
        "Valid summary counts",
        bool(valid_summary.all()),
        (
            f"Valid rows: {int(valid_summary.sum()):,}"
            f"/{len(summary):,}"
        ),
    )

    if valid_summary.all():
        computed_share = reversals / comparisons

        v.add(
            component,
            "Summary reversal rates",
            numeric_close(
                summary["reversal_share"],
                computed_share
            ),
        )

def validate_rankings(detail, summary, v):
    component = "Model-ranking reversals"

    required_detail = [
        "coarse_difference",
        "refined_difference",
        "ranking_reversal",
        "tie_involved",
    ]

    required_summary = [
        "reversals", "comparisons",
        "reversal_share", "ties",
    ]

    if not v.required_columns(
        detail, required_detail, component
    ):
        return

    if not v.required_columns(
        summary, required_summary, component
    ):
        return

    coarse = pd.to_numeric(
        detail.coarse_difference, errors="coerce"
    ).to_numpy()

    refined = pd.to_numeric(
        detail.refined_difference, errors="coerce"
    ).to_numpy()

    finite = np.isfinite(coarse) & np.isfinite(refined)

    v.add(
        component,
        "Finite score differences",
        bool(finite.all()),
        f"{finite.sum():,}/{len(finite):,}",
    )

    if not finite.all():
        return

    original_sign = np.where(
        np.abs(coarse) <= TIE_TOL,
        0,
        np.sign(coarse),
    )

    refined_sign = np.where(
        np.abs(refined) <= TIE_TOL,
        0,
        np.sign(refined),
    )

    recomputed = (
        original_sign * refined_sign == -1
    )

    strict = (
        ((coarse > 0) & (refined < 0))
        | ((coarse < 0) & (refined > 0))
    )

    stored = parse_bool(
        detail.ranking_reversal
    ).to_numpy()

    stored_ties = parse_bool(
        detail.tie_involved
    ).to_numpy()

    recomputed_ties = (
        (original_sign == 0)
        | (refined_sign == 0)
    )

    v.add(
        component,
        "Original ranking logic",
        bool(np.array_equal(stored, recomputed)),
        f"Disagreements: "
        f"{np.count_nonzero(stored != recomputed)}",
    )

    v.add(
        component,
        "Tie detection consistency",
        bool(np.array_equal(
            stored_ties, recomputed_ties
        )),
    )

    v.add(
        component,
        "Detail/summary reversal counts",
        int(stored.sum()) == int(summary.reversals.sum()),
        f"Detail={stored.sum()}, "
        f"summary={summary.reversals.sum()}",
    )

    v.add(
        component,
        "Detail/summary comparison counts",
        len(detail) == int(summary.comparisons.sum()),
    )

    v.add(
        component,
        "Detail/summary tie counts",
        int(stored_ties.sum()) == int(summary.ties.sum()),
    )

    v.add(
        component,
        "Strict sign versus tolerance-based reversals",
        bool(np.array_equal(strict, recomputed)),
        f"Strict={strict.sum()}, "
        f"tolerance-based={recomputed.sum()}",
        severity="warning",
    )

    v.add(
        component,
        "Summary reversal rates",
        numeric_close(
            summary.reversal_share,
            summary.reversals / summary.comparisons,
        ),
    )


def validate_margin_table(margins, ranking_detail, v):
    component = "Ranking margin sensitivity"

    required = [
        "margin",
        "all_valid_comparisons",
        "eligible_comparisons",
        "reversals",
        "conditional_rate_pct",
        "all_comparison_rate_pct",
    ]

    if not v.required_columns(
        margins, required, component
    ):
        return

    coarse = pd.to_numeric(
        ranking_detail.coarse_difference,
        errors="coerce",
    ).to_numpy()

    refined = pd.to_numeric(
        ranking_detail.refined_difference,
        errors="coerce",
    ).to_numpy()

    valid = np.isfinite(coarse) & np.isfinite(refined)
    coarse = coarse[valid]
    refined = refined[valid]

    strict = (
        ((coarse > 0) & (refined < 0))
        | ((coarse < 0) & (refined > 0))
    )

    gap = np.minimum(
        np.abs(coarse),
        np.abs(refined),
    )

    all_correct = True
    errors = []

    for row in margins.itertuples():
        eligible = gap > row.margin
        expected_eligible = int(eligible.sum())
        expected_reversals = int(
            (strict & eligible).sum()
        )

        correct = (
            row.all_valid_comparisons == len(coarse)
            and row.eligible_comparisons == expected_eligible
            and row.reversals == expected_reversals
        )

        if not correct:
            all_correct = False
            errors.append(row.margin)

    v.add(
        component,
        "Ranking margin counts",
        all_correct,
        (
            "All margins match"
            if all_correct
            else f"Mismatched margins: {errors}"
        ),
    )


def main():
    args = parse_args()
    results = args.results
    output = args.output

    v = Validator()

    exp1 = load_optional(
        results,
        "experiment1_structural.csv",
        ["experiment1"],
        v,
    )

    exp2 = load_optional(
        results,
        "experiment2_summary.csv",
        ["experiment2"],
        v,
    )

    exp3 = load_optional(
        results,
        "experiment3_summary.csv",
        ["experiment3"],
        v,
    )

    exp4 = load_optional(
        results,
        "experiment4_matched_population_summary.csv",
        ["matched_population", "experiment4"],
        v,
    )

    verdict_detail = load_optional(
        results,
        "verdict_detail.csv",
        ["decisions"],
        v,
    )

    verdict_summary = load_optional(
        results,
        "verdict_summary.csv",
        ["decisions"],
        v,
    )

    ranking_detail = load_optional(
        results,
        "ranking_detail.csv",
        ["decisions"],
        v,
    )

    ranking_summary = load_optional(
        results,
        "ranking_summary.csv",
        ["decisions"],
        v,
    )

    margin_summary = load_optional(
        results,
        "ranking_margin_summary.csv",
        ["validation"],
        v,
    )

    if exp1 is not None:
        validate_experiment1(exp1, v)

    if exp2 is not None:
        validate_experiment2(exp2, v)

    if exp3 is not None:
        validate_experiment3(exp3, v)

    if exp4 is not None:
        validate_experiment4(exp4, v)

    if (
        verdict_detail is not None
        and verdict_summary is not None
    ):
        validate_verdicts(
            verdict_detail,
            verdict_summary,
            v,
        )

    if (
        ranking_detail is not None
        and ranking_summary is not None
    ):
        validate_rankings(
            ranking_detail,
            ranking_summary,
            v,
        )

    if (
        margin_summary is not None
        and ranking_detail is not None
    ):
        validate_margin_table(
            margin_summary,
            ranking_detail,
            v,
        )

    report = v.report(output)

    print("\n" + "=" * 45)
    print("Validation summary")
    print("=" * 45)
    print(f"Passed:   {report['passed']}")
    print(f"Failed:   {report['failed']}")
    print(f"Warnings: {report['warnings']}")
    print(f"Reports saved to: {output}")

    if report["failed"] > 0:
        sys.exit(1)

    if args.strict and report["warnings"] > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
