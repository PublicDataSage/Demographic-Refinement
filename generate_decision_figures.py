
#!/usr/bin/env python3
"""Generate FAccT decision-level figures and ranking robustness outputs.

Required inputs (from decision_analyses.py):
    verdict_summary.csv
    ranking_summary.csv
    ranking_detail.csv

Outputs:
    figures/decisions/fig_decision_reversals.png
    figures/decisions/fig_decision_reversals.pdf

    results/validation/ranking_margin_summary.csv
    results/validation/ranking_margin_by_metric.csv
    results/validation/ranking_flag_disagreements.csv
    results/validation/ranking_validation_summary.csv

The stored ranking reversal flag uses a numerical tie tolerance of 1e-12.
We preserve that flag for reproducing the original results, while
independently checking strict sign-based reversals.

All percentages are descriptive summaries of the evaluated configurations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


TIE_TOLERANCE = 1e-12
DEFAULT_MARGINS = [0.0, 0.001, 0.005, 0.01]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate decision-level figures and ranking audits."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/decisions"),
        help="Directory containing decision_analyses.py CSV outputs.",
    )

    parser.add_argument(
        "--figures",
        type=Path,
        default=Path("figures/decisions"),
        help="Output directory for figures.",
    )

    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("results/validation"),
        help="Output directory for validation tables.",
    )

    parser.add_argument(
        "--margins",
        type=float,
        nargs="+",
        default=DEFAULT_MARGINS,
        help="Minimum absolute score gaps on both partitions.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    return parser.parse_args()


def require_columns(df, columns, name):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )


def load_inputs(directory: Path):
    paths = {
        "verdict": directory / "verdict_summary.csv",
        "ranking": directory / "ranking_summary.csv",
        "detail": directory / "ranking_detail.csv",
    }

    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {label} input: {path}\n"
                "Run decision_analyses.py first, or specify --input."
            )

    verdict = pd.read_csv(paths["verdict"])
    ranking = pd.read_csv(paths["ranking"])
    detail = pd.read_csv(paths["detail"])

    require_columns(
        verdict,
        ["tolerance", "reversals", "comparisons"],
        "verdict_summary.csv",
    )

    require_columns(
        ranking,
        ["m", "reversals", "comparisons"],
        "ranking_summary.csv",
    )

    require_columns(
        detail,
        [
            "metric",
            "coarse_difference",
            "refined_difference",
            "ranking_reversal",
            "tie_involved",
        ],
        "ranking_detail.csv",
    )

    return verdict, ranking, detail


def aggregate_rates(frame, group_column):
    summary = (
        frame.groupby(group_column, as_index=False, dropna=False)
        [["reversals", "comparisons"]]
        .sum()
        .sort_values(group_column)
    )

    summary["rate"] = np.where(
        summary["comparisons"] > 0,
        100.0 * summary["reversals"] / summary["comparisons"],
        np.nan,
    )

    return summary


def configure_plot():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def generate_decision_figure(
    verdict,
    ranking,
    outdir: Path,
    dpi: int,
):
    outdir.mkdir(parents=True, exist_ok=True)

    configure_plot()

    v = aggregate_rates(verdict, "tolerance")
    r = aggregate_rates(ranking, "m")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.6, 4.5),
        constrained_layout=True,
    )

    ax = axes[0]
    ax.plot(
        v["tolerance"],
        v["rate"],
        marker="o",
        markersize=6,
        linewidth=2,
    )

    ax.set_xlabel(r"Audit tolerance $\tau$")
    ax.set_ylabel("Verdict reversal rate (%)")
    ax.set_xticks(v["tolerance"])
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    axes[0].text(
        0.02,
        0.97,
        "(a)",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    ax = axes[1]
    ax.plot(
        r["m"],
        r["rate"],
        marker="s",
        markersize=6,
        linewidth=2,
    )

    ax.set_xlabel(r"Refinement factor $m$")
    ax.set_ylabel("Ranking reversal rate (%)")
    ax.set_xticks(r["m"])
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    axes[1].text(
        0.02,
        0.97,
        "(b)",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    png = outdir / "fig_decision_reversals.png"
    pdf = outdir / "fig_decision_reversals.pdf"

    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)

    v.to_csv(outdir / "verdict_by_tolerance.csv", index=False)
    r.to_csv(outdir / "ranking_by_refinement.csv", index=False)

    print(f"Created: {png}")
    print(f"Created: {pdf}")

    return v, r


def parse_boolean(series: pd.Series):
    """Handle Boolean columns loaded from CSV as bool or strings."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = series.astype("string").str.strip().str.lower()

    allowed = {"true", "false", "1", "0"}
    unexpected = normalized.dropna()[~normalized.dropna().isin(allowed)]

    if not unexpected.empty:
        raise ValueError(
            "Unexpected Boolean values: "
            f"{unexpected.unique().tolist()[:10]}"
        )

    return normalized.isin(["true", "1"]).fillna(False).astype(bool)


def ranking_margin_analysis(
    detail,
    outdir: Path,
    margins,
):
    outdir.mkdir(parents=True, exist_ok=True)

    frame = detail.copy()

    for col in ["coarse_difference", "refined_difference"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    original_rows = len(frame)

    valid_mask = (
        np.isfinite(frame["coarse_difference"])
        & np.isfinite(frame["refined_difference"])
    )

    frame = frame.loc[valid_mask].copy()

    if frame.empty:
        raise ValueError("No valid ranking comparisons found.")

    coarse = frame["coarse_difference"].to_numpy(dtype=float)
    refined = frame["refined_difference"].to_numpy(dtype=float)

    # Strict mathematical sign change, with exact zero excluded.
    strict = (
        ((coarse > 0) & (refined < 0))
        | ((coarse < 0) & (refined > 0))
    )

    # Original decision_analyses.py convention:
    # scores within 1e-12 of zero are numerical ties.
    coarse_sign = np.where(
        np.abs(coarse) <= TIE_TOLERANCE,
        0,
        np.sign(coarse),
    )

    refined_sign = np.where(
        np.abs(refined) <= TIE_TOLERANCE,
        0,
        np.sign(refined),
    )

    tolerance_based = coarse_sign * refined_sign == -1

    frame["strict_reversal"] = strict
    frame["tolerance_based_reversal"] = tolerance_based

    frame["stored_reversal"] = parse_boolean(
        frame["ranking_reversal"]
    )

    frame["stored_tie"] = parse_boolean(
        frame["tie_involved"]
    )

    frame["recomputed_tie"] = (
        (coarse_sign == 0) | (refined_sign == 0)
    )

    frame["minimum_gap"] = np.minimum(
        np.abs(coarse),
        np.abs(refined),
    )

    frame["stored_vs_tolerance_disagreement"] = (
        frame["stored_reversal"]
        != frame["tolerance_based_reversal"]
    )

    frame["stored_vs_strict_disagreement"] = (
        frame["stored_reversal"]
        != frame["strict_reversal"]
    )

    disagreements = frame.loc[
        frame["stored_vs_strict_disagreement"]
        | frame["stored_vs_tolerance_disagreement"]
        | (frame["stored_tie"] != frame["recomputed_tie"])
    ].copy()

    disagreements.to_csv(
        outdir / "ranking_flag_disagreements.csv",
        index=False,
    )

    margin_rows = []
    metric_rows = []

    for margin in sorted(set(margins)):
        if margin < 0:
            raise ValueError("Margins must be nonnegative.")

        eligible = frame.loc[
            frame["minimum_gap"] > margin
        ].copy()

        total = len(frame)
        n_eligible = len(eligible)
        reversals = int(eligible["strict_reversal"].sum())

        margin_rows.append({
            "margin": margin,
            "all_valid_comparisons": total,
            "eligible_comparisons": n_eligible,
            "reversals": reversals,
            "conditional_rate_pct": (
                100 * reversals / n_eligible
                if n_eligible else np.nan
            ),
            "all_comparison_rate_pct": (
                100 * reversals / total
                if total else np.nan
            ),
        })

        for metric, group in frame.groupby("metric"):
            metric_eligible = group.loc[
                group["minimum_gap"] > margin
            ]

            n = len(metric_eligible)
            rev = int(metric_eligible["strict_reversal"].sum())

            metric_rows.append({
                "metric": metric,
                "margin": margin,
                "all_valid_comparisons": len(group),
                "eligible_comparisons": n,
                "reversals": rev,
                "conditional_rate_pct": (
                    100 * rev / n if n else np.nan
                ),
                "all_comparison_rate_pct": (
                    100 * rev / len(group)
                ),
            })

    margin_summary = pd.DataFrame(margin_rows)
    metric_summary = pd.DataFrame(metric_rows)

    margin_summary.to_csv(
        outdir / "ranking_margin_summary.csv",
        index=False,
    )

    metric_summary.to_csv(
        outdir / "ranking_margin_by_metric.csv",
        index=False,
    )

    validation = pd.DataFrame([{
        "original_detail_rows": original_rows,
        "valid_detail_rows": len(frame),
        "nonfinite_rows_removed": original_rows - len(frame),
        "stored_reversals": int(frame["stored_reversal"].sum()),
        "tolerance_based_reversals": int(
            frame["tolerance_based_reversal"].sum()
        ),
        "strict_reversals": int(frame["strict_reversal"].sum()),
        "stored_ties": int(frame["stored_tie"].sum()),
        "recomputed_ties": int(frame["recomputed_tie"].sum()),
        "stored_vs_tolerance_disagreements": int(
            frame["stored_vs_tolerance_disagreement"].sum()
        ),
        "stored_vs_strict_disagreements": int(
            frame["stored_vs_strict_disagreement"].sum()
        ),
    }])

    validation.to_csv(
        outdir / "ranking_validation_summary.csv",
        index=False,
    )

    print("\nRanking validation:")
    print(validation.to_string(index=False))

    print("\nRanking margin sensitivity:")
    print(
        margin_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return margin_summary, metric_summary, validation


def reconcile_ranking_summary(
    ranking_summary,
    detail,
    validation_dir: Path,
):
    """Check that the original summary aggregates match detail flags."""
    stored = parse_boolean(detail["ranking_reversal"])
    summary_count = int(
        pd.to_numeric(
            ranking_summary["reversals"],
            errors="raise",
        ).sum()
    )

    detail_count = int(stored.sum())

    report = pd.DataFrame([{
        "ranking_summary_reversals": summary_count,
        "ranking_detail_reversals": detail_count,
        "difference": summary_count - detail_count,
        "summary_comparisons": int(
            ranking_summary["comparisons"].sum()
        ),
        "detail_comparisons": len(detail),
    }])

    report.to_csv(
        validation_dir / "ranking_summary_reconciliation.csv",
        index=False,
    )

    print("\nSummary/detail reconciliation:")
    print(report.to_string(index=False))

    if (
        summary_count != detail_count
        or int(ranking_summary["comparisons"].sum()) != len(detail)
    ):
        print(
            "WARNING: Ranking detail and summary counts differ. "
            "Investigate before reporting pooled results."
        )

    return report


def main():
    args = parse_args()

    args.figures.mkdir(parents=True, exist_ok=True)
    args.validation.mkdir(parents=True, exist_ok=True)

    verdict, ranking, detail = load_inputs(args.input)

    print("Generating decision-reversal figures...")
    generate_decision_figure(
        verdict,
        ranking,
        args.figures,
        args.dpi,
    )

    print("\nChecking ranking margins...")
    ranking_margin_analysis(
        detail,
        args.validation,
        args.margins,
    )

    reconcile_ranking_summary(
        ranking,
        detail,
        args.validation,
    )

    print("\nAll outputs saved successfully.")


if __name__ == "__main__":
    main()
