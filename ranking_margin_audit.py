
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MARGINS = [0.0, 0.001, 0.005, 0.01]


def summarize(data, group_cols, margin):
    eligible = data[data["min_gap"] > margin].copy()
    if not group_cols:
        return pd.DataFrame([{
            "margin": margin,
            "eligible": len(eligible),
            "reversals": int(eligible["strict_reversal"].sum()),
            "conditional_rate": (
                eligible["strict_reversal"].mean()
                if len(eligible) else np.nan
            ),
            "overall_rate": (
                eligible["strict_reversal"].sum() / len(data)
                if len(data) else np.nan
            ),
            "retained_fraction": (
                len(eligible) / len(data)
                if len(data) else np.nan
            )
        }])

    output = (
        eligible.groupby(group_cols, dropna=False)
        .agg(
            eligible=("strict_reversal", "size"),
            reversals=("strict_reversal", "sum")
        )
        .reset_index()
    )
    output["margin"] = margin
    output["conditional_rate"] = (
        output["reversals"] / output["eligible"]
    )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/decision/ranking_detail.csv"
    )
    parser.add_argument(
        "--output",
        default="results/validation"
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    required = [
        "dataset", "metric", "m",
        "coarse_difference", "refined_difference",
        "ranking_reversal"
    ]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["coarse_difference"] = pd.to_numeric(
        df["coarse_difference"], errors="coerce"
    )
    df["refined_difference"] = pd.to_numeric(
        df["refined_difference"], errors="coerce"
    )

    valid = (
        np.isfinite(df["coarse_difference"]) &
        np.isfinite(df["refined_difference"])
    )

    invalid_count = int((~valid).sum())
    df = df.loc[valid].copy()

    coarse = df["coarse_difference"]
    refined = df["refined_difference"]

    df["strict_reversal"] = (coarse * refined) < 0
    df["min_gap"] = np.minimum(
        coarse.abs(), refined.abs()
    )

    df["stored_reversal"] = (
        df["ranking_reversal"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    disagreements = df[
        df["strict_reversal"] != df["stored_reversal"]
    ].copy()

    disagreements.to_csv(
        out / "ranking_flag_disagreements.csv",
        index=False
    )

    overall = []
    grouped = []

    for margin in MARGINS:
        overall.append(summarize(df, [], margin))

        for cols in [
            ["dataset"],
            ["metric"],
            ["m"],
            ["dataset", "metric"]
        ]:
            part = summarize(df, cols, margin)
            part["grouping"] = "+".join(cols)
            grouped.append(part)

    overall = pd.concat(overall, ignore_index=True)
    grouped = pd.concat(grouped, ignore_index=True)

    overall.to_csv(
        out / "ranking_margin_summary.csv",
        index=False
    )
    grouped.to_csv(
        out / "ranking_margin_breakdown.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(
        overall["margin"],
        100 * overall["conditional_rate"],
        marker="o",
        label="Among eligible comparisons"
    )
    ax.plot(
        overall["margin"],
        100 * overall["overall_rate"],
        marker="s",
        linestyle="--",
        label="Among all comparisons"
    )

    ax.set_xlabel("Minimum fairness-score margin")
    ax.set_ylabel("Ranking reversal rate (%)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(
        out / "ranking_margin_sensitivity.pdf",
        bbox_inches="tight"
    )
    fig.savefig(
        out / "ranking_margin_sensitivity.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(f"Analyzed comparisons: {len(df):,}")
    print(f"Nonfinite comparisons excluded: {invalid_count}")
    print(f"Flag disagreements: {len(disagreements):,}")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
