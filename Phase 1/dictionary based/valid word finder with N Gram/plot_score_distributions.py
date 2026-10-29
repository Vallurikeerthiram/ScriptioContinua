"""
Plot 1D score distributions from a hybrid-results Excel sheet.

How this file works:
This script reads an Excel file produced by the hybrid export script and creates
separate one-dimensional plots for selected score columns. Each word is drawn as
a point on a horizontal score axis. Vertical tick marks are also drawn so dense
regions are easier to see.

Coloring behavior:
- If `ground_truth_label` exists, green means ground-truth valid and red means
  ground-truth invalid.
- Otherwise, green and red represent predicted validity.

Manual run examples:
`python plot_score_distributions.py --input-file "results/spreadsheets/special_test_hybrid_results.xlsx" --output-dir "results/plots"`
`python plot_score_distributions.py --input-file "results/spreadsheets/special_test_2_hybrid_results.xlsx" --output-dir "results/plots/special_test_2" --prefix "special_test_2_"`
`python plot_score_distributions.py --input-file "results/spreadsheets/test_word_count_hybrid_results.xlsx" --output-dir "results/plots/test_word_count" --prefix "test_word_count_"`

Example idea:
If many valid words cluster near high hybrid scores while invalid words stay on
the low side, the resulting plot gives a quick visual sense of how separable the
classes are.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCORE_COLUMNS = [
    ("hybrid_score", "Hybrid Score"),
    ("pronounceability_score", "Pronounceability Score"),
    ("zipf_score", "Zipf Score"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 1D score plots from a hybrid-results Excel file.")
    parser.add_argument("--input-file", type=Path, default=Path("results/spreadsheets/special_test_hybrid_results.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    parser.add_argument("--prefix", type=str, default="")
    return parser.parse_args()


def make_plot(df: pd.DataFrame, column: str, title: str, output_path: Path) -> None:
    if "ground_truth_label" in df.columns:
        valid_mask = df["ground_truth_label"] == 1
        invalid_mask = ~valid_mask
        valid_label = "Valid (ground truth)"
        invalid_label = "Invalid (ground truth)"
        color_column = "ground_truth_label"
    else:
        valid_mask = df["predicted_label"] == 1
        invalid_mask = ~valid_mask
        valid_label = "Valid (predicted)"
        invalid_label = "Invalid (predicted)"
        color_column = "predicted_label"

    fig, ax = plt.subplots(figsize=(12, 2.8))

    ax.scatter(
        df.loc[valid_mask, column],
        [0] * int(valid_mask.sum()),
        color="forestgreen",
        s=55,
        alpha=0.85,
        label=valid_label,
    )
    ax.scatter(
        df.loc[invalid_mask, column],
        [0] * int(invalid_mask.sum()),
        color="crimson",
        s=55,
        alpha=0.85,
        label=invalid_label,
    )

    for _, row in df.iterrows():
        color = "forestgreen" if row[color_column] == 1 else "crimson"
        ax.vlines(row[column], -0.08, 0.08, colors=color, alpha=0.35, linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Score / Probability")
    ax.set_yticks([])
    ax.set_ylim(-0.2, 0.2)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", ncol=2, frameon=False)

    if column in {"hybrid_score", "pronounceability_score"}:
        ax.set_xlim(-0.02, 1.02)
    else:
        min_score = float(df[column].min())
        max_score = float(df[column].max())
        padding = max(0.1, (max_score - min_score) * 0.05)
        ax.set_xlim(min_score - padding, max_score + padding)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_excel(args.input_file)
    args.output_dir.mkdir(exist_ok=True)

    for column, title in SCORE_COLUMNS:
        filename = f"{args.prefix}{column}_1d_plot.png"
        output_path = args.output_dir / filename
        make_plot(df, column, title, output_path)
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
