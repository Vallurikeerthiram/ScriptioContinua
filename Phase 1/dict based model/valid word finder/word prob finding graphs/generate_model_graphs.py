"""
Generate one reference-style graph per model column from a scored Excel file.

The workbook may contain multiple sheets. All sheets that contain `word` and
`label` columns are merged before plotting so the graphs cover the full dataset.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation
import pandas as pd


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "test_dataset_corpus_trained_scores_2026-04-29.xlsx"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "model graphs"
EXCLUDED_COLUMNS = {"word", "label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create one graph per numeric model column from a scored Excel workbook.")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--precision", type=int, default=6, help="Decimal precision used when stacking near-identical scores.")
    return parser.parse_args()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "plot"


def prettify_title(column_name: str) -> str:
    name = column_name.replace(".py", "")
    name = name.replace("_", " ")
    return " ".join(part.capitalize() for part in name.split())


def load_scored_rows(path: Path) -> pd.DataFrame:
    workbook = pd.ExcelFile(path)
    frames: list[pd.DataFrame] = []
    for sheet_name in workbook.sheet_names:
        df = workbook.parse(sheet_name)
        if {"word", "label"}.issubset(df.columns):
            frames.append(df.copy())
    if not frames:
        raise SystemExit("No sheets with both 'word' and 'label' columns were found.")
    return pd.concat(frames, ignore_index=True)


def detect_model_columns(df: pd.DataFrame) -> list[str]:
    model_columns: list[str] = []
    for column in df.columns:
        if column in EXCLUDED_COLUMNS:
            continue
        numeric_series = pd.to_numeric(df[column], errors="coerce")
        if numeric_series.notna().any():
            model_columns.append(column)
    return model_columns


def build_stacked_points(df: pd.DataFrame, column: str, precision: int) -> pd.DataFrame:
    plot_df = df[["word", "label", column]].copy()
    plot_df["score"] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df["label"] = pd.to_numeric(plot_df["label"], errors="coerce")
    plot_df = plot_df.dropna(subset=["score", "label"])
    plot_df = plot_df[plot_df["label"].isin([0, 1])].copy()
    plot_df["stack_key"] = plot_df["score"].round(precision)
    plot_df["stack_index"] = plot_df.groupby(["label", "stack_key"]).cumcount() + 1
    plot_df["y"] = plot_df.apply(
        lambda row: row["stack_index"] if int(row["label"]) == 1 else -row["stack_index"],
        axis=1,
    )
    return plot_df


def set_reasonable_xlim(ax, scores: pd.Series, column: str) -> None:
    min_score = float(scores.min())
    max_score = float(scores.max())

    if column == "character_backoff_language_model.py":
        positive_scores = scores[scores > 0]
        if not positive_scores.empty:
            min_positive = float(positive_scores.min())
            max_positive = float(positive_scores.max())
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(LogFormatterSciNotation())
            ax.set_xlim(min_positive * 0.8, max_positive * 1.25)
        return

    if 0.0 <= min_score and max_score <= 1.0:
        ax.set_xlim(-0.02, 1.02)
        return

    if math.isclose(min_score, max_score):
        padding = max(0.1, abs(min_score) * 0.1 + 0.1)
    else:
        padding = max(0.05, (max_score - min_score) * 0.08)
    ax.set_xlim(min_score - padding, max_score + padding)


def make_plot(df: pd.DataFrame, column: str, output_path: Path, precision: int) -> None:
    plot_df = build_stacked_points(df, column, precision)
    if plot_df.empty:
        return

    valid_df = plot_df[plot_df["label"] == 1]
    invalid_df = plot_df[plot_df["label"] == 0]
    max_stack = int(plot_df["stack_index"].max())

    fig_height = max(4.8, min(10.0, 4.0 + (0.18 * max_stack)))
    fig, ax = plt.subplots(figsize=(14, fig_height))

    ax.axhline(0, color="black", linewidth=1.0, alpha=0.7)
    ax.vlines(valid_df["score"], 0, valid_df["y"], colors="forestgreen", linewidth=1.4, alpha=0.75)
    ax.vlines(invalid_df["score"], 0, invalid_df["y"], colors="crimson", linewidth=1.4, alpha=0.75)

    ax.scatter(
        valid_df["score"],
        valid_df["y"],
        color="forestgreen",
        s=26,
        alpha=0.9,
        label="Label 1: meaningful words",
        zorder=3,
    )
    ax.scatter(
        invalid_df["score"],
        invalid_df["y"],
        color="crimson",
        s=26,
        alpha=0.9,
        label="Label 0: meaningless words",
        zorder=3,
    )

    ax.set_title(prettify_title(column), fontsize=14, pad=12)
    if column == "character_backoff_language_model.py":
        ax.set_xlabel("Model Probability (log scale)")
    else:
        ax.set_xlabel("Model Score / Probability")
    ax.set_ylabel("Stacked overlap count")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.grid(axis="y", linestyle=":", alpha=0.18)
    ax.legend(loc="upper right", frameon=False)

    set_reasonable_xlim(ax, plot_df["score"], column)
    ax.set_ylim(-(max_stack + 1), max_stack + 1)

    y_ticks = list(range(-max_stack, 0)) + [0] + list(range(1, max_stack + 1))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(abs(value)) if value != 0 else "0" for value in y_ticks])

    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_scored_rows(args.input_file)
    model_columns = detect_model_columns(df)
    if not model_columns:
        raise SystemExit("No numeric model columns were found in the input workbook.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for column in model_columns:
        output_path = args.output_dir / f"{slugify(column)}_graph.png"
        make_plot(df, column, output_path, args.precision)
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
