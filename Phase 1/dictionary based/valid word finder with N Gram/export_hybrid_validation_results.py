"""
Export helper for hybrid validation results.

How this file works:
This script loads an input Excel file, runs every word through the hybrid
lexical-plus-pronounceability validator, and writes a new Excel file containing
the scores and predicted labels. It supports both labeled sheets and unlabeled
word-count sheets.

If the input has:
- `word` and `label`, the output includes ground-truth fields.
- `word` and `count`, the output preserves count and writes only predictions.

This script is useful when you want a spreadsheet you can inspect manually,
plot, sort, or compare across models.

Manual run examples:
`python export_hybrid_validation_results.py --input-file "datasets/special test.xlsx" --output-file "results/spreadsheets/special_test_hybrid_results.xlsx"`
`python export_hybrid_validation_results.py --input-file "datasets/special test 2.xlsx" --output-file "results/spreadsheets/special_test_2_hybrid_results.xlsx"`
`python export_hybrid_validation_results.py --input-file "datasets/test_word_count.xlsx" --output-file "results/spreadsheets/test_word_count_hybrid_results.xlsx"`

Example idea:
If the row contains `apple, 1`, the script scores `apple`, stores the hybrid
score, predicted validity, lexical score pieces, pronounceability score, and
also keeps the ground-truth label so you can compare them in Excel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hybrid_lexical_pronounceability_validator import HybridWordValidator
from lexical_frequency_validator import build_lexicon
from syllable_backoff_pronounceability_validator import PronounceabilityValidator
from character_backoff_language_model import load_words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export hybrid validator results for a labeled Excel sheet.")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, default=Path("datasets/train_word_count.xlsx"))
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--wordlist", choices=["small", "large", "best"], default="large")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_excel(args.input_file)
    train_words = load_words(args.train_file)
    lexicon = build_lexicon(wordlist=args.wordlist)
    pronounceability = PronounceabilityValidator(threshold=args.threshold)
    pronounceability.fit(train_words)
    validator = HybridWordValidator(lexicon=lexicon, pronounceability=pronounceability)

    rows = []
    for row in df.to_dict(orient="records"):
        result = validator.score_word(row.get("word"))
        has_label = "label" in row
        has_count = "count" in row
        rows.append(
            {
                "word": row.get("word"),
                "count": int(row["count"]) if has_count and row["count"] is not None else 1,
                "hybrid_score": float(result["score"]),
                "predicted_validity": "valid" if bool(result["is_valid"]) else "invalid",
                "predicted_label": int(bool(result["is_valid"])),
                "normalized_word": result.get("normalized_word"),
                "source": result.get("source"),
                "zipf_score": float(result.get("zipf_score", 0.0) or 0.0),
                "pronounceability_score": float(result.get("pronounceability_score", 0.0) or 0.0),
            }
        )
        if has_label:
            ground_truth = int(row["label"])
            rows[-1]["ground_truth"] = "valid" if ground_truth == 1 else "invalid"
            rows[-1]["ground_truth_label"] = ground_truth

    out_df = pd.DataFrame(rows)
    out_df.to_excel(args.output_file, index=False)

    print(f"written: {args.output_file}")
    print(f"rows: {len(out_df)}")


if __name__ == "__main__":
    main()
