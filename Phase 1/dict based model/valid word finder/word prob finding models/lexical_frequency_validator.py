"""
Lexical frequency validator learned from the training corpus.

How this file works:
This module builds a lexicon directly from the training sentences. Each word is
assigned an observed count, empirical probability, and a normalized frequency
score derived from the count distribution. This removes the dependency on any
predefined external dictionary.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from character_backoff_language_model import DEFAULT_TRAIN_PATH, load_words, normalize_word


def build_lexicon(
    train_path: Path = DEFAULT_TRAIN_PATH,
    train_sheet: str | None = None,
) -> dict[str, dict[str, float | int]]:
    weighted_words = load_words(train_path, sheet_name=train_sheet)
    total_count = sum(count for _, count in weighted_words)
    max_count = max((count for _, count in weighted_words), default=1)
    log_denom = math.log1p(max_count) or 1.0

    lexicon: dict[str, dict[str, float | int]] = {}
    for word, count in weighted_words:
        probability = count / total_count if total_count else 0.0
        normalized_score = math.log1p(count) / log_denom
        lexicon[word] = {
            "count": count,
            "probability": probability,
            "normalized_score": normalized_score,
        }
    return lexicon


def export_lexicon(lexicon: dict[str, dict[str, float | int]], output_path: Path) -> None:
    rows = [
        {
            "word": word,
            "count": int(metrics["count"]),
            "probability": float(metrics["probability"]),
            "normalized_score": float(metrics["normalized_score"]),
        }
        for word, metrics in sorted(lexicon.items())
    ]
    df = pd.DataFrame(rows)
    if output_path.suffix.lower() == ".xlsx":
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)


def normalize_zipf(score: float) -> float:
    return max(0.0, min(1.0, score))


def validate_word(word: str, lexicon: dict[str, dict[str, float | int]]) -> dict[str, object]:
    normalized = normalize_word(word)
    if not normalized:
        return {
            "word": word,
            "normalized_word": None,
            "is_valid": False,
            "count": 0,
            "probability": 0.0,
            "normalized_score": 0.0,
        }

    entry = lexicon.get(normalized)
    return {
        "word": word,
        "normalized_word": normalized,
        "is_valid": entry is not None,
        "count": int(entry["count"]) if entry else 0,
        "probability": float(entry["probability"]) if entry else 0.0,
        "normalized_score": normalize_zipf(float(entry["normalized_score"])) if entry else 0.0,
    }


def evaluate_dataset(path: Path, lexicon: dict[str, dict[str, float | int]]) -> dict[str, float | int]:
    words = load_words(path)
    valid_rows = 0
    valid_weight = 0
    total_rows = 0
    total_weight = 0
    score_sum = 0.0
    weighted_score_sum = 0.0

    for word, count in words:
        result = validate_word(word, lexicon)
        total_rows += 1
        total_weight += count
        score_sum += result["normalized_score"]
        weighted_score_sum += result["normalized_score"] * count
        if result["is_valid"]:
            valid_rows += 1
            valid_weight += count

    return {
        "rows": total_rows,
        "total_weight": total_weight,
        "valid_row_fraction": valid_rows / total_rows if total_rows else 0.0,
        "valid_weight_fraction": valid_weight / total_weight if total_weight else 0.0,
        "average_score": score_sum / total_rows if total_rows else 0.0,
        "weighted_average_score": weighted_score_sum / total_weight if total_weight else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corpus-trained lexical frequency validator.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--train-sheet", type=str, default=None)
    parser.add_argument("--export", type=Path, help="Optional output path for the lexicon (.csv or .xlsx).")
    parser.add_argument("--word", type=str, help="Optional single word to validate.")
    parser.add_argument("--eval-file", type=Path, help="Optional Excel file to evaluate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lexicon = build_lexicon(train_path=args.train, train_sheet=args.train_sheet)
    print(f"lexicon_words: {len(lexicon)}")

    if args.export is not None:
        export_lexicon(lexicon, args.export)
        print(f"exported_to: {args.export}")

    if args.word is not None:
        result = validate_word(args.word, lexicon)
        print(f"word: {result['word']}")
        print(f"normalized_word: {result['normalized_word']}")
        print(f"is_valid: {result['is_valid']}")
        print(f"count: {result['count']}")
        print(f"probability: {result['probability']:.16e}")
        print(f"normalized_score: {result['normalized_score']:.6f}")

    if args.eval_file is not None:
        metrics = evaluate_dataset(args.eval_file, lexicon)
        print(f"eval_file: {args.eval_file}")
        print(f"rows: {metrics['rows']}")
        print(f"total_weight: {metrics['total_weight']}")
        print(f"valid_row_fraction: {metrics['valid_row_fraction']:.6f}")
        print(f"valid_weight_fraction: {metrics['valid_weight_fraction']:.6f}")
        print(f"average_score: {metrics['average_score']:.6f}")
        print(f"weighted_average_score: {metrics['weighted_average_score']:.6f}")


if __name__ == "__main__":
    main()
