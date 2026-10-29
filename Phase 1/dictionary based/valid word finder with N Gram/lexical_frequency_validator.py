"""
Lexical frequency validator based on wordfreq.

How this file works:
This module builds a lexicon of known English words using the `wordfreq`
package. Each word is assigned a Zipf frequency score. The validator then marks
an input word as valid if it appears in that lexicon and also exposes a
normalized frequency score between 0 and 1.

This is the cleanest dictionary-style validator in the folder. It does not
check pronounceability or phoneme structure. It only asks whether the word is
known and how common it is.

Manual run examples:
`python lexical_frequency_validator.py --word apple`
`python lexical_frequency_validator.py --eval-file "datasets/special test.xlsx"`
`python lexical_frequency_validator.py --export "results/spreadsheets/lexicon.xlsx"`

Example idea:
For `apple`, the word is found in the lexicon, so `is_valid=True` and the Zipf
score is converted to a normalized familiarity score. For a nonce word such as
`blost`, the word is usually absent and the score becomes zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from wordfreq import iter_wordlist, zipf_frequency

from character_backoff_language_model import load_words, normalize_word


def build_lexicon(
    language: str = "en",
    wordlist: str = "large",
    max_words: int | None = None,
) -> dict[str, float]:
    lexicon: dict[str, float] = {}
    for index, raw_word in enumerate(iter_wordlist(language, wordlist=wordlist)):
        if max_words is not None and index >= max_words:
            break

        word = normalize_word(raw_word)
        if not word:
            continue

        lexicon[word] = zipf_frequency(word, language, wordlist=wordlist)
    return lexicon


def export_lexicon(lexicon: dict[str, float], output_path: Path) -> None:
    rows = [
        {"word": word, "zipf_score": zipf_score, "normalized_score": normalize_zipf(zipf_score)}
        for word, zipf_score in sorted(lexicon.items())
    ]
    df = pd.DataFrame(rows)
    if output_path.suffix.lower() == ".xlsx":
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)


def normalize_zipf(zipf_score: float) -> float:
    return max(0.0, min(1.0, zipf_score / 8.0))


def validate_word(word: str, lexicon: dict[str, float]) -> dict[str, object]:
    normalized = normalize_word(word)
    if not normalized:
        return {
            "word": word,
            "normalized_word": None,
            "is_valid": False,
            "zipf_score": 0.0,
            "normalized_score": 0.0,
        }

    zipf_score = lexicon.get(normalized, 0.0)
    return {
        "word": word,
        "normalized_word": normalized,
        "is_valid": normalized in lexicon,
        "zipf_score": zipf_score,
        "normalized_score": normalize_zipf(zipf_score),
    }


def evaluate_dataset(path: Path, lexicon: dict[str, float]) -> dict[str, float | int]:
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
    parser = argparse.ArgumentParser(description="Lexicon-based English word validator.")
    parser.add_argument("--wordlist", choices=["small", "large", "best"], default="large")
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--export", type=Path, help="Optional output path for the lexicon (.csv or .xlsx).")
    parser.add_argument("--word", type=str, help="Optional single word to validate.")
    parser.add_argument("--eval-file", type=Path, help="Optional Excel file to evaluate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lexicon = build_lexicon(wordlist=args.wordlist, max_words=args.max_words)
    print(f"lexicon_words: {len(lexicon)}")

    if args.export is not None:
        export_lexicon(lexicon, args.export)
        print(f"exported_to: {args.export}")

    if args.word is not None:
        result = validate_word(args.word, lexicon)
        print(f"word: {result['word']}")
        print(f"normalized_word: {result['normalized_word']}")
        print(f"is_valid: {result['is_valid']}")
        print(f"zipf_score: {result['zipf_score']:.4f}")
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
