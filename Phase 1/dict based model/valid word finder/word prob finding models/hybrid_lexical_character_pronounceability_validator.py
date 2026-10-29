"""
Hybrid lexical plus character-pronounceability word validator.

How this file works:
This file preserves the older hybrid approach that combines:

1. Lexical familiarity from the wordfreq lexicon.
2. Character-based pronounceability from the older pronounceability validator.

Decision logic:
- If the word is present in the lexicon, lexical familiarity and character-based
  pronounceability are combined.
- If the word is absent from the lexicon, the validator falls back to the
  character-based pronounceability score alone.

This file exists so you can compare the older character-based hybrid with the
newer syllable-based hybrid side by side.

Manual run examples:
`python hybrid_lexical_character_pronounceability_validator.py --word apple`
`python hybrid_lexical_character_pronounceability_validator.py --word snorple`
`python hybrid_lexical_character_pronounceability_validator.py --eval-file "datasets/special test.xlsx"`
`python hybrid_lexical_character_pronounceability_validator.py --eval-file "datasets/special test 2.xlsx"`

Example idea:
For `apple`, the validator looks up lexical frequency, computes a normalized
lexical score, gets the character-based pronounceability score, adds a common
word bonus, and returns a final combined score.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from character_backoff_language_model import DEFAULT_TRAIN_PATH, load_words, normalize_word
from character_backoff_pronounceability_validator import CharacterPronounceabilityValidator
from lexical_frequency_validator import build_lexicon, normalize_zipf


class HybridWordValidator:
    def __init__(
        self,
        lexicon: dict[str, dict[str, float | int]],
        pronounceability: CharacterPronounceabilityValidator,
    ) -> None:
        self.lexicon = lexicon
        self.pronounceability = pronounceability

    def score_word(self, raw_word: str) -> dict[str, object]:
        word = normalize_word(raw_word)
        if not word:
            return {
                "word": raw_word,
                "normalized_word": None,
                "is_valid": False,
                "score": 0.0,
                "source": "invalid_input",
                "zipf_score": 0.0,
                "pronounceability_score": 0.0,
            }

        pronounce_result = self.pronounceability.score_word(word)
        pronounce_score = pronounce_result["score"]

        if word in self.lexicon:
            lexical_entry = self.lexicon[word]
            lexical_probability = float(lexical_entry["probability"])
            familiarity_score = normalize_zipf(float(lexical_entry["normalized_score"]))
            common_word_bonus = 0.1 if int(lexical_entry["count"]) >= 5 else 0.0
            combined_score = min(
                1.0,
                (0.5 * pronounce_score) + (0.5 * familiarity_score) + common_word_bonus,
            )
            return {
                "word": raw_word,
                "normalized_word": word,
                "is_valid": combined_score >= self.pronounceability.threshold,
                "score": combined_score,
                "source": "lexicon+pronounceability",
                "lexical_probability": lexical_probability,
                "lexical_count": int(lexical_entry["count"]),
                "familiarity_score": familiarity_score,
                "pronounceability_score": pronounce_score,
            }

        return {
            "word": raw_word,
            "normalized_word": word,
            "is_valid": pronounce_result["is_pronounceable"],
            "score": pronounce_result["score"],
            "source": "pronounceability",
            "lexical_probability": 0.0,
            "lexical_count": 0,
            "pronounceability_score": pronounce_result["score"],
        }


def evaluate_dataset(path: Path, validator: HybridWordValidator) -> dict[str, float | int]:
    words = load_words(path)
    total_rows = 0
    total_weight = 0
    score_sum = 0.0
    weighted_score_sum = 0.0
    valid_rows = 0
    valid_weight = 0

    for word, count in words:
        result = validator.score_word(word)
        total_rows += 1
        total_weight += count
        score_sum += result["score"]
        weighted_score_sum += result["score"] * count
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
    parser = argparse.ArgumentParser(
        description="Hybrid word validator using corpus lexical familiarity plus character-based pronounceability."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--train-sheet", type=str, default=None)
    parser.add_argument("--word", type=str, help="Optional word to score.")
    parser.add_argument("--eval-file", type=Path, help="Optional Excel file to evaluate.")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--score-only", action="store_true", help="Print only the final score for --word.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_words = load_words(args.train, sheet_name=args.train_sheet)
    lexicon = build_lexicon(train_path=args.train, train_sheet=args.train_sheet)
    pronounceability = CharacterPronounceabilityValidator(threshold=args.threshold)
    pronounceability.fit(train_words)
    validator = HybridWordValidator(lexicon=lexicon, pronounceability=pronounceability)

    if args.word is not None:
        result = validator.score_word(args.word)
        if args.score_only:
            print(f"{result['score']:.6f}")
            return

        print(f"training_words: {len(train_words)}")
        print(f"lexicon_words: {len(lexicon)}")
        print(f"threshold: {args.threshold:.2f}")
        print(f"word: {result['word']}")
        print(f"normalized_word: {result['normalized_word']}")
        print(f"is_valid: {result['is_valid']}")
        print(f"score: {result['score']:.6f}")
        print(f"source: {result['source']}")
        print(f"lexical_probability: {result['lexical_probability']:.16e}")
        print(f"pronounceability_score: {result['pronounceability_score']:.6f}")

    if args.eval_file is not None:
        print(f"training_words: {len(train_words)}")
        print(f"lexicon_words: {len(lexicon)}")
        print(f"threshold: {args.threshold:.2f}")
        metrics = evaluate_dataset(args.eval_file, validator)
        print(f"eval_file: {args.eval_file}")
        print(f"rows: {metrics['rows']}")
        print(f"total_weight: {metrics['total_weight']}")
        print(f"valid_row_fraction: {metrics['valid_row_fraction']:.6f}")
        print(f"valid_weight_fraction: {metrics['valid_weight_fraction']:.6f}")
        print(f"average_score: {metrics['average_score']:.6f}")
        print(f"weighted_average_score: {metrics['weighted_average_score']:.6f}")


if __name__ == "__main__":
    main()
