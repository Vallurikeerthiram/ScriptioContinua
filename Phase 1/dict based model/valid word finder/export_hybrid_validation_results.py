"""
Export helper for corpus-trained validation results.

How this file works:
This script trains the six requested models from the sentence workbook's
training sheet, scores every word from the test workbook, and writes a new
Excel workbook with one score column per model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from character_backoff_language_model import DEFAULT_TRAIN_PATH, SimpleBackoffChainModel, make_readable_copy, normalize_word
from character_backoff_pronounceability_validator import CharacterPronounceabilityValidator
from hybrid_lexical_character_pronounceability_validator import HybridWordValidator as HybridCharacterValidator
from hybrid_lexical_pronounceability_validator import HybridWordValidator
from lexical_frequency_validator import build_lexicon
from syllable_backoff_pronounceability_validator import PronounceabilityValidator
from character_backoff_language_model import load_words

PARENT_VALIDATOR_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_VALIDATOR_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_VALIDATOR_DIR))

from orthographic_phonotactic_validator import PhonotacticWordlikenessValidator
from phoneme_sequence_wordlikeness_validator import PhonemeWordlikenessValidator


DEFAULT_INPUT_FILE = Path(__file__).resolve().parents[1] / "test dataset.xlsx"
DEFAULT_OUTPUT_FILE = Path(__file__).resolve().parents[1] / "test_dataset_corpus_trained_scores.xlsx"
OUTPUT_SCORE_COLUMNS = [
    "character_backoff_language_model.py",
    "character_backoff_pronounceability_validator.py",
    "hybrid_lexical_character_pronounceability_validator.py",
    "hybrid_lexical_pronounceability_validator.py",
    "lexical_frequency_validator.py",
    "orthographic_phonotactic_validator.py",
    "phoneme_sequence_wordlikeness_validator.py",
    "syllable_backoff_pronounceability_validator.py",
    "lexical_probability_from_train",
]
SOURCE_COLUMNS_TO_REPLACE = set(OUTPUT_SCORE_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export corpus-trained scores for the requested validators.")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--train-sheet", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.35)
    return parser.parse_args()


def load_test_rows(path: Path) -> dict[str, pd.DataFrame]:
    readable_path = make_readable_copy(path)
    workbook = pd.ExcelFile(readable_path)
    sheets: dict[str, pd.DataFrame] = {}
    for sheet_name in workbook.sheet_names:
        df = workbook.parse(sheet_name)
        if "word" in df.columns:
            sheets[sheet_name] = df.copy()
    return sheets


def main() -> None:
    args = parse_args()
    train_words = load_words(args.train_file, sheet_name=args.train_sheet)
    lexicon = build_lexicon(train_path=args.train_file, train_sheet=args.train_sheet)

    char_model = SimpleBackoffChainModel()
    char_model.fit(train_words)

    char_pronounceability = CharacterPronounceabilityValidator(threshold=args.threshold)
    char_pronounceability.fit(train_words)

    syllable_pronounceability = PronounceabilityValidator(threshold=args.threshold)
    syllable_pronounceability.fit(train_words)

    hybrid_character_validator = HybridCharacterValidator(
        lexicon=lexicon,
        pronounceability=char_pronounceability,
    )
    hybrid_validator = HybridWordValidator(
        lexicon=lexicon,
        pronounceability=syllable_pronounceability,
    )
    orthographic_validator = PhonotacticWordlikenessValidator()
    orthographic_validator.fit(train_words)

    phoneme_validator = PhonemeWordlikenessValidator()
    phoneme_validator.fit(train_words)

    sheet_outputs: dict[str, pd.DataFrame] = {}
    total_rows = 0
    for sheet_name, df in load_test_rows(args.input_file).items():
        rows = []
        for row in df.to_dict(orient="records"):
            base_row = {
                key: value
                for key, value in row.items()
                if key not in SOURCE_COLUMNS_TO_REPLACE
            }
            word = row.get("word")
            normalized_word = normalize_word(word)
            lexical_result = hybrid_character_validator.lexicon.get(normalized_word) if normalized_word else None
            char_probability, _ = char_model.word_probability(normalized_word) if normalized_word else (0.0, [])
            char_pron_result = char_pronounceability.score_word(word)
            hybrid_char_result = hybrid_character_validator.score_word(word)
            hybrid_result = hybrid_validator.score_word(word)
            orthographic_result = orthographic_validator.score_word(word)
            phoneme_result = phoneme_validator.score_word(word)
            syllable_result = syllable_pronounceability.score_word(word)
            lexical_score = 0.0
            lexical_probability = 0.0
            if lexical_result is not None:
                lexical_score = float(lexical_result["normalized_score"])
                lexical_probability = float(lexical_result["probability"])

            output_row = dict(base_row)
            output_row.update(
                {
                    "character_backoff_language_model.py": float(char_probability),
                    "character_backoff_pronounceability_validator.py": float(char_pron_result["score"]),
                    "hybrid_lexical_character_pronounceability_validator.py": float(hybrid_char_result["score"]),
                    "hybrid_lexical_pronounceability_validator.py": float(hybrid_result["score"]),
                    "lexical_frequency_validator.py": lexical_score,
                    "lexical_probability_from_train": lexical_probability,
                    "orthographic_phonotactic_validator.py": float(orthographic_result["score"]),
                    "phoneme_sequence_wordlikeness_validator.py": float(phoneme_result["score"]),
                    "syllable_backoff_pronounceability_validator.py": float(syllable_result["score"]),
                }
            )
            rows.append(output_row)

        output_df = pd.DataFrame(rows)
        preferred_columns = ["word"]
        if "label" in output_df.columns:
            preferred_columns.append("label")
        preferred_columns.extend(column for column in OUTPUT_SCORE_COLUMNS if column in output_df.columns)
        remaining_columns = [column for column in output_df.columns if column not in preferred_columns]
        output_df = output_df[preferred_columns + remaining_columns]
        sheet_outputs[sheet_name] = output_df
        total_rows += len(output_df)

    with pd.ExcelWriter(args.output_file) as writer:
        for sheet_name, output_df in sheet_outputs.items():
            output_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"written: {args.output_file}")
    print(f"sheets: {len(sheet_outputs)}")
    print(f"rows: {total_rows}")


if __name__ == "__main__":
    main()
