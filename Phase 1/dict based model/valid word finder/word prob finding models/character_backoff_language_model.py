"""
Character-level backoff language model for word scoring.

How this file works:
This module learns letter continuation patterns from weighted words. The words
can come from either a direct word/count Excel file or a sentence workbook such
as `SENT_ID.xlsx`, where words are learned from the `train` sheet text.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


WORD_RE = re.compile(r"[a-z]+")
DEFAULT_TRAIN_PATH = Path(__file__).resolve().parents[3] / "DATASET" / "18000 rows" / "new dataset" / "SENT_ID.xlsx"


def make_readable_copy(path: Path) -> Path:
    try:
        with path.open("rb"):
            return path
    except PermissionError:
        temp_dir = Path(tempfile.mkdtemp(prefix="simple_backoff_"))
        copied = temp_dir / path.name
        shutil.copy2(path, copied)
        return copied


def normalize_word(value: object) -> str | None:
    if value is None:
        return None
    text = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode("ascii")
        .strip()
        .lower()
    )
    if not text:
        return None
    match = WORD_RE.fullmatch(text)
    return match.group(0) if match else None


def tokenize_text(text: object) -> list[str]:
    normalized_text = (
        unicodedata.normalize("NFKD", str(text or ""))
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return WORD_RE.findall(normalized_text)


def dataframe_to_weighted_words(df: pd.DataFrame) -> list[tuple[str, int]]:
    normalized_columns = {str(column).strip().lower(): column for column in df.columns}
    if "word" in normalized_columns:
        word_column = normalized_columns["word"]
        count_column = normalized_columns.get("count")
        records: list[tuple[str, int]] = []
        for row in df.itertuples(index=False):
            word = normalize_word(getattr(row, str(word_column), None))
            if not word:
                continue
            raw_count = getattr(row, str(count_column), 1) if count_column is not None else 1
            try:
                count = int(raw_count)
            except (TypeError, ValueError):
                count = 1
            if count > 0:
                records.append((word, count))
        return records

    for text_key in ("sent_ori", "sentence", "text", "content"):
        if text_key in normalized_columns:
            text_column = normalized_columns[text_key]
            counter: Counter[str] = Counter()
            for value in df[text_column].tolist():
                counter.update(tokenize_text(value))
            return sorted(counter.items())

    return []


def load_words(path: Path, sheet_name: str | None = None) -> list[tuple[str, int]]:
    readable_path = make_readable_copy(path)
    workbook = pd.ExcelFile(readable_path)
    sheets = [sheet_name] if sheet_name is not None else (["train"] if "train" in workbook.sheet_names else workbook.sheet_names)

    merged_counts: Counter[str] = Counter()
    for current_sheet in sheets:
        df = workbook.parse(current_sheet)
        merged_counts.update(dict(dataframe_to_weighted_words(df)))
    return sorted((word, count) for word, count in merged_counts.items() if count > 0)


class SimpleBackoffChainModel:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.context_totals: dict[str, int] = {}

    def fit(self, weighted_words: list[tuple[str, int]]) -> None:
        self.counts = defaultdict(Counter)
        for word, weight in weighted_words:
            self._add_word(word, weight)
        self.context_totals = {
            context: sum(next_char_counts.values())
            for context, next_char_counts in self.counts.items()
        }

    def _add_word(self, word: str, weight: int) -> None:
        for i in range(len(word)):
            next_char = word[i]
            for start in range(i):
                context = word[start:i]
                self.counts[context][next_char] += weight

    def probability_of_next_char(self, context: str, next_char: str) -> tuple[float, str]:
        current = context
        while current:
            if current in self.counts:
                total = self.context_totals[current]
                count = self.counts[current][next_char]
                if total > 0 and count > 0:
                    return count / total, current
            current = current[1:]
        return self.epsilon, ""

    def word_probability(self, word: str) -> tuple[float, list[dict[str, object]]]:
        probability = 1.0
        steps: list[dict[str, object]] = []
        for i, next_char in enumerate(word):
            context = word[:i]
            char_probability, used_context = self.probability_of_next_char(context, next_char)
            probability *= char_probability
            steps.append(
                {
                    "position": i,
                    "char": next_char,
                    "original_context": context,
                    "used_context": used_context,
                    "probability": char_probability,
                }
            )
        return probability, steps


def evaluate_dataset(
    model: SimpleBackoffChainModel,
    words: list[tuple[str, int]],
) -> dict[str, float | int]:
    total_rows = 0
    total_weight = 0
    unweighted_sum = 0.0
    weighted_sum = 0.0

    for word, count in words:
        probability, _ = model.word_probability(word)
        total_rows += 1
        total_weight += count
        unweighted_sum += probability
        weighted_sum += probability * count

    return {
        "rows": total_rows,
        "total_weight": total_weight,
        "unweighted_average_probability": unweighted_sum / total_rows if total_rows else 0.0,
        "weighted_average_probability": weighted_sum / total_weight if total_weight else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple character-level chain-rule backoff model.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--train-sheet", type=str, default=None, help="Optional workbook sheet name. Defaults to 'train' when present.")
    parser.add_argument("--word", type=str, help="Word to score.")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="Fallback probability when no context matches.")
    parser.add_argument("--show-steps", action="store_true", help="Print per-character backoff details.")
    parser.add_argument("--eval-file", type=Path, help="Optional Excel file to evaluate after training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_words = load_words(args.train, sheet_name=args.train_sheet)

    model = SimpleBackoffChainModel(epsilon=args.epsilon)
    model.fit(train_words)

    print(f"training_source: {args.train}")
    print(f"training_words: {len(train_words)}")

    if args.word is not None:
        word = normalize_word(args.word)
        if not word:
            raise SystemExit("Please provide a lowercase alphabetic word, for example: --word keert")

        probability, steps = model.word_probability(word)
        print(f"word: {word}")
        print(f"probability: {probability:.16e}")
        if probability > 0.0:
            print(f"log_probability: {math.log(probability):.16f}")
        else:
            print("log_probability: -inf")

        if args.show_steps:
            print("steps:")
            for step in steps:
                used_context = step["used_context"] if step["used_context"] else "<epsilon>"
                print(
                    f"  pos={step['position']} char='{step['char']}' "
                    f"context='{step['original_context']}' used='{used_context}' "
                    f"p={step['probability']:.16e}"
                )

    if args.eval_file is not None:
        eval_words = load_words(args.eval_file)
        metrics = evaluate_dataset(model, eval_words)
        print(f"eval_file: {args.eval_file}")
        print(f"eval_rows: {metrics['rows']}")
        print(f"eval_total_weight: {metrics['total_weight']}")
        print(f"eval_unweighted_average_probability: {metrics['unweighted_average_probability']:.16e}")
        print(f"eval_weighted_average_probability: {metrics['weighted_average_probability']:.16e}")


if __name__ == "__main__":
    main()
