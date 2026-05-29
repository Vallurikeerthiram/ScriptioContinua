"""
Character-level backoff language model for word scoring.

How this file works:
This module learns letter continuation patterns from a training Excel file that
contains words and counts. For each position in a word, it stores how often a
next character follows a given left context. When scoring a new word, it tries
to use the longest available context first and backs off to shorter contexts if
that exact context was never seen in training. If nothing matches, it uses a
small epsilon probability.

This model is the low-level utility used by some of the other validators. It is
not syllable-aware or phoneme-aware. It simply models words as sequences of
characters.

Typical training flow:
1. Load weighted words from an Excel file.
2. Add counts for all character transitions.
3. For a test word, multiply the probabilities of each next character given its
   context.
4. Optionally inspect the backoff steps to see which context was actually used.

Manual run examples:
`python character_backoff_language_model.py --word apple`
`python character_backoff_language_model.py --word apple --show-steps`
`python character_backoff_language_model.py --eval-file "datasets/test_word_count.xlsx"`
`python character_backoff_language_model.py --train-source wordfreq --word hello`

Example idea:
If the model scores `apple`, it first predicts `a`, then `p` after `a`, then
the next `p` after `ap`, then `l` after `app`, and so on. If `app` was not seen
but `pp` or `p` was seen, it backs off until it finds a known context.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from wordfreq import iter_wordlist, word_frequency


WORD_RE = re.compile(r"[a-z]+")


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
    text = str(value).strip().lower()
    if not text:
        return None
    match = WORD_RE.fullmatch(text)
    return match.group(0) if match else None


def load_words(path: Path) -> list[tuple[str, int]]:
    df = pd.read_excel(make_readable_copy(path))
    records: list[tuple[str, int]] = []
    for row in df.itertuples(index=False):
        word = normalize_word(getattr(row, "word", None))
        if not word:
            continue
        try:
            count = int(getattr(row, "count", 1))
        except (TypeError, ValueError):
            count = 1
        if count > 0:
            records.append((word, count))
    return records


def load_wordfreq_words(
    language: str = "en",
    wordlist: str = "large",
    count_scale: int = 1_000_000,
    max_words: int | None = None,
) -> list[tuple[str, int]]:
    records: list[tuple[str, int]] = []
    for index, raw_word in enumerate(iter_wordlist(language, wordlist=wordlist)):
        if max_words is not None and index >= max_words:
            break

        word = normalize_word(raw_word)
        if not word:
            continue

        frequency = word_frequency(word, language, wordlist=wordlist, minimum=0.0)
        count = max(1, int(round(frequency * count_scale)))
        records.append((word, count))
    return records


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
    parser.add_argument("--train", type=Path, default=Path("datasets/train_word_count.xlsx"))
    parser.add_argument("--train-source", choices=["excel", "wordfreq"], default="excel")
    parser.add_argument("--wordfreq-list", choices=["small", "large", "best"], default="large")
    parser.add_argument("--wordfreq-max-words", type=int, default=None)
    parser.add_argument("--wordfreq-count-scale", type=int, default=1_000_000)
    parser.add_argument("--word", type=str, help="Word to score.")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="Fallback probability when no context matches.")
    parser.add_argument("--show-steps", action="store_true", help="Print per-character backoff details.")
    parser.add_argument("--eval-file", type=Path, help="Optional Excel file to evaluate after training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.train_source == "wordfreq":
        train_words = load_wordfreq_words(
            language="en",
            wordlist=args.wordfreq_list,
            count_scale=args.wordfreq_count_scale,
            max_words=args.wordfreq_max_words,
        )
    else:
        train_words = load_words(args.train)

    model = SimpleBackoffChainModel(epsilon=args.epsilon)
    model.fit(train_words)

    print(f"training_source: {args.train_source}")
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
