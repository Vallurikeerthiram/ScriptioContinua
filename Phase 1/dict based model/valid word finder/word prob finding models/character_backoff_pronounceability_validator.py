"""
Character-based pronounceability validator.

How this file works:
This validator treats a word as a sequence of letters and estimates whether it
looks pronounceable in English-like spelling. It uses two signals:

1. Character backoff probability:
   A character-level language model estimates how plausible the letter sequence
   is based on training words.
2. Vowel/consonant chunk plausibility:
   The word is split into vowel and consonant chunks such as `str` + `ea` + `m`.
   Chunks seen during training are rewarded, and unusual long unseen chunks are
   penalized.

The final score is a weighted mix of the character score and the chunk score.
This is the older pronounceability approach that existed before the
syllable-level version was introduced.

Manual run examples:
`python character_backoff_pronounceability_validator.py --word apple`
`python character_backoff_pronounceability_validator.py --word blost --show-details`
`python character_backoff_pronounceability_validator.py --eval-file "datasets/special test.xlsx"`

Example idea:
For `apple`, the file first scores the letter sequence with the character
backoff model, then splits the word into chunks like `a` + `ppl` + `e`, checks
whether those chunk patterns were seen in training, and combines both signals
into one pronounceability score.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter
from pathlib import Path

from character_backoff_language_model import DEFAULT_TRAIN_PATH, SimpleBackoffChainModel, load_words, normalize_word


VOWELS = set("aeiouy")


def split_into_sound_chunks(word: str) -> list[tuple[str, str]]:
    if not word:
        return []

    chunks: list[tuple[str, str]] = []
    current_type = "V" if word[0] in VOWELS else "C"
    current_chars = [word[0]]

    for char in word[1:]:
        chunk_type = "V" if char in VOWELS else "C"
        if chunk_type == current_type:
            current_chars.append(char)
        else:
            chunks.append((current_type, "".join(current_chars)))
            current_type = chunk_type
            current_chars = [char]

    chunks.append((current_type, "".join(current_chars)))
    return chunks


def logistic(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class CharacterPronounceabilityValidator:
    def __init__(self, epsilon: float = 1e-8, threshold: float = 0.35) -> None:
        self.char_model = SimpleBackoffChainModel(epsilon=epsilon)
        self.threshold = threshold
        self.vowel_chunks: Counter[str] = Counter()
        self.consonant_chunks: Counter[str] = Counter()
        self.max_seen_consonant_run = 0
        self.avg_logprob_mean = -10.0
        self.avg_logprob_std = 1.0

    def fit(self, weighted_words: list[tuple[str, int]]) -> None:
        self.char_model.fit(weighted_words)
        self.vowel_chunks = Counter()
        self.consonant_chunks = Counter()
        self.max_seen_consonant_run = 0

        avg_log_probs: list[float] = []
        for word, count in weighted_words:
            chunks = split_into_sound_chunks(word)
            for chunk_type, chunk in chunks:
                if chunk_type == "V":
                    self.vowel_chunks[chunk] += count
                else:
                    self.consonant_chunks[chunk] += count
                    self.max_seen_consonant_run = max(self.max_seen_consonant_run, len(chunk))

            probability, _ = self.char_model.word_probability(word)
            avg_log_prob = math.log(probability) / max(1, len(word))
            repeats = min(count, 20)
            avg_log_probs.extend([avg_log_prob] * repeats)

        if avg_log_probs:
            self.avg_logprob_mean = statistics.mean(avg_log_probs)
            self.avg_logprob_std = statistics.pstdev(avg_log_probs) or 1.0

    def char_plausibility(self, word: str) -> tuple[float, float]:
        probability, _ = self.char_model.word_probability(word)
        avg_log_prob = math.log(probability) / max(1, len(word))
        z_score = (avg_log_prob - self.avg_logprob_mean) / self.avg_logprob_std
        return logistic(z_score), avg_log_prob

    def chunk_plausibility(self, word: str) -> tuple[float, list[dict[str, object]]]:
        chunks = split_into_sound_chunks(word)
        if not chunks:
            return 0.0, []

        details: list[dict[str, object]] = []
        chunk_scores: list[float] = []

        has_vowel = any(chunk_type == "V" for chunk_type, _ in chunks)
        if not has_vowel:
            return 0.0, [{"type": "rule", "chunk": "", "score": 0.0, "reason": "no_vowel_chunk"}]

        for chunk_type, chunk in chunks:
            if chunk_type == "V":
                count = self.vowel_chunks.get(chunk, 0)
                if count > 0:
                    score = 1.0
                    reason = "seen_vowel_chunk"
                elif len(chunk) == 1:
                    score = 0.45
                    reason = "unseen_single_vowel"
                else:
                    score = 0.15
                    reason = "unseen_vowel_cluster"
            else:
                count = self.consonant_chunks.get(chunk, 0)
                if count > 0:
                    score = 1.0
                    reason = "seen_consonant_chunk"
                elif len(chunk) <= 2:
                    score = 0.35
                    reason = "short_unseen_consonant_chunk"
                else:
                    score = 0.05
                    reason = "long_unseen_consonant_chunk"

                if len(chunk) > self.max_seen_consonant_run:
                    score *= 0.2
                    reason = "too_long_consonant_run"

            chunk_scores.append(score)
            details.append(
                {
                    "type": chunk_type,
                    "chunk": chunk,
                    "score": score,
                    "reason": reason,
                }
            )

        return sum(chunk_scores) / len(chunk_scores), details

    def score_word(self, raw_word: str) -> dict[str, object]:
        word = normalize_word(raw_word)
        if not word:
            return {
                "word": raw_word,
                "normalized_word": None,
                "is_pronounceable": False,
                "score": 0.0,
                "char_score": 0.0,
                "chunk_score": 0.0,
                "avg_log_prob_per_char": float("-inf"),
                "details": [],
            }

        char_score, avg_log_prob = self.char_plausibility(word)
        chunk_score, details = self.chunk_plausibility(word)
        combined = 0.75 * char_score + 0.25 * chunk_score
        return {
            "word": raw_word,
            "normalized_word": word,
            "is_pronounceable": combined >= self.threshold,
            "score": combined,
            "char_score": char_score,
            "chunk_score": chunk_score,
            "avg_log_prob_per_char": avg_log_prob,
            "details": details,
        }


def evaluate_dataset(path: Path, validator: CharacterPronounceabilityValidator) -> dict[str, float | int]:
    words = load_words(path)
    total_rows = 0
    total_weight = 0
    accepted_rows = 0
    accepted_weight = 0
    score_sum = 0.0
    weighted_score_sum = 0.0

    for word, count in words:
        result = validator.score_word(word)
        total_rows += 1
        total_weight += count
        score_sum += result["score"]
        weighted_score_sum += result["score"] * count
        if result["is_pronounceable"]:
            accepted_rows += 1
            accepted_weight += count

    return {
        "rows": total_rows,
        "total_weight": total_weight,
        "accepted_row_fraction": accepted_rows / total_rows if total_rows else 0.0,
        "accepted_weight_fraction": accepted_weight / total_weight if total_weight else 0.0,
        "average_score": score_sum / total_rows if total_rows else 0.0,
        "weighted_average_score": weighted_score_sum / total_weight if total_weight else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Character-level pronounceability validator.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--train-sheet", type=str, default=None)
    parser.add_argument("--word", type=str, help="Optional word to score.")
    parser.add_argument("--eval-file", type=Path, help="Optional Excel file to evaluate.")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--show-details", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_words = load_words(args.train, sheet_name=args.train_sheet)
    validator = CharacterPronounceabilityValidator(threshold=args.threshold)
    validator.fit(train_words)

    print(f"training_words: {len(train_words)}")
    print(f"threshold: {validator.threshold:.2f}")

    if args.word is not None:
        result = validator.score_word(args.word)
        print(f"word: {result['word']}")
        print(f"normalized_word: {result['normalized_word']}")
        print(f"is_pronounceable: {result['is_pronounceable']}")
        print(f"score: {result['score']:.6f}")
        print(f"char_score: {result['char_score']:.6f}")
        print(f"chunk_score: {result['chunk_score']:.6f}")
        if math.isfinite(result['avg_log_prob_per_char']):
            print(f"avg_log_prob_per_char: {result['avg_log_prob_per_char']:.6f}")
        else:
            print("avg_log_prob_per_char: -inf")

        if args.show_details:
            print("details:")
            for detail in result["details"]:
                print(
                    f"  type={detail['type']} chunk='{detail['chunk']}' "
                    f"score={detail['score']:.3f} reason={detail['reason']}"
                )

    if args.eval_file is not None:
        metrics = evaluate_dataset(args.eval_file, validator)
        print(f"eval_file: {args.eval_file}")
        print(f"rows: {metrics['rows']}")
        print(f"total_weight: {metrics['total_weight']}")
        print(f"accepted_row_fraction: {metrics['accepted_row_fraction']:.6f}")
        print(f"accepted_weight_fraction: {metrics['accepted_weight_fraction']:.6f}")
        print(f"average_score: {metrics['average_score']:.6f}")
        print(f"weighted_average_score: {metrics['weighted_average_score']:.6f}")


if __name__ == "__main__":
    main()
