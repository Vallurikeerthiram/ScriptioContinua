"""
Orthographic phonotactic wordlikeness validator.

This file exposes the reusable `PhonotacticWordlikenessValidator` class under
the canonical module name expected by the rest of the valid-word-finder tools.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter

from wordfreq import iter_wordlist, zipf_frequency

from character_backoff_pronounceability_validator import VOWELS, split_into_sound_chunks
from character_backoff_language_model import load_words, normalize_word


def logistic(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def zscore_to_unit(value: float, mean: float, std: float) -> float:
    return logistic((value - mean) / std)


class PhonotacticWordlikenessValidator:
    def __init__(self, max_ngram: int = 4, alpha: float = 0.1) -> None:
        self.max_ngram = max_ngram
        self.alpha = alpha
        self.ngram_counts: dict[int, Counter[str]] = {n: Counter() for n in range(1, max_ngram + 1)}
        self.ngram_totals: dict[int, int] = {}
        self.chunk_counts: Counter[str] = Counter()
        self.vocab_letters: set[str] = set()
        self.familiarity_lexicon: dict[str, float] = {}
        self.train_ngram_mean = -8.0
        self.train_ngram_std = 1.0
        self.train_chunk_mean = -6.0
        self.train_chunk_std = 1.0

    def fit(self, weighted_words: list[tuple[str, int]]) -> None:
        self.ngram_counts = {n: Counter() for n in range(1, self.max_ngram + 1)}
        self.chunk_counts = Counter()
        self.vocab_letters = set()

        for word, count in weighted_words:
            self.vocab_letters.update(word)
            padded = f"^{word}$"
            for n in range(1, self.max_ngram + 1):
                if len(padded) < n:
                    continue
                for i in range(len(padded) - n + 1):
                    self.ngram_counts[n][padded[i:i + n]] += count

            for n in range(2, 5):
                if len(word) < n:
                    continue
                for i in range(len(word) - n + 1):
                    self.chunk_counts[word[i:i + n]] += count

        self.ngram_totals = {n: sum(counter.values()) for n, counter in self.ngram_counts.items()}
        self.familiarity_lexicon = self._build_wordfreq_lexicon()
        self._fit_score_calibration(weighted_words)

    def _build_wordfreq_lexicon(self) -> dict[str, float]:
        lexicon: dict[str, float] = {}
        for raw_word in iter_wordlist("en", wordlist="large"):
            word = normalize_word(raw_word)
            if not word:
                continue
            lexicon[word] = zipf_frequency(word, "en", wordlist="large")
        return lexicon

    def _smoothed_ngram_prob(self, gram: str) -> float:
        n = len(gram)
        total = self.ngram_totals.get(n, 0)
        count = self.ngram_counts[n].get(gram, 0)
        vocab_size = max(1, len(self.vocab_letters) + 2)
        return (count + self.alpha) / (total + self.alpha * (vocab_size ** n))

    def _word_ngram_logscore(self, word: str) -> float:
        padded = f"^{word}$"
        log_sum = 0.0
        grams = 0
        for n in range(2, self.max_ngram + 1):
            if len(padded) < n:
                continue
            weight = n - 1
            for i in range(len(padded) - n + 1):
                prob = self._smoothed_ngram_prob(padded[i:i + n])
                log_sum += weight * math.log(prob)
                grams += weight
        return log_sum / max(1, grams)

    def _word_chunk_logscore(self, word: str) -> float:
        values: list[float] = []
        for n in range(2, 5):
            if len(word) < n:
                continue
            for i in range(len(word) - n + 1):
                chunk = word[i:i + n]
                count = self.chunk_counts.get(chunk, 0)
                values.append(math.log(count + 1.0))
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _shape_score(self, word: str) -> float:
        chunks = split_into_sound_chunks(word)
        if not chunks or not any(char in VOWELS for char in word):
            return 0.0

        consonant_runs = [len(chunk) for kind, chunk in chunks if kind == "C"]
        vowel_runs = [len(chunk) for kind, chunk in chunks if kind == "V"]

        score = 1.0
        if consonant_runs:
            score -= 0.18 * max(0, max(consonant_runs) - 3)
        if vowel_runs:
            score -= 0.15 * max(0, max(vowel_runs) - 2)
        if len(chunks) == 1:
            score -= 0.25
        if word.endswith("q") or (word.startswith("q") and len(word) == 1):
            score -= 0.2
        return max(0.0, min(1.0, score))

    def _familiarity_score(self, word: str) -> float:
        zipf = self.familiarity_lexicon.get(word, 0.0)
        return max(0.0, min(1.0, zipf / 8.0))

    def _fit_score_calibration(self, weighted_words: list[tuple[str, int]]) -> None:
        ngram_values: list[float] = []
        chunk_values: list[float] = []
        for word, count in weighted_words:
            ngram_score = self._word_ngram_logscore(word)
            chunk_score = self._word_chunk_logscore(word)
            repeats = min(count, 20)
            ngram_values.extend([ngram_score] * repeats)
            chunk_values.extend([chunk_score] * repeats)

        if ngram_values:
            self.train_ngram_mean = statistics.mean(ngram_values)
            self.train_ngram_std = statistics.pstdev(ngram_values) or 1.0
        if chunk_values:
            self.train_chunk_mean = statistics.mean(chunk_values)
            self.train_chunk_std = statistics.pstdev(chunk_values) or 1.0

    def score_word(self, raw_word: str) -> dict[str, float | bool | str]:
        word = normalize_word(raw_word)
        if not word:
            return {
                "word": str(raw_word),
                "score": 0.0,
                "is_wordlike": False,
                "ngram_score": 0.0,
                "chunk_score": 0.0,
                "shape_score": 0.0,
                "familiarity_score": 0.0,
            }

        raw_ngram = self._word_ngram_logscore(word)
        raw_chunk = self._word_chunk_logscore(word)
        ngram_score = zscore_to_unit(raw_ngram, self.train_ngram_mean, self.train_ngram_std)
        chunk_score = zscore_to_unit(raw_chunk, self.train_chunk_mean, self.train_chunk_std)
        shape_score = self._shape_score(word)
        familiarity_score = self._familiarity_score(word)

        combined = (
            0.45 * ngram_score
            + 0.25 * chunk_score
            + 0.15 * shape_score
            + 0.15 * familiarity_score
        )
        return {
            "word": word,
            "score": combined,
            "is_wordlike": combined >= 0.5,
            "ngram_score": ngram_score,
            "chunk_score": chunk_score,
            "shape_score": shape_score,
            "familiarity_score": familiarity_score,
        }


def evaluate_dataset(path, validator: PhonotacticWordlikenessValidator) -> dict[str, float | int]:
    words = load_words(path)
    total_rows = 0
    total_weight = 0
    score_sum = 0.0
    weighted_score_sum = 0.0
    accepted_rows = 0
    accepted_weight = 0

    for word, count in words:
        result = validator.score_word(word)
        score = float(result["score"])
        total_rows += 1
        total_weight += count
        score_sum += score
        weighted_score_sum += score * count
        if result["is_wordlike"]:
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
