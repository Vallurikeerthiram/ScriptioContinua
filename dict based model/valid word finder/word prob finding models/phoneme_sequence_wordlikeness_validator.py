"""
Phoneme-sequence wordlikeness validator.

How this file works:
This validator converts a word into phonemes using `cmudict` when possible and
`g2p_en` as fallback. It then models the phoneme sequence directly rather than
the raw letters. The score mixes:

1. Phoneme n-gram plausibility.
2. Onset/coda and consonant-vowel shape plausibility.
3. A small lexical familiarity bonus from word frequency.

This file is useful when you want a more sound-based notion of wordlikeness
than simple spelling statistics.

Manual run examples:
`python phoneme_sequence_wordlikeness_validator.py --word apple`
`python phoneme_sequence_wordlikeness_validator.py --word drindle`
`python phoneme_sequence_wordlikeness_validator.py --eval-file "datasets/special test.xlsx"`

Example idea:
For `apple`, the script derives phonemes such as `AE P AH L`, computes how
typical the phoneme transitions are, checks whether the onset and coda patterns
look familiar, adds a small lexical bonus, and returns a final wordlikeness
score.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter
from pathlib import Path

import cmudict
from g2p_en import G2p
from wordfreq import zipf_frequency

from character_backoff_language_model import load_words, normalize_word


ARPABET_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY", "IH", "IY", "OW",
    "OY", "UH", "UW",
}


def strip_stress(phone: str) -> str:
    return "".join(ch for ch in phone if not ch.isdigit())


def is_vowel_phone(phone: str) -> bool:
    return strip_stress(phone) in ARPABET_VOWELS


def logistic(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class PhonemeWordlikenessValidator:
    def __init__(self, max_ngram: int = 3, alpha: float = 0.1, threshold: float = 0.45) -> None:
        self.max_ngram = max_ngram
        self.alpha = alpha
        self.threshold = threshold
        self.cmudict_entries = cmudict.dict()
        self.g2p = G2p()
        self.phone_ngram_counts: dict[int, Counter[tuple[str, ...]]] = {
            n: Counter() for n in range(1, max_ngram + 1)
        }
        self.phone_ngram_totals: dict[int, int] = {}
        self.onset_counts: Counter[tuple[str, ...]] = Counter()
        self.coda_counts: Counter[tuple[str, ...]] = Counter()
        self.pattern_counts: Counter[str] = Counter()
        self.phone_vocab: set[str] = set()
        self.train_phone_mean = -10.0
        self.train_phone_std = 1.0

    def _phones_for_word(self, word: str) -> tuple[tuple[str, ...] | None, str]:
        pronunciations = self.cmudict_entries.get(word)
        if pronunciations:
            phones = tuple(strip_stress(phone) for phone in pronunciations[0])
            return phones, "cmudict"

        try:
            generated = [
                strip_stress(phone)
                for phone in self.g2p(word)
                if phone and phone != " " and any(ch.isalpha() for ch in phone)
            ]
        except Exception:
            generated = []

        if generated:
            return tuple(generated), "g2p"
        return None, "fallback_letters"

    def fit(self, weighted_words: list[tuple[str, int]]) -> None:
        self.phone_ngram_counts = {n: Counter() for n in range(1, self.max_ngram + 1)}
        self.onset_counts = Counter()
        self.coda_counts = Counter()
        self.pattern_counts = Counter()
        self.phone_vocab = set()

        phone_scores: list[float] = []
        for word, count in weighted_words:
            phones, _ = self._phones_for_word(word)
            if not phones:
                continue

            self.phone_vocab.update(phones)
            padded = ("<s>",) + phones + ("</s>",)
            for n in range(1, self.max_ngram + 1):
                if len(padded) < n:
                    continue
                for i in range(len(padded) - n + 1):
                    self.phone_ngram_counts[n][padded[i:i + n]] += count

            onset, coda, pattern = self._extract_shape_features(phones)
            self.onset_counts[onset] += count
            self.coda_counts[coda] += count
            self.pattern_counts[pattern] += count

        self.phone_ngram_totals = {n: sum(counter.values()) for n, counter in self.phone_ngram_counts.items()}

        for word, count in weighted_words:
            phones, _ = self._phones_for_word(word)
            if not phones:
                continue
            score = self._raw_phone_ngram_score(phones)
            repeats = min(count, 20)
            phone_scores.extend([score] * repeats)

        if phone_scores:
            self.train_phone_mean = statistics.mean(phone_scores)
            self.train_phone_std = statistics.pstdev(phone_scores) or 1.0

    def _extract_shape_features(self, phones: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...], str]:
        first_vowel = None
        last_vowel = None
        pattern_parts: list[str] = []

        for index, phone in enumerate(phones):
            kind = "V" if is_vowel_phone(phone) else "C"
            pattern_parts.append(kind)
            if kind == "V":
                if first_vowel is None:
                    first_vowel = index
                last_vowel = index

        if first_vowel is None:
            return phones, (), "".join(pattern_parts)

        onset = phones[:first_vowel]
        coda = phones[last_vowel + 1:]
        return onset, coda, "".join(pattern_parts)

    def _smoothed_phone_prob(self, gram: tuple[str, ...]) -> float:
        n = len(gram)
        total = self.phone_ngram_totals.get(n, 0)
        count = self.phone_ngram_counts[n].get(gram, 0)
        vocab_size = max(1, len(self.phone_vocab) + 2)
        return (count + self.alpha) / (total + self.alpha * (vocab_size ** n))

    def _raw_phone_ngram_score(self, phones: tuple[str, ...]) -> float:
        padded = ("<s>",) + phones + ("</s>",)
        log_sum = 0.0
        grams = 0
        for n in range(2, self.max_ngram + 1):
            if len(padded) < n:
                continue
            weight = n - 1
            for i in range(len(padded) - n + 1):
                prob = self._smoothed_phone_prob(padded[i:i + n])
                log_sum += weight * math.log(prob)
                grams += weight
        return log_sum / max(1, grams)

    def _phone_score(self, phones: tuple[str, ...]) -> float:
        raw = self._raw_phone_ngram_score(phones)
        z = (raw - self.train_phone_mean) / self.train_phone_std
        return logistic(z)

    def _shape_score(self, phones: tuple[str, ...]) -> float:
        onset, coda, pattern = self._extract_shape_features(phones)
        if not any(is_vowel_phone(phone) for phone in phones):
            return 0.0

        score = 0.0
        score += 0.4 if self.onset_counts.get(onset, 0) > 0 else max(0.0, 0.4 - 0.1 * max(0, len(onset) - 2))
        score += 0.3 if self.coda_counts.get(coda, 0) > 0 else max(0.0, 0.3 - 0.1 * max(0, len(coda) - 2))
        score += 0.3 if self.pattern_counts.get(pattern, 0) > 0 else 0.1
        return max(0.0, min(1.0, score))

    def _lexical_bonus(self, word: str) -> float:
        return max(0.0, min(1.0, zipf_frequency(word, "en", wordlist="large") / 8.0))

    def score_word(self, raw_word: str) -> dict[str, float | bool | str]:
        word = normalize_word(raw_word)
        if not word:
            return {
                "word": str(raw_word),
                "score": 0.0,
                "is_wordlike": False,
                "phone_score": 0.0,
                "shape_score": 0.0,
                "lexical_bonus": 0.0,
                "source": "invalid_input",
            }

        phones, source = self._phones_for_word(word)
        if phones:
            phone_score = self._phone_score(phones)
            shape_score = self._shape_score(phones)
        else:
            phone_score = 0.15 if any(ch in "aeiouy" for ch in word) else 0.0
            shape_score = 0.35 if any(ch in "aeiouy" for ch in word) else 0.0
            source = "fallback_letters"

        lexical_bonus = self._lexical_bonus(word)
        score = 0.6 * phone_score + 0.25 * shape_score + 0.15 * lexical_bonus
        return {
            "word": word,
            "score": score,
            "is_wordlike": score >= self.threshold,
            "phone_score": phone_score,
            "shape_score": shape_score,
            "lexical_bonus": lexical_bonus,
            "source": source,
        }


def evaluate_dataset(path: Path, validator: PhonemeWordlikenessValidator) -> dict[str, float | int]:
    words = load_words(path)
    total_rows = 0
    total_weight = 0
    score_sum = 0.0
    weighted_score_sum = 0.0

    for word, count in words:
        result = validator.score_word(word)
        score = float(result["score"])
        total_rows += 1
        total_weight += count
        score_sum += score
        weighted_score_sum += score * count

    return {
        "rows": total_rows,
        "total_weight": total_weight,
        "average_score": score_sum / total_rows if total_rows else 0.0,
        "weighted_average_score": weighted_score_sum / total_weight if total_weight else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phoneme-based wordlikeness validator.")
    parser.add_argument("--train", type=Path, default=Path("datasets/train_word_count.xlsx"))
    parser.add_argument("--word", type=str, help="Optional word to score.")
    parser.add_argument("--eval-file", type=Path, help="Optional Excel file to evaluate.")
    parser.add_argument("--score-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_words = load_words(args.train)
    validator = PhonemeWordlikenessValidator()
    validator.fit(train_words)

    if args.word is not None:
        result = validator.score_word(args.word)
        if args.score_only:
            print(f"{result['score']:.6f}")
            return
        print(f"training_words: {len(train_words)}")
        print(f"word: {result['word']}")
        print(f"is_wordlike: {result['is_wordlike']}")
        print(f"score: {result['score']:.6f}")
        print(f"phone_score: {result['phone_score']:.6f}")
        print(f"shape_score: {result['shape_score']:.6f}")
        print(f"lexical_bonus: {result['lexical_bonus']:.6f}")
        print(f"source: {result['source']}")
        return

    if args.eval_file is not None:
        metrics = evaluate_dataset(args.eval_file, validator)
        print(f"training_words: {len(train_words)}")
        print(f"eval_file: {args.eval_file}")
        print(f"rows: {metrics['rows']}")
        print(f"total_weight: {metrics['total_weight']}")
        print(f"average_score: {metrics['average_score']:.6f}")
        print(f"weighted_average_score: {metrics['weighted_average_score']:.6f}")


if __name__ == "__main__":
    main()
