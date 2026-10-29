"""
Syllable-level backoff pronounceability validator.

How this file works:
This file is the newer pronounceability model. Instead of working purely at the
letter level, it tries to represent a word as a sequence of syllable-like units.

Pipeline:
1. Convert a word to phonemes using `cmudict` or `g2p_en`.
2. Split the phoneme sequence into syllable-like groups.
3. Convert each syllable into a token such as `P-AH-L`.
4. Train a backoff chain model over syllable tokens.
5. Score new words using the syllable sequence probability.
6. Add a small syllable inventory familiarity score.

This file is what the current hybrid validator uses as its pronounceability
component.

Manual run examples:
`python syllable_backoff_pronounceability_validator.py --word apple --show-details`
`python syllable_backoff_pronounceability_validator.py --word aishwarya --show-details`
`python syllable_backoff_pronounceability_validator.py --eval-file "datasets/special test.xlsx"`

Example idea:
For `apple`, the module obtains phonemes, syllabifies them into something like
`AE` and `P-AH-L`, scores how likely that syllable sequence is under the trained
backoff model, checks whether those syllable tokens were familiar in training,
and combines the two scores into a pronounceability score.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import cmudict
from g2p_en import G2p

from character_backoff_language_model import load_words, normalize_word


ARPABET_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY", "IH", "IY", "OW",
    "OY", "UH", "UW",
}

COMMON_ONSETS = {
    ("B",), ("BL",), ("BR",), ("CH",), ("D",), ("DH",), ("DR",), ("F",), ("FL",), ("FR",),
    ("G",), ("GL",), ("GR",), ("HH",), ("JH",), ("K",), ("KL",), ("KR",), ("L",), ("M",),
    ("N",), ("P",), ("PL",), ("PR",), ("R",), ("S",), ("SH",), ("SK",), ("SL",), ("SM",),
    ("SN",), ("SP",), ("ST",), ("SW",), ("T",), ("TH",), ("TR",), ("V",), ("W",), ("Y",),
    ("Z",), ("ZH",), ("S", "K"), ("S", "L"), ("S", "M"), ("S", "N"), ("S", "P"), ("S", "T"),
    ("S", "W"), ("P", "L"), ("P", "R"), ("B", "L"), ("B", "R"), ("D", "R"), ("F", "L"),
    ("F", "R"), ("G", "L"), ("G", "R"), ("K", "L"), ("K", "R"), ("T", "R"), ("TH", "R"),
    ("SH", "R"),
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


def split_cluster(cluster: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if len(cluster) <= 1:
        return (), cluster

    for onset_len in (2, 1):
        if len(cluster) >= onset_len and cluster[-onset_len:] in COMMON_ONSETS:
            return cluster[:-onset_len], cluster[-onset_len:]
    return cluster[:-1], cluster[-1:]


def syllabify_phones(phones: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    vowel_positions = [index for index, phone in enumerate(phones) if is_vowel_phone(phone)]
    if not vowel_positions:
        return (phones,) if phones else ()

    syllables: list[tuple[str, ...]] = []
    start = 0

    for i, vowel_index in enumerate(vowel_positions):
        if i == len(vowel_positions) - 1:
            syllables.append(phones[start:])
            break

        next_vowel_index = vowel_positions[i + 1]
        interlude = phones[vowel_index + 1:next_vowel_index]
        coda, onset = split_cluster(interlude)
        syllable_end = vowel_index + 1 + len(coda)
        syllables.append(phones[start:syllable_end])
        start = syllable_end
        if onset:
            start = next_vowel_index - len(onset)

    return tuple(syllable for syllable in syllables if syllable)


class SyllableBackoffChainModel:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon
        self.counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.context_totals: dict[tuple[str, ...], int] = {}

    def fit(self, weighted_sequences: list[tuple[tuple[str, ...], int]]) -> None:
        self.counts = defaultdict(Counter)
        for sequence, weight in weighted_sequences:
            self._add_sequence(sequence, weight)
        self.context_totals = {
            context: sum(next_token_counts.values())
            for context, next_token_counts in self.counts.items()
        }

    def _add_sequence(self, sequence: tuple[str, ...], weight: int) -> None:
        for i, next_token in enumerate(sequence):
            context = sequence[:i]
            self.counts[context][next_token] += weight

    def probability_of_next_token(self, context: tuple[str, ...], next_token: str) -> tuple[float, tuple[str, ...]]:
        current = context
        while current:
            if current in self.counts:
                total = self.context_totals[current]
                count = self.counts[current][next_token]
                if total > 0 and count > 0:
                    return count / total, current
            current = current[1:]

        root_counts = self.counts.get((), Counter())
        root_total = self.context_totals.get((), 0)
        root_count = root_counts.get(next_token, 0)
        if root_total > 0 and root_count > 0:
            return root_count / root_total, ()
        return self.epsilon, ()

    def sequence_probability(self, sequence: tuple[str, ...]) -> tuple[float, list[dict[str, object]]]:
        probability = 1.0
        steps: list[dict[str, object]] = []
        for i, next_token in enumerate(sequence):
            context = sequence[:i]
            token_probability, used_context = self.probability_of_next_token(context, next_token)
            probability *= token_probability
            steps.append(
                {
                    "position": i,
                    "token": next_token,
                    "original_context": context,
                    "used_context": used_context,
                    "probability": token_probability,
                }
            )
        return probability, steps


class PronounceabilityValidator:
    def __init__(self, epsilon: float = 1e-8, threshold: float = 0.35) -> None:
        self.syllable_model = SyllableBackoffChainModel(epsilon=epsilon)
        self.threshold = threshold
        self.cmudict_entries = cmudict.dict()
        self.g2p = G2p()
        self.syllable_counts: Counter[str] = Counter()
        self.avg_logprob_mean = -10.0
        self.avg_logprob_std = 1.0

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

    def _syllables_for_word(self, word: str) -> tuple[tuple[str, ...], str]:
        phones, source = self._phones_for_word(word)
        if not phones:
            return ((word,),), source

        syllables = syllabify_phones(phones)
        if not syllables:
            return ((word,),), source
        return syllables, source

    def fit(self, weighted_words: list[tuple[str, int]]) -> None:
        syllable_sequences: list[tuple[tuple[str, ...], int]] = []
        self.syllable_counts = Counter()

        for word, count in weighted_words:
            syllables, _ = self._syllables_for_word(word)
            sequence = tuple("-".join(syllable) for syllable in syllables)
            if not sequence:
                continue
            syllable_sequences.append((sequence, count))
            for syllable in sequence:
                self.syllable_counts[syllable] += count

        self.syllable_model.fit(syllable_sequences)

        avg_log_probs: list[float] = []
        for sequence, count in syllable_sequences:
            probability, _ = self.syllable_model.sequence_probability(sequence)
            avg_log_prob = math.log(probability) / max(1, len(sequence))
            repeats = min(count, 20)
            avg_log_probs.extend([avg_log_prob] * repeats)

        if avg_log_probs:
            self.avg_logprob_mean = statistics.mean(avg_log_probs)
            self.avg_logprob_std = statistics.pstdev(avg_log_probs) or 1.0

    def syllable_plausibility(self, syllable_sequence: tuple[str, ...]) -> tuple[float, float]:
        probability, _ = self.syllable_model.sequence_probability(syllable_sequence)
        avg_log_prob = math.log(probability) / max(1, len(syllable_sequence))
        z_score = (avg_log_prob - self.avg_logprob_mean) / self.avg_logprob_std
        return logistic(z_score), avg_log_prob

    def syllable_inventory_score(self, syllable_sequence: tuple[str, ...]) -> float:
        if not syllable_sequence:
            return 0.0

        values = []
        for syllable in syllable_sequence:
            count = self.syllable_counts.get(syllable, 0)
            values.append(1.0 if count > 0 else 0.2)
        return sum(values) / len(values)

    def score_word(self, raw_word: str) -> dict[str, object]:
        word = normalize_word(raw_word)
        if not word:
            return {
                "word": raw_word,
                "normalized_word": None,
                "is_pronounceable": False,
                "score": 0.0,
                "syllable_score": 0.0,
                "inventory_score": 0.0,
                "avg_log_prob_per_syllable": float("-inf"),
                "syllables": [],
                "source": "invalid_input",
            }

        syllables, source = self._syllables_for_word(word)
        syllable_sequence = tuple("-".join(syllable) for syllable in syllables)
        syllable_score, avg_log_prob = self.syllable_plausibility(syllable_sequence)
        inventory_score = self.syllable_inventory_score(syllable_sequence)
        combined = 0.8 * syllable_score + 0.2 * inventory_score

        return {
            "word": raw_word,
            "normalized_word": word,
            "is_pronounceable": combined >= self.threshold,
            "score": combined,
            "syllable_score": syllable_score,
            "inventory_score": inventory_score,
            "avg_log_prob_per_syllable": avg_log_prob,
            "syllables": list(syllable_sequence),
            "source": source,
        }


def evaluate_dataset(path: Path, validator: PronounceabilityValidator) -> dict[str, float | int]:
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
    parser = argparse.ArgumentParser(description="Syllable-level backoff pronounceability validator.")
    parser.add_argument("--train", type=Path, default=Path("datasets/train_word_count.xlsx"))
    parser.add_argument("--word", type=str, help="Optional word to score.")
    parser.add_argument("--eval-file", type=Path, help="Optional Excel file to evaluate.")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--show-details", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_words = load_words(args.train)
    validator = PronounceabilityValidator(threshold=args.threshold)
    validator.fit(train_words)

    print(f"training_words: {len(train_words)}")
    print(f"threshold: {validator.threshold:.2f}")

    if args.word is not None:
        result = validator.score_word(args.word)
        print(f"word: {result['word']}")
        print(f"normalized_word: {result['normalized_word']}")
        print(f"is_pronounceable: {result['is_pronounceable']}")
        print(f"score: {result['score']:.6f}")
        print(f"syllable_score: {result['syllable_score']:.6f}")
        print(f"inventory_score: {result['inventory_score']:.6f}")
        if math.isfinite(result["avg_log_prob_per_syllable"]):
            print(f"avg_log_prob_per_syllable: {result['avg_log_prob_per_syllable']:.6f}")
        else:
            print("avg_log_prob_per_syllable: -inf")
        print(f"source: {result['source']}")

        if args.show_details:
            print(f"syllables: {', '.join(result['syllables'])}")

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
