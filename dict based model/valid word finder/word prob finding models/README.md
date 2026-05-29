# Word Probability Finding Models

This folder contains the core Python implementations for various word-validity and wordlikeness models.

## Models

- **`character_backoff_language_model.py`**: Base character-level backoff language model.
- **`character_backoff_pronounceability_validator.py`**: Character-based pronounceability scoring.
- **`syllable_backoff_pronounceability_validator.py`**: Syllable-level pronounceability scoring using phonemes.
- **`lexical_frequency_validator.py`**: Lexicon-based validation using Zipf frequency.
- **`hybrid_lexical_character_pronounceability_validator.py`**: Combines lexical familiarity with character-based pronounceability.
- **`hybrid_lexical_pronounceability_validator.py`**: Combines lexical familiarity with syllable-based pronounceability.
- **`phoneme_sequence_wordlikeness_validator.py`**: Sound-based wordlikeness model using phoneme n-grams.
- **`orthographic_phonotactic_validator.py`**: Spelling-pattern-based wordlikeness model.
