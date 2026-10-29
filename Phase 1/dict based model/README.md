# Dictionary-Based Word Segmentation

This folder implements word segmentation using dictionary-driven techniques. It utilizes a Trie data structure for efficient word lookup and various validators to estimate the likelihood of character sequences being valid words.

## Subdirectories

- **`dict/`**: Contains various word lists, dictionaries, and the Trie implementation.
- **`valid word finder/`**: Implements the logic for validating words based on orthographic, phonotactic, and linguistic models.

## Methodology

The dictionary-based approach typically involves:
1.  Building a Trie from a large corpus of English words.
2.  Traversing the scriptio continua text and identifying potential word boundaries using the Trie.
3.  Using validation models (trained on character frequencies and phoneme sequences) to disambiguate or validate word choices.
