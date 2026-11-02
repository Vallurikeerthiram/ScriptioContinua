# Semantic Trie Engine and Linguistic Dictionaries

This directory contains the universal Trie engine implementation and linguistic databases supporting semantic verification in Phase 1 word segmentation.

---

## Files

- **`universal_trie.py`**: High-performance prefix tree implementation utilizing `__slots__` and dictionary branching. Incorporates strict Part-of-Speech verification, definition validation, and regex fallback matching for dates, numbers, and URLs.
- **`english_dictionary.csv`**: Comprehensive lexical database mapping English words to grammatical definitions and semantic classifications.
- **`mobypos.txt`**: Moby Part-of-Speech database containing morphological categorization for over 230,000 English words.
- **`web_words_1M.txt`**: Large-scale frequency-sorted wordlist extracted from web corpora, providing background frequency priors.
