# Dictionary-Based Semantic Trie Engine

This module provides a deterministic alternative to word segmentation using a high-precision Trie-based engine. It validates alphanumeric strings against millions of verified linguistic records to ensure semantic correctness.

## Contents

*   **`dict/universal_trie.py`**: The core engine. It manages the Trie structure, loads dictionaries, and performs semantic analysis.
*   **`dict/*.txt` / `dict/*.csv`**: Linguistic datasets used for verification, including Moby Part-of-Speech, Web Words, and English definitions.
*   **`SENT_based_split.xlsx`**: Ground truth data used to validate the Trie's segmentation performance.

---

## Methodology

### 1. High-Precision Semantic Trie (`UniversalTrie`)
Unlike standard Tries that only check for the existence of a word, this engine implements a **Strict Semantic Verification** policy:
*   **POS Verification**: A word is only "Accepted" if it is mapped to a valid Part-of-Speech tag (e.g., Noun, Verb, Adjective).
*   **Lexical Mapping**: The engine verifies word candidates against definitions to confirm semantic identity.
*   **Trie Architecture**: Uses `__slots__` and efficient dictionary-based branching to manage millions of words with minimal memory overhead.

### 2. Multi-Dataset Integration
The engine builds its internal model by merging several diverse data sources:
1.  **Lexical Definitions**: Loads definitions from `english_dictionary.csv`.
2.  **Grammatical Specificity**: Integrates Moby POS data for morphological verification.
3.  **Background Vocabulary**: Uses large-scale frequency lists (`web_words_1M.txt`) for indexing, but rejects them if they lack specific semantic metadata.

### 3. Pattern-Based Fallback
To handle entities not typically found in a dictionary, the engine includes high-precision regex patterns for:
*   **URLs/Web Links**
*   **Date Formats**
*   **Currency & Financial Values**
*   **Numerical Data**

---

## How to Run

### Interactive Analysis
Navigate to the `dict/` subfolder and run the engine:
```bash
python universal_trie.py
```
*   **Input**: Enter any string or sequence of characters.
*   **Output**: The engine will return if the string is "Accepted," its verified Part-of-Speech types, and its primary definitions.

### Automated Validation
The engine can be integrated into a word segmentation pipeline by recursively checking substrings in a "Scriptio Continua" string to identify valid semantic boundaries.
