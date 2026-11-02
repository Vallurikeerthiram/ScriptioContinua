# Phase 1: Sentence-Level Word Boundary Recovery

Phase 1 establishes the foundational research workflow of the *Scriptio Continua* project. It investigates whether computational models can restore spaces and word boundaries from continuous character strings without receiving pre-existing words, tokenizers, or word embeddings.

---

## The Phase 1 Problem Formulation

In Phase 1, input text is assumed to be an isolated English sentence stripped of whitespace and punctuation:

```text
Original:    The quick brown fox jumps over the lazy dog.
Normalized:  thequickbrownfoxjumpsoverthelazydog
Target:      the quick brown fox jumps over the lazy dog
```

The model operates strictly at the character level, learning character embeddings and predicting boundary transitions for every character in the sequence.

### Tagging Paradigms

1. **Binary State (`STATE_2`)**:
   - `0`: Character is within a word.
   - `1`: Character is the terminal letter of a word (a space is inserted after this character during reconstruction).

2. **BIES Scheme (`STATE_4`)**:
   - `B`: Begin character of a multi-character word.
   - `I`: Intermediate character of a multi-character word.
   - `E`: End character of a multi-character word.
   - `S`: Single-character word (e.g., "a", "I").

---

## Directory Organization

```text
Phase 1/
|-- DATASET/                      # Data scraping and preprocessing pipeline
|   |-- pullArticles.py           # Simple Wikipedia category scraper
|   |-- Preprocessing.py          # Cleaning, normalization, and label generation
|   |-- graphs.py                 # Corpus distribution visualizer
|   |-- 8000 rows/                # Intermediate 8,000-sentence dataset split
|   `-- 18000 rows/               # Expanded 18,000-sentence dataset split
|
|-- Models/                       # Neural and CRF sequence tagger experiments
|   |-- plain DL models/          # BiLSTM, BiGRU, BiRNN, CNN without CRF
|   |-- DL models_CRF/            # Sequence taggers enhanced with Linear-Chain CRF
|   |-- 1st run/                  # Initial baseline experiments
|   `-- resul;ts/                 # Consolidated benchmark output spreadsheets
|
|-- plain DL/                     # Standalone PyTorch sequence tagging module
|   |-- train_models.py           # Training and evaluation runner
|   `-- SENT_based_split.xlsx     # Sentence train/val/test split
|
|-- dict based model/             # Algorithmic Trie baselines with phonotactic validators
|   |-- dict/                     # Word frequency dictionaries and Trie structures
|   `-- valid word finder/        # Orthographic and language model validation
|
|-- dictionary based/             # Semantic Trie engine with Part-of-Speech verification
|   |-- universal_trie.py         # Moby POS and definition-aware Trie
|   `-- dict/                     # POS and dictionary definition data
|
|-- LLM based/                    # Few-shot prompting baselines via Ollama
|   |-- type 1/                   # Direct readable-text restoration
|   |-- type 2/                   # Binary (0/1) state sequence prediction
|   |-- type 3/                   # BIES state sequence prediction
|   `-- shared/                   # Shared prompt builder, cache, and Ollama caller
|
|-- Literature survey/            # Background research papers, surveys, and notes
|-- extras/                       # Utility scripts for metrics calculation
`-- scripts/                      # Helper automation scripts
```

---

## Restorative Approaches Evaluated in Phase 1

Phase 1 benchmarks four fundamentally distinct restoration paradigms:

### 1. Character-Level Neural Sequence Taggers
- **Architectures**: Bidirectional LSTM, Bidirectional GRU, Vanilla Bidirectional RNN, and 1D-CNN.
- **Embedding**: Trainable 64-dimensional character embedding learned from scratch.
- **Feature Extraction**: Recurrent layers (hidden dimension 128) and convolutional filters capture local character n-grams and bidirectional context.

### 2. Linear-Chain Conditional Random Fields (CRF)
- Appends a trainable transition matrix $A_{i,j}$ over the BIES emission scores.
- Solves global sequence decoding via the Viterbi algorithm, enforcing transition grammar (e.g., forbidding `B` $\to$ `B` transitions without an intervening `E`).

### 3. High-Precision Semantic Trie Engines
- Non-neural deterministic baselines.
- Traverses continuous strings using prefix Trees (Tries) indexed against millions of verified words.
- Integrates Part-of-Speech (POS) tags, phonetic sequence likelihoods, and character language models to score ambiguous segmentation candidates.

### 4. Few-Shot Large Language Models (LLMs)
- Zero-training prompting baselines using local LLMs (Qwen 3.5, Gemma 4, DeepSeek-R1) via Ollama.
- Evaluates three output contracts:
  - **Type 1**: Direct text restoration (inserting spaces directly).
  - **Type 2**: Predicting 0/1 binary strings.
  - **Type 3**: Predicting BIES tag sequences.

---

## Quick Start for Phase 1

### 1. Generate the Sentence-Level Dataset
```powershell
Set-Location "Phase 1/DATASET"
python pullArticles.py
python Preprocessing.py
```

### 2. Train Neural Sequence Taggers
```powershell
Set-Location "Phase 1/plain DL"
python train_models.py
```

### 3. Run the Semantic Trie Engine
```powershell
Set-Location "Phase 1/dictionary based/dict"
python universal_trie.py
```

### 4. Run Few-Shot LLM Baselines (via Ollama)
Ensure Ollama is running locally, then execute:
```powershell
# Type 1: Direct text restoration smoke test
python "Phase 1/LLM based/type 1/run_type1.py" --model qwen3.5:4b --limit 10

# Type 2: Binary state prediction
python "Phase 1/LLM based/type 2/run_type2.py" --model qwen3.5:4b --limit 10

# Type 3: BIES state prediction
python "Phase 1/LLM based/type 3/run_type3.py" --model qwen3.5:4b --limit 10
```

---

## Submodule Documentation

- [Dataset Pipeline Documentation](DATASET/Readme.md)
- [Deep Learning Sequence Taggers](plain%20DL/readme.md)
- [CRF-Based Models Documentation](Models/DL%20models_CRF/README.md)
- [Semantic Trie Engine Documentation](dictionary%20based/Readme.md)
- [Few-Shot LLM Pipeline Documentation](LLM%20based/README.md)
- [Literature Survey and Background](Literature%20survey/README.md)
