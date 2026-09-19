# Phase 2: Multi-Sentence, Line-Oriented & Cross-Lingual Scriptio Continua

This directory contains experimental artifacts, datasets, neural sequence tagging models, CRF decoders, and cross-lingual evaluation pipelines for **Phase 2** of the Scriptio Continua research project.

> [!NOTE]
> This branch is an active **Beta / Work-in-Progress** release for Phase 2 experiments.

---

## Directory Structure

```text
Phase 2/
|-- English/
|   |-- 50_Characters/              # Line-oriented / multi-sentence experiments (50-char window)
|   |   |-- 2state_4state.py         # Standard neural sequence taggers (BiLSTM, GRU, RNN, CNN)
|   |   |-- 4state_CRF.py            # CRF-enhanced BIES sequence decoder
|   |   |-- 2000_articles_50_character_dataset_ART_ID_split_70_15_15.xlsx
|   |   |-- dataset/                 # 2000-article 50-character source dataset
|   |   `-- Output Analysis/         # Detailed prediction analysis workbooks
|   |
|   |-- Sentence_Level/             # Sentence-level reference experiments
|   |   |-- 2state_4state.py         # Sentence-wise neural sequence taggers
|   |   |-- 4state_crf.py            # Sentence-wise CRF models
|   |   |-- Sent_based_split.xlsx    # Sentence-based train/val/test splits
|   |   `-- Dataset/                 # Sorted sentence dataset with paragraph IDs
|   |
|   `-- Dataset/                    # Raw Simple Wikipedia 2000-article domain dataset
|
`-- Telugu/                         # Cross-lingual expansion to Indic continuous script
    |-- dataset/                    # Telugu sentence-level dataset (first 100 articles)
    |-- pkl/                        # Trained model weights & vocabularies
    |   |-- 2_state/                # Binary state models (BiLSTM, GRU, RNN, CNN, vocab)
    |   |-- 4_state/                # BIES 4-state models (BiLSTM, GRU, RNN, CNN, vocab)
    |   `-- 4_state_crf/            # BIES + CRF models (BiLSTM, GRU, RNN, CNN, vocab)
    |-- results_with_metrics/       # Evaluation metrics spreadsheets
    |-- results_without_metrics/    # Raw prediction outputs
    |-- telugu2-state.ipynb         # 2-state training & evaluation notebook
    |-- telugu4-state.ipynb         # 4-state BIES training & evaluation notebook
    `-- telugu4-state-crf.ipynb     # 4-state CRF training & evaluation notebook
```

---

## Key Experimental Focus Areas

### 1. English Line-Oriented (50 Characters)
Evaluates character-level word-boundary recovery on fixed-width continuous lines with potential line wraps and multi-sentence spans. Compares:
- Binary boundary state (`STATE_2`) vs. BIES boundary state (`STATE_4`).
- Plain neural architectures (BiLSTM, BiGRU, Bidirectional Vanilla RNN, CNN) vs. Linear-Chain CRF decoding.

### 2. English Sentence-Level Baseline
Reference benchmarks on sentence-bounded continuous text without line wrapping, establishing upper-bound segmentation performance under clean boundaries.

### 3. Telugu Cross-Lingual Evaluation
Evaluates the continuous script segmentation workflow on Telugu, exploring:
- Akshara and Unicode character-level boundary prediction.
- Morphological and orthographic challenges in agglutinative Indic scripts.
- Pre-trained model weights saved as pickle files (`.pkl`) ready for evaluation and inference.
