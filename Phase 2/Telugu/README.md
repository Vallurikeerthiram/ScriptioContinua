# Telugu Scriptio Continua Research Track

This directory contains datasets, sequence tagging models, CRF decoders, and detailed linguistic analysis for restoring word boundaries in continuous Telugu script (*scriptio continua*).

---

## Linguistic and Technical Context

Telugu is a Dravidian language with an alphasyllabic (abugida) writing system and rich agglutinative morphology. Applying *scriptio continua* restoration to Telugu presents linguistic challenges fundamentally distinct from English:

1. **Akshara vs. Character Representation**:
   - In English, a character is simply a single ASCII letter (e.g., `a`, `b`, `c`).
   - In Telugu, the core orthographic unit is the **akshara** (syllabic unit). An akshara consists of an independent vowel or a base consonant optionally combined with dependent vowel signs (*matras*), virama (*halant* / `్`), anusvara (`ం`), and consonant conjuncts (*vattulu* / సంయుక్తాక్షరాలు, ద్విత్వాక్షరాలు).
   - Splitting strictly by raw Unicode code points severs vowel modifiers and conjunct markers from their host consonants, corrupting the script. Hence, this framework enforces akshara-aware tokenization and boundary prediction.

2. **Agglutinative Morphology and Sandhi**:
   - Telugu forms complex compound words through *sandhi* (phonetic fusion at morpheme boundaries) and inflectional agglutination (case markers, postpositions, and pronominal suffixes appended directly to nominal stems).
   - Identifying where one word ends and the next begins requires learning complex character and akshara co-occurrence patterns.

---

## Directory Overview

```text
Telugu/
|-- Complete Dataset/               # Large-scale production benchmark (5 lakh aksharas)
|   |-- dataset/                    # 70/15/15 sentence-level split dataset
|   |-- Detailed Results and Analysis/ # Character-level, complexity, and frequency workbooks
|   |-- model predictions without metrics/ # Raw model output spreadsheets
|   |-- notebooks and code files/   # Jupyter notebooks and saved PyTorch .pkl models
|   |-- results summaries/          # Consolidated evaluation metrics
|   |-- count_unique_telugu.py      # Akshara extraction and frequency analysis script
|   `-- telugu_akshara_frequencies.xlsx # Complete akshara frequency workbook
|
`-- sample dataset work/            # Initial 100-article prototyping track
    |-- dataset/                    # Sample dataset split
    |-- pkl/                        # Saved model weights for sample runs
    |-- results_with_metrics/       # Metric workbooks for 2-state, 4-state, 4-state CRF
    |-- results_without_metrics/    # Raw prediction outputs
    |-- telugu2-state.ipynb         # 2-state training notebook
    |-- telugu4-state.ipynb         # 4-state BIES notebook
    `-- telugu4-state-crf.ipynb     # 4-state CRF notebook
```

---

## Methodological Summary

### 1. Labeling Schemes
- **2-State (Binary)**:
  - `0`: Non-boundary akshara.
  - `1`: Word-terminal akshara (a space belongs immediately after this unit).
- **4-State (BIES)**:
  - `B` (*Begin*): Initial akshara of a multi-akshara word.
  - `I` (*Inside*): Internal akshara of a multi-akshara word.
  - `E` (*End*): Final akshara of a multi-akshara word.
  - `S` (*Single*): Standalone single-akshara word.

### 2. Model Architectures
- **BiLSTM**: Bidirectional LSTM modeling bilateral contextual character dependencies.
- **BiGRU**: Bidirectional Gated Recurrent Unit offering fast convergence and parameter efficiency.
- **BiRNN**: Standard bidirectional Elman recurrent network baseline.
- **CNN**: 1D Convolutional Neural Network capturing local character n-gram receptive fields.
- **Linear-Chain CRF**: Transition matrix layer enforcing valid BIES sequence paths during 4-state decoding.

---

## Performance Highlights on Complete Dataset (Test Split)

| State Paradigm | Architecture | Macro F1 | Accuracy | Precision | Recall | BLEU | BERTScore F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Telugu 2-State** | **GRU** | **0.7574** | **0.8150** | 0.7892 | 0.7538 | 0.2354 | 0.8444 |
| **Telugu 2-State** | **BiLSTM** | 0.7522 | 0.8091 | 0.7792 | 0.7485 | 0.2458 | 0.8486 |
| **Telugu 2-State** | **CNN** | 0.6417 | 0.7459 | 0.7009 | 0.6401 | 0.1802 | 0.8756 |
| **Telugu 2-State** | **RNN** | 0.6350 | 0.7378 | 0.6961 | 0.6323 | 0.1704 | 0.8747 |
| **Telugu 4-State** | **GRU** | **0.6807** | **0.7257** | 0.6928 | 0.6856 | 0.2673 | 0.8291 |
| **Telugu 4-State** | **BiLSTM** | 0.6698 | 0.7157 | 0.6806 | 0.6756 | 0.2770 | 0.8393 |
| **Telugu 4-State** | **RNN** | 0.5839 | 0.6333 | 0.6070 | 0.5896 | 0.2353 | 0.8389 |
| **Telugu 4-State** | **CNN** | 0.5565 | 0.6051 | 0.5692 | 0.5614 | **0.3121** | **0.8895** |
| **Telugu 4-State + CRF** | **GRU** | **0.6239** | **0.6674** | 0.6280 | 0.6300 | 0.2530 | 0.8244 |
| **Telugu 4-State + CRF** | **BiLSTM** | 0.5900 | 0.6288 | 0.5946 | 0.5982 | 0.2512 | 0.8239 |
| **Telugu 4-State + CRF** | **RNN** | 0.5495 | 0.5878 | 0.5531 | 0.5550 | 0.2715 | 0.8722 |
| **Telugu 4-State + CRF** | **CNN** | 0.3542 | 0.3897 | 0.3550 | 0.3623 | 0.1329 | 0.7518 |

---

## Subdirectory Navigation

- [`Complete Dataset/`](Complete%20Dataset/README.md): Production corpus, 12 model evaluations, akshara analysis, and saved model weights.
- [`sample dataset work/`](sample%20dataset%20work/README.md): 100-article prototype track, initial notebooks, and exploratory runs.
