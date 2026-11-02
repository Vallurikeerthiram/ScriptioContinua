# English Word Segmentation

This directory contains experiments, datasets, and analysis for character-level word-boundary recovery in English *scriptio continua*.

---

## Research Tracks

English word segmentation in Phase 2 is evaluated across two distinct structural settings:

```text
Word Segmentation/
|-- 50_Characters/              # Line-oriented, fixed-width window evaluation
|   |-- 2state_4state.py        # Plain neural models (BiLSTM, GRU, RNN, CNN)
|   |-- 4state_CRF.py           # CRF-constrained BIES decoding
|   |-- Output Analysis/        # Per-model detailed prediction workbooks
|   `-- dataset/                # 2,000-article 50-character slice dataset
|
|-- Sentence_Level/             # Sentence-level reference baseline
|   |-- 2state_4state.py        # Sentence-bounded neural taggers
|   |-- 4state_crf.py           # Sentence-bounded CRF decoders
|   |-- Sent_based_split.xlsx   # Ground-truth sentence split
|   `-- Dataset/                # Paragraph-sorted sentence corpus
|
`-- Dataset/
    `-- 2000 articles simple_wikipedia_random_domain_dataset.xlsx
```

---

## Track Comparison

| Dimension | 50-Character Track | Sentence-Level Track |
| :--- | :--- | :--- |
| **Input Unit** | Fixed 50-character chunks with line wrapping | Isolated, complete sentence strings |
| **Boundary Characteristics** | Words may be truncated or wrapped across lines; partial words at start/end | Clean sentence boundaries; all words are grammatically complete |
| **Realistic Context** | Closer to archival document lines, inscriptions, or OCR columns | Controlled theoretical baseline establishing the clean upper bound |
| **Labeling Schemes** | Binary (`STATE_2`) and BIES (`STATE_4`) | Binary (`STATE_2`) and BIES (`STATE_4`) |
| **Decoding Approaches** | Plain neural classification and Linear-Chain CRF | Plain neural classification and Linear-Chain CRF |

Refer to the respective subdirectories for technical specifications, scripts, and output workbooks:
- [`50_Characters/`](50_Characters/README.md)
- [`Sentence_Level/`](Sentence_Level/README.md)
