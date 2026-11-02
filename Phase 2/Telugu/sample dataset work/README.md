# Telugu Sample Dataset Prototyping Track

This directory contains the initial exploratory prototype for the Telugu *scriptio continua* word segmentation track, evaluated on a 100-article corpus slice.

---

## Purpose and Scope

Before scaling up to the complete 14,480-sentence corpus, this module served to:
1. Validate Telugu akshara and Unicode character tokenization pipelines.
2. Confirm that character-level neural taggers (BiLSTM, GRU, RNN, CNN) successfully converge on Indic text.
3. Establish the first baseline comparisons between 2-state binary tagging, 4-state BIES tagging, and CRF transition decoding.
4. Verify that text reconstruction evaluation routines (BLEU, ROUGE, BERTScore) handle Telugu Unicode properly without glyph dissociation.

---

## Directory Organization

```text
sample dataset work/
|-- dataset/
|   `-- telugu_sentence_level_first_100_articles_split (1).xlsx
|                                       # First 100 articles split into TRAIN, VAL, TEST
|-- pkl/                                # Saved model weights for the 100-article models
|   |-- 2_state/                        # BiLSTM, CNN, GRU, RNN, and vocab .pkl files
|   |-- 4_state/                        # BiLSTM, CNN, GRU, RNN, and vocab .pkl files
|   `-- 4_state_crf/                    # BiLSTM, CNN, GRU, RNN, and vocab .pkl files
|
|-- results_without_metrics/            # Raw prediction output spreadsheets
|   |-- 2state.xlsx
|   |-- 4state.xlsx
|   `-- 4state_crf.xlsx
|
|-- results_with_metrics/               # Full metric spreadsheets
|   |-- telugu_2state_scores.xlsx
|   |-- telugu_4state_scores.xlsx
|   `-- telugu_4state_crf_scores.xlsx
|
|-- telugu2-state.ipynb                 # Interactive notebook for 2-state prototyping
|-- telugu4-state.ipynb                 # Interactive notebook for 4-state prototyping
`-- telugu4-state-crf.ipynb             # Interactive notebook for 4-state CRF prototyping
```

---

## Transition to Production Complete Dataset

Having validated the architecture and convergence behavior on this 100-article sample, the experiment was scaled up to the full corpus. Researchers seeking production-grade model checkpoints, larger vocabulary coverage, and definitive benchmarks should refer to:
- [`Phase 2/Telugu/Complete Dataset/`](../Complete%20Dataset/README.md)
