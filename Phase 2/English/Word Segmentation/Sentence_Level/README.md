# Sentence-Level Word Segmentation Baseline

This directory contains the sentence-bounded word segmentation baseline for Phase 2.

---

## Role in the Benchmark

While the 50-character track evaluates realistic line-truncated text, the sentence-level track operates on complete, isolated English sentences. It serves as an experimental control, establishing the upper performance bound when every input string represents a grammatically complete unit without cross-line hyphenation or word boundary clipping.

---

## Directory Contents

```text
Sentence_Level/
|-- 2state_4state.py         # Plain neural tagger implementation (BiLSTM, GRU, RNN, CNN)
|-- 2state_4state.xlsx       # Benchmark results for 2-state and 4-state models
|-- 4state_crf.py            # CRF sequence tagger implementation
|-- 4state_crf.xlsx          # Benchmark results for 4-state + CRF models
|-- Sent_based_split.xlsx    # Sentence dataset split into TRAIN, VAL, and TEST sheets
`-- Dataset/
    `-- 2000_articles_sentences_final_sorted_with_para_id.xlsx
                             # Full sentence corpus with article and paragraph IDs
```

---

## Dataset Characteristics

- **Total Articles**: 2,000 articles
- **Total Sentences**: 20,747 sentence records
- **Total Continuous Characters**: 1,571,692 characters
- **Source Words**: 325,790 word tokens
- **Schema**:
  - `ART_ID`: Article identifier
  - `PARA_ID`: Paragraph identifier within the article
  - `SENT_ID`: Sentence identifier
  - `SENT_ORI`: Original sentence text with punctuation and whitespace
  - `SCRIPT_CONTIN`: Normalized continuous alphanumeric characters
  - `STATE_2`: Binary boundary labels
  - `STATE_4`: BIES boundary labels

---

## Running Experiments

### Train Plain Sequence Taggers
```powershell
python "Phase 2/English/Word Segmentation/Sentence_Level/2state_4state.py"
```

### Train CRF-Enhanced Sequence Decoders
```powershell
python "Phase 2/English/Word Segmentation/Sentence_Level/4state_crf.py"
```

Results are logged directly to `2state_4state.xlsx` and `4state_crf.xlsx`.
