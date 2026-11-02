# 50-Character Window Line-Level Word Segmentation

This directory contains experimental code, datasets, and detailed prediction logs for word segmentation evaluated on fixed-length 50-character windows.

---

## Motivation

In real-world manuscripts, stone inscriptions, and unsegmented OCR scans, text is organized into physical lines or reading columns rather than pre-isolated sentences. A single 50-character slice can:
- Begin in the middle of a word;
- End in the middle of a word;
- Contain sentence boundaries without punctuation clues.

This track tests whether character-level neural taggers and sequence decoders can accurately predict word boundaries when sentence context is truncated.

---

## Directory Contents

```text
50_Characters/
|-- 2000_articles_50_character_dataset_ART_ID_split_70_15_15.xlsx  # 70/15/15 article-split dataset
|-- 2state_4state.py                                               # Plain neural tagger training script
|-- 2state_4state.xlsx                                             # Results for plain 2-state and 4-state models
|-- 4state_CRF.py                                                  # CRF-enhanced BIES tagger training script
|-- 4state+crf.xlsx                                                # Results for 4-state + CRF models
|-- dataset/
|   `-- 2000_articles_50_character_dataset.xlsx                    # Complete unsplit 50-character dataset
`-- Output Analysis/
    |-- state2_line_level_models_detailed_results.xlsx             # Per-sample predictions for 2-state models
    |-- state4_line_level_models_detailed_results.xlsx             # Per-sample predictions for 4-state models
    `-- state4_crf_line_level_models_detailed_results.xlsx         # Per-sample predictions for 4-state CRF
```

---

## Modeling Workflow

### 1. Tagging Schemes
- **`STATE_2`**: Binary character classification where `0` denotes an internal character and `1` marks the final character of a word.
- **`STATE_4` (BIES)**: Four-state classification:
  - `B`: Begin character of a multi-character word.
  - `I`: Inside character of a multi-character word.
  - `E`: End character of a multi-character word.
  - `S`: Single-character word.

### 2. Neural Architectures
- **BiLSTM**: Bidirectional Long Short-Term Memory network capturing forward and backward character sequences.
- **BiGRU**: Bidirectional Gated Recurrent Unit offering computational efficiency with gated memory.
- **BiRNN**: Standard bidirectional Elman recurrent network.
- **CNN**: 1D Convolutional Neural Network with kernel size 3 detecting local character n-grams.

### 3. CRF Sequence Layer
`4state_CRF.py` appends a Linear-Chain CRF on top of each neural encoder to model state transition likelihoods, preventing invalid sequences such as `B` directly followed by `B` without intervening `E` or `I`.

---

## How to Run

### Plain Neural Taggers (2-State and 4-State)
```powershell
python "Phase 2/English/Word Segmentation/50_Characters/2state_4state.py"
```

### CRF-Enhanced Sequence Taggers (4-State BIES)
```powershell
python "Phase 2/English/Word Segmentation/50_Characters/4state_CRF.py"
```

*Note: GPU execution via CUDA is strongly recommended for the CRF variants due to the forward-backward algorithm computation over sequence transitions.*
