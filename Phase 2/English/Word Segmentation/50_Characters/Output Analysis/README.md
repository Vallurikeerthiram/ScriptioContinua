# 50-Character Word Segmentation Output Analysis

This directory contains detailed, per-sample prediction logs for word segmentation models evaluated on 50-character window continuous text.

---

## Analysis Workbooks

### 1. `state2_line_level_models_detailed_results.xlsx`
- **Model Family**: Plain neural taggers (BiLSTM, BiGRU, BiRNN, 1D-CNN) trained on the binary `STATE_2` scheme.
- **Contents**: Full test set predictions matching each continuous 50-character line to its ground-truth binary string and model-predicted binary output.

### 2. `state4_line_level_models_detailed_results.xlsx`
- **Model Family**: Plain neural taggers trained on the BIES `STATE_4` scheme.
- **Contents**: Character-by-character BIES label predictions alongside ground truth, enabling error categorization across word beginnings, interiors, endings, and single-letter words.

### 3. `state4_crf_line_level_models_detailed_results.xlsx`
- **Model Family**: CRF-enhanced sequence taggers trained on BIES `STATE_4`.
- **Contents**: Viterbi-decoded state sequences, illustrating how CRF transition constraints resolve impossible tag transitions present in unconstrained neural outputs.
