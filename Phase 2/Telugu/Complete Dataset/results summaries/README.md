# Telugu Complete Dataset Results Summaries

This directory contains consolidated evaluation metrics summarizing model accuracy, macro F1, precision, recall, and text reconstruction scores across the complete Telugu dataset.

---

## File Inventory

### 1. `telugu_2state_predictions_scores.xlsx`
- **Scope**: Summarizes performance for all four 2-state models (BiLSTM, CNN, GRU, RNN).
- **Contents**:
  - Macro and micro Precision, Recall, F1 score, and Accuracy.
  - Per-split metric breakdowns for `train`, `test`, and `Evaluation`.
  - Reconstruction metrics: Sentence BLEU, ROUGE-1/2/L, METEOR, and BERTScore (Precision, Recall, F1).

### 2. `telugu_4state_predictions_summary.csv`
- **Scope**: Exported CSV summary of the 4-state BIES neural taggers without CRF.
- **Key Columns**:
  - `STATE_MODEL`, `MODEL`, `SPLIT`, `ROWS`, `BLEU`, `ROUGE_1_F1`, `ROUGE_2_F1`, `ROUGE_L_F1`, `METEOR`, `BERTSCORE_P`, `BERTSCORE_R`, `BERTSCORE_F1`, `F1`, `PRECISION`, `ACCURACY`, `RECALL`.

### 3. `telugu_4state_crf_predictions_summary.csv`
- **Scope**: Exported CSV summary of the 4-state BIES models enhanced with Linear-Chain CRF sequence decoding.
- **Key Comparisons**: Directly comparable to `telugu_4state_predictions_summary.csv` to measure the exact delta introduced by CRF transition constraints on Telugu text.

---

## Metric Summary on Test Set

| Paradigm | Top Architecture | Macro F1 | Accuracy | BLEU | BERTScore F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 2-State | GRU | **0.7574** | **0.8150** | 0.2354 | 0.8444 |
| 4-State (BIES) | GRU | **0.6807** | **0.7257** | 0.2673 | 0.8291 |
| 4-State + CRF | GRU | **0.6239** | **0.6674** | 0.2530 | 0.8244 |
