# Raw Telugu Model Predictions

This directory stores character-level prediction outputs generated during inference on the test split of the complete Telugu dataset.

---

## Files

- **`telugu_2state_predictions.xlsx`**:
  - Contains row-by-row predictions from the four 2-state models (BiLSTM, BiGRU, BiRNN, CNN).
  - Columns include the raw continuous script string, ground-truth binary label sequence (`0`/`1`), and predicted label sequences from each model.
- **`telugu_4state_predictions.xlsx`**:
  - Contains row-by-row predictions from the four 4-state models (BiLSTM, BiGRU, BiRNN, CNN) using the BIES scheme (`B`, `I`, `E`, `S`).
  - Columns align the input continuous characters with ground-truth BIES tags and individual model predictions.
- **`telugu_4state_crf_predictions.xlsx`**:
  - Contains row-by-row predictions from the four 4-state models enhanced with Linear-Chain CRF sequence decoding.
  - Viterbi-decoded state paths are recorded alongside the input text and ground-truth sequences.

---

## Utility

These spreadsheets contain raw predictions without summary aggregation, allowing researchers to:
- Perform error analysis on specific linguistic edge cases (e.g., compound words, low-frequency conjuncts).
- Implement custom sequence evaluation metrics or post-processing rules.
- Inspect exact discrepancies between plain neural predictions and CRF-constrained predictions.
