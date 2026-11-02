# Telugu Character-Level Analysis

This directory contains fine-grained analyses of model performance across individual aksharas and akshara frequency distributions in Telugu *scriptio continua*.

---

## Directory Contents

### Notebook
- **`characterfrequencymetric1.ipynb`**:
  - Computes akshara frequency counts across the complete corpus.
  - Stratifies test set precision, recall, and F1 by individual akshara tokens.
  - Analyzes correlation between akshara frequency and segmentation accuracy.

### Workbooks
- **`telugu_akshara_frequencies.xlsx`**: Master frequency table recording occurrence counts for every distinct akshara found in the dataset.
- **`telugu_2_state_character_metrics.xlsx`**: Character-level precision, recall, and F1 for all four 2-state models.
- **`telugu_4_state_character_metrics.xlsx`**: Character-level precision, recall, and F1 for all four 4-state models.
- **`telugu_akshara_metrics_12_models_combined.xlsx`**: Side-by-side comparison of all 12 model configurations mapped to each akshara.
- **`telugu_akshara_metrics_12_models_detailed.xlsx`**: Comprehensive metrics matrix including support counts, true positives, false positives, and false negatives per akshara.
