# Telugu Linguistic Stratification Workbooks

This directory contains analytical workbooks that stratify Telugu segmentation performance across linguistic complexity classes and frequency percentiles.

---

## Workbook Inventory

### 1. `complexity_grouped_metrics.xlsx`
- **Focus**: Evaluates performance categorized by akshara structural complexity:
  - **Simple Aksharas**: Base consonants and vowels without subscript conjuncts.
  - **Conjunct Aksharas (*Vattulu*)**: Consonant clusters bound by virama (`్`), including *dvitva* (geminate consonants) and *samyukta* (heterogeneous clusters).

### 2. `frequency_grouped_metrics.xlsx`
- **Focus**: Evaluates performance categorized by corpus frequency:
  - High frequency (top 10%).
  - Medium frequency (interquartile range).
  - Low frequency (long-tail and rare scholarly terms).

### 3. `char_model_metrics_with_complexity_avg.xlsx`
- Aggregates precision, recall, and F1 across complexity tiers, computing macro-averages per model architecture.

### 4. `char_model_metrics_with_frequency_avg.xlsx`
- Aggregates precision, recall, and F1 across frequency tiers, computing macro-averages per model architecture.

### 5. `telugu_2_state_character_metrics.xlsx` & `telugu_4_state_character_metrics.xlsx`
- Baseline character-level metric tables for 2-state and 4-state models.
