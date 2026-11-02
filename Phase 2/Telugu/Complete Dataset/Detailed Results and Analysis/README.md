# Telugu Detailed Results and Analysis

This directory contains multi-dimensional evaluations of Telugu *scriptio continua* models, analyzing segmentation performance across character frequencies, structural complexity, and sentence-level reconstruction quality.

---

## Directory Structure

```text
Detailed Results and Analysis/
|-- character level analysis/                   # Akshara frequency breakdown and metrics
|   |-- characterfrequencymetric1.ipynb         # Frequency vs performance analysis notebook
|   |-- telugu_2_state_character_metrics.xlsx   # 2-state character precision, recall, F1
|   |-- telugu_4_state_character_metrics.xlsx   # 4-state character precision, recall, F1
|   |-- telugu_akshara_frequencies.xlsx         # Akshara frequency count across corpus
|   |-- telugu_akshara_metrics_12_models_combined.xlsx # Consolidated 12-model akshara metrics
|   `-- telugu_akshara_metrics_12_models_detailed.xlsx # Per-akshara metric breakdown
|
|-- sentence level metrics/                     # Full sentence reconstruction and evaluation
|   |-- notebook/
|   |   `-- __notebook_source__ (1).ipynb       # Sentence-level metric calculation notebook
|   `-- scoresoutputs/
|       |-- all_models_summary.csv              # Test, train, eval metrics across all 12 models
|       |-- all_models_summary.xlsx             # Excel version of complete model summary
|       |-- telugu_2_state_model_predictions_detailed_results.xlsx
|       |-- telugu_4_state_model_predictions_detailed_results.xlsx
|       `-- telugu_4_state_crf_model_predictions_detailed_results.xlsx
|
|-- Workbooks/                                  # Linguistic stratification workbooks
|   |-- char_model_metrics_with_complexity_avg.xlsx # Metrics stratified by akshara complexity
|   |-- char_model_metrics_with_frequency_avg.xlsx  # Metrics stratified by corpus frequency
|   |-- complexity_grouped_metrics.xlsx         # Simple vs conjunct akshara grouping
|   |-- frequency_grouped_metrics.xlsx          # High, medium, and low frequency tiers
|   |-- telugu_2_state_character_metrics.xlsx
|   `-- telugu_4_state_character_metrics.xlsx
|
`-- Telugumodelanalysis.xlsx                    # High-level model comparison summary
```

---

## Analysis Dimensions

### 1. Linguistic Complexity Analysis (`Workbooks/complexity_grouped_metrics.xlsx`)
Aksharas are stratified by their structural orthographic complexity:
- **Simple Aksharas**: Independent vowels (e.g., `అ`, `ఆ`) and base consonants with or without vowel matras (e.g., `క`, `కి`, `కు`).
- **Complex / Conjunct Aksharas (*Samūktha / Dvitva*)**: Consonants carrying virama and subordinate consonants (*vattulu*, e.g., `క్క`, `ర్త్య`, `ష్ట్ర`).
- **Findings**: Recurrent taggers (BiLSTM and GRU) exhibit minimal performance degradation on complex conjuncts compared to CNNs, indicating that sequential recurrence retains memory across multi-byte Unicode conjunct sequences.

### 2. Frequency-Grouped Analysis (`Workbooks/frequency_grouped_metrics.xlsx`)
Aksharas are categorized into frequency deciles:
- High frequency (top 10% most common aksharas).
- Medium frequency.
- Rare / long-tail aksharas (hapax legomena and low-frequency scholarly compounds).
- **Findings**: Boundary precision remains robust for frequent aksharas (>0.82 F1 in 2-state GRU), while rare aksharas suffer higher recall degradation.

### 3. Sentence-Level Reconstruction Metrics (`sentence level metrics/scoresoutputs/`)
For each test sentence, the predicted boundary sequence is converted back into segmented Telugu text with whitespace, and compared against the ground truth using:
- **BLEU**: Word and character n-gram precision.
- **ROUGE (1, 2, L)**: Recall-oriented n-gram and longest common subsequence overlap.
- **METEOR**: Harmonic mean of precision and recall with stemming and synonym alignment.
- **BERTScore**: Semantic cosine similarity using contextual embeddings.
