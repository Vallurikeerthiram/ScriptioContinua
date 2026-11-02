# Telugu Sentence-Level Metrics and Summaries

This directory contains evaluation notebooks and consolidated score outputs measuring full-sentence text reconstruction quality for Telugu word segmentation models.

---

## Directory Organization

```text
sentence level metrics/
|-- notebook/
|   `-- __notebook_source__ (1).ipynb   # Jupyter notebook computing sentence BLEU, ROUGE, METEOR, BERTScore
`-- scoresoutputs/
    |-- all_models_summary.csv          # Master performance table across all 12 models and splits
    |-- all_models_summary.xlsx         # Master Excel summary workbook
    |-- telugu_2_state_model_predictions_detailed_results.xlsx
    |-- telugu_4_state_model_predictions_detailed_results.xlsx
    `-- telugu_4_state_crf_model_predictions_detailed_results.xlsx
```

---

## Metrics Computed

For every reconstructed Telugu sentence:
1. **F1, Precision, Recall, Accuracy**: Token-level classification metrics on boundary predictions.
2. **BLEU**: Word n-gram precision against ground-truth reference text.
3. **ROUGE-1, ROUGE-2, ROUGE-L**: Unigram, bigram, and longest common subsequence recall and F1.
4. **METEOR**: Explicit alignment-based score balancing precision and recall.
5. **BERTScore (P, R, F1)**: Contextual embedding cosine similarity measuring preservation of semantic meaning.
