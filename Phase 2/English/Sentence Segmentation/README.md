# English Sentence Segmentation

This directory contains research models, notebooks, and evaluation workbooks for sentence boundary restoration from continuous English script (*scriptio continua*).

---

## Overview

Sentence segmentation in continuous text is significantly more challenging than word boundary recovery. While words follow local phonotactic and orthographic constraints, sentence boundaries require capturing long-range syntactic structure, clause completeness, and semantic discourse markers without punctuation or spacing clues.

To address this challenge, Phase 2 evaluates **16 hybrid neural architectures** that combine:
1. **First-stage Neural Encoders**: BiLSTM, BiGRU, Bidirectional vanilla RNN, and 1D-CNN to capture local character-level n-gram features.
2. **Second-stage Pretrained Transformer Backbones**: BERT, RoBERTa, DeBERTa-v3, and ELECTRA to inject deep contextual representations.
3. **Structured Sequence Decoding**: Linear-Chain Conditional Random Fields (CRF) to jointly score label transition sequences and enforce valid boundary syntax.

---

## Directory Structure

```text
Sentence Segmentation/
|-- dataset/
|   `-- 2000_articles_50_character_dataset_ART_ID_split_70_15_15 (2).xlsx
|-- notebook and code files/
|   |-- scriptio-continua-proj except cnn.ipynb  # Recurrent architectures (BiLSTM, BiGRU, BiRNN)
|   `-- scriptio-continua-proj-cnn.ipynb         # Convolutional architectures (CNN + Transformers)
`-- results/
    |-- All_16_Models_Summary.xlsx               # Consolidated benchmark metrics across all 16 models
    |-- __huggingface_repos__.json               # Metadata of downloaded HuggingFace models
    |-- BiGRU_CRF_BERT.xlsx
    |-- BiGRU_CRF_DeBERTa_v3.xlsx
    |-- BiGRU_CRF_ELECTRA.xlsx
    |-- BiGRU_CRF_RoBERTa.xlsx
    |-- BiLSTM_CRF_BERT.xlsx
    |-- BiLSTM_CRF_DeBERTa_v3.xlsx
    |-- BiLSTM_CRF_ELECTRA.xlsx
    |-- BiLSTM_CRF_RoBERTa.xlsx
    |-- BiRNN_CRF_BERT.xlsx
    |-- BiRNN_CRF_DeBERTa_v3.xlsx
    |-- BiRNN_CRF_ELECTRA.xlsx
    |-- BiRNN_CRF_RoBERTa.xlsx
    |-- CNN_CRF_BERT.xlsx
    |-- CNN_CRF_DeBERTa_v3.xlsx
    |-- CNN_CRF_ELECTRA.xlsx
    `-- CNN_CRF_RoBERTa.xlsx
```

---

## 16-Model Experimental Benchmark

The table below summarizes the test set evaluation results across all 16 hybrid model configurations on the 2,000-article benchmark:

| Neural Layer 1 | Transformer Layer 2 | Overall F1 | Accuracy | Sent Break F1 | Sent Break Acc | Sent Break Exact Match | BLEU | BERTScore F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN + CRF** | **DeBERTa_v3** | 0.8738 | 0.9076 | **0.8373** | **0.9797** | **0.8621** | 0.4696 | 0.9420 |
| **CNN + CRF** | **RoBERTa** | 0.8738 | 0.9076 | **0.8144** | 0.9764 | 0.8408 | 0.4696 | 0.9420 |
| **BiLSTM + CRF** | **DeBERTa_v3** | 0.9623 | 0.9747 | **0.7895** | 0.9719 | 0.8321 | 0.8157 | 0.9841 |
| **BiLSTM + CRF** | **RoBERTa** | 0.9623 | 0.9747 | 0.7822 | 0.9712 | 0.8250 | 0.8157 | 0.9841 |
| **BiGRU + CRF** | **RoBERTa** | **0.9644** | **0.9759** | 0.7774 | 0.9704 | 0.8226 | **0.8194** | **0.9850** |
| **BiGRU + CRF** | **DeBERTa_v3** | **0.9644** | **0.9759** | 0.7763 | 0.9701 | 0.8200 | **0.8194** | **0.9850** |
| **BiRNN + CRF** | **DeBERTa_v3** | 0.9451 | 0.9627 | 0.7437 | 0.9658 | 0.8009 | 0.7435 | 0.9768 |
| **BiRNN + CRF** | **RoBERTa** | 0.9451 | 0.9627 | 0.7371 | 0.9665 | 0.8048 | 0.7435 | 0.9768 |
| **CNN + CRF** | **ELECTRA** | 0.8738 | 0.9076 | 0.7057 | 0.9615 | 0.7529 | 0.4696 | 0.9420 |
| **CNN + CRF** | **BERT** | 0.8738 | 0.9076 | 0.6964 | 0.9624 | 0.7561 | 0.4696 | 0.9420 |
| **BiGRU + CRF** | **ELECTRA** | 0.9644 | 0.9759 | 0.6931 | 0.9581 | 0.7452 | 0.8194 | 0.9850 |
| **BiGRU + CRF** | **BERT** | 0.9644 | 0.9759 | 0.6876 | 0.9585 | 0.7462 | 0.8194 | 0.9850 |
| **BiLSTM + CRF** | **ELECTRA** | 0.9623 | 0.9747 | 0.6870 | 0.9591 | 0.7483 | 0.8157 | 0.9841 |
| **BiLSTM + CRF** | **BERT** | 0.9623 | 0.9747 | 0.6765 | 0.9587 | 0.7430 | 0.8157 | 0.9841 |
| **BiRNN + CRF** | **ELECTRA** | 0.9451 | 0.9627 | 0.6525 | 0.9546 | 0.7285 | 0.7435 | 0.9768 |
| **BiRNN + CRF** | **BERT** | 0.9451 | 0.9627 | 0.6317 | 0.9507 | 0.7056 | 0.7435 | 0.9768 |

### Key Findings
1. **Sentence Break Identification**: **CNN + CRF + DeBERTa_v3** achieved the highest sentence break F1 score (**0.8373**) and exact match accuracy (**86.21%**), demonstrating that local convolution combined with disentangled attention representations excel at detecting discourse boundaries.
2. **Overall Character Boundary Recovery**: **BiGRU + CRF + RoBERTa** and **BiGRU + CRF + DeBERTa_v3** achieved the highest overall sequence tagging F1 (**0.9644**), BLEU (**0.8194**), and BERTScore F1 (**0.9850**).
3. **Transformer Impact**: DeBERTa-v3 and RoBERTa consistently outperformed standard BERT and ELECTRA across all primary metric categories.

---

## Evaluation Metrics

The evaluation captures both character-level boundary accuracy and reconstructed sentence fidelity:
- **F1 / Precision / Recall / Accuracy**: Micro and macro classification metrics over character tag transitions.
- **SENT_BREAK_F1**: Specific harmonic mean of precision and recall for boundary characters that represent sentence terminations.
- **SENT_BREAK_EXACT_MATCH**: Percentage of sentences whose boundaries were predicted with zero offsets.
- **BLEU & ROUGE (1, 2, L)**: N-gram overlap between predicted reconstructed text and original ground truth.
- **BERTScore (P, R, F1)**: Semantic cosine similarity between contextual embeddings of predicted and reference sentences.

---

## Running the Notebooks

1. Open `notebook and code files/scriptio-continua-proj except cnn.ipynb` in VS Code or Jupyter to run the recurrent model experiments (BiLSTM, BiGRU, BiRNN).
2. Open `notebook and code files/scriptio-continua-proj-cnn.ipynb` to run the convolutional model experiments.
3. Model predictions are written to individual Excel workbooks in `results/`, and aggregated via `All_16_Models_Summary.xlsx`.
