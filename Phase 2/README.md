# Phase 2: Multi-Sentence, Line-Oriented, Hybrid Transformers, and Telugu Scriptio Continua

Phase 2 advances the *Scriptio Continua* research framework from isolated sentence processing into complex document-like conditions, hybrid transformer-neural architectures, and cross-lingual Indic script segmentation.

---

## Technical Overview

While Phase 1 investigated baseline word boundary restoration on isolated, clean English sentences, Phase 2 tackles real-world complexities encountered in digitized archives, manuscripts, and non-Latin scripts:

1. **Document-Level & Line-Oriented Word Segmentation**: Testing neural taggers on fixed-length 50-character chunks with line wrapping and arbitrary sentence splits.
2. **Hybrid Transformer Sentence Break Recovery**: Deploying 16 hybrid combinations of character-level sequence taggers (BiLSTM, BiGRU, BiRNN, 1D-CNN) and pretrained Transformer backbones (BERT, RoBERTa, DeBERTa-v3, ELECTRA) with CRF decoding to jointly predict word boundaries and sentence breaks.
3. **Telugu Cross-Lingual Expansion**: Extending the framework to Telugu, an agglutinative Dravidian language requiring akshara-level phonetic tokenization, complex conjunct handling, and dedicated vocabulary representations.

---

## Directory Organization

```text
Phase 2/
|-- English/
|   |-- Dataset/                    # 2,000-article Simple Wikipedia corpus and split datasets
|   |   |-- 2000 articles simple_wikipedia_random_domain_dataset.xlsx
|   |   `-- 2000_articles_50_character_dataset_ART_ID_split_70_15_15 (2).xlsx
|   |
|   |-- Sentence Segmentation/      # Hybrid Transformer + Neural + CRF models
|   |   |-- dataset/                # 50-character line dataset
|   |   |-- notebook and code files/# Training notebooks for recurrent and CNN models
|   |   `-- results/                # 16 hybrid model results & All_16_Models_Summary.xlsx
|   |
|   `-- Word Segmentation/          # Word boundary recovery across settings
|       |-- 50_Characters/          # 50-character window line-level models & outputs
|       |-- Sentence_Level/         # Sentence-bounded reference baseline
|       `-- Dataset/                # Full article and sorted sentence datasets
|
`-- Telugu/
    |-- Complete Dataset/           # Production-scale corpus (14,480 sentences, 5 lakh aksharas)
    |   |-- dataset/                # 70/15/15 sentence split
    |   |-- Detailed Results and Analysis/ # Character, complexity, and sentence workbooks
    |   |-- model predictions without metrics/ # Raw prediction spreadsheets
    |   |-- notebooks and code files/# Saved PyTorch weights (.pkl) and notebooks
    |   |-- results summaries/      # Aggregated metric workbooks and CSV files
    |   |-- count_unique_telugu.py  # Akshara tokenization regex script
    |   `-- telugu_akshara_frequencies.xlsx
    |
    `-- sample dataset work/        # 100-article prototype track
        |-- dataset/                # Initial sample split
        |-- pkl/                    # Prototype model weights
        |-- results_with_metrics/   # Prototype evaluation workbooks
        `-- notebooks (.ipynb)      # Interactive 2-state, 4-state, and CRF notebooks
```

---

## Key Experimental Results

### 1. English Sentence Segmentation Benchmark (16 Hybrid Models)

Evaluated on the test split of the 2,000-article dataset:

| Architecture (Layer 1) | Transformer Backbone (Layer 2) | Overall F1 | Overall Acc | Sent Break F1 | Sent Break Acc | Exact Match | BLEU | BERTScore F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN + CRF** | **DeBERTa_v3** | 0.8738 | 0.9076 | **0.8373** | **0.9797** | **0.8621** | 0.4696 | 0.9420 |
| **CNN + CRF** | **RoBERTa** | 0.8738 | 0.9076 | **0.8144** | 0.9764 | 0.8408 | 0.4696 | 0.9420 |
| **BiLSTM + CRF** | **DeBERTa_v3** | 0.9623 | 0.9747 | **0.7895** | 0.9719 | 0.8321 | 0.8157 | 0.9841 |
| **BiLSTM + CRF** | **RoBERTa** | 0.9623 | 0.9747 | 0.7822 | 0.9712 | 0.8250 | 0.8157 | 0.9841 |
| **BiGRU + CRF** | **RoBERTa** | **0.9644** | **0.9759** | 0.7774 | 0.9704 | 0.8226 | **0.8194** | **0.9850** |
| **BiGRU + CRF** | **DeBERTa_v3** | **0.9644** | **0.9759** | 0.7763 | 0.9701 | 0.8200 | **0.8194** | **0.9850** |
| **BiRNN + CRF** | **DeBERTa_v3** | 0.9451 | 0.9627 | 0.7437 | 0.9658 | 0.8009 | 0.7435 | 0.9768 |
| **BiRNN + CRF** | **RoBERTa** | 0.9451 | 0.9627 | 0.7371 | 0.9665 | 0.8048 | 0.7435 | 0.9768 |

*DeBERTa-v3 coupled with CNN+CRF yields superior sentence break detection, while BiGRU+CRF coupled with RoBERTa/DeBERTa-v3 achieves the highest overall token reconstruction fidelity.*

### 2. Telugu Word Segmentation Benchmark (Complete Dataset, Test Split)

Evaluated on 2,172 test sentences from the production Telugu corpus:

| Tagging State | Neural Encoder | Macro F1 | Accuracy | Precision | Recall | BLEU | BERTScore F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2-State** | **GRU** | **0.7574** | **0.8150** | 0.7892 | 0.7538 | 0.2354 | 0.8444 |
| **2-State** | **BiLSTM** | 0.7522 | 0.8091 | 0.7792 | 0.7485 | 0.2458 | 0.8486 |
| **2-State** | **CNN** | 0.6417 | 0.7459 | 0.7009 | 0.6401 | 0.1802 | 0.8756 |
| **2-State** | **RNN** | 0.6350 | 0.7378 | 0.6961 | 0.6323 | 0.1704 | 0.8747 |
| **4-State (BIES)** | **GRU** | **0.6807** | **0.7257** | 0.6928 | 0.6856 | 0.2673 | 0.8291 |
| **4-State (BIES)** | **BiLSTM** | 0.6698 | 0.7157 | 0.6806 | 0.6756 | 0.2770 | 0.8393 |
| **4-State (BIES)** | **CNN** | 0.5565 | 0.6051 | 0.5692 | 0.5614 | **0.3121** | **0.8895** |
| **4-State + CRF** | **GRU** | **0.6239** | **0.6674** | 0.6280 | 0.6300 | 0.2530 | 0.8244 |
| **4-State + CRF** | **BiLSTM** | 0.5900 | 0.6288 | 0.5946 | 0.5982 | 0.2512 | 0.8239 |

---

## Modules and Navigation

- [English Phase 2 Overview](English/README.md)
  - [English Datasets](English/Dataset/README.md)
  - [English Sentence Segmentation (16 Models)](English/Sentence%20Segmentation/README.md)
  - [English Word Segmentation](English/Word%20Segmentation/README.md)
    - [50-Character Line-Level Word Segmentation](English/Word%20Segmentation/50_Characters/README.md)
    - [Sentence-Level Word Segmentation Baseline](English/Word%20Segmentation/Sentence_Level/README.md)
- [Telugu Phase 2 Overview](Telugu/README.md)
  - [Telugu Complete Dataset Module](Telugu/Complete%20Dataset/README.md)
    - [Detailed Results and Linguistic Analysis](Telugu/Complete%20Dataset/Detailed%20Results%20and%20Analysis/README.md)
    - [Notebooks and Pre-Trained Weights](Telugu/Complete%20Dataset/notebooks%20and%20code%20files/README.md)
    - [Raw Predictions](Telugu/Complete%20Dataset/model%20predictions%20without%20metrics/README.md)
    - [Results Summaries](Telugu/Complete%20Dataset/results%20summaries/README.md)
  - [Telugu Sample Dataset Prototyping](Telugu/sample%20dataset%20work/README.md)
