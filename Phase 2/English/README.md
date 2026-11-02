# English Phase 2 Research Module

This directory contains the English language experimental pipelines, datasets, models, and results for Phase 2 of the *Scriptio Continua* research project.

---

## Architecture and Tracks

Phase 2 for English expands beyond isolated sentence word-boundary recovery into complex document-like structures and multi-stage restoration tasks:

```text
English/
|-- Dataset/                    # Raw 2,000-article corpus and 50-character split data
|   |-- 2000 articles simple_wikipedia_random_domain_dataset.xlsx
|   `-- 2000_articles_50_character_dataset_ART_ID_split_70_15_15 (2).xlsx
|
|-- Sentence Segmentation/      # Hybrid Transformer + Neural + CRF sentence break recovery
|   |-- dataset/
|   |-- notebook and code files/
|   `-- results/                # 16 hybrid model results & All_16_Models_Summary.xlsx
|
`-- Word Segmentation/          # Word boundary prediction across document settings
    |-- 50_Characters/          # Line-level 50-character window experiments
    |-- Sentence_Level/         # Sentence-bounded reference experiments
    `-- Dataset/
```

---

## Key Experimental Focus

### 1. Sentence Segmentation (Hybrid Transformers + CRF)
Recovers both sentence boundaries and word spaces from continuous text using 16 combinations of:
- **Neural Sequence Encoders**: BiLSTM, BiGRU, BiRNN, 1D-CNN
- **Pretrained Transformers**: BERT (`bert-base-uncased`), RoBERTa (`roberta-base`), DeBERTa-v3 (`microsoft/deberta-v3-small`), ELECTRA (`google/electra-base-discriminator`)
- **Sequence Decoders**: Linear-Chain CRF

*Best performance: CNN + CRF + DeBERTa-v3 achieved 0.8373 Sentence Break F1 and 86.21% Exact Match.*

### 2. Line-Level Word Segmentation (50-Character Windows)
Evaluates whether neural taggers can segment text when words and sentences are clipped by fixed-width lines (simulating physical manuscript lines and column-wrapped OCR).

### 3. Sentence-Level Baseline
Provides the controlled upper-bound benchmark on clean, complete sentence strings.

---

## Detailed Track Documentation

- [English Datasets Documentation](Dataset/README.md)
- [Sentence Segmentation (16 Hybrid Models)](Sentence%20Segmentation/README.md)
- [Word Segmentation Overview](Word%20Segmentation/README.md)
  - [50-Character Line-Level Word Segmentation](Word%20Segmentation/50_Characters/README.md)
  - [Sentence-Level Word Segmentation Baseline](Word%20Segmentation/Sentence_Level/README.md)
