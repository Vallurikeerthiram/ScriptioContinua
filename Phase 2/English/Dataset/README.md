# English Phase 2 Datasets

This directory contains the primary English corpora used for Phase 2 word and sentence segmentation benchmarks.

---

## Dataset Inventory

### 1. Simple Wikipedia Random Domain Dataset
- **Filename**: `2000 articles simple_wikipedia_random_domain_dataset.xlsx`
- **Description**: A collection of 2,000 full Simple Wikipedia articles spanning multiple thematic categories (science, history, technology, literature, geography, arts).
- **Source**: Extracted and filtered using the Phase 1 article scraper with content deduplication and paragraph quality thresholds.
- **Purpose**: Provides the ground-truth document-level text for constructing line-oriented and sentence-segmented experimental datasets.

### 2. 50-Character Window Dataset (Split 70/15/15)
- **Filename**: `2000_articles_50_character_dataset_ART_ID_split_70_15_15 (2).xlsx`
- **Description**: The 2,000 articles segmented into fixed-width 50-character continuous chunks with strict article-level separation.
- **Split Distribution**:
  - **Train**: 70% of articles (1,400 articles)
  - **Validation**: 15% of articles (300 articles)
  - **Test**: 15% of articles (300 articles)
- **Zero Article Leakage**: The train, validation, and test splits contain non-overlapping article identifiers (`ART_ID`), guaranteeing unbiased evaluation.
- **Columns**:
  - `ART_ID`: Unique integer identifier for the article.
  - `LINE_ID` / `CHUNK_ID`: Sequential index of the 50-character slice within the article.
  - `SENT_ORI`: Original English text corresponding to the slice.
  - `SCRIPT_CONTIN`: Normalized continuous alphanumeric text without spaces or punctuation.
  - `STATE_2`: Binary sequence (0 = non-boundary, 1 = word boundary).
  - `STATE_4`: BIES sequence (B = Begin, I = Inside, E = End, S = Single-character word).

---

## Usage in Experiments

These workbooks are ingested by:
1. **Sentence Segmentation Experiments**: `Phase 2/English/Sentence Segmentation/`
2. **Word Segmentation 50-Character Experiments**: `Phase 2/English/Word Segmentation/50_Characters/`
