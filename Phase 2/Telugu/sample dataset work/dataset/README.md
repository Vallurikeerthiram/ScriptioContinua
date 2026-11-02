# Telugu Sample Dataset

This directory contains the dataset used for the initial 100-article prototype track.

---

## Dataset File

- **Filename**: `telugu_sentence_level_first_100_articles_split (1).xlsx`
- **Scope**: Sentence records extracted from the first 100 articles of the Telugu corpus.
- **Splits**: Contains non-overlapping `TRAIN`, `VAL`, and `TEST` partitions.
- **Columns**:
  - `SENT_ID`: Sentence identifier.
  - `SENT_ORI`: Original spaced sentence.
  - `SCRIPT_CONTIN`: Continuous Telugu text.
  - `STATE_2`: Binary boundary labels (`0`/`1`).
  - `STATE_4`: BIES boundary labels (`B`/`I`/`E`/`S`).

For the full production dataset (14,480 sentences, 5 lakh aksharas), refer to [`Phase 2/Telugu/Complete Dataset/dataset/`](../../Complete%20Dataset/dataset/README.md).
