# Telugu Production Dataset

This directory contains the primary benchmark dataset used for training, evaluating, and testing Telugu *scriptio continua* word segmentation models on the complete dataset.

---

## Dataset File

- **Filename**: `telugu_sentence_dataset_70_15_15_split.xlsx`
- **Total Sentences**: 14,480 sentences
- **Corpus Size**: Over 500,000 continuous akshara units

---

## Split Structure

The workbook contains three dedicated sheets with non-overlapping sentence records:

| Sheet Name | Row Count | Purpose |
| :--- | ---: | :--- |
| `train` | 10,136 | Model training and loss backpropagation |
| `test` | 2,172 | Primary held-out benchmark evaluation |
| `Evaluation` | 2,172 | Secondary validation set for cross-split stability verification |

---

## Column Descriptions

- `SENT_ID`: Unique integer identifier for each sentence record.
- `SENT_ORI`: Original grammatically segmented Telugu text with whitespace and punctuation.
- `SCRIPT_CONTIN`: Normalized continuous Telugu text without whitespace or punctuation.
- `STATE_2`: Space-separated binary labels (`0` = non-boundary akshara, `1` = word-terminal akshara).
- `STATE_4`: Space-separated BIES labels (`B` = Begin, `I` = Inside, `E` = End, `S` = Single-akshara word).
