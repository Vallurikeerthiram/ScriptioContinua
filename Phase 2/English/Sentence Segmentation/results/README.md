# English Sentence Segmentation Results

This directory contains the prediction workbooks and consolidated performance metrics for the 16 hybrid Transformer + Sequence Tagger + CRF models evaluated on English sentence segmentation.

---

## File Inventory

### Master Summary
- **`All_16_Models_Summary.xlsx`**: Master table aggregating all 16 model combinations across training, validation, and test splits. Includes overall boundary F1, accuracy, precision, recall, sentence break specific F1, sentence break exact match, BLEU, ROUGE-1/2/L, METEOR, and BERTScore (P, R, F1).

### Model Spreadsheets
Each workbook contains detailed character-level and sentence-level predictions for that specific architecture across train, validation, and test sets:
- **BiGRU + CRF Variants**:
  - `BiGRU_CRF_BERT.xlsx`
  - `BiGRU_CRF_DeBERTa_v3.xlsx`
  - `BiGRU_CRF_ELECTRA.xlsx`
  - `BiGRU_CRF_RoBERTa.xlsx`
- **BiLSTM + CRF Variants**:
  - `BiLSTM_CRF_BERT.xlsx`
  - `BiLSTM_CRF_DeBERTa_v3.xlsx`
  - `BiLSTM_CRF_ELECTRA.xlsx`
  - `BiLSTM_CRF_RoBERTa.xlsx`
- **BiRNN + CRF Variants**:
  - `BiRNN_CRF_BERT.xlsx`
  - `BiRNN_CRF_DeBERTa_v3.xlsx`
  - `BiRNN_CRF_ELECTRA.xlsx`
  - `BiRNN_CRF_RoBERTa.xlsx`
- **CNN + CRF Variants**:
  - `CNN_CRF_BERT.xlsx`
  - `CNN_CRF_DeBERTa_v3.xlsx`
  - `CNN_CRF_ELECTRA.xlsx`
  - `CNN_CRF_RoBERTa.xlsx`

### HuggingFace Provenance
- **`__huggingface_repos__.json`**: Records the exact repository IDs, commit hashes, and model weights downloaded from the Hugging Face Hub (`microsoft/deberta-v3-small`, `bert-base-uncased`, `roberta-base`, `google/electra-base-discriminator`) ensuring full experimental reproducibility.
