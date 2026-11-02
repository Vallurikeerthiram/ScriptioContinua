# English Sentence Segmentation Notebooks

This directory contains Jupyter notebooks used to train and evaluate the 16 hybrid models for sentence segmentation on continuous English script.

---

## Notebooks

### 1. `scriptio-continua-proj except cnn.ipynb`
- **Architectures Covered**: Recurrent neural networks (Bidirectional LSTM, Bidirectional GRU, Bidirectional vanilla RNN) integrated with pretrained Transformer contextual embeddings and Linear-Chain CRF decoders.
- **Transformers Evaluated**: `bert-base-uncased`, `roberta-base`, `microsoft/deberta-v3-small`, `google/electra-base-discriminator`.
- **Workflow**:
  - Ingests the 50-character window dataset.
  - Embeds characters using learned character embeddings concatenated with subword token alignments from the transformer.
  - Passes sequences through the bidirectional recurrent layer.
  - Computes transition scores using the CRF layer.
  - Generates evaluation metrics and exports prediction workbooks.

### 2. `scriptio-continua-proj-cnn.ipynb`
- **Architectures Covered**: 1D Convolutional Neural Network (CNN) integrated with pretrained Transformer contextual embeddings and Linear-Chain CRF decoders.
- **Workflow**:
  - Applies 1D convolution (`kernel_size=3`, `padding=1`) to capture local character tri-grams.
  - Fuses convolutional feature maps with transformer representations.
  - Decodes sentence break and word boundary transitions via CRF Viterbi decoding.
  - Achieved the highest sentence break F1 score (**0.8373**) and exact match accuracy (**86.21%**) when paired with DeBERTa-v3.
