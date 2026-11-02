# Telugu Production Notebooks and Pre-Trained Models

This directory contains Jupyter training/evaluation notebooks and serialized PyTorch model weights (`.pkl`) for Telugu *scriptio continua* word segmentation on the complete dataset.

---

## Directory Organization

```text
notebooks and code files/
|-- 2-state/
|   |-- 2-state-telugu-fulll (1).ipynb   # Training and evaluation notebook for 2-state models
|   |-- telugu_2state_bilstm.pkl         # Trained BiLSTM weights
|   |-- telugu_2state_cnn.pkl            # Trained 1D-CNN weights
|   |-- telugu_2state_gru.pkl            # Trained BiGRU weights
|   |-- telugu_2state_rnn.pkl            # Trained Vanilla BiRNN weights
|   `-- telugu_2state_vocab.pkl          # Character / akshara vocabulary dictionary
|
|-- 4-state/
|   |-- 4-state-telugu-full.ipynb        # Training and evaluation notebook for 4-state BIES models
|   |-- telugu_4state_bilstm.pkl         # Trained BiLSTM weights
|   |-- telugu_4state_cnn.pkl            # Trained 1D-CNN weights
|   |-- telugu_4state_gru.pkl            # Trained BiGRU weights
|   |-- telugu_4state_rnn.pkl            # Trained Vanilla BiRNN weights
|   `-- telugu_4state_vocab.pkl          # Character / akshara vocabulary dictionary
|
`-- 4-state-crf/
    |-- telugu-4-state-crf-full.ipynb    # Training and evaluation notebook for 4-state + CRF
    |-- telugu_4state_crf_bilstm.pkl     # Trained BiLSTM + CRF weights
    |-- telugu_4state_crf_cnn.pkl        # Trained 1D-CNN + CRF weights
    |-- telugu_4state_crf_gru.pkl        # Trained BiGRU + CRF weights
    |-- telugu_4state_crf_rnn.pkl        # Trained Vanilla BiRNN + CRF weights
    `-- telugu_4state_crf_vocab.pkl      # Character / akshara vocabulary dictionary
```

---

## Model Specifications

- **Embedding Layer**: Trainable character/akshara embeddings of dimension 64.
- **Recurrent Hidden Size**: 128 (bidirectional output dimension = 256).
- **CNN Architecture**: 1D convolution with 128 filters, kernel size 3, padding 1, followed by ReLU.
- **Dropout**: 0.2 across layers.
- **Optimizer**: Adam with learning rate 1e-3.
- **Decoding**:
  - 2-state & 4-state: Softmax per-character classification cross-entropy loss.
  - 4-state CRF: Negative log-likelihood sequence loss with Viterbi decoding at inference.

---

## Loading and Inference Guide

To load a pre-trained model and run inference on unseen unspaced Telugu text:

```python
import pickle
import torch

# 1. Load the vocabulary
vocab_path = "notebooks and code files/2-state/telugu_2state_vocab.pkl"
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)

# 2. Load the trained model weights
model_path = "notebooks and code files/2-state/telugu_2state_gru.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

model.eval()

# 3. Tokenize input continuous text
continuous_text = "రాముడుమంచిబాలుడు"  # "Ramudu manchi baludu" without spaces
char_indices = [vocab.get(char, vocab.get("<UNK>", 0)) for char in continuous_text]
input_tensor = torch.tensor([char_indices], dtype=torch.long)

# 4. Predict boundaries
with torch.no_grad():
    outputs = model(input_tensor)
    predictions = torch.argmax(outputs, dim=-1).squeeze(0).tolist()

# 5. Reconstruct text with spaces (where prediction == 1)
reconstructed = []
for char, pred in zip(continuous_text, predictions):
    reconstructed.append(char)
    if pred == 1:
        reconstructed.append(" ")

print("Reconstructed text:", "".join(reconstructed).strip())
```
