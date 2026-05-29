# Deep Learning Module: Technical Specification

This module implements a character-level sequence tagging framework using PyTorch to solve the word segmentation problem. It transforms continuous alphanumeric strings back into readable sentences by predicting word boundaries at the character level.

---

## 1. Core Methodology: Sequence Tagging

The problem is treated as a "Many-to-Many" sequence labeling task. Given an input sequence of $N$ characters $C = \{c_1, c_2, ..., c_n\}$, the model predicts a sequence of $N$ labels $L = \{l_1, l_2, ..., l_n\}$.

### Labeling Schemes
The system evaluates two distinct labeling strategies:
*   **`state2` (Binary Segmentation)**:
    *   `0`: Non-boundary character.
    *   `1`: Word boundary (the character is the terminal letter of a word).
*   **`state4` (BIES Scheme)**:
    *   `B` (Begin): First character of a multi-character word.
    *   `I` (Inside): Intermediate character of a multi-character word.
    *   `E` (End): Last character of a multi-character word.
    *   `S` (Single): A standalone single-character word (e.g., "a", "I").

---

## 2. Neural Architectures

The framework provides five specialized implementations of `SequenceTaggerBase`:

| Model | Description | Implementation Details |
| :--- | :--- | :--- |
| **BiLSTM** | Bi-directional Long Short-Term Memory | 1 layer, bidirectional, captures long-range context from both directions. |
| **RNN** | Vanilla Recurrent Neural Network | Uses `tanh` activation; serves as a baseline for recurrent modeling. |
| **GRU** | Gated Recurrent Unit | A streamlined recurrent architecture optimized for vanishing gradient mitigation. |
| **CNN** | 1D Convolutional Neural Network | `kernel_size=3`, `padding=1`. Uses ReLU activation to detect local character n-grams. |
| **CRF** | Conditional Random Field | Includes a custom `CRFLayer` with transition matrices to ensure label consistency (e.g., prevents 'I' from following 'E'). |

---

## 3. Data Specification (`SENT_based_split.xlsx`)

The training script expects an Excel file with the following sheet/column structure:
*   **Sheets**: `TRAIN`, `VAL`, `TEST`
*   **Required Columns**:
    *   `SCRIPT_CONTIN`: The raw continuous string (input).
    *   `SENT_ORI`: The original ground truth sentence (for reconstruction validation).
    *   `STATE_2`: Space-separated integers (e.g., `0 0 1 0 1`).
    *   `STATE_4`: Space-separated labels (e.g., `B I E B E`).
    *   `ART_ID`, `PARA_ID`, `SENT_ID`: Metadata for tracking sentence origin.

---

## 4. Hyperparameters & Configuration

Default settings in the `Config` class:
*   **Embedding Dim**: 64 (Trainable Character Embeddings)
*   **Hidden Dim**: 128 (for RNN/LSTM/GRU)
*   **CNN Channels**: 128
*   **Dropout**: 0.2
*   **Batch Size**: 64
*   **Optimizer**: Adam (`lr=1e-3`, `weight_decay=1e-5`)
*   **Patience**: 2 (Early stopping based on Validation F1)

---

## 5. Execution & Output

### How to Run
```bash
python train_models.py
```

### Execution Pipeline:
1.  **Vocab Building**: Extracts all unique characters from the `TRAIN` split.
2.  **Embedding Initialization**: Initializes weights using Xavier Uniform distribution.
3.  **Training Loop**: Iterates through epochs, calculating Cross-Entropy Loss (or CRF Negative Log-Likelihood).
4.  **Reconstruction**: Post-processing logic converts predicted labels back into spaced text.
5.  **Metrics Generation**: Compares predicted text vs. `SENT_ORI` using BLEU, METEOR, ROUGE-L, and **BERTScore** (using `bert-score` library).

### Output
The script generates `model_comparison_results.xlsx`, which contains:
*   **Summary Sheets**: Aggregated performance metrics for every model/scheme combination.
*   **Prediction Sheets**: Side-by-side comparison of `ORIGINAL_OUTPUT` vs `PREDICTED_OUTPUT`.
*   **Confusion Matrices**: Detailed error analysis for label transitions.

---
**Note on `code.py`**: The directory contains a file named `code.py`. To avoid conflicts with the Python Standard Library `code` module, `train_models.py` uses a custom preloader to ensure the system `code` module is loaded correctly.
