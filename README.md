# Scriptio Continua

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A character-first computational research framework for restoring readable text from continuous writing (*scriptio continua*), progressing from word boundary recovery to sentence segmentation, hybrid transformer decoding, and cross-lingual Indic script restoration.

---

## Executive Summary

*Scriptio Continua* investigates a fundamental problem in historical manuscript analysis, epigraphy, and noisy optical character recognition (OCR): **how do we recover readable, structured text when the source contains no dependable spaces between words or sentences?**

In ancient Latin, Greek, Sanskrit, Telugu, and numerous other writing traditions, manuscripts and inscriptions were inscribed without word separators or punctuation. Standard natural language processing pipelines fail catastrophically on such text because tokenizers, subword encoders, and word embeddings presuppose the existence of discrete word boundaries.

This project formulates text restoration as a **character-level and akshara-level sequence labeling problem**:
- **English Benchmark**: Establishes controlled validation on isolated sentences (Phase 1), progresses to line-oriented 50-character chunks with arbitrary breaks, and deploys 16 hybrid Transformer + Neural + CRF models for joint sentence boundary recovery (Phase 2).
- **Telugu Cross-Lingual Track**: Extends the methodology to Telugu, an agglutinative Dravidian language requiring phonetic syllabic tokenization (*aksharas*), complex consonant conjunct handling (*vattulu* / సంయుక్తాక్షరాలు), and dedicated vocabulary embeddings.

---

## At a Glance

| Dimension | Details |
| :--- | :--- |
| **Input** | Continuous character strings without whitespace or punctuation (e.g., `thequickbrownfox` or `రాముడుమంచిబాలుడు`) |
| **Prediction Targets** | Word boundaries (Binary `0/1` or BIES `STATE_4`) and sentence termination breaks |
| **Core Constraint** | Word embeddings require tokens; continuous text has no tokens. Embeddings must be learned at the character or akshara level. |
| **Methodologies** | Character-level sequence taggers (BiLSTM, BiGRU, BiRNN, 1D-CNN), Linear-Chain CRFs, Hybrid Transformers (BERT, RoBERTa, DeBERTa-v3, ELECTRA), Semantic Tries with POS verification, and Few-Shot LLMs (Qwen, Gemma, DeepSeek via Ollama). |
| **Phase 1** | Sentence-level English word boundary recovery across neural, dictionary, and LLM baselines. |
| **Phase 2** | Line-oriented 50-character English chunks, 16 hybrid Transformer sentence-break models, and full-scale Telugu segmentation (5 lakh aksharas). |
| **Languages** | English (Latin script) and Telugu (Brahmic / Telugu abugida script). |
| **Status** | Active research repository and benchmark suite with pre-trained PyTorch weights and evaluation workbooks. |

---

## Problem Formulation & Labeling Schemes

### 1. The Character-First Pipeline

```text
Continuous Input:   thequickbrownfox
       |
       v
Trainable Character / Akshara Embeddings (dim = 64)
       |
       v
Sequence Encoders (BiLSTM / BiGRU / BiRNN / 1D-CNN)
       |
       +---> [Optional] Pretrained Transformer Context (DeBERTa / RoBERTa / BERT / ELECTRA)
       |
       v
Structured Sequence Decoder (Softmax or Linear-Chain CRF)
       |
       v
Boundary Tag Sequence -> Reconstructed Text with Spaces
```

### 2. State Representations

#### Binary Segmentation (`STATE_2`)
Each character is classified as:
- `0`: Non-boundary (internal letter of a word).
- `1`: Word terminal (the final letter of a word; a space is inserted immediately following).

```text
Input:    t h e q u i c k b r o w n f o x
STATE_2:  0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 1
Output:   the quick brown fox
```

#### Four-State BIES Scheme (`STATE_4`)
Models intra-word position and standalone words:
- `B` (*Begin*): Initial character of a multi-character word.
- `I` (*Inside*): Intermediate character of a multi-character word.
- `E` (*End*): Final character of a multi-character word.
- `S` (*Single*): Standalone single-character word (e.g., English "a", "I").

```text
Input:    t h e q u i c k a f o x
STATE_4:  B I E B I I I E S B I E
Output:   the quick a fox
```

#### Telugu Akshara Formulation
In Telugu, segmenting by raw Unicode code points splits base consonants from their vowel diacritics (*matras*) or virama (*halant* / `్`). The framework applies phonetic syllabic regex grouping (`count_unique_telugu.py`) to treat each akshara as an atomic sequence unit, preserving linguistic integrity across simple and conjunct (*vattulu*) characters.

---

## Repository Architecture

```text
ScriptioContinua/
|-- Phase 1/                               # Foundational sentence-level experiments
|   |-- DATASET/                           # Wikipedia scraping and preprocessing
|   |-- Models/                            # Sequence taggers and CRF experiments
|   |-- plain DL/                          # Standalone PyTorch training pipeline
|   |-- dict based model/                  # Trie with phonotactic validators
|   |-- dictionary based/                  # Universal semantic Trie with POS verification
|   |-- LLM based/                         # Few-shot prompting baselines via Ollama
|   |-- Literature survey/                 # Academic surveys and background papers
|   |-- extras/                            # Metric generation utilities
|   `-- scripts/                           # Automation scripts
|
|-- Phase 2/                               # Multi-sentence, line-oriented, and Telugu research
|   |-- English/
|   |   |-- Dataset/                       # 2,000-article corpus & 50-char window splits
|   |   |-- Sentence Segmentation/         # 16 Hybrid Transformer + Neural + CRF models
|   |   `-- Word Segmentation/
|   |       |-- 50_Characters/             # Line-level 50-character window models & logs
|   |       |-- Sentence_Level/            # Sentence-bounded reference baseline
|   |       `-- Dataset/                   # Ground-truth sentence data
|   |
|   `-- Telugu/
|       |-- Complete Dataset/              # Production corpus (14,480 sentences, 5 lakh aksharas)
|       |   |-- dataset/                   # 70/15/15 sentence split
|       |   |-- Detailed Results and Analysis/ # Character, complexity, and sentence workbooks
|       |   |-- model predictions without metrics/ # Raw prediction spreadsheets
|       |   |-- notebooks and code files/  # Saved PyTorch .pkl models and notebooks
|       |   |-- results summaries/         # Consolidated score tables
|       |   |-- count_unique_telugu.py     # Akshara extraction and tokenization script
|       |   `-- telugu_akshara_frequencies.xlsx
|       |
|       `-- sample dataset work/           # Initial 100-article prototyping track
|
|-- LICENSE                                # MIT License
`-- README.md                              # Master repository documentation
```

---

## Benchmark Highlights

### 1. English Sentence Segmentation (16 Hybrid Models)
Evaluated on test records from the 2,000-article corpus:

| Architecture | Transformer Backbone | Overall F1 | Overall Acc | Sent Break F1 | Sent Break Acc | Exact Match | BLEU | BERTScore F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN + CRF** | **DeBERTa_v3** | 0.8738 | 0.9076 | **0.8373** | **0.9797** | **0.8621** | 0.4696 | 0.9420 |
| **CNN + CRF** | **RoBERTa** | 0.8738 | 0.9076 | **0.8144** | 0.9764 | 0.8408 | 0.4696 | 0.9420 |
| **BiLSTM + CRF** | **DeBERTa_v3** | 0.9623 | 0.9747 | **0.7895** | 0.9719 | 0.8321 | 0.8157 | 0.9841 |
| **BiLSTM + CRF** | **RoBERTa** | 0.9623 | 0.9747 | 0.7822 | 0.9712 | 0.8250 | 0.8157 | 0.9841 |
| **BiGRU + CRF** | **RoBERTa** | **0.9644** | **0.9759** | 0.7774 | 0.9704 | 0.8226 | **0.8194** | **0.9850** |
| **BiGRU + CRF** | **DeBERTa_v3** | **0.9644** | **0.9759** | 0.7763 | 0.9701 | 0.8200 | **0.8194** | **0.9850** |

*Primary insight: 1D convolutions combined with DeBERTa-v3 attention representations excel at pinpointing discourse breaks (0.8373 F1, 86.21% exact sentence match), whereas recurrent encoders (BiGRU/BiLSTM) paired with RoBERTa/DeBERTa-v3 maximize total sequence reconstruction (0.8194 BLEU, 0.9850 BERTScore F1).*

### 2. Telugu Word Segmentation (Complete Dataset, 14,480 Sentences)
Evaluated on the 2,172 test sentences:

| State Scheme | Architecture | Macro F1 | Accuracy | Precision | Recall | BLEU | BERTScore F1 |
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

## Installation & Environment Setup

### 1. Clone the Repository
```powershell
git clone https://github.com/Vallurikeerthiram/ScriptioContinua.git
Set-Location ScriptioContinua
```

### 2. Configure Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install Dependencies
```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install transformers datasets accelerate
python -m pip install pandas openpyxl nltk bert-score wordfreq g2p_en tqdm beautifulsoup4 requests
```

---

## Reproduction and Execution Guide

### English Phase 1: Isolated Sentence Tagging
```powershell
# Run plain deep learning models (BiLSTM, GRU, RNN, CNN)
Set-Location "Phase 1/plain DL"
python train_models.py

# Run few-shot LLM baselines (requires Ollama running)
Set-Location "../LLM based"
python "type 1/run_type1.py" --model qwen3.5:4b --limit 10
```

### English Phase 2: Word and Sentence Segmentation
```powershell
# 50-Character window line-level word segmentation
Set-Location "Phase 2/English/Word Segmentation/50_Characters"
python 2state_4state.py
python 4state_CRF.py

# Sentence segmentation using hybrid Transformer models
# Open and run the interactive notebooks:
# Phase 2/English/Sentence Segmentation/notebook and code files/scriptio-continua-proj except cnn.ipynb
# Phase 2/English/Sentence Segmentation/notebook and code files/scriptio-continua-proj-cnn.ipynb
```

### Telugu Phase 2: Complete Dataset Inference
To load pre-trained Telugu model checkpoints and run inference:
```python
import pickle
import torch

vocab_path = "Phase 2/Telugu/Complete Dataset/notebooks and code files/2-state/telugu_2state_vocab.pkl"
model_path = "Phase 2/Telugu/Complete Dataset/notebooks and code files/2-state/telugu_2state_gru.pkl"

with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)

with open(model_path, "rb") as f:
    model = pickle.load(f)

model.eval()

text = "రాముడుమంచిబాలుడు"
indices = [vocab.get(ch, vocab.get("<UNK>", 0)) for ch in text]
inputs = torch.tensor([indices], dtype=torch.long)

with torch.no_grad():
    preds = torch.argmax(model(inputs), dim=-1).squeeze(0).tolist()

reconstructed = "".join([ch + (" " if p == 1 else "") for ch, p in zip(text, preds)]).strip()
print("Restored Text:", reconstructed)
```

---

## Detailed Documentation Directory Links

- [Phase 1 Master Documentation](Phase%201/README.md)
  - [Dataset Scraping & Cleaning](Phase%201/DATASET/Readme.md)
  - [Neural & CRF Sequence Taggers](Phase%201/plain%20DL/readme.md)
  - [Semantic Trie Engine](Phase%201/dictionary%20based/Readme.md)
  - [Few-Shot LLM Prompting Baselines](Phase%201/LLM%20based/README.md)
- [Phase 2 Master Documentation](Phase%202/README.md)
  - [English Phase 2 Overview](Phase%202/English/README.md)
    - [English Datasets](Phase%202/English/Dataset/README.md)
    - [16 Hybrid Transformer Sentence Segmentation Models](Phase%202/English/Sentence%20Segmentation/README.md)
    - [50-Character Line-Level Word Segmentation](Phase%202/English/Word%20Segmentation/50_Characters/README.md)
    - [Sentence-Level Word Segmentation Baseline](Phase%202/English/Word%20Segmentation/Sentence_Level/README.md)
  - [Telugu Phase 2 Overview](Phase%202/Telugu/README.md)
    - [Complete Telugu Dataset Benchmark](Phase%202/Telugu/Complete%20Dataset/README.md)
    - [Linguistic Complexity & Frequency Analysis](Phase%202/Telugu/Complete%20Dataset/Detailed%20Results%20and%20Analysis/README.md)
    - [Pre-Trained Weights & Notebooks](Phase%202/Telugu/Complete%20Dataset/notebooks%20and%20code%20files/README.md)
    - [Sample Dataset Prototyping](Phase%202/Telugu/sample%20dataset%20work/README.md)

---

## License & Citation

This project is licensed under the [MIT License](LICENSE).

Developed and maintained by **Valluri Keerthi Ram**.
