# Scriptio Continua: Word Segmentation Framework

**Scriptio Continua** is a comprehensive research project designed to solve the automated word segmentation problem for continuous character strings in English. This framework evaluates and compares modern **Deep Learning Sequence Tagging** architectures, **Dictionary-Driven Semantic Trie validation**, and **Few-Shot Large Language Model (LLM)** prompting.

---

## 🚀 Key Features

- **Automated Data Pipeline**: Multi-domain Wikipedia scraper with built-in deduplication (SHA256) and character-level transformation.
- **Neural Segmentation Suite**: Character-level sequence labeling models including BiLSTM, CNN, CRF, GRU, and RNN.
- **LLM Few-Shot Pipeline**: Leverages models like Qwen and Gemma via Ollama for direct restoration and state labeling without fine-tuning.
- **Semantic Validation Engine**: A high-precision Trie that verifies word candidates against lexical definitions and Part-of-Speech (POS) tags.
- **Dual Labeling Schemes**: Support for both Binary (Boundary/Non-boundary) and BIES (Begin, Inside, End, Single) tagging.
- **Exhaustive Benchmarking**: Rigorous evaluation using standard NLP metrics: BLEU, METEOR, ROUGE-L, and BERTScore.

---

## 📂 Project Architecture

- **`DATASET/`**: Automated data generation and ETL (Scraping, Transformation, Statistics).
- **`dict based model/`**: Algorithmic module using a Semantic Trie engine and various word-likelihood validators (orthographic, phonotactic, pronounceability).
- **`Models/`**: Deep Learning module containing plain neural networks (BiLSTM, GRU, etc.) and CRF-enhanced architectures.
- **`LLM based/`**: Few-shot prompting pipeline for word segmentation using Large Language Models via Ollama.
- **`Literature survey/`**: Collection of research papers and reports related to word segmentation and scriptio continua.
- **`extras/`**: Utility scripts for generating metrics and processing workbooks.
- **`scripts/`**: Miscellaneous automation and utility scripts.

---

## 🛠️ Technical Stack

- **Core**: Python 3.8+
- **Deep Learning**: PyTorch, TensorFlow/Keras
- **LLM Interface**: Ollama
- **Data Analysis**: Pandas, NumPy, OpenPyXL
- **Natural Language Processing**: NLTK, BeautifulSoup4, `bert-score`, `wordfreq`, `g2p_en`
- **Scraping & Utilities**: Requests, TQDM, Hashlib

---

## 🏃 Getting Started

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Vallurikeerthiram/ScriptioContinua.git
cd ScriptioContinua

# Install dependencies
pip install torch pandas requests beautifulsoup4 nltk bert-score openpyxl wordfreq g2p_en
```

### 2. End-to-End Pipeline
1.  **Scrape**: Run `python DATASET/pullArticles.py` to build the raw corpus.
2.  **Preprocess**: Run `python DATASET/Preprocessing.py` to create the Scriptio Continua dataset.
3.  **Train DL Models**: Navigate to `Models/plain DL models/` or `Models/DL models_CRF/` and run the respective training scripts.
4.  **Run LLM Pipeline**: Navigate to `LLM based/` and use the `run_typeX.py` scripts with Ollama.
5.  **Dictionary Validation**: Explore `dict based model/valid word finder/` for dictionary and pronounceability-based segmentation.

---

## 📊 Evaluation Metrics Benchmarking

The project benchmarks all models against:
- **Sequence Metrics**: Accuracy, Precision, Recall, and F1-Score of the predicted labels.
- **Reconstruction Metrics**: 
    - **BLEU / ROUGE / METEOR**: Overlap and alignment between reconstructed and original text.
    - **BERTScore**: Semantic similarity using contextual embeddings.

---
**Maintained by:** [Valluri Keerthi Ram](https://github.com/Vallurikeerthiram)
