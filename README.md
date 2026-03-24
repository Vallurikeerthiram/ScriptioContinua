# Scriptio Continua: Character-Level Word Segmentation Framework

**Scriptio Continua** is a comprehensive NLP research project designed to solve the automated word segmentation problem for continuous character strings. The framework evaluates and compares modern **Deep Learning Sequence Tagging** architectures against a **Deterministic Semantic Trie-based validation engine**.

---

## 🚀 Key Features

- **Automated Data Pipeline**: Multi-domain Wikipedia scraper with built-in deduplication (SHA256) and character-level transformation.
- **Neural Segmentation Suite**: Five character-level sequence labeling models (BiLSTM, CNN, CRF, GRU, RNN).
- **Dual Labeling Schemes**: Support for both Binary (Boundary/Non-boundary) and BIES (Begin, Inside, End, Single) tagging.
- **Semantic Validation Engine**: A high-precision Trie that verifies word candidates against millions of POS tags and lexical definitions.
- **Exhaustive Benchmarking**: Rigorous evaluation using NLP metrics: BLEU, METEOR, ROUGE-L, and BERTScore.

---

## 📂 Project Architecture

```text
ScriptioContinua/
├── dataset/                  # Data Generation & ETL
│   ├── pullArticles.py       # Wikipedia scraper (Domain-specific)
│   ├── Preprocessing.py      # Transformation & Statistics extraction
│   └── graphs.py             # Data distribution analysis
├── plain DL/                 # Deep Learning Module
│   ├── train_models.py       # Unified training/eval framework (PyTorch)
│   ├── SENT_based_split.xlsx # Structured dataset for modeling
│   └── code.py               # Preprocessing utilities
└── dictionary based/         # Algorithmic Module
    ├── dict/
    │   ├── universal_trie.py # Semantic Trie engine (Strict validation)
    │   └── *.csv, *.txt      # Lexical datasets (Moby POS, 1M Web Words)
    └── SENT_based_split.xlsx # Validation ground truth
```

---

## 🛠️ Technical Stack

- **Core**: Python 3.8+
- **Deep Learning**: PyTorch
- **Data Analysis**: Pandas, NumPy, OpenPyXL
- **Natural Language Processing**: NLTK, BeautifulSoup4, `bert-score`
- **Scraping & Utilities**: Requests, TQDM, Hashlib

---

## 🧪 Detailed Methodology

### 1. Data Transformation (The ETL Phase)
The system retrieves diverse articles from Simple Wikipedia and transforms them into "Scriptio Continua" state:
1.  **Cleaning**: Stripping brackets, special characters, and non-alphanumeric symbols.
2.  **Degradation**: Complete removal of whitespaces and conversion to lowercase.
3.  **Labeling**: Generating ground-truth label sequences for both Binary and BIES schemes.

### 2. Deep Learning (The Modeling Phase)
We treat word segmentation as a **Character-Level Sequence Labeling** task. Models ingest character embeddings and predict the likelihood of a word boundary at each position.
- **Architectures**: Evaluates BiLSTM (long-range dependencies), CNN (local n-grams), and CRF (label transition consistency).
- **Optimization**: Uses Adam optimizer with weighted decay and early stopping based on Validation F1 scores.

### 3. Semantic Trie (The Deterministic Phase)
A custom-built `UniversalTrie` acts as a verification engine. A word is only "accepted" if it passes a strict semantic check:
- **Lexical Check**: Does the string exist in the dictionary?
- **Semantic Check**: Does the string have a verified Part-of-Speech (POS) or Definition?
- **Pattern Check**: Does it match high-precision regex for URLs, Dates, or Currency?

---

## 🏃 Getting Started

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Vallurikeerthiram/ScriptioContinua.git
cd ScriptioContinua

# Install dependencies
pip install torch pandas requests beautifulsoup4 nltk bert-score openpyxl
```

### 2. End-to-End Pipeline
1.  **Scrape**: Run `python dataset/pullArticles.py` to build the raw corpus.
2.  **Preprocess**: Run `python dataset/Preprocessing.py` to create the Scriptio Continua dataset.
3.  **Segment & Train**: Navigate to `plain DL/` and run `python train_models.py` to train all five models and generate the benchmark report (`model_comparison_results.xlsx`).
4.  **Verify**: Use `python dictionary based/dict/universal_trie.py` for interactive semantic validation.

---

## 📊 Evaluation Metrics

The project benchmarks all models against:
- **Sequence Metrics**: Accuracy, Precision, Recall, and F1-Score of the predicted labels.
- **Reconstruction Metrics**: 
    - **BLEU / ROUGE / METEOR**: Overlap and alignment between reconstructed and original text.
    - **BERTScore**: Semantic similarity using contextual embeddings.

---
**Maintained by:** [Valluri Keerthi Ram](https://github.com/Vallurikeerthiram)
