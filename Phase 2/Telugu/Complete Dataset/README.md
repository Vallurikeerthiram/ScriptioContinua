# Telugu Complete Dataset Module

This directory houses the complete production-scale Telugu *scriptio continua* corpus, model artifacts, evaluation notebooks, and fine-grained character/sentence analysis.

---

## Directory Map

```text
Complete Dataset/
|-- dataset/
|   `-- telugu_sentence_dataset_70_15_15_split.xlsx  # 14,480-sentence corpus (TRAIN, TEST, EVALUATION)
|
|-- Detailed Results and Analysis/                   # Multi-tier linguistic and performance analysis
|   |-- character level analysis/                   # Akshara frequency metrics and notebook
|   |-- sentence level metrics/                     # Detailed sentence prediction scores & summaries
|   |-- Workbooks/                                  # Complexity and frequency-grouped workbooks
|   `-- Telugumodelanalysis.xlsx                    # High-level model comparison workbook
|
|-- model predictions without metrics/               # Raw character-by-character prediction logs
|   |-- telugu_2state_predictions.xlsx
|   |-- telugu_4state_predictions.xlsx
|   `-- telugu_4state_crf_predictions.xlsx
|
|-- notebooks and code files/                       # Execution notebooks and pre-trained .pkl weights
|   |-- 2-state/                                    # 2-state notebook, BiLSTM, CNN, GRU, RNN, vocab .pkl
|   |-- 4-state/                                    # 4-state notebook, BiLSTM, CNN, GRU, RNN, vocab .pkl
|   `-- 4-state-crf/                                # 4-state CRF notebook, BiLSTM, CNN, GRU, RNN, vocab .pkl
|
|-- results summaries/                              # Consolidated score summaries
|   |-- telugu_2state_predictions_scores.xlsx
|   |-- telugu_4state_predictions_summary.csv
|   `-- telugu_4state_crf_predictions_summary.csv
|
|-- count_unique_telugu.py                          # Akshara tokenization & unique count script
`-- telugu_akshara_frequencies.xlsx                 # Comprehensive frequency distribution across corpus
```

---

## Dataset Statistics

The primary dataset (`telugu_sentence_dataset_70_15_15_split.xlsx`) comprises 14,480 total sentence records distributed across three strictly non-overlapping splits:

| Split Sheet | Sentence Rows | Purpose |
| :--- | ---: | :--- |
| `train` | 10,136 | Model parameter optimization |
| `test` | 2,172 | Primary unbiased benchmark evaluation |
| `Evaluation` | 2,172 | Secondary validation and cross-split stability check |

### Akshara Tokenization Logic
Telugu text is tokenized into phonetic syllabic units (aksharas) using the regular expression defined in `count_unique_telugu.py`:
```python
AKSHARA_REGEX = re.compile(
    r'[అ-హౘ-ౚ0-9A-Za-z]'           # Base consonant, independent vowel, or alphanumeric
    r'[ఁ-ఃా-ౌౕౖ]*'               # Attached vowel signs (matras) and diacritics
    r'(?:్[క-హౘ-ౚ][ఁ-ఃా-ౌౕౖ]*)*'   # Consonant conjuncts (vattulu / halant + consonant)
    r'్?'                            # Optional trailing virama
    r'[ఁ-ః]?'                        # Optional trailing anusvara / visarga
    r'|.'                            # Fallback to single character
)
```

Across the corpus, this extracts hundreds of thousands of distinct akshara occurrences, which are indexed in `telugu_akshara_frequencies.xlsx`.

---

## Benchmark Results (Test Split)

Full comparative metrics recorded in `Detailed Results and Analysis/sentence level metrics/scoresoutputs/all_models_summary.csv`:

| Model Suite | Architecture | F1 Score | Accuracy | Precision | Recall | BLEU | ROUGE-L F1 | METEOR | BERTScore F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2-State** | **GRU** | **0.7574** | **0.8150** | 0.7892 | 0.7538 | 0.2354 | 0.3043 | 0.2892 | 0.8444 |
| **2-State** | **BiLSTM** | 0.7522 | 0.8091 | 0.7792 | 0.7485 | 0.2458 | 0.3186 | 0.3035 | 0.8486 |
| **2-State** | **CNN** | 0.6417 | 0.7459 | 0.7009 | 0.6401 | 0.1802 | 0.3070 | 0.2671 | 0.8756 |
| **2-State** | **RNN** | 0.6350 | 0.7378 | 0.6961 | 0.6323 | 0.1704 | 0.2789 | 0.2416 | 0.8747 |
| **4-State** | **GRU** | **0.6807** | **0.7257** | 0.6928 | 0.6856 | 0.2673 | 0.3386 | 0.3455 | 0.8291 |
| **4-State** | **BiLSTM** | 0.6698 | 0.7157 | 0.6806 | 0.6756 | 0.2770 | 0.3526 | 0.3569 | 0.8393 |
| **4-State** | **RNN** | 0.5839 | 0.6333 | 0.6070 | 0.5896 | 0.2353 | 0.3079 | 0.3057 | 0.8389 |
| **4-State** | **CNN** | 0.5565 | 0.6051 | 0.5692 | 0.5614 | **0.3121** | **0.4534** | **0.4524** | **0.8895** |
| **4-State + CRF** | **GRU** | **0.6239** | **0.6674** | 0.6280 | 0.6300 | 0.2530 | 0.3118 | 0.3069 | 0.8244 |
| **4-State + CRF** | **BiLSTM** | 0.5900 | 0.6288 | 0.5946 | 0.5982 | 0.2512 | 0.3162 | 0.3191 | 0.8239 |
| **4-State + CRF** | **RNN** | 0.5495 | 0.5878 | 0.5531 | 0.5550 | 0.2715 | 0.3895 | 0.3857 | 0.8722 |
| **4-State + CRF** | **CNN** | 0.3542 | 0.3897 | 0.3550 | 0.3623 | 0.1329 | 0.1110 | 0.1135 | 0.7518 |

---

## Subdirectory Details

- [Detailed Results and Analysis](Detailed%20Results%20and%20Analysis/README.md): Character-level akshara frequency, complexity tiers, and sentence score summaries.
- [Model Predictions Without Metrics](model%20predictions%20without%20metrics/README.md): Raw predicted label sequences for all test instances.
- [Notebooks and Code Files](notebooks%20and%20code%20files/README.md): Training notebooks and saved `.pkl` model checkpoints ready for loading.
- [Results Summaries](results%20summaries/README.md): Score summary sheets and CSV metric exports.
