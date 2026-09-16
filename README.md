# Scriptio Continua

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A character-first research framework for turning continuous writing into readable text, beginning with word-boundary recovery and progressing toward sentence, punctuation, and casing restoration.

Scriptio Continua investigates an old but still difficult language-processing problem: how do we recover readable text when the input contains no dependable spaces between words? This matters for historical manuscripts and inscriptions, digitized archival material, noisy OCR, and any language source where modern formatting cannot be assumed.

The project uses English as a controlled first benchmark to validate the full research workflow before applying it to genuinely unsegmented historical-language material. A Telugu word-segmentation track based on the Phase 1 task is being developed in parallel.

## At a glance

| Topic | Summary |
| --- | --- |
| Input | Continuous character strings such as `thequickbrownfox` |
| Current output | Word boundaries represented as binary or BIES character tags, then reconstructed with spaces |
| Core constraint | Word embeddings require word tokens; continuous text has no reliable word tokens yet |
| Main approaches | Character-level neural taggers, CRF-constrained decoding, dictionary/trie validation, and few-shot LLM baselines |
| Phase 1 | Word segmentation from already isolated English sentences |
| Phase 2 | Line-oriented and multi-sentence English records, compared with the sentence-wise setting |
| Long-term goal | Joint recovery of spaces, sentence boundaries, punctuation, and casing for historical continuous-script sources |
| Project status | Research prototype and benchmark repository; not a production transcription service |

## The problem

Historically, many writing systems and media used little or no separation between words. A modern reader or NLP pipeline is accustomed to receiving:

```text
The quick brown fox jumps over the lazy dog.
```

but may instead encounter:

```text
thequickbrownfoxjumpsoverthelazydog
```

The first missing layer is word segmentation. Without it, conventional word-level NLP cannot reliably create embeddings, parse syntax, retrieve information, translate text, or run downstream language models.

Scriptio Continua therefore begins at the character level:

```text
Continuous text
      -> character vocabulary and trainable character embeddings
      -> one tag per character
      -> reconstructed word boundaries
      -> readable text with spaces
```

For example:

```text
Input:         thequickbrownfox
STATE_2:       0010000100001001   (1 means "end of a word")
Reconstruction: the quick brown fox
```

The project also supports a richer BIES label scheme:

| Tag | Meaning |
| --- | --- |
| `B` | Beginning of a multi-character word |
| `I` | Inside a multi-character word |
| `E` | End of a multi-character word |
| `S` | A single-character word |

## Research progression

### Phase 1: establish word-boundary recovery

Phase 1 assumes the input is already a single sentence. It builds the English Scriptio Continua dataset and compares several ways to restore word boundaries:

- character-level BiLSTM, RNN, CNN, and GRU sequence taggers;
- CRF-enhanced sequence taggers for structured BIES decoding;
- dictionary and semantic-Trie validation methods; and
- few-shot LLM baselines through Ollama, including Qwen, Gemma, and DeepSeek configurations.

This phase answers the first question: can a model infer spaces without receiving pre-existing words or word embeddings?

### Phase 2: move from sentences to line-oriented records

Phase 2 makes the setting closer to document input. Its source workbook contains article records with line wrapping and, in many cases, multiple sentences per record. It compares the original sentence-wise setting with a harder line-oriented/multi-sentence setting while retaining character-level word-boundary prediction.

The released Phase 2 assets are the folders `50 char`, `sentence wise`, and `dataset_stats_outputs` at the repository root after merge.

### What the current code does and does not do

| Capability | Status in the checked code |
| --- | --- |
| Recover word spaces from continuous English input | Implemented and evaluated |
| Compare `STATE_2` and BIES (`STATE_4`) labels | Implemented |
| Compare plain neural encoders and CRF decoders | Implemented |
| Process line-wrapped/multi-sentence records as a data setting | Implemented |
| Predict sentence breaks | Not yet implemented as a model output |
| Predict punctuation | Not yet implemented as a model output |
| Restore capitalization | Not yet implemented as a model output |
| End-to-end inference from raw document text alone | Not yet implemented |

This distinction is intentional and important. The project vision is joint text restoration, but the current benchmark measures the word-boundary layer first.

## System architecture

```text
                         DATA CREATION AND LABELING

Simple Wikipedia / curated text
        -> cleaning and sentence processing
        -> continuous-script transformation
        -> STATE_2 and STATE_4/BIES character labels

                         RESTORATION APPROACHES

continuous characters
        +-------------------+---------------------+------------------+
        |                   |                     |                  |
        v                   v                     v                  v
Character neural      CRF-constrained       Dictionary /        Few-shot LLM
sequence taggers      BIES decoding         Trie validation      prompting
        |                   |                     |                  |
        +-------------------+---------------------+------------------+
                                    |
                                    v
                      predicted boundary labels and spaces
                                    |
                                    v
                       readable, word-segmented output
```

### Character-level neural models

The neural experiments learn embeddings over characters rather than words. The main encoder families are:

- BiLSTM
- bidirectional vanilla RNN
- CNN
- bidirectional GRU

For fair comparisons, the Phase 2 scripts use the same trainable 64-dimensional character embedding for every model. Default settings include a hidden size of 128, 128 CNN channels, dropout of 0.2, batch size of 64, Adam optimization, eight maximum epochs, and early stopping with patience two.

### Structured prediction with CRF

The CRF variants model dependencies between neighboring BIES labels. This helps prevent implausible tag transitions and turns boundary prediction into a sequence-level decoding problem rather than independent per-character classification.

### Dictionary and Trie baselines

The dictionary-based modules build a Trie over word lists and combine it with lexical-frequency, orthographic, phonotactic, pronunciation, and character-language-model validators. These methods provide an interpretable non-neural comparison point.

### Few-shot LLM baselines

The LLM pipeline uses a train sheet for examples and a test sheet for inference. It supports three output contracts:

1. direct readable-text restoration;
2. binary `0/1` boundary labels; and
3. BIES boundary labels.

This makes it possible to compare compact character models with prompt-based models without requiring LLM fine-tuning. Ollama is used for local or configured model access.

## Data and labels

### Phase 1 dataset pipeline

[`DATASET`](DATASET) contains the collection and preprocessing workflow:

1. `pullArticles.py` gathers Simple Wikipedia articles across domains and removes duplicate or near-duplicate content using fingerprints and SHA256 checks.
2. `Preprocessing.py` cleans text, tokenizes it into sentences, normalizes it, and creates continuous-script examples with metadata.
3. The resulting sentence-level records are used to produce binary and BIES word-boundary labels.

### Phase 2 sentence-wise reference corpus

[`sentence wise/2000_articles_sentences_final.xlsx`](<sentence wise/2000_articles_sentences_final.xlsx>) provides the reference sentence-level setting used in this workspace.

| Property | Value |
| --- | ---: |
| Articles | 2,000 |
| Sentence records | 20,747 |
| Continuous-script characters | 1,571,692 |
| Source words | 325,790 |
| Main columns | `Art_id`, `Sent_id`, `Sentence`, `scriptio_continua`, `2state`, `4state` |

### Phase 2 line-oriented / multi-sentence corpus

[`50 char/SENT_based_split.xlsx`](<50 char/SENT_based_split.xlsx>) stores 2,000 non-overlapping article records. Each record may have line wrapping and multiple sentences.

| Split | Article records | Article overlap with other splits |
| --- | ---: | --- |
| Train | 1,400 | None |
| Validation | 300 | None |
| Test | 300 | None |

The companion statistics workbook records 46,213 physical lines. Many lines contain only part of a sentence or span sentence boundaries, which is why this setting is more representative of document-like input than Phase 1's isolated sentences.

The workbook contract is:

| Column | Description |
| --- | --- |
| `ART_ID` | Article identifier |
| `SENT_ORI` | Ground-truth formatted text, including punctuation and stored line layout |
| `SCRIPT_CONTIN` | Continuous-script counterpart; stored cells can retain line breaks |
| `STATE_2` | Binary word-boundary tag sequence aligned to normalized continuous characters |
| `STATE_4` | BIES word-boundary tag sequence aligned to normalized continuous characters |

### Phase 2 preprocessing note

The multi-sentence scripts normalize input to letters, marks, and numbers; whitespace and punctuation are removed before character encoding. They use the ground-truth `SENT_ORI` field to locate sentence spans and expand each record into sentence-level samples before training.

As a result, Phase 2 currently evaluates word-boundary recovery on examples derived from line-oriented records. It does not learn sentence-break labels from end-of-line positions, nor does it reconstruct punctuation or casing. The included Phase 2 examples also preserve case and digits in `SCRIPT_CONTIN`, so they are not a capitalization-restoration benchmark.

### Data-quality and reproducibility note

Before training, the multi-sentence scripts reject rows whose normalized character count does not match both label sequences. With the included workbook, the retained population is:

| Split | Source records | Rejected for label-length mismatch | Expanded sentence samples used |
| --- | ---: | ---: | ---: |
| Train | 1,400 | 964 | 3,299 |
| Validation | 300 | 205 | 682 |
| Test | 300 | 202 | 684 |

The stored Phase 2 multi-sentence metrics therefore apply to the retained, expanded population. This filtering is documented so future experiments can repair the alignment differences, version a clean split, and make stronger controlled comparisons.

## Experimental evidence

Phase 2 evaluates two label schemes, four neural encoders, and CRF-enhanced BIES decoders. The result workbooks include:

- character-label macro precision, recall, F1, and accuracy;
- reconstruction BLEU;
- ROUGE-1, ROUGE-2, and ROUGE-L F1;
- METEOR; and
- BERTScore precision, recall, and F1 when the optional dependency is available.

The strongest stored BIES+CRF test result in each Phase 2 track is shown below. Scores are recorded experiment outputs, not deployment guarantees.

| Track | Best stored model | Test macro F1 | Test accuracy | BLEU | BERTScore F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Sentence-wise reference | GRU + CRF | 0.9050 | 0.9631 | 0.7936 | 0.9764 |
| Line-oriented / multi-sentence | GRU + CRF | 0.8063 | 0.8971 | 0.5280 | 0.9365 |

The two Phase 2 tracks are not a single controlled ablation of line context: their preprocessing and retained populations differ. They should be interpreted as benchmark evidence for two related settings, not as a direct causal comparison.

Detailed artifacts:

- Phase 2 sentence-wise `STATE_2` and `STATE_4`: [model_comparison_results (1).xlsx](<sentence wise/model_comparison_results (1).xlsx>)
- Phase 2 sentence-wise BIES + CRF: [model_4state_crf_results (1).xlsx](<sentence wise/model_4state_crf_results (1).xlsx>)
- Phase 2 line-oriented `STATE_2` and `STATE_4`: [model_comparison_results_multi_sentence.xlsx](<50 char/model_comparison_results_multi_sentence.xlsx>)
- Phase 2 line-oriented BIES + CRF: [model_comparison_results_CRF_multi_sentence.xlsx](<50 char/model_comparison_results_CRF_multi_sentence.xlsx>)
- Dataset and cross-track statistics: [dataset_stats_outputs](dataset_stats_outputs)

## Repository map

The repository brings research assets, data preparation, models, and experiment records together in one place.

```text
ScriptioContinua/
|
|-- DATASET/                      # Simple Wikipedia collection and preprocessing
|-- Models/                       # Phase 1 neural and CRF experiments
|   |-- plain DL models/
|   |-- DL models_CRF/
|   `-- 1st run/
|
|-- LLM based/                    # Few-shot Ollama experiments
|   |-- type 1/                   # Direct readable-text restoration
|   |-- type 2/                   # Binary boundary labels
|   |-- type 3/                   # BIES boundary labels
|   `-- shared/
|
|-- dict based model/             # Trie and linguistic validation baselines
|
|-- sentence wise/                # Phase 2 sentence-level reference experiments
|-- 50 char/                      # Phase 2 line-oriented / multi-sentence experiments
|-- dataset_stats_outputs/         # Phase 2 data and result comparisons
|
|-- Literature survey/             # Research background
`-- extras/                        # Metrics and workbook utilities
```

Some older exploratory folders remain in the repository to preserve research history. The directories above are the recommended starting points.

## Quick start

### 1. Clone and create an environment

```powershell
git clone https://github.com/Vallurikeerthiram/ScriptioContinua.git
Set-Location ScriptioContinua

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch numpy pandas openpyxl requests beautifulsoup4 nltk bert-score wordfreq g2p_en tqdm
```

The repository contains multiple research routes. Install additional packages only when a specific submodule requests them; some older experiments may also use TensorFlow/Keras.

### 2. Build a sentence-level corpus (Phase 1)

```powershell
Set-Location DATASET
python pullArticles.py
python Preprocessing.py
```

See [`DATASET/Readme.md`](<DATASET/Readme.md>) for source-data controls, filtering, deduplication, and output details.

### 3. Run neural word-segmentation experiments

The Phase 1 model scripts are documented in:

- [`Models/plain DL models`](<Models/plain DL models>) for standard neural taggers;
- [`Models/DL models_CRF`](<Models/DL models_CRF>) for CRF-enhanced models.

For the Phase 2 line-oriented experiments, run from the folder containing the supplied workbook:

```powershell
Set-Location "50 char"
python train_models_multi_sentence.py
```

The BIES+CRF Phase 2 experiment requires a CUDA-enabled PyTorch installation:

```powershell
Set-Location "50 char"
python train_models_CRF_multi_sentence.py
```

The non-CRF script can fall back to CPU. Both scripts write to their existing result-workbook names, so set a new `OUTPUT_PATH` before running if the stored artifacts must be preserved.

### 4. Run an LLM baseline

Install and start Ollama, then choose one of the supported tasks. A small smoke test can be run from the repository root with:

```powershell
python "LLM based/type 1/run_type1.py" --model qwen3.5:4b --limit 10
```

Use `type 2` for binary state output and `type 3` for BIES output. The LLM documentation explains model selection, batch sizing, resume support, and Excel outputs: [`LLM based/README.md`](<LLM based/README.md>).

### Historical Phase 2 sentence-wise scripts

[`sentence wise/train_model.py`](<sentence wise/train_model.py>) and [`sentence wise/train_4state_crf.py`](<sentence wise/train_4state_crf.py>) are preserved experiment scripts. Their paths point to a historical absolute `c:/PROJECT_PHASE/.../SENT_based_split.xlsx` location that is not bundled as a directly runnable input in this folder.

To reproduce those runs, update `EXCEL_PATH` and `OUTPUT_PATH` and provide `TRAIN`, `VAL`, and `TEST` sheets containing at least `ART_ID`, `PARA_ID`, `SENT_ID`, `SENT_ORI`, `SCRIPT_CONTIN`, `STATE_2`, and `STATE_4`. The raw sentence-reference workbook has a different schema and is not a drop-in replacement. The CRF script requires CUDA by default.

## Roadmap

1. Repair and version the line-oriented label alignment so every source record is available for training and evaluation.
2. Make data paths and experiment configuration command-line driven rather than source-code constants.
3. Add an explicit line-boundary feature when document layout is meaningful.
4. Train a joint model that predicts word boundaries, sentence boundaries, punctuation, and casing together.
5. Build inference that accepts raw continuous input without access to ground-truth `SENT_ORI`.
6. Extend the validated workflow to Telugu and evaluate it on truly historical continuous-script material.

## License and contact

This repository is available under the [MIT License](LICENSE).

Maintained by [Valluri Keerthi Ram](https://github.com/Vallurikeerthiram).
