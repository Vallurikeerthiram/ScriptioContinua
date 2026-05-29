# DATASET: Simple Wikipedia Scriptio Continua Dataset

This folder contains the tools and data used to build the scriptio continua dataset from Simple Wikipedia articles.

## Contents

- **`pullArticles.py`**: A script to fetch random articles from Simple Wikipedia based on various domains (sports, science, technology, etc.). It ensures articles meet minimum length and quality requirements.
- **`Preprocessing.py`**: Processes the raw articles into sentence-level scriptio continua. It:
  - Tokenizes content into sentences.
  - Removes brackets and normalizes spaces.
  - Converts text to lowercase and removes all non-alphanumeric characters.
  - Computes statistics like word count, stopword count, and character counts.
- **`graphs.py`**: Likely used for visualizing dataset statistics.
- **`8000 rows/`**: Contains a dataset split with approximately 8,000 rows of processed sentences.
- **`18000 rows/`**: Contains a larger dataset split with approximately 18,000 rows.

## Data Workflow

1.  Run `pullArticles.py` to generate `simple_wikipedia_random_domain_dataset.xlsx`.
2.  Run `Preprocessing.py` to generate the sentence-level dataset with scriptio continua versions and ground truth labels.

## Outputs

The processed datasets include columns for:
- Original sentence
- Sentence without brackets/excess whitespace
- Scriptio continua version (lowercase, no spaces, no punctuation)
- Various counts (stopwords, spaces, words, characters, numbers, letters, special characters)
