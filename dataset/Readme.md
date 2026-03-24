# Dataset Generation & Preprocessing

This module provides an automated pipeline for creating high-quality, domain-specific "Scriptio Continua" datasets from Simple Wikipedia articles.

## Contents

*   **`pullArticles.py`**: The scraper. It uses the Simple Wikipedia API to fetch articles across domains like science, technology, and sports.
*   **`Preprocessing.py`**: The transformer. It cleans the text, tokenizes sentences, and produces the final "Scriptio Continua" format for model training.
*   **`graphs.py`**: Visualization tool for analyzing dataset statistics (distributions, word frequencies, etc.).

---

## Methodology

### 1. Data Scraping (`pullArticles.py`)
The script identifies Wikipedia categories and scrapes articles using:
*   **Domain Filtering**: Keyword-based classification into categories (science, sports, politics, etc.).
*   **Quality Control**: Minimum paragraph lengths (150 chars) and a defined range of paragraphs (2–10) per article.
*   **Integrity Checks**: SHA256 hashing and fingerprinting (using the first 300 words) ensure that identical or near-identical content is not duplicated.
*   **Rate Limiting**: Integrated sleep cycles and user-agent rotation for reliable API usage.

### 2. Preprocessing & Transformation (`Preprocessing.py`)
This script converts clean English sentences into the degraded "Scriptio Continua" state:
1.  **Cleaning**: Removes brackets (parentheses, square, curly) and excessive whitespace.
2.  **Normalization**: Lowercases all text.
3.  **Segmentation**: Tokenizes the article into individual sentences using NLTK.
4.  **Encoding**: Strips all non-alphanumeric characters and spaces.
5.  **Statistical Calculation**: Computes metrics (stopword count, word count, character count, digits/letters/special characters) to provide metadata for each sentence.

---

## How to Run

### Step 1: Scrape Articles
```bash
python pullArticles.py
```
*   Config: Edit `TARGET_COUNT` (default: 2000) or `OUTPUT_FILE` in the script as needed.

### Step 2: Transform to Scriptio Continua
```bash
python Preprocessing.py
```
*   Input: `simple_wikipedia_random_domain_dataset.xlsx`.
*   Output: `simple_wikipedia_sentence_level_scriptio_continua.xlsx`.
