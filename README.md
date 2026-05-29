# ScriptioContinua: Word Segmentation for English Scriptio Continua

This project focuses on the restoration of spaces in English text that has been converted to "Scriptio Continua" (text without spaces, punctuation, or capitalization). It explores various methodologies for word segmentation, including dictionary-based models, deep learning architectures (including CRF), and Large Language Model (LLM) few-shot prompting.

## Project Structure

- **`DATASET/`**: Contains scripts and raw data for creating the dataset from Simple Wikipedia.
- **`dict based model/`**: Implements word segmentation using dictionary-driven approaches and word likelihood validators.
- **`Models/`**: Contains Deep Learning models, including plain neural networks and Conditional Random Fields (CRF) for sequence labeling.
- **`LLM based/`**: A pipeline for few-shot word segmentation using Large Language Models via Ollama.
- **`Literature survey/`**: Collection of research papers and reports related to word segmentation and scriptio continua.
- **`extras/`**: Utility scripts for generating metrics and processing workbooks.
- **`scripts/`**: Miscellaneous shell scripts for automation.

## Key Methodologies

1.  **Dictionary-Based**: Uses a trie-based dictionary and various validators (orthographic, phonotactic, pronounceability) to identify valid word boundaries.
2.  **Deep Learning**: Employs sequence labeling techniques where each character is classified (e.g., 0/1 for word ends or BIES for Beginning, Inside, End, Single-word).
3.  **LLM Few-Shot**: Leverages the zero/few-shot capabilities of models like Qwen and Gemma to restore sentences directly or provide state labels.

## Getting Started

Refer to the README files within each subdirectory for specific instructions on running the models and processing the data.
