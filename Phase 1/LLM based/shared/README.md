# Shared LLM Pipeline Logic

This folder contains the common logic used by all LLM-based task types.

## Files

- **`fewshot_ollama_pipeline.py`**: The core engine that:
  - Loads data from Excel workbooks.
  - Builds few-shot prompts based on the training data.
  - Interfaces with Ollama to run inference.
  - Parses JSON responses from the models.
  - Saves results back to Excel and maintains a JSONL cache for resuming.
