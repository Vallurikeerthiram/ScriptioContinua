# LLM Task Type 1: Direct Sentence Restoration

This folder implements the "Direct Restoration" task where the LLM is given a scriptio continua string and asked to output the original sentence with spaces restored.

## Key Files

- **`run_type1.py`**: The entry point for running Type 1 experiments.
- **`run_type1_requested_models.py`**: A variant script for running specific models.
- **`add_type1_metrics_sheets.py`**: Adds metric calculation sheets to the results workbooks.

## Usage

```powershell
python run_type1.py --model qwen3.5:4b --limit 10
```
