# LLM Task Type 2: 0/1 State Labeling

This folder implements the "0/1 Labeling" task where the LLM is asked to provide a sequence of 0s and 1s corresponding to the characters in the scriptio continua input, where 1 indicates a word boundary (end of a word).

## Key Files

- **`run_type2.py`**: The entry point for running Type 2 experiments.

## Usage

```powershell
python run_type2.py --model qwen3.5:4b --limit 10
```
