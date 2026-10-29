# LLM Task Type 3: BIES State Labeling

This folder implements the "BIES Labeling" task where the LLM is asked to provide a sequence of B, I, E, S labels for each character in the scriptio continua input:
- **B**: Beginning of a word.
- **I**: Inside a word.
- **E**: End of a word.
- **S**: Single-character word.

## Key Files

- **`run_type3.py`**: The entry point for running Type 3 experiments.

## Usage

```powershell
python run_type3.py --model qwen3.5:4b --limit 10
```
