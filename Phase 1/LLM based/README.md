# LLM Based Few-Shot Pipeline

This folder contains a few-shot, no-training pipeline for your `SENT_ID.xlsx` dataset.

Folder layout:

- `type 1/run_type1.py`: input scriptio continua, expect direct sentence output.
- `type 2/run_type2.py`: input scriptio continua, expect 0/1 state output.
- `type 3/run_type3.py`: input scriptio continua, expect BIES output.
- `shared/fewshot_ollama_pipeline.py`: shared loader, prompt builder, Ollama caller, and Excel writer.
- `project_brief.txt`: the project description sent to the model.
- `model_setup.md`: notes on which requested models fit this laptop.

All runs use:

- Train sheet for few-shot examples.
- Test sheet for inference.
- Excel output with the columns you requested:
  - `sent ID`
  - `scriptio continua sentence`
  - `ground truth sent`
  - `ground truth 4 state`
  - `ground truth 2 state`
  - `output direct output`
  - `output 4 state`
  - `output 2 state`

Example commands:

```powershell
& 'C:\Users\kamma\AppData\Local\Programs\Python\Python312\python.exe' '.\LLM based\type 1\run_type1.py' --model qwen3.5:4b --limit 10
& 'C:\Users\kamma\AppData\Local\Programs\Python\Python312\python.exe' '.\LLM based\type 2\run_type2.py' --model qwen3.5:4b --limit 10
& 'C:\Users\kamma\AppData\Local\Programs\Python\Python312\python.exe' '.\LLM based\type 3\run_type3.py' --model qwen3.5:4b --limit 10
```

Useful options:

- `--limit 10` for a smoke test before the full test sheet.
- `--resume` to continue from the JSONL cache if a run is interrupted.
- `--batch-size 1` for safest parsing, or increase it later if you want to experiment.
- `--model gemma4:e2b` to switch to the Gemma 4 edge model.
