from pathlib import Path
import sys


SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from fewshot_ollama_pipeline import run_cli


if __name__ == "__main__":
    run_cli(default_task="type3", default_output_dir=Path(__file__).resolve().parent)
