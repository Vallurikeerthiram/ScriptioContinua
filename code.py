from __future__ import annotations

import importlib.util
import sysconfig
from pathlib import Path


if __name__ == "code":
    stdlib_code_path = Path(sysconfig.get_path("stdlib")) / "code.py"
    spec = importlib.util.spec_from_file_location(__name__, stdlib_code_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load stdlib code module from {stdlib_code_path}")
    stdlib_code_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stdlib_code_module)
    globals().update(stdlib_code_module.__dict__)
else:
    from train_models import run_experiment

    if __name__ == "__main__":
        run_experiment()
