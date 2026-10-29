import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER_PATH = SCRIPT_DIR / "run_type1.py"
DEFAULT_PYTHON = Path(r"C:\Users\kamma\AppData\Local\Programs\Python\Python312\python.exe")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Type 1 sentence restoration for three requested models and "
            "save one Excel workbook per model."
        )
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--gemma-model", default="gemma4:e2b")
    parser.add_argument("--llama-model", default="llama4:scout")
    parser.add_argument("--deepseek-model", default="deepseek-v3.2:cloud")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--project-brief", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--few-shot-count", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def build_model_plan(args: argparse.Namespace) -> List[Tuple[str, str]]:
    return [
        ("Gemma 4", args.gemma_model),
        ("Llama 4", args.llama_model),
        ("DeepSeek", args.deepseek_model),
    ]


def build_command(args: argparse.Namespace, model_tag: str) -> List[str]:
    cmd: List[str] = [
        str(args.python),
        str(RUNNER_PATH),
        "--model",
        model_tag,
        "--output-dir",
        str(args.output_dir),
        "--few-shot-count",
        str(args.few_shot_count),
        "--batch-size",
        str(args.batch_size),
        "--timeout",
        str(args.timeout),
        "--sleep",
        str(args.sleep),
    ]

    if args.dataset is not None:
        cmd.extend(["--dataset", str(args.dataset)])
    if args.project_brief is not None:
        cmd.extend(["--project-brief", str(args.project_brief)])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.resume:
        cmd.append("--resume")

    return cmd


def ensure_paths(args: argparse.Namespace) -> None:
    if not args.python.exists():
        raise FileNotFoundError(f"Python not found: {args.python}")
    if not RUNNER_PATH.exists():
        raise FileNotFoundError(f"Type 1 runner not found: {RUNNER_PATH}")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    ensure_paths(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for label, model_tag in build_model_plan(args):
        print(f"\n=== Running {label} ({model_tag}) ===")
        command = build_command(args, model_tag)
        print("Command:", " ".join(f'"{part}"' if " " in part else part for part in command))
        completed = subprocess.run(command, cwd=SCRIPT_DIR.parent.parent)
        if completed.returncode != 0:
            print(f"{label} failed with exit code {completed.returncode}.", file=sys.stderr)
            return completed.returncode

    print("\nFinished all three model runs.")
    print("Expected output files:")
    for _, model_tag in build_model_plan(args):
        model_slug = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model_tag)
        print(args.output_dir / f"type1_{model_slug}_test_results.xlsx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
