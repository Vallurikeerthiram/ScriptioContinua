import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = ROOT_DIR / "DATASET" / "18000 rows" / "new dataset" / "SENT_ID.xlsx"
DEFAULT_PROJECT_BRIEF_PATH = ROOT_DIR / "LLM based" / "project_brief.txt"

SCRIPT_NORMALIZER = re.compile(r"[^a-z0-9]+")
STATE2_PATTERN = re.compile(r"[01]")
STATE4_PATTERN = re.compile(r"[SBEMI]", re.IGNORECASE)


@dataclass
class SentenceRecord:
    art_id: str
    para_id: str
    sent_id: str
    sentence: str
    script: str
    state2: str
    state4: str


def normalize_script(text: str) -> str:
    return SCRIPT_NORMALIZER.sub("", str(text).lower())


def parse_state2(value: str, n_chars: int) -> List[str]:
    flags = STATE2_PATTERN.findall(str(value))
    if len(flags) < n_chars:
        flags += ["0"] * (n_chars - len(flags))
    return flags[:n_chars]


def parse_state4(value: str, n_chars: int) -> List[str]:
    tags = [tag.upper() for tag in STATE4_PATTERN.findall(str(value))]
    tags = ["M" if tag == "I" else tag for tag in tags]
    if len(tags) < n_chars:
        tags += ["M"] * (n_chars - len(tags))
    return tags[:n_chars]


def format_state2(flags: List[str]) -> str:
    return " ".join(flags)


def format_state4(tags: List[str]) -> str:
    return " ".join(tags)


def state4_to_state2(tags: List[str]) -> List[str]:
    return ["1" if tag in {"S", "E"} else "0" for tag in tags]


def state2_to_state4(flags: List[str]) -> List[str]:
    if not flags:
        return []

    boundaries = [index for index, flag in enumerate(flags) if flag == "1"]
    if not boundaries or boundaries[-1] != len(flags) - 1:
        boundaries.append(len(flags) - 1)

    tags: List[str] = []
    start = 0
    for end in boundaries:
        word_len = (end - start) + 1
        if word_len <= 0:
            continue
        if word_len == 1:
            tags.append("S")
        else:
            tags.append("B")
            tags.extend(["M"] * max(word_len - 2, 0))
            tags.append("E")
        start = end + 1

    if len(tags) < len(flags):
        remainder = len(flags) - len(tags)
        if remainder == 1:
            tags.append("S")
        else:
            tags.append("B")
            tags.extend(["M"] * max(remainder - 2, 0))
            tags.append("E")
    return tags[: len(flags)]


def state2_to_sentence(script: str, flags: List[str]) -> str:
    pieces: List[str] = []
    n_chars = min(len(script), len(flags))
    for index in range(n_chars):
        pieces.append(script[index])
        if flags[index] == "1" and index < n_chars - 1:
            pieces.append(" ")
    return re.sub(r"\s+", " ", "".join(pieces)).strip()


def state4_to_sentence(script: str, tags: List[str]) -> str:
    words: List[str] = []
    current: List[str] = []
    n_chars = min(len(script), len(tags))
    for index in range(n_chars):
        ch = script[index]
        tag = "M" if tags[index] == "I" else tags[index]
        if tag == "S":
            if current:
                words.append("".join(current))
                current = []
            words.append(ch)
        elif tag == "B":
            if current:
                words.append("".join(current))
            current = [ch]
        elif tag == "E":
            if current:
                current.append(ch)
                words.append("".join(current))
                current = []
            else:
                words.append(ch)
        else:
            if not current:
                current = [ch]
            else:
                current.append(ch)
    if current:
        words.append("".join(current))
    return re.sub(r"\s+", " ", " ".join(words)).strip()


def extract_normalized_words(text: str) -> List[str]:
    words: List[str] = []
    for chunk in re.split(r"\s+", str(text).strip()):
        normalized = normalize_script(chunk)
        if normalized:
            words.append(normalized)
    return words


def words_to_state4(script: str, words: List[str]) -> Optional[List[str]]:
    joined = "".join(words)
    if joined != script:
        return None

    tags: List[str] = []
    for word in words:
        if len(word) == 1:
            tags.append("S")
        else:
            tags.append("B")
            tags.extend(["M"] * max(len(word) - 2, 0))
            tags.append("E")
    return tags


def convert_direct_output(script: str, raw_prediction: str) -> Dict[str, str]:
    output_sentence = str(raw_prediction).strip()
    normalized_words = extract_normalized_words(output_sentence)
    status = "exact_normalized_match"

    if normalized_words and "".join(normalized_words) == script:
        usable_words = normalized_words
    elif normalized_words and sum(len(word) for word in normalized_words) == len(script):
        usable_words = []
        cursor = 0
        for word in normalized_words:
            usable_words.append(script[cursor : cursor + len(word)])
            cursor += len(word)
        status = "repaired_from_word_lengths"
    else:
        usable_words = [script] if script else []
        status = "fallback_single_word"

    predicted_state4 = words_to_state4(script, usable_words) or state2_to_state4(
        ["0"] * max(len(script) - 1, 0) + (["1"] if script else [])
    )
    predicted_state2 = state4_to_state2(predicted_state4)

    return {
        "output_sentence": output_sentence or " ".join(usable_words),
        "output_state2": format_state2(predicted_state2),
        "output_state4": format_state4(predicted_state4),
        "prediction_status": status,
    }


def convert_state2_output(script: str, raw_prediction: str) -> Dict[str, str]:
    flags = parse_state2(raw_prediction, len(script))
    predicted_state4 = state2_to_state4(flags)
    return {
        "output_sentence": state2_to_sentence(script, flags),
        "output_state2": format_state2(flags),
        "output_state4": format_state4(predicted_state4),
        "prediction_status": "parsed_state2",
    }


def convert_state4_output(script: str, raw_prediction: str) -> Dict[str, str]:
    tags = parse_state4(raw_prediction, len(script))
    predicted_state2 = state4_to_state2(tags)
    return {
        "output_sentence": state4_to_sentence(script, tags),
        "output_state2": format_state2(predicted_state2),
        "output_state4": format_state4(tags),
        "prediction_status": "parsed_state4",
    }


def load_records(dataset_path: Path, sheet_name: str) -> List[SentenceRecord]:
    df = pd.read_excel(dataset_path, sheet_name=sheet_name)
    records: List[SentenceRecord] = []
    for row in df.to_dict(orient="records"):
        script = normalize_script(row.get("SCRIPT_CONTIN", ""))
        if not script:
            continue
        records.append(
            SentenceRecord(
                art_id=str(row.get("ART_ID", "")),
                para_id=str(row.get("PARA_ID", "")),
                sent_id=str(row.get("SENT_ID", "")),
                sentence=str(row.get("SENT_ORI", "")).strip(),
                script=script,
                state2=format_state2(parse_state2(str(row.get("STATE_2", "")), len(script))),
                state4=format_state4(parse_state4(str(row.get("STATE_4", "")), len(script))),
            )
        )
    return records


def select_few_shot_examples(train_records: List[SentenceRecord], count: int) -> List[SentenceRecord]:
    if not train_records:
        return []

    by_length = sorted(train_records, key=lambda record: len(record.script))
    selectors = [
        lambda record: len(record.script) <= 25,
        lambda record: 26 <= len(record.script) <= 50,
        lambda record: 51 <= len(record.script) <= 90,
        lambda record: 91 <= len(record.script) <= 140,
        lambda record: bool(re.search(r"\d", record.sentence)),
        lambda record: any(mark in record.sentence for mark in [",", ";", ":", "(", ")"]),
    ]

    chosen: List[SentenceRecord] = []
    seen_ids = set()
    for selector in selectors:
        match = next((record for record in by_length if selector(record) and record.sent_id not in seen_ids), None)
        if match is None:
            continue
        chosen.append(match)
        seen_ids.add(match.sent_id)
        if len(chosen) >= count:
            return chosen[:count]

    if len(chosen) < count:
        step = max(len(by_length) // max(count, 1), 1)
        for index in range(0, len(by_length), step):
            record = by_length[index]
            if record.sent_id in seen_ids:
                continue
            chosen.append(record)
            seen_ids.add(record.sent_id)
            if len(chosen) >= count:
                break

    return chosen[:count]


def format_examples(task_type: str, examples: List[SentenceRecord]) -> str:
    blocks: List[str] = []
    for index, example in enumerate(examples, start=1):
        if task_type == "type1":
            answer = example.sentence
            output_key = "output_sentence"
        elif task_type == "type2":
            answer = example.state2
            output_key = "output_state2"
        else:
            answer = example.state4
            output_key = "output_state4"

        block = "\n".join(
            [
                f"Example {index}",
                "{",
                f'  "sent_id": "{example.sent_id}",',
                f'  "script": "{example.script}",',
                f'  "ground_truth_sentence": {json.dumps(example.sentence)},',
                f'  "{output_key}": {json.dumps(answer)}',
                "}",
            ]
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def build_system_message(project_brief: str, task_type: str) -> str:
    task_specific = {
        "type1": (
            "Task type 1: restore a readable sentence from scriptio continua. "
            "The normalized prediction must preserve the exact character order of the input script."
        ),
        "type2": (
            "Task type 2: output one 0 or 1 label per input character. "
            "Use 1 at the end of every word, including the final character."
        ),
        "type3": (
            "Task type 3: output one B, M, E, or S label per input character using BIES word-boundary tags."
        ),
    }[task_type]
    return "\n".join(
        [
            project_brief.strip(),
            "",
            task_specific,
            "Return only valid JSON and no extra commentary.",
        ]
    )


def build_user_message(task_type: str, examples: List[SentenceRecord], batch: List[SentenceRecord]) -> str:
    if task_type == "type1":
        schema_note = (
            'Each JSON item must contain "sent_id" and "prediction". '
            '"prediction" must be the restored sentence as a string.'
        )
    elif task_type == "type2":
        schema_note = (
            'Each JSON item must contain "sent_id" and "prediction". '
            '"prediction" must be a space-separated 0/1 label sequence with one label per character.'
        )
    else:
        schema_note = (
            'Each JSON item must contain "sent_id" and "prediction". '
            '"prediction" must be a space-separated B/M/E/S label sequence with one label per character.'
        )

    items_to_solve = [
        {
            "sent_id": record.sent_id,
            "script": record.script,
        }
        for record in batch
    ]
    return "\n".join(
        [
            "Few-shot examples from the train sheet:",
            format_examples(task_type, examples),
            "",
            schema_note,
            "Solve the following items and return only a JSON array:",
            json.dumps(items_to_solve, ensure_ascii=True, indent=2),
        ]
    )


def extract_json_payload(raw_text: str):
    cleaned = str(raw_text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    for opener, closer in [("[", "]"), ("{", "}")]:
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start == -1 or end == -1 or end < start:
            continue
        candidate = cleaned[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("Could not parse JSON payload from model response.")


def estimate_num_predict(task_type: str, records: List[SentenceRecord]) -> int:
    total_chars = sum(len(record.script) for record in records)
    if task_type == "type1":
        return max(160, (total_chars // 2) + 96)
    return max(320, (total_chars * 4) + 96)


def chat_with_ollama(
    model: str,
    system_message: str,
    user_message: str,
    timeout_s: int,
    num_predict: int,
) -> str:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": "20m",
        "options": {
            "temperature": 0,
            "top_p": 0.9,
            "num_predict": num_predict,
        },
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    }
    request = urllib.request.Request(
        url="http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Ollama HTTP error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama at http://127.0.0.1:11434. Make sure the Ollama app is running."
        ) from exc

    return str(body.get("message", {}).get("content", "")).strip()


def convert_prediction(task_type: str, script: str, raw_prediction: str) -> Dict[str, str]:
    if task_type == "type1":
        return convert_direct_output(script, raw_prediction)
    if task_type == "type2":
        return convert_state2_output(script, raw_prediction)
    return convert_state4_output(script, raw_prediction)


def load_project_brief(project_brief_path: Path) -> str:
    return project_brief_path.read_text(encoding="utf-8")


def load_existing_cache(cache_path: Path) -> Dict[str, Dict[str, str]]:
    if not cache_path.exists():
        return {}

    cached_rows: Dict[str, Dict[str, str]] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        cached_rows[str(row["sent ID"])] = row
    return cached_rows


def append_cache(cache_path: Path, row: Dict[str, str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def save_workbook(rows: List[Dict[str, str]], metadata: Dict[str, str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(rows)
    metadata_df = pd.DataFrame(
        [{"key": key, "value": value} for key, value in metadata.items()]
    )
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="results", index=False)
        metadata_df.to_excel(writer, sheet_name="metadata", index=False)


def make_output_row(
    record: SentenceRecord,
    task_type: str,
    model: str,
    raw_response: str,
    converted: Dict[str, str],
) -> Dict[str, str]:
    return {
        "sent ID": record.sent_id,
        "scriptio continua sentence": record.script,
        "ground truth sent": record.sentence,
        "ground truth 4 state": record.state4,
        "ground truth 2 state": record.state2,
        "output direct output": converted["output_sentence"],
        "output 4 state": converted["output_state4"],
        "output 2 state": converted["output_state2"],
        "task type": task_type,
        "model": model,
        "prediction status": converted["prediction_status"],
        "raw model response": raw_response,
    }


def run_pipeline(
    task_type: str,
    model: str,
    dataset_path: Path,
    output_dir: Path,
    project_brief_path: Path,
    few_shot_count: int,
    batch_size: int,
    timeout_s: int,
    limit: Optional[int],
    resume: bool,
    sleep_s: float,
) -> Path:
    project_brief = load_project_brief(project_brief_path)
    train_records = load_records(dataset_path, "train")
    test_records = load_records(dataset_path, "test")
    if limit is not None:
        test_records = test_records[:limit]

    examples = select_few_shot_examples(train_records, few_shot_count)
    system_message = build_system_message(project_brief, task_type)

    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"{task_type}_{model_slug}_test_cache.jsonl"
    output_path = output_dir / f"{task_type}_{model_slug}_test_results.xlsx"

    existing = load_existing_cache(cache_path) if resume else {}
    rows: List[Dict[str, str]] = list(existing.values())
    completed_ids = set(existing)

    pending = [record for record in test_records if record.sent_id not in completed_ids]
    total = len(test_records)

    for start in range(0, len(pending), max(batch_size, 1)):
        batch = pending[start : start + max(batch_size, 1)]
        user_message = build_user_message(task_type, examples, batch)
        batch_num_predict = estimate_num_predict(task_type, batch)
        try:
            response_text = chat_with_ollama(
                model,
                system_message,
                user_message,
                timeout_s,
                batch_num_predict,
            )
            payload = extract_json_payload(response_text)
            if isinstance(payload, dict):
                payload = [payload]
            if not isinstance(payload, list):
                raise ValueError("Model payload is not a JSON list.")

            predictions = {
                str(item.get("sent_id", "")): str(item.get("prediction", ""))
                for item in payload
                if isinstance(item, dict)
            }
            missing = [record.sent_id for record in batch if record.sent_id not in predictions]
            if missing:
                raise ValueError(f"Missing predictions for: {', '.join(missing)}")

            batch_rows = []
            for record in batch:
                converted = convert_prediction(task_type, record.script, predictions[record.sent_id])
                row = make_output_row(record, task_type, model, predictions[record.sent_id], converted)
                batch_rows.append(row)
        except Exception:
            batch_rows = []
            for record in batch:
                single_user_message = build_user_message(task_type, examples, [record])
                response_text = chat_with_ollama(
                    model,
                    system_message,
                    single_user_message,
                    timeout_s,
                    estimate_num_predict(task_type, [record]),
                )
                try:
                    payload = extract_json_payload(response_text)
                    if isinstance(payload, list):
                        item = payload[0]
                    elif isinstance(payload, dict):
                        item = payload
                    else:
                        raise ValueError("Single-item payload is not valid JSON.")
                    raw_prediction = str(item.get("prediction", ""))
                except Exception:
                    raw_prediction = response_text.strip()
                converted = convert_prediction(task_type, record.script, raw_prediction)
                batch_rows.append(make_output_row(record, task_type, model, raw_prediction, converted))

        for row in batch_rows:
            rows.append(row)
            append_cache(cache_path, row)
            done = len(rows)
            print(f"[{done}/{total}] saved {row['sent ID']}")

        save_workbook(
            rows=sorted(rows, key=lambda row: row["sent ID"]),
            metadata={
                "task_type": task_type,
                "model": model,
                "dataset_path": str(dataset_path),
                "project_brief_path": str(project_brief_path),
                "few_shot_count": str(few_shot_count),
                "batch_size": str(batch_size),
                "limit": "" if limit is None else str(limit),
            },
            output_path=output_path,
        )

        if sleep_s > 0:
            time.sleep(sleep_s)

    return output_path


def run_cli(default_task: str, default_output_dir: Path) -> None:
    parser = argparse.ArgumentParser(description="Few-shot Ollama runner for scriptio continua tasks.")
    parser.add_argument("--task-type", default=default_task, choices=["type1", "type2", "type3"])
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--project-brief", type=Path, default=DEFAULT_PROJECT_BRIEF_PATH)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--few-shot-count", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    result_path = run_pipeline(
        task_type=args.task_type,
        model=args.model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        project_brief_path=args.project_brief,
        few_shot_count=args.few_shot_count,
        batch_size=args.batch_size,
        timeout_s=args.timeout,
        limit=args.limit,
        resume=args.resume,
        sleep_s=args.sleep,
    )
    print(f"Saved results to: {result_path}")


if __name__ == "__main__":
    run_cli(default_task="type1", default_output_dir=ROOT_DIR / "LLM based" / "type 1")
