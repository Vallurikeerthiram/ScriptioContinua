from __future__ import annotations

import math
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import pandas as pd
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT.parents[1] / "DATASET" / "18000 rows" / "new dataset" / "SENT_ID.xlsx"
RESULTS_DIR = ROOT / "results" / "spreadsheets"

MODE_FILES = {
    "cm": RESULTS_DIR / "orthographic_cm_test_results.xlsx",
    "coh": RESULTS_DIR / "orthographic_coh_test_results.xlsx",
    "gs": RESULTS_DIR / "orthographic_gs_test_results.xlsx",
    "lm": RESULTS_DIR / "orthographic_lm_test_results.xlsx",
}

MODE_SHEETS = {
    "cm": "cm best output",
    "coh": "coh best output",
    "gs": "gs best output",
    "lm": "lm best output",
}

MODE_SCORE_COLUMNS = {
    "cm": "best_cm",
    "coh": "best_coh",
    "gs": "best_gs",
    "lm": "best_lm",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")
SHEET_X_VALUES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
XML_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def normalize_text(value: object) -> str:
    return " ".join(TOKEN_RE.findall(str(value or "").lower()))


def normalize_script(value: object) -> str:
    return "".join(TOKEN_RE.findall(str(value or "").lower()))


def tokenize_text(value: object) -> list[str]:
    return TOKEN_RE.findall(str(value or "").lower())


def parse_state4(value: object, n_chars: int) -> list[str]:
    tags = re.findall(r"[SBEMI]", str(value).upper())
    tags = ["M" if tag == "I" else tag for tag in tags]
    if len(tags) < n_chars:
        tags += ["M"] * (n_chars - len(tags))
    return tags[:n_chars]


def tokens_from_state4(script: str, raw_labels: object) -> list[str]:
    labels = parse_state4(raw_labels, len(script))
    tokens: list[str] = []
    current: list[str] = []

    for index, char in enumerate(script):
        tag = labels[index]
        if tag == "S":
            if current:
                tokens.append("".join(current))
                current = []
            tokens.append(char)
        elif tag == "B":
            if current:
                tokens.append("".join(current))
            current = [char]
        elif tag == "E":
            if current:
                current.append(char)
                tokens.append("".join(current))
                current = []
            else:
                tokens.append(char)
        else:
            if current:
                current.append(char)
            else:
                current = [char]

    if current:
        tokens.append("".join(current))
    return [token for token in tokens if token]


def state4_from_tokens(tokens: list[str]) -> str:
    labels: list[str] = []
    for token in tokens:
        if len(token) <= 1:
            labels.append("S")
        else:
            labels.append("B")
            if len(token) > 2:
                labels.extend("M" for _ in range(len(token) - 2))
            labels.append("E")
    return " ".join(labels)


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(curr[-1], prev[j]))
        prev = curr
    return prev[-1]


def rouge_n_f1(reference: list[str], candidate: list[str], n: int) -> float:
    if len(reference) < n or len(candidate) < n:
        return 0.0
    ref_counts = Counter(tuple(reference[i:i + n]) for i in range(len(reference) - n + 1))
    cand_counts = Counter(tuple(candidate[i:i + n]) for i in range(len(candidate) - n + 1))
    overlap = sum(min(count, cand_counts[gram]) for gram, count in ref_counts.items())
    ref_total = sum(ref_counts.values())
    cand_total = sum(cand_counts.values())
    if ref_total == 0 or cand_total == 0 or overlap == 0:
        return 0.0
    precision = overlap / cand_total
    recall = overlap / ref_total
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(reference: list[str], candidate: list[str]) -> float:
    if not reference or not candidate:
        return 0.0
    lcs = lcs_length(reference, candidate)
    if lcs == 0:
        return 0.0
    precision = lcs / len(candidate)
    recall = lcs / len(reference)
    return 2 * precision * recall / (precision + recall)


def meteor_score_simple(reference: list[str], candidate: list[str]) -> float:
    if not reference or not candidate:
        return 0.0

    ref_positions: dict[str, list[int]] = {}
    for index, token in enumerate(reference):
        ref_positions.setdefault(token, []).append(index)

    matched_ref_indices: list[int] = []
    used_counter: Counter[str] = Counter()
    matches = 0
    for token in candidate:
        positions = ref_positions.get(token, [])
        use_index = used_counter[token]
        if use_index < len(positions):
            matched_ref_indices.append(positions[use_index])
            used_counter[token] += 1
            matches += 1

    if matches == 0:
        return 0.0

    precision = matches / len(candidate)
    recall = matches / len(reference)
    f_mean = (10 * precision * recall) / (recall + 9 * precision) if (recall + 9 * precision) else 0.0

    chunks = 1
    for left, right in zip(matched_ref_indices, matched_ref_indices[1:]):
        if right != left + 1:
            chunks += 1
    penalty = 0.5 * ((chunks / matches) ** 3)
    return (1.0 - penalty) * f_mean


def macro_state_metrics(reference: list[str], predicted: list[str]) -> dict[str, float]:
    labels = ["B", "M", "E", "S"]
    max_len = max(len(reference), len(predicted), 1)
    if len(reference) < max_len:
        reference = reference + ["M"] * (max_len - len(reference))
    if len(predicted) < max_len:
        predicted = predicted + ["M"] * (max_len - len(predicted))

    accuracy = sum(int(left == right) for left, right in zip(reference, predicted)) / max_len

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for label in labels:
        tp = sum(1 for ref, pred in zip(reference, predicted) if ref == label and pred == label)
        fp = sum(1 for ref, pred in zip(reference, predicted) if ref != label and pred == label)
        fn = sum(1 for ref, pred in zip(reference, predicted) if ref == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "f1": sum(f1s) / len(f1s),
        "precision": sum(precisions) / len(precisions),
        "accuracy": accuracy,
        "recall": sum(recalls) / len(recalls),
    }


def sentence_metrics(reference_sentence: str, output_sentence: str) -> dict[str, float]:
    ref_tokens = tokenize_text(reference_sentence)
    out_tokens = tokenize_text(output_sentence)

    if ref_tokens and out_tokens:
        bleu = sentence_bleu([ref_tokens], out_tokens, smoothing_function=SmoothingFunction().method1)
    else:
        bleu = 0.0

    return {
        "bleu": bleu,
        "rouge_1_f1": rouge_n_f1(ref_tokens, out_tokens, 1),
        "rouge_2_f1": rouge_n_f1(ref_tokens, out_tokens, 2),
        "rouge_l_f1": rouge_l_f1(ref_tokens, out_tokens),
        "meteor": meteor_score_simple(ref_tokens, out_tokens),
    }


def output_state4_for_script(script_contin: str, output_sentence: str) -> str:
    output_tokens = tokenize_text(output_sentence)
    if not output_tokens:
        output_tokens = [script_contin] if script_contin else []
    return state4_from_tokens(output_tokens)


def load_test_rows() -> dict[int, dict[str, object]]:
    df = pd.read_excel(DATASET_PATH, sheet_name="test")
    rows: dict[int, dict[str, object]] = {}
    for row_number, row in enumerate(df.itertuples(index=False), start=1):
        script = normalize_script(getattr(row, "SCRIPT_CONTIN", ""))
        ground_truth_tokens = tokens_from_state4(script, getattr(row, "STATE_4", ""))
        rows[row_number] = {
            "art_id": getattr(row, "ART_ID", ""),
            "para_id": getattr(row, "PARA_ID", ""),
            "sent_id": getattr(row, "SENT_ID", ""),
            "script_contin": script,
            "ground_truth_sentence": " ".join(ground_truth_tokens),
            "ground_truth_state_4": " ".join(parse_state4(getattr(row, "STATE_4", ""), len(script))),
        }
    return rows


def parse_best_output_sheet(path: Path, sheet_name: str) -> list[dict[str, object]]:
    with zipfile.ZipFile(path) as zf:
        workbook_root = ET.fromstring(zf.read("xl/workbook.xml"))
        rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rel_root}

        target_sheet = None
        for sheet in workbook_root.find("a:sheets", XML_NS):
            if sheet.attrib["name"] == sheet_name:
                target_sheet = sheet
                break
        if target_sheet is None:
            raise SystemExit(f"Sheet '{sheet_name}' not found in {path}.")

        relation_id = target_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        target = rel_map[relation_id].lstrip("/")
        xml_path = target if target.startswith("xl/") else f"xl/{target}"

        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in shared_root:
                text_parts = [node.text or "" for node in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")]
                shared_strings.append("".join(text_parts))

        rows: list[dict[str, object]] = []
        headers: list[str] | None = None
        context = ET.iterparse(zf.open(xml_path), events=("end",))
        for _, elem in context:
            if elem.tag != "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row":
                continue

            values: list[str | None] = []
            for cell in elem:
                if cell.tag != "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c":
                    continue
                cell_type = cell.attrib.get("t")
                value: str | None = None
                if cell_type == "inlineStr":
                    is_node = cell.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}is")
                    if is_node is not None:
                        parts = [node.text or "" for node in is_node.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")]
                        value = "".join(parts)
                else:
                    v_node = cell.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                    if v_node is not None:
                        value = v_node.text
                        if cell_type == "s" and value is not None:
                            value = shared_strings[int(value)]
                values.append(value)

            if headers is None:
                headers = [str(value or "") for value in values]
            else:
                row_dict = {headers[index]: values[index] if index < len(values) else None for index in range(len(headers))}
                rows.append(row_dict)

            elem.clear()

    return rows


def add_bert_scores(
    records: list[dict[str, object]],
    batch_size: int = 64,
    cache: dict[tuple[str, str], tuple[float, float, float]] | None = None,
) -> None:
    if cache is None:
        cache = {}

    unique_pairs: dict[tuple[str, str], None] = {}
    for record in records:
        key = (str(record["ground_truth_for_script_continua"]), str(record["output_sentence"]))
        if key not in cache:
            unique_pairs[key] = None

    pair_list = list(unique_pairs.keys())
    if pair_list:
        for start in range(0, len(pair_list), batch_size):
            batch = pair_list[start:start + batch_size]
            candidates = [pair[1] or "" for pair in batch]
            references = [pair[0] or "" for pair in batch]
            precision, recall, f1 = bert_score(
                candidates,
                references,
                lang="en",
                model_type="distilbert-base-uncased",
                verbose=False,
                device="cpu",
            )
            for pair, p, r, f in zip(batch, precision.tolist(), recall.tolist(), f1.tolist()):
                cache[pair] = (float(p), float(r), float(f))

    for record in records:
        key = (str(record["ground_truth_for_script_continua"]), str(record["output_sentence"]))
        p, r, f = cache[key]
        record["bertscore_p"] = p
        record["bertscore_r"] = r
        record["bertscore_f1"] = f


def build_mode_records(
    mode: str,
    test_rows: dict[int, dict[str, object]],
    bert_cache: dict[tuple[str, str], tuple[float, float, float]],
) -> list[dict[str, object]]:
    workbook_rows = parse_best_output_sheet(MODE_FILES[mode], MODE_SHEETS[mode])
    records: list[dict[str, object]] = []

    for row in workbook_rows:
        row_number = int(float(str(row.get("row_number") or "0")))
        if row_number not in test_rows:
            continue
        test_info = test_rows[row_number]
        output_sentence = str(row.get("output") or "")
        output_state_4 = output_state4_for_script(str(test_info["script_contin"]), output_sentence)

        sentence_score_map = sentence_metrics(str(test_info["ground_truth_sentence"]), output_sentence)
        state_score_map = macro_state_metrics(
            str(test_info["ground_truth_state_4"]).split(),
            output_state_4.split(),
        )

        records.append(
            {
                "art_id": test_info["art_id"],
                "para_id": test_info["para_id"],
                "sent_id": test_info["sent_id"],
                "script_continue_text": test_info["script_contin"],
                "ground_truth_for_script_continua": test_info["ground_truth_sentence"],
                "output_sentence": output_sentence,
                "ground_truth_state_4": test_info["ground_truth_state_4"],
                "output_state_4": output_state_4,
                "x_percent": int(float(str(row.get("x_percent") or "0"))),
                "top_k_per_start": int(float(str(row.get("top_k_per_start") or "0"))),
                "candidate_count": int(float(str(row.get("candidate_count") or "0"))),
                MODE_SCORE_COLUMNS[mode]: float(str(row.get(MODE_SCORE_COLUMNS[mode]) or "0")),
                "bleu": sentence_score_map["bleu"],
                "rouge_1_f1": sentence_score_map["rouge_1_f1"],
                "rouge_2_f1": sentence_score_map["rouge_2_f1"],
                "rouge_l_f1": sentence_score_map["rouge_l_f1"],
                "meteor": sentence_score_map["meteor"],
                "f1": state_score_map["f1"],
                "precision": state_score_map["precision"],
                "accuracy": state_score_map["accuracy"],
                "recall": state_score_map["recall"],
            }
        )

    add_bert_scores(records, cache=bert_cache)
    return records


def write_mode_workbook(mode: str, records: list[dict[str, object]]) -> Path:
    output_path = RESULTS_DIR / f"orthographic_{mode}_test_metrics.xlsx"
    ordered_columns = [
        "art_id",
        "para_id",
        "sent_id",
        "script_continue_text",
        "ground_truth_for_script_continua",
        "output_sentence",
        "ground_truth_state_4",
        "output_state_4",
        "x_percent",
        "top_k_per_start",
        "candidate_count",
        MODE_SCORE_COLUMNS[mode],
        "bleu",
        "rouge_1_f1",
        "rouge_2_f1",
        "rouge_l_f1",
        "meteor",
        "bertscore_p",
        "bertscore_r",
        "bertscore_f1",
        "f1",
        "precision",
        "accuracy",
        "recall",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for x_value in SHEET_X_VALUES:
            sheet_rows = [record for record in records if int(record["x_percent"]) == x_value]
            df = pd.DataFrame(sheet_rows, columns=ordered_columns)
            df.to_excel(writer, sheet_name=f"x={x_value}", index=False)
    return output_path


def main() -> None:
    test_rows = load_test_rows()
    bert_cache: dict[tuple[str, str], tuple[float, float, float]] = {}
    for mode in ("cm", "coh", "gs", "lm"):
        print(f"building_metrics_for: {mode}")
        records = build_mode_records(mode, test_rows, bert_cache)
        output_path = write_mode_workbook(mode, records)
        print(f"written: {output_path}")
        print(f"rows: {len(records)}")


if __name__ == "__main__":
    main()
