from __future__ import annotations

import importlib.util
import math
import random
import sys
import sysconfig
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def preload_stdlib_code_module() -> None:
    if "code" in sys.modules and getattr(sys.modules["code"], "__file__", "") != __file__:
        return
    stdlib_dir = Path(sysconfig.get_path("stdlib"))
    stdlib_code_path = stdlib_dir / "code.py"
    spec = importlib.util.spec_from_file_location("code", stdlib_code_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["code"] = module


preload_stdlib_code_module()

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


EXCEL_PATH = Path("SENT_based_split.xlsx")
OUTPUT_PATH = Path("model_comparison_results.xlsx")
RANDOM_SEED = 42
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_LABEL = -100


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    excel_path: Path = EXCEL_PATH
    output_path: Path = OUTPUT_PATH
    require_gpu: bool = True
    embedding_dim: int = 64
    hidden_dim: int = 128
    cnn_channels: int = 128
    dropout: float = 0.2
    batch_size: int = 64
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 2
    num_workers: int = 0
    bert_score_batch_size: int = 16


def parse_state2(state_text: str) -> List[int]:
    return [int(token) for token in str(state_text).strip().split() if token]


def parse_state4(state_text: str) -> List[str]:
    labels = []
    for token in str(state_text).strip().split():
        label = token.upper()
        labels.append("I" if label == "M" else label)
    return labels


def reconstruct_from_state2(script_contin: str, labels: Sequence[int]) -> str:
    words: List[str] = []
    current: List[str] = []
    for char, label in zip(script_contin, labels):
        current.append(char)
        if int(label) == 1:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return " ".join(word for word in words if word)


def reconstruct_from_state4(script_contin: str, labels: Sequence[str]) -> str:
    words: List[str] = []
    current: List[str] = []
    for char, raw_label in zip(script_contin, labels):
        label = "I" if raw_label.upper() == "M" else raw_label.upper()
        if label == "S":
            if current:
                words.append("".join(current))
                current = []
            words.append(char)
        elif label == "B":
            if current:
                words.append("".join(current))
            current = [char]
        elif label == "I":
            current = current + [char] if current else [char]
        elif label == "E":
            current = current + [char] if current else [char]
            words.append("".join(current))
            current = []
        else:
            current = current + [char] if current else [char]
    if current:
        words.append("".join(current))
    return " ".join(word for word in words if word)


def clean_text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def load_split_data(excel_path: Path) -> Dict[str, pd.DataFrame]:
    sheets: Dict[str, pd.DataFrame] = {}
    workbook = pd.ExcelFile(excel_path)
    for sheet_name in ("TRAIN", "VAL", "TEST"):
        df = workbook.parse(sheet_name)
        df["SENT_ORI"] = df["SENT_ORI"].map(clean_text)
        df["SCRIPT_CONTIN"] = df["SCRIPT_CONTIN"].map(clean_text)
        df["STATE_2_LIST"] = df["STATE_2"].map(parse_state2)
        df["STATE_4_LIST"] = df["STATE_4"].map(parse_state4)

        valid_rows = []
        for row in df.itertuples(index=False):
            valid_rows.append(
                len(row.SCRIPT_CONTIN) > 0
                and len(row.STATE_2_LIST) == len(row.SCRIPT_CONTIN)
                and len(row.STATE_4_LIST) == len(row.SCRIPT_CONTIN)
            )
        invalid_count = len(valid_rows) - sum(valid_rows)
        if invalid_count:
            warnings.warn(
                f"{sheet_name}: dropped {invalid_count} invalid rows where script and label lengths differed."
            )
        df = df.loc[valid_rows].reset_index(drop=True)

        df["EXPECTED_SPLIT_STATE2"] = [
            reconstruct_from_state2(script, labels)
            for script, labels in zip(df["SCRIPT_CONTIN"], df["STATE_2_LIST"])
        ]
        df["EXPECTED_SPLIT_STATE4"] = [
            reconstruct_from_state4(script, labels)
            for script, labels in zip(df["SCRIPT_CONTIN"], df["STATE_4_LIST"])
        ]
        sheets[sheet_name.lower()] = df
    return sheets


def build_char_vocab(train_df: pd.DataFrame) -> Dict[str, int]:
    counter = Counter()
    for text in train_df["SCRIPT_CONTIN"]:
        counter.update(text)
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for char in sorted(counter):
        vocab[char] = len(vocab)
    return vocab


def create_embedding_weights(vocab_size: int, embedding_dim: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    weights = torch.empty(vocab_size, embedding_dim)
    nn.init.xavier_uniform_(weights)
    weights[0].zero_()
    return weights


class SentenceDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        char_vocab: Dict[str, int],
        label_scheme: str,
        label_to_id: Dict[object, int],
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.char_vocab = char_vocab
        self.label_scheme = label_scheme
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.dataframe.iloc[index]
        script = row["SCRIPT_CONTIN"]
        char_ids = [self.char_vocab.get(char, self.char_vocab[UNK_TOKEN]) for char in script]
        if self.label_scheme == "state2":
            labels = [self.label_to_id[label] for label in row["STATE_2_LIST"]]
            expected_split = row["EXPECTED_SPLIT_STATE2"]
        else:
            labels = [self.label_to_id[label] for label in row["STATE_4_LIST"]]
            expected_split = row["EXPECTED_SPLIT_STATE4"]
        return {
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "length": len(char_ids),
            "art_id": row["ART_ID"],
            "para_id": row["PARA_ID"],
            "sent_id": row["SENT_ID"],
            "script_contin": script,
            "sent_ori": row["SENT_ORI"],
            "expected_split": expected_split,
        }


def collate_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    padded_chars = pad_sequence([item["char_ids"] for item in batch], batch_first=True, padding_value=0)
    padded_labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=PAD_LABEL)
    mask = padded_labels.ne(PAD_LABEL)
    return {
        "char_ids": padded_chars,
        "labels": padded_labels,
        "mask": mask,
        "art_ids": [item["art_id"] for item in batch],
        "para_ids": [item["para_id"] for item in batch],
        "sent_ids": [item["sent_id"] for item in batch],
        "script_contin": [item["script_contin"] for item in batch],
        "sent_ori": [item["sent_ori"] for item in batch],
        "expected_split": [item["expected_split"] for item in batch],
    }


class TrainableEmbedding(nn.Module):
    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(weights.clone(), freeze=False, padding_idx=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class SequenceTaggerBase(nn.Module):
    def __init__(self, embedding_weights: torch.Tensor, dropout: float, num_labels: int) -> None:
        super().__init__()
        self.embedding = TrainableEmbedding(embedding_weights)
        self.dropout = nn.Dropout(dropout)
        self.num_labels = num_labels


class BiLSTMTagger(SequenceTaggerBase):
    def __init__(self, embedding_weights: torch.Tensor, hidden_dim: int, dropout: float, num_labels: int) -> None:
        super().__init__(embedding_weights, dropout, num_labels)
        self.encoder = nn.LSTM(
            input_size=embedding_weights.size(1),
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.encoder(self.dropout(self.embedding(inputs)))
        return self.classifier(self.dropout(outputs))


class RNNTagger(SequenceTaggerBase):
    def __init__(self, embedding_weights: torch.Tensor, hidden_dim: int, dropout: float, num_labels: int) -> None:
        super().__init__(embedding_weights, dropout, num_labels)
        self.encoder = nn.RNN(
            input_size=embedding_weights.size(1),
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            nonlinearity="tanh",
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.encoder(self.dropout(self.embedding(inputs)))
        return self.classifier(self.dropout(outputs))


class GRUTagger(SequenceTaggerBase):
    def __init__(self, embedding_weights: torch.Tensor, hidden_dim: int, dropout: float, num_labels: int) -> None:
        super().__init__(embedding_weights, dropout, num_labels)
        self.encoder = nn.GRU(
            input_size=embedding_weights.size(1),
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.encoder(self.dropout(self.embedding(inputs)))
        return self.classifier(self.dropout(outputs))


class CNNTagger(SequenceTaggerBase):
    def __init__(self, embedding_weights: torch.Tensor, cnn_channels: int, dropout: float, num_labels: int) -> None:
        super().__init__(embedding_weights, dropout, num_labels)
        self.conv = nn.Conv1d(embedding_weights.size(1), cnn_channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(cnn_channels, num_labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embedding(inputs)).transpose(1, 2)
        x = self.activation(self.conv(x)).transpose(1, 2)
        return self.classifier(self.dropout(x))


class CRFLayer(nn.Module):
    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_log_partition(emissions, mask)
        return torch.mean(denominator - numerator)

    def _compute_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0, :].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        for timestep in range(1, emissions.size(1)):
            active = mask[:, timestep]
            prev_tags = tags[:, timestep - 1]
            curr_tags = tags[:, timestep]
            emit = emissions[:, timestep, :].gather(1, curr_tags.unsqueeze(1)).squeeze(1)
            score += (self.transitions[prev_tags, curr_tags] + emit) * active
        seq_ends = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, seq_ends.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        return score

    def _compute_log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = self.start_transitions + emissions[:, 0]
        for timestep in range(1, emissions.size(1)):
            next_score = score.unsqueeze(2) + self.transitions + emissions[:, timestep].unsqueeze(1)
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, timestep].unsqueeze(1), next_score, score)
        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        score = self.start_transitions + emissions[:, 0]
        history: List[torch.Tensor] = []
        for timestep in range(1, emissions.size(1)):
            next_score = score.unsqueeze(2) + self.transitions + emissions[:, timestep].unsqueeze(1)
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[:, timestep].unsqueeze(1), next_score, score)
            history.append(indices)
        score = score + self.end_transitions
        best_last_tags = score.argmax(dim=1)
        seq_ends = mask.long().sum(dim=1) - 1
        decoded_paths: List[List[int]] = []
        for batch_idx in range(emissions.size(0)):
            best_tag = best_last_tags[batch_idx].item()
            best_path = [best_tag]
            for timestep in range(seq_ends[batch_idx].item() - 1, -1, -1):
                best_tag = history[timestep][batch_idx][best_tag].item()
                best_path.append(best_tag)
            decoded_paths.append(list(reversed(best_path)))
        return decoded_paths


class CRFTagger(nn.Module):
    def __init__(self, embedding_weights: torch.Tensor, hidden_dim: int, dropout: float, num_labels: int) -> None:
        super().__init__()
        self.embedding = TrainableEmbedding(embedding_weights)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(embedding_weights.size(1), hidden_dim)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.crf = CRFLayer(num_labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embedding(inputs))
        x = self.activation(self.projection(x))
        return self.classifier(self.dropout(x))

    def loss(self, inputs: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        safe_tags = tags.masked_fill(~mask, 0)
        return self.crf(self.forward(inputs), safe_tags, mask)


def build_model(model_name: str, embedding_weights: torch.Tensor, config: Config, num_labels: int) -> nn.Module:
    if model_name == "bilstm":
        return BiLSTMTagger(embedding_weights, config.hidden_dim, config.dropout, num_labels)
    if model_name == "rnn":
        return RNNTagger(embedding_weights, config.hidden_dim, config.dropout, num_labels)
    if model_name == "gru":
        return GRUTagger(embedding_weights, config.hidden_dim, config.dropout, num_labels)
    if model_name == "cnn":
        return CNNTagger(embedding_weights, config.cnn_channels, config.dropout, num_labels)
    if model_name == "crf":
        return CRFTagger(embedding_weights, config.hidden_dim, config.dropout, num_labels)
    raise ValueError(f"Unknown model name: {model_name}")


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)
    total_loss = 0.0
    for batch in loader:
        inputs = batch["char_ids"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)
        optimizer.zero_grad()
        if isinstance(model, CRFTagger):
            loss = model.loss(inputs, labels, mask)
        else:
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def predict_with_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    id_to_label: Dict[int, object],
    label_scheme: str,
) -> Tuple[float, pd.DataFrame]:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)
    total_loss = 0.0
    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["char_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            if isinstance(model, CRFTagger):
                emissions = model(inputs)
                safe_labels = labels.masked_fill(~mask, 0)
                loss = model.crf(emissions, safe_labels, mask)
                predicted_ids = model.crf.decode(emissions, mask)
            else:
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                predicted_tensor = logits.argmax(dim=-1)
                predicted_ids = [predicted_tensor[i][mask[i]].tolist() for i in range(predicted_tensor.size(0))]
            total_loss += loss.item()

            for idx, pred_seq in enumerate(predicted_ids):
                true_seq = labels[idx][mask[idx]].tolist()
                pred_labels = [id_to_label[item] for item in pred_seq]
                true_labels = [id_to_label[item] for item in true_seq]
                script = batch["script_contin"][idx]
                if label_scheme == "state2":
                    predicted_output = reconstruct_from_state2(script, pred_labels)
                    original_output = reconstruct_from_state2(script, true_labels)
                else:
                    predicted_output = reconstruct_from_state4(script, pred_labels)
                    original_output = reconstruct_from_state4(script, true_labels)
                rows.append(
                    {
                        "ART_ID": batch["art_ids"][idx],
                        "PARA_ID": batch["para_ids"][idx],
                        "SENT_ID": batch["sent_ids"][idx],
                        "SCRIPT_CONTIN": script,
                        "SENT_ORI": batch["sent_ori"][idx],
                        "ORIGINAL_OUTPUT": original_output,
                        "PREDICTED_OUTPUT": predicted_output,
                        "TRUE_LABELS": " ".join(str(x) for x in true_labels),
                        "PRED_LABELS": " ".join(str(x) for x in pred_labels),
                    }
                )
    return total_loss / max(len(loader), 1), pd.DataFrame(rows)


def compute_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], label_ids: Sequence[int]) -> np.ndarray:
    label_index = {label_id: idx for idx, label_id in enumerate(label_ids)}
    matrix = np.zeros((len(label_ids), len(label_ids)), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[label_index[true_label], label_index[pred_label]] += 1
    return matrix


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], label_ids: Sequence[int]) -> Dict[str, object]:
    matrix = compute_confusion_matrix(y_true, y_pred, label_ids)
    total = matrix.sum()
    precision_list: List[float] = []
    recall_list: List[float] = []
    f1_list: List[float] = []
    for idx in range(len(label_ids)):
        tp = matrix[idx, idx]
        fp = matrix[:, idx].sum() - tp
        fn = matrix[idx, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        precision_list.append(float(precision))
        recall_list.append(float(recall))
        f1_list.append(float(f1))
    return {
        "precision": float(np.mean(precision_list)),
        "recall": float(np.mean(recall_list)),
        "f1": float(np.mean(f1_list)),
        "accuracy": float(np.trace(matrix) / total) if total else 0.0,
        "confusion_matrix": matrix,
    }


def ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)) if len(tokens) >= n else Counter()


def compute_bleu_score(references: Sequence[str], candidates: Sequence[str], max_n: int = 4) -> float:
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    ref_length = 0
    cand_length = 0
    for reference, candidate in zip(references, candidates):
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        ref_length += len(ref_tokens)
        cand_length += len(cand_tokens)
        for n in range(1, max_n + 1):
            ref_ng = ngrams(ref_tokens, n)
            cand_ng = ngrams(cand_tokens, n)
            total_counts[n - 1] += max(sum(cand_ng.values()), 1)
            clipped_counts[n - 1] += sum(min(count, ref_ng[gram]) for gram, count in cand_ng.items())
    if cand_length == 0:
        return 0.0
    precisions = [(clip + 1.0) / (total + 1.0) for clip, total in zip(clipped_counts, total_counts)]
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    bp = 1.0 if cand_length > ref_length else math.exp(1 - (ref_length / cand_length))
    return float(geo_mean * bp)


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def compute_rouge_scores(references: Sequence[str], candidates: Sequence[str]) -> Dict[str, float]:
    rouge1_scores: List[float] = []
    rouge2_scores: List[float] = []
    rouge_l_scores: List[float] = []
    for reference, candidate in zip(references, candidates):
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        for n, bucket in ((1, rouge1_scores), (2, rouge2_scores)):
            ref_ng = ngrams(ref_tokens, n)
            cand_ng = ngrams(cand_tokens, n)
            overlap = sum(min(count, cand_ng[gram]) for gram, count in ref_ng.items())
            ref_count = sum(ref_ng.values())
            cand_count = sum(cand_ng.values())
            precision = overlap / cand_count if cand_count else 0.0
            recall = overlap / ref_count if ref_count else 0.0
            bucket.append((2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0)
        lcs = lcs_length(ref_tokens, cand_tokens)
        precision = lcs / len(cand_tokens) if cand_tokens else 0.0
        recall = lcs / len(ref_tokens) if ref_tokens else 0.0
        rouge_l_scores.append((2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0)
    return {
        "rouge1_f1": float(np.mean(rouge1_scores)) if rouge1_scores else 0.0,
        "rouge2_f1": float(np.mean(rouge2_scores)) if rouge2_scores else 0.0,
        "rougeL_f1": float(np.mean(rouge_l_scores)) if rouge_l_scores else 0.0,
    }


def chunk_count(match_flags: List[int]) -> int:
    chunks = 0
    in_chunk = False
    for flag in match_flags:
        if flag and not in_chunk:
            chunks += 1
            in_chunk = True
        elif not flag:
            in_chunk = False
    return chunks


def compute_meteor_score(reference: str, candidate: str) -> float:
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    if not ref_tokens or not cand_tokens:
        return 0.0
    ref_counter = Counter(ref_tokens)
    cand_counter = Counter(cand_tokens)
    matches = sum(min(ref_counter[token], cand_counter[token]) for token in ref_counter)
    if matches == 0:
        return 0.0
    precision = matches / len(cand_tokens)
    recall = matches / len(ref_tokens)
    f_mean = (10 * precision * recall / (recall + 9 * precision)) if (recall + 9 * precision) else 0.0
    ref_positions: Dict[str, List[int]] = {}
    for idx, token in enumerate(ref_tokens):
        ref_positions.setdefault(token, []).append(idx)
    used_positions = set()
    flags = [0] * len(cand_tokens)
    last_position = -1
    for idx, token in enumerate(cand_tokens):
        for pos in ref_positions.get(token, []):
            if pos not in used_positions and pos >= last_position:
                used_positions.add(pos)
                flags[idx] = 1
                last_position = pos
                break
    chunks = chunk_count(flags)
    penalty = 0.5 * ((chunks / matches) ** 3)
    return float((1 - penalty) * f_mean)


def compute_text_generation_metrics(predictions_df: pd.DataFrame, bert_batch_size: int) -> Dict[str, float]:
    references = predictions_df["ORIGINAL_OUTPUT"].tolist()
    candidates = predictions_df["PREDICTED_OUTPUT"].tolist()
    scores = {
        "bleu": compute_bleu_score(references, candidates),
        "meteor": float(np.mean([compute_meteor_score(ref, cand) for ref, cand in zip(references, candidates)])),
        **compute_rouge_scores(references, candidates),
        "bertscore_precision": float("nan"),
        "bertscore_recall": float("nan"),
        "bertscore_f1": float("nan"),
    }
    try:
        from bert_score import score as bert_score

        precision, recall, f1 = bert_score(
            candidates,
            references,
            lang="en",
            verbose=False,
            batch_size=bert_batch_size,
        )
        scores["bertscore_precision"] = float(precision.mean().item())
        scores["bertscore_recall"] = float(recall.mean().item())
        scores["bertscore_f1"] = float(f1.mean().item())
    except Exception as exc:
        warnings.warn(
            "BERTScore skipped. Install compatible `bert-score`, `transformers`, and `torch` to enable it. "
            f"Reason: {exc}"
        )
    return scores


def normalize_label_key(raw_label: str, label_to_id: Dict[object, int]) -> object:
    sample_key = next(iter(label_to_id.keys()))
    return int(raw_label) if isinstance(sample_key, int) else raw_label


def flatten_labels(predictions_df: pd.DataFrame, label_to_id: Dict[object, int]) -> Tuple[List[int], List[int]]:
    y_true: List[int] = []
    y_pred: List[int] = []
    for row in predictions_df.itertuples(index=False):
        y_true.extend(label_to_id[normalize_label_key(label, label_to_id)] for label in str(row.TRUE_LABELS).split())
        y_pred.extend(label_to_id[normalize_label_key(label, label_to_id)] for label in str(row.PRED_LABELS).split())
    return y_true, y_pred


def quick_label_f1(predictions_df: pd.DataFrame, label_to_id: Dict[object, int], id_to_label: Dict[int, object]) -> float:
    y_true, y_pred = flatten_labels(predictions_df, label_to_id)
    return classification_metrics(y_true, y_pred, list(id_to_label.keys()))["f1"]


def evaluate_predictions(
    predictions_df: pd.DataFrame,
    label_to_id: Dict[object, int],
    id_to_label: Dict[int, object],
    bert_batch_size: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    y_true, y_pred = flatten_labels(predictions_df, label_to_id)
    label_ids = list(id_to_label.keys())
    class_scores = classification_metrics(y_true, y_pred, label_ids)
    text_scores = compute_text_generation_metrics(predictions_df, bert_batch_size)
    confusion_df = pd.DataFrame(
        class_scores["confusion_matrix"],
        index=[f"TRUE_{id_to_label[label_id]}" for label_id in label_ids],
        columns=[f"PRED_{id_to_label[label_id]}" for label_id in label_ids],
    )
    summary = {
        "precision": class_scores["precision"],
        "recall": class_scores["recall"],
        "f1": class_scores["f1"],
        "accuracy": class_scores["accuracy"],
        **text_scores,
    }
    return summary, confusion_df


def create_data_loaders(
    splits: Dict[str, pd.DataFrame],
    char_vocab: Dict[str, int],
    label_scheme: str,
    label_to_id: Dict[object, int],
    config: Config,
) -> Dict[str, DataLoader]:
    loaders = {}
    for split_name, dataframe in splits.items():
        loaders[split_name] = DataLoader(
            SentenceDataset(dataframe, char_vocab, label_scheme, label_to_id),
            batch_size=config.batch_size,
            shuffle=(split_name == "train"),
            num_workers=config.num_workers,
            pin_memory=(torch.cuda.is_available()),
            collate_fn=collate_batch,
        )
    return loaders


def fit_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    config: Config,
    label_to_id: Dict[object, int],
    id_to_label: Dict[int, object],
    label_scheme: str,
) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = None
    best_val_f1 = -1.0
    patience_counter = 0
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, device)
        val_loss, val_predictions = predict_with_model(model, loaders["val"], device, id_to_label, label_scheme)
        val_f1 = quick_label_f1(val_predictions, label_to_id, id_to_label)
        print(
            f"[{label_scheme.upper()}][{model.__class__.__name__}] "
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def safe_sheet_name(name: str) -> str:
    return name[:31]


def write_results_to_excel(
    output_path: Path,
    summary_frames: Dict[str, pd.DataFrame],
    prediction_frames: Dict[str, pd.DataFrame],
    confusion_frames: Dict[str, pd.DataFrame],
) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, dataframe in summary_frames.items():
            dataframe.to_excel(writer, sheet_name=safe_sheet_name(sheet_name), index=False)
        for sheet_name, dataframe in prediction_frames.items():
            dataframe.to_excel(writer, sheet_name=safe_sheet_name(sheet_name), index=False)
        for sheet_name, dataframe in confusion_frames.items():
            dataframe.to_excel(writer, sheet_name=safe_sheet_name(sheet_name))


def run_experiment() -> None:
    set_seed(RANDOM_SEED)
    config = Config()
    device = select_device()
    if config.require_gpu and device.type != "cuda":
        raise RuntimeError(
            "GPU was requested but CUDA is not available in this PyTorch install. "
            "Install a CUDA-enabled PyTorch build, then run the script again."
        )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using device: cuda ({gpu_name})")
    else:
        print(f"Using device: {device}")
    print("Embedding choice for all models: trainable character embedding with the same vocabulary, dimension, and initialization.")

    splits = load_split_data(config.excel_path)
    char_vocab = build_char_vocab(splits["train"])
    embedding_weights = create_embedding_weights(len(char_vocab), config.embedding_dim, RANDOM_SEED)

    label_spaces = {
        "state2": {"label_to_id": {0: 0, 1: 1}, "id_to_label": {0: 0, 1: 1}},
        "state4": {"label_to_id": {"B": 0, "I": 1, "E": 2, "S": 3}, "id_to_label": {0: "B", 1: "I", 2: "E", 3: "S"}},
    }
    model_names = ["bilstm", "rnn", "cnn", "gru", "crf"]
    summary_frames: Dict[str, pd.DataFrame] = {}
    prediction_frames: Dict[str, pd.DataFrame] = {}
    confusion_frames: Dict[str, pd.DataFrame] = {}

    for label_scheme in ("state2", "state4"):
        label_to_id = label_spaces[label_scheme]["label_to_id"]
        id_to_label = label_spaces[label_scheme]["id_to_label"]
        loaders = create_data_loaders(splits, char_vocab, label_scheme, label_to_id, config)
        summary_rows: List[Dict[str, object]] = []
        all_predictions: List[pd.DataFrame] = []

        for model_name in model_names:
            print(f"\nTraining {model_name.upper()} on {label_scheme.upper()} ...")
            model = build_model(model_name, embedding_weights, config, len(label_to_id)).to(device)
            model = fit_model(model, loaders, device, config, label_to_id, id_to_label, label_scheme)

            for split_name in ("train", "val", "test"):
                _, predictions_df = predict_with_model(model, loaders[split_name], device, id_to_label, label_scheme)
                summary, confusion_df = evaluate_predictions(
                    predictions_df,
                    label_to_id,
                    id_to_label,
                    config.bert_score_batch_size,
                )
                summary_rows.append(
                    {
                        "MODEL": model_name.upper(),
                        "LABEL_SCHEME": label_scheme.upper(),
                        "SPLIT": split_name.upper(),
                        "EMBEDDING": f"trainable_char_embedding_{config.embedding_dim}d",
                        "BLEU": summary["bleu"],
                        "ROUGE_1_F1": summary["rouge1_f1"],
                        "ROUGE_2_F1": summary["rouge2_f1"],
                        "ROUGE_L_F1": summary["rougeL_f1"],
                        "METEOR": summary["meteor"],
                        "BERTSCORE_P": summary["bertscore_precision"],
                        "BERTSCORE_R": summary["bertscore_recall"],
                        "BERTSCORE_F1": summary["bertscore_f1"],
                        "F1": summary["f1"],
                        "PRECISION": summary["precision"],
                        "ACCURACY": summary["accuracy"],
                        "RECALL": summary["recall"],
                    }
                )
                predictions_df.insert(0, "MODEL", model_name.upper())
                predictions_df.insert(1, "LABEL_SCHEME", label_scheme.upper())
                predictions_df.insert(2, "SPLIT", split_name.upper())
                all_predictions.append(predictions_df)
                confusion_frames[f"cm_{label_scheme}_{model_name}_{split_name}"] = confusion_df

        summary_frames[f"summary_{label_scheme}"] = pd.DataFrame(summary_rows)
        prediction_frames[f"predictions_{label_scheme}"] = pd.concat(all_predictions, ignore_index=True)

    write_results_to_excel(config.output_path, summary_frames, prediction_frames, confusion_frames)
    print(f"\nFinished. Output written to: {config.output_path.resolve()}")


if __name__ == "__main__":
    run_experiment()
