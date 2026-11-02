from __future__ import annotations

import argparse
import importlib.util
import math
import random
import sys
import sysconfig
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def preload_stdlib_code_module() -> None:
    if "code" in sys.modules and getattr(sys.modules["code"], "__file__", "") != __file__:
        return
    stdlib_code_path = Path(sysconfig.get_path("stdlib")) / "code.py"
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


DEFAULT_EXCEL_PATH = Path("2000_articles_50_character_dataset_ART_ID_split_70_15_15.xlsx")
DEFAULT_OUTPUT_PATH = Path("line_wise_4state_and_crf_results.xlsx")
RANDOM_SEED = 42
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_LABEL = -100


@dataclass
class Config:
    excel_path: Path = DEFAULT_EXCEL_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH
    window_size: int = 4
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
    use_bertscore: bool = False
    invalid_transition_penalty_weight: float = 2.0
    require_gpu: bool = False
    train_rows: int = 0
    val_rows: int = 0
    test_rows: int = 0


MODEL_DISPLAY_NAMES = {
    "bilstm": "BiLSTM",
    "cnn": "CNN",
    "bigru": "BiGRU",
    "birnn": "BiRNN",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(require_gpu: bool) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_gpu and device.type != "cuda":
        warnings.warn("GPU was requested but CUDA is not available. Falling back to CPU.")
    return device


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def parse_state2(state_text: object) -> List[int]:
    text = str(state_text).strip()
    tokens = text.split()
    if len(tokens) <= 1:
        tokens = list(text)
    return [int(token) for token in tokens if token != ""]


def parse_state4(state_text: object) -> List[str]:
    labels: List[str] = []
    text = str(state_text).strip()
    tokens = text.split()
    if len(tokens) <= 1:
        tokens = list(text)
    for token in tokens:
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
        label = "I" if str(raw_label).upper() == "M" else str(raw_label).upper()
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


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Art Id": "ART_ID",
        "Art_id": "ART_ID",
        "ART_ID": "ART_ID",
        "Para Id": "PARA_ID",
        "para_id": "PARA_ID",
        "PARA_ID": "PARA_ID",
        "Line Id": "LINE_ID",
        "Line_id": "LINE_ID",
        "LINE_ID": "LINE_ID",
        "sent ids": "SENT_IDS",
        "Sent_id": "SENT_IDS",
        "SENT_ID": "SENT_IDS",
        "Ori Sent": "SENT_ORI",
        "Sentence": "SENT_ORI",
        "SENT_ORI": "SENT_ORI",
        "Scrip Sent": "SCRIPT_CONTIN",
        "scriptio_continua": "SCRIPT_CONTIN",
        "SCRIPT_CONTIN": "SCRIPT_CONTIN",
        "2State": "STATE_2",
        "2state": "STATE_2",
        "STATE_2": "STATE_2",
        "4State": "STATE_4",
        "4state": "STATE_4",
        "STATE_4": "STATE_4",
    }
    normalized = df.rename(columns={col: rename_map.get(col, col) for col in df.columns}).copy()
    required = ["ART_ID", "PARA_ID", "LINE_ID", "SENT_ORI", "SCRIPT_CONTIN", "STATE_2", "STATE_4"]
    missing = [col for col in required if col not in normalized.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")
    if "SENT_IDS" not in normalized.columns:
        normalized["SENT_IDS"] = ""
    return normalized


def load_split_data(excel_path: Path) -> Dict[str, pd.DataFrame]:
    workbook = pd.ExcelFile(excel_path)
    sheet_lookup = {sheet.lower(): sheet for sheet in workbook.sheet_names}
    aliases = {"train": "train", "validation": "val", "val": "val", "test": "test"}
    split_sheets: Dict[str, str] = {}
    for raw_name, split_name in aliases.items():
        if raw_name in sheet_lookup and split_name not in split_sheets:
            split_sheets[split_name] = sheet_lookup[raw_name]
    missing = [split for split in ("train", "val", "test") if split not in split_sheets]
    if missing:
        raise ValueError(f"Workbook must contain Train, Validation/Val, and Test sheets. Missing: {missing}")

    splits: Dict[str, pd.DataFrame] = {}
    for split_name, sheet_name in split_sheets.items():
        df = normalize_columns(workbook.parse(sheet_name, dtype=str))
        for col in ("SENT_ORI", "SCRIPT_CONTIN", "SENT_IDS", "STATE_2", "STATE_4"):
            df[col] = df[col].map(clean_text)
        df["PARA_ID"] = pd.to_numeric(df["PARA_ID"], errors="coerce")
        df["LINE_ID"] = pd.to_numeric(df["LINE_ID"], errors="coerce")
        df["STATE_2_LIST"] = df["STATE_2"].map(parse_state2)
        df["STATE_4_LIST"] = df["STATE_4"].map(parse_state4)
        valid_rows = [
            len(row.SCRIPT_CONTIN) > 0
            and len(row.STATE_2_LIST) == len(row.SCRIPT_CONTIN)
            and len(row.STATE_4_LIST) == len(row.SCRIPT_CONTIN)
            for row in df.itertuples(index=False)
        ]
        invalid_count = len(valid_rows) - sum(valid_rows)
        if invalid_count:
            warnings.warn(f"{split_name}: dropped {invalid_count} rows with mismatched text/label lengths.")
        df = df.loc[valid_rows].copy()
        df["EXPECTED_SPLIT_STATE2"] = [
            reconstruct_from_state2(script, labels)
            for script, labels in zip(df["SCRIPT_CONTIN"], df["STATE_2_LIST"])
        ]
        df["EXPECTED_SPLIT_STATE4"] = [
            reconstruct_from_state4(script, labels)
            for script, labels in zip(df["SCRIPT_CONTIN"], df["STATE_4_LIST"])
        ]
        splits[split_name] = df.sort_values(["ART_ID", "PARA_ID", "LINE_ID"]).reset_index(drop=True)
    return splits


def limit_split_rows(splits: Dict[str, pd.DataFrame], config: Config) -> Dict[str, pd.DataFrame]:
    row_limits = {"train": config.train_rows, "val": config.val_rows, "test": config.test_rows}
    limited: Dict[str, pd.DataFrame] = {}
    for split_name, dataframe in splits.items():
        limit = row_limits[split_name]
        limited[split_name] = dataframe.head(limit).copy().reset_index(drop=True) if limit > 0 else dataframe.copy()
        print(
            f"{split_name.upper()}: using {len(limited[split_name])} rows "
            f"from {len(dataframe)} available rows.",
            flush=True,
        )
    return limited


def build_char_vocab(train_df: pd.DataFrame) -> Dict[str, int]:
    counter = Counter()
    for text in train_df["SCRIPT_CONTIN"]:
        counter.update(text)
    counter.update(" ")
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


def rolling_context_rows(dataframe: pd.DataFrame, label_scheme: str, window_size: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    expected_col = "EXPECTED_SPLIT_STATE2" if label_scheme == "state2" else "EXPECTED_SPLIT_STATE4"
    label_col = "STATE_2_LIST" if label_scheme == "state2" else "STATE_4_LIST"
    for _, group in dataframe.groupby(["ART_ID", "PARA_ID"], sort=False):
        history: List[str] = []
        for row in group.itertuples(index=False):
            current_script = row.SCRIPT_CONTIN
            context = history[-max(window_size - 1, 0) :]
            input_text = " ".join([*context, current_script]) if context else current_script
            current_start = len(input_text) - len(current_script)
            labels = list(getattr(row, label_col))
            full_labels = [PAD_LABEL] * current_start + labels
            rows.append(
                {
                    "input_text": input_text,
                    "current_start": current_start,
                    "labels": full_labels,
                    "art_id": row.ART_ID,
                    "para_id": row.PARA_ID,
                    "line_id": row.LINE_ID,
                    "sent_ids": row.SENT_IDS,
                    "script_contin": current_script,
                    "sent_ori": row.SENT_ORI,
                    "expected_split": getattr(row, expected_col),
                }
            )
            history.append(getattr(row, expected_col))
    return rows


class LineWiseDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        char_vocab: Dict[str, int],
        label_scheme: str,
        window_size: int,
        label_to_id: Dict[object, int],
    ) -> None:
        self.examples = rolling_context_rows(dataframe, label_scheme, window_size)
        self.char_vocab = char_vocab
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        example = self.examples[index]
        char_ids = [self.char_vocab.get(char, self.char_vocab[UNK_TOKEN]) for char in str(example["input_text"])]
        label_ids = [
            PAD_LABEL if label == PAD_LABEL else self.label_to_id[label]
            for label in example["labels"]
        ]
        return {
            **example,
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "labels_tensor": torch.tensor(label_ids, dtype=torch.long),
        }


def collate_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    padded_chars = pad_sequence([item["char_ids"] for item in batch], batch_first=True, padding_value=0)
    padded_labels = pad_sequence([item["labels_tensor"] for item in batch], batch_first=True, padding_value=PAD_LABEL)
    return {
        "char_ids": padded_chars,
        "labels": padded_labels,
        "mask": padded_labels.ne(PAD_LABEL),
        "items": list(batch),
    }


class SequenceTaggerBase(nn.Module):
    def __init__(
        self,
        embedding_weights: torch.Tensor,
        dropout: float,
        num_labels: int,
        use_crf: bool = False,
        crf_kwargs: Dict[str, torch.Tensor] | None = None,
        invalid_penalty_masks: Dict[str, torch.Tensor] | None = None,
        invalid_penalty_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_weights.clone(), freeze=False, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.num_labels = num_labels
        self.use_crf = use_crf
        self.crf = CRFLayer(num_labels, **(crf_kwargs or {})) if use_crf else None
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)
        self.invalid_penalty_masks = invalid_penalty_masks or {}
        self.invalid_penalty_weight = invalid_penalty_weight

    def loss(self, inputs: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emissions = self.forward(inputs)
        if not self.use_crf:
            return self.criterion(emissions.view(-1, emissions.size(-1)), tags.view(-1))
            
        batch_size = emissions.size(0)
        valid_lens = mask.long().sum(dim=1)
        max_len = valid_lens.max().item()
        
        packed_emissions = emissions.new_zeros(batch_size, max_len, emissions.size(2))
        packed_tags = tags.new_zeros(batch_size, max_len)
        packed_mask = mask.new_zeros(batch_size, max_len, dtype=torch.bool)
        
        for i in range(batch_size):
            vlen = valid_lens[i].item()
            if vlen > 0:
                first_true = mask[i].nonzero(as_tuple=True)[0][0].item()
                packed_emissions[i, :vlen] = emissions[i, first_true : first_true + vlen]
                packed_tags[i, :vlen] = tags[i, first_true : first_true + vlen]
                packed_mask[i, :vlen] = True
                
        safe_tags = packed_tags.masked_fill(~packed_mask, 0)
        loss = self.crf(packed_emissions, safe_tags, packed_mask)
        if self.invalid_penalty_masks and self.invalid_penalty_weight > 0:
            loss = loss + self.invalid_penalty_weight * compute_invalid_bies_penalty(
                packed_emissions,
                packed_mask,
                self.invalid_penalty_masks,
            )
        return loss

    def decode(self, inputs: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        emissions = self.forward(inputs)
        if not self.use_crf:
            predicted = emissions.argmax(dim=-1)
            return [predicted[i][mask[i]].tolist() for i in range(predicted.size(0))]
            
        batch_size = emissions.size(0)
        valid_lens = mask.long().sum(dim=1)
        max_len = valid_lens.max().item()
        
        packed_emissions = emissions.new_zeros(batch_size, max_len, emissions.size(2))
        packed_mask = mask.new_zeros(batch_size, max_len, dtype=torch.bool)
        
        for i in range(batch_size):
            vlen = valid_lens[i].item()
            if vlen > 0:
                first_true = mask[i].nonzero(as_tuple=True)[0][0].item()
                packed_emissions[i, :vlen] = emissions[i, first_true : first_true + vlen]
                packed_mask[i, :vlen] = True
                
        return self.crf.decode(packed_emissions, packed_mask)


class BiLSTMTagger(SequenceTaggerBase):
    def __init__(
        self,
        embedding_weights: torch.Tensor,
        hidden_dim: int,
        dropout: float,
        num_labels: int,
        use_crf: bool = False,
        crf_kwargs: Dict[str, torch.Tensor] | None = None,
        invalid_penalty_masks: Dict[str, torch.Tensor] | None = None,
        invalid_penalty_weight: float = 0.0,
    ) -> None:
        super().__init__(
            embedding_weights,
            dropout,
            num_labels,
            use_crf,
            crf_kwargs,
            invalid_penalty_masks,
            invalid_penalty_weight,
        )
        self.encoder = nn.LSTM(embedding_weights.size(1), hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.encoder(self.dropout(self.embedding(inputs)))
        return self.classifier(self.dropout(outputs))


class BiRNNTagger(SequenceTaggerBase):
    def __init__(
        self,
        embedding_weights: torch.Tensor,
        hidden_dim: int,
        dropout: float,
        num_labels: int,
        use_crf: bool = False,
        crf_kwargs: Dict[str, torch.Tensor] | None = None,
        invalid_penalty_masks: Dict[str, torch.Tensor] | None = None,
        invalid_penalty_weight: float = 0.0,
    ) -> None:
        super().__init__(
            embedding_weights,
            dropout,
            num_labels,
            use_crf,
            crf_kwargs,
            invalid_penalty_masks,
            invalid_penalty_weight,
        )
        self.encoder = nn.RNN(
            embedding_weights.size(1),
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            nonlinearity="tanh",
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.encoder(self.dropout(self.embedding(inputs)))
        return self.classifier(self.dropout(outputs))


class BiGRUTagger(SequenceTaggerBase):
    def __init__(
        self,
        embedding_weights: torch.Tensor,
        hidden_dim: int,
        dropout: float,
        num_labels: int,
        use_crf: bool = False,
        crf_kwargs: Dict[str, torch.Tensor] | None = None,
        invalid_penalty_masks: Dict[str, torch.Tensor] | None = None,
        invalid_penalty_weight: float = 0.0,
    ) -> None:
        super().__init__(
            embedding_weights,
            dropout,
            num_labels,
            use_crf,
            crf_kwargs,
            invalid_penalty_masks,
            invalid_penalty_weight,
        )
        self.encoder = nn.GRU(embedding_weights.size(1), hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.encoder(self.dropout(self.embedding(inputs)))
        return self.classifier(self.dropout(outputs))


class CNNTagger(SequenceTaggerBase):
    def __init__(
        self,
        embedding_weights: torch.Tensor,
        cnn_channels: int,
        dropout: float,
        num_labels: int,
        use_crf: bool = False,
        crf_kwargs: Dict[str, torch.Tensor] | None = None,
        invalid_penalty_masks: Dict[str, torch.Tensor] | None = None,
        invalid_penalty_weight: float = 0.0,
    ) -> None:
        super().__init__(
            embedding_weights,
            dropout,
            num_labels,
            use_crf,
            crf_kwargs,
            invalid_penalty_masks,
            invalid_penalty_weight,
        )
        self.conv = nn.Conv1d(embedding_weights.size(1), cnn_channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(cnn_channels, num_labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embedding(inputs)).transpose(1, 2)
        x = self.activation(self.conv(x)).transpose(1, 2)
        return self.classifier(self.dropout(x))


class CRFLayer(nn.Module):
    def __init__(
        self,
        num_tags: int,
        transition_mask: torch.Tensor | None = None,
        start_mask: torch.Tensor | None = None,
        end_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.register_buffer(
            "transition_mask",
            transition_mask if transition_mask is not None else torch.ones(num_tags, num_tags, dtype=torch.bool),
        )
        self.register_buffer(
            "start_mask",
            start_mask if start_mask is not None else torch.ones(num_tags, dtype=torch.bool),
        )
        self.register_buffer(
            "end_mask",
            end_mask if end_mask is not None else torch.ones(num_tags, dtype=torch.bool),
        )
        self.constraint_value = -1e4
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def _masked_start_transitions(self) -> torch.Tensor:
        return self.start_transitions.masked_fill(~self.start_mask, self.constraint_value)

    def _masked_end_transitions(self) -> torch.Tensor:
        return self.end_transitions.masked_fill(~self.end_mask, self.constraint_value)

    def _masked_transitions(self) -> torch.Tensor:
        return self.transitions.masked_fill(~self.transition_mask, self.constraint_value)

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_log_partition(emissions, mask)
        return torch.mean(denominator - numerator)

    def _compute_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        start_transitions = self._masked_start_transitions()
        end_transitions = self._masked_end_transitions()
        transitions = self._masked_transitions()
        score = start_transitions[tags[:, 0]]
        score += emissions[:, 0, :].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        for timestep in range(1, emissions.size(1)):
            active = mask[:, timestep]
            prev_tags = tags[:, timestep - 1]
            curr_tags = tags[:, timestep]
            emit = emissions[:, timestep, :].gather(1, curr_tags.unsqueeze(1)).squeeze(1)
            score += (transitions[prev_tags, curr_tags] + emit) * active
        seq_ends = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, seq_ends.unsqueeze(1)).squeeze(1)
        score += end_transitions[last_tags]
        return score

    def _compute_log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        start_transitions = self._masked_start_transitions()
        end_transitions = self._masked_end_transitions()
        transitions = self._masked_transitions()
        score = start_transitions + emissions[:, 0]
        for timestep in range(1, emissions.size(1)):
            next_score = score.unsqueeze(2) + transitions + emissions[:, timestep].unsqueeze(1)
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, timestep].unsqueeze(1), next_score, score)
        score = score + end_transitions
        return torch.logsumexp(score, dim=1)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        start_transitions = self._masked_start_transitions()
        end_transitions = self._masked_end_transitions()
        transitions = self._masked_transitions()
        score = start_transitions + emissions[:, 0]
        history: List[torch.Tensor] = []
        for timestep in range(1, emissions.size(1)):
            next_score = score.unsqueeze(2) + transitions
            best_score, best_path = next_score.max(dim=1)
            best_score = best_score + emissions[:, timestep]
            score = torch.where(mask[:, timestep].unsqueeze(1), best_score, score)
            history.append(best_path)
        score = score + end_transitions
        best_last_tags = score.argmax(dim=1)

        paths: List[List[int]] = []
        lengths = mask.long().sum(dim=1).tolist()
        for batch_idx, length in enumerate(lengths):
            tag = best_last_tags[batch_idx]
            path = [int(tag.item())]
            for hist in reversed(history[: max(length - 1, 0)]):
                tag = hist[batch_idx][tag]
                path.append(int(tag.item()))
            paths.append(list(reversed(path)))
        return paths


def build_bies_crf_constraints(label_to_id: Dict[object, int]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    num_labels = len(label_to_id)
    transition_mask = torch.zeros(num_labels, num_labels, dtype=torch.bool)
    start_mask = torch.zeros(num_labels, dtype=torch.bool)
    end_mask = torch.zeros(num_labels, dtype=torch.bool)

    valid_next = {
        "B": ("I", "E"),
        "I": ("I", "E"),
        "E": ("B", "S"),
        "S": ("B", "S"),
    }
    for label in ("B", "S"):
        start_mask[label_to_id[label]] = True
    for label in ("E", "S"):
        end_mask[label_to_id[label]] = True
    for current_label, next_labels in valid_next.items():
        for next_label in next_labels:
            transition_mask[label_to_id[current_label], label_to_id[next_label]] = True

    return (
        {"transition_mask": transition_mask, "start_mask": start_mask, "end_mask": end_mask},
        {"transition": ~transition_mask, "start": ~start_mask, "end": ~end_mask},
    )


def compute_invalid_bies_penalty(
    emissions: torch.Tensor,
    mask: torch.Tensor,
    invalid_masks: Dict[str, torch.Tensor],
) -> torch.Tensor:
    probabilities = torch.softmax(emissions, dim=-1)
    lengths = mask.long().sum(dim=1)
    start_penalty = probabilities[:, 0, :][:, invalid_masks["start"].to(probabilities.device)].sum(dim=1)
    last_positions = (lengths - 1).clamp_min(0)
    last_probabilities = probabilities[torch.arange(probabilities.size(0), device=probabilities.device), last_positions]
    end_penalty = last_probabilities[:, invalid_masks["end"].to(probabilities.device)].sum(dim=1)

    transition_penalty = torch.zeros(probabilities.size(0), device=probabilities.device)
    transition_invalid = invalid_masks["transition"].to(probabilities.device)
    for timestep in range(1, probabilities.size(1)):
        active = mask[:, timestep].float()
        pair_probabilities = probabilities[:, timestep - 1].unsqueeze(2) * probabilities[:, timestep].unsqueeze(1)
        invalid_mass = pair_probabilities.masked_select(transition_invalid.unsqueeze(0)).view(probabilities.size(0), -1).sum(dim=1)
        transition_penalty = transition_penalty + invalid_mass * active

    return ((start_penalty + end_penalty + transition_penalty) / lengths.float().clamp_min(1.0)).mean()


def build_model(
    model_name: str,
    embedding_weights: torch.Tensor,
    config: Config,
    num_labels: int,
    use_crf: bool = False,
    label_to_id: Dict[object, int] | None = None,
) -> nn.Module:
    crf_kwargs: Dict[str, torch.Tensor] = {}
    invalid_penalty_masks: Dict[str, torch.Tensor] | None = None
    invalid_penalty_weight = 0.0
    if use_crf:
        if label_to_id is None:
            raise ValueError("label_to_id is required when use_crf=True.")
        crf_kwargs, invalid_penalty_masks = build_bies_crf_constraints(label_to_id)
        invalid_penalty_weight = config.invalid_transition_penalty_weight
    if model_name == "bilstm":
        return BiLSTMTagger(
            embedding_weights,
            config.hidden_dim,
            config.dropout,
            num_labels,
            use_crf,
            crf_kwargs,
            invalid_penalty_masks,
            invalid_penalty_weight,
        )
    if model_name == "cnn":
        return CNNTagger(
            embedding_weights,
            config.cnn_channels,
            config.dropout,
            num_labels,
            use_crf,
            crf_kwargs,
            invalid_penalty_masks,
            invalid_penalty_weight,
        )
    if model_name == "bigru":
        return BiGRUTagger(
            embedding_weights,
            config.hidden_dim,
            config.dropout,
            num_labels,
            use_crf,
            crf_kwargs,
            invalid_penalty_masks,
            invalid_penalty_weight,
        )
    if model_name == "birnn":
        return BiRNNTagger(
            embedding_weights,
            config.hidden_dim,
            config.dropout,
            num_labels,
            use_crf,
            crf_kwargs,
            invalid_penalty_masks,
            invalid_penalty_weight,
        )
    raise ValueError(f"Unknown model name: {model_name}")


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        inputs = batch["char_ids"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)
        optimizer.zero_grad()
        loss = model.loss(inputs, labels, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def predict_batch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    id_to_label: Dict[int, object],
    label_scheme: str,
) -> Tuple[float, pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["char_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            total_loss += model.loss(inputs, labels, mask).item()
            predicted_ids = model.decode(inputs, mask)
            for idx, (item, pred_seq) in enumerate(zip(batch["items"], predicted_ids)):
                true_seq = labels[idx][mask[idx]].tolist()
                pred_labels = [id_to_label[item_id] for item_id in pred_seq]
                true_labels = [id_to_label[item_id] for item_id in true_seq]
                script = item["script_contin"]
                predicted_output = (
                    reconstruct_from_state2(script, pred_labels)
                    if label_scheme == "state2"
                    else reconstruct_from_state4(script, pred_labels)
                )
                original_output = (
                    reconstruct_from_state2(script, true_labels)
                    if label_scheme == "state2"
                    else reconstruct_from_state4(script, true_labels)
                )
                rows.append(
                    {
                        "ART_ID": item["art_id"],
                        "PARA_ID": item["para_id"],
                        "LINE_ID": item["line_id"],
                        "SENT_IDS": item["sent_ids"],
                        "INPUT_TEXT": item["input_text"],
                        "CURRENT_LINE_START": item["current_start"],
                        "SCRIPT_CONTIN": script,
                        "SENT_ORI": item["sent_ori"],
                        "ORIGINAL_OUTPUT": original_output,
                        "PREDICTED_OUTPUT": predicted_output,
                        "TRUE_LABELS": " ".join(str(x) for x in true_labels),
                        "PRED_LABELS": " ".join(str(x) for x in pred_labels),
                    }
                )
    return total_loss / max(len(loader), 1), pd.DataFrame(rows)


def predict_linewise_cascade(
    model: nn.Module,
    dataframe: pd.DataFrame,
    char_vocab: Dict[str, int],
    device: torch.device,
    id_to_label: Dict[int, object],
    label_scheme: str,
    window_size: int,
) -> pd.DataFrame:
    model.eval()
    rows: List[Dict[str, object]] = []
    expected_col = "EXPECTED_SPLIT_STATE2" if label_scheme == "state2" else "EXPECTED_SPLIT_STATE4"
    label_col = "STATE_2_LIST" if label_scheme == "state2" else "STATE_4_LIST"
    with torch.no_grad():
        for _, group in dataframe.groupby(["ART_ID", "PARA_ID"], sort=False):
            predicted_history: List[str] = []
            for row in group.itertuples(index=False):
                context = predicted_history[-max(window_size - 1, 0) :]
                input_text = " ".join([*context, row.SCRIPT_CONTIN]) if context else row.SCRIPT_CONTIN
                current_start = len(input_text) - len(row.SCRIPT_CONTIN)
                char_ids = [char_vocab.get(char, char_vocab[UNK_TOKEN]) for char in input_text]
                inputs = torch.tensor([char_ids], dtype=torch.long, device=device)
                mask = torch.ones_like(inputs, dtype=torch.bool, device=device)
                decoded_ids = model.decode(inputs, mask)[0][current_start : current_start + len(row.SCRIPT_CONTIN)]
                pred_labels = [id_to_label[item] for item in decoded_ids]
                true_labels = list(getattr(row, label_col))
                predicted_output = (
                    reconstruct_from_state2(row.SCRIPT_CONTIN, pred_labels)
                    if label_scheme == "state2"
                    else reconstruct_from_state4(row.SCRIPT_CONTIN, pred_labels)
                )
                original_output = getattr(row, expected_col)
                rows.append(
                    {
                        "ART_ID": row.ART_ID,
                        "PARA_ID": row.PARA_ID,
                        "LINE_ID": row.LINE_ID,
                        "SENT_IDS": row.SENT_IDS,
                        "INPUT_TEXT": input_text,
                        "CURRENT_LINE_START": current_start,
                        "SCRIPT_CONTIN": row.SCRIPT_CONTIN,
                        "SENT_ORI": row.SENT_ORI,
                        "ORIGINAL_OUTPUT": original_output,
                        "PREDICTED_OUTPUT": predicted_output,
                        "TRUE_LABELS": " ".join(str(x) for x in true_labels),
                        "PRED_LABELS": " ".join(str(x) for x in pred_labels),
                    }
                )
                predicted_history.append(predicted_output)
    return pd.DataFrame(rows)


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
    brevity_penalty = 1.0 if cand_length > ref_length else math.exp(1 - (ref_length / cand_length))
    return float(geo_mean * brevity_penalty)


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
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
    return float((10 * precision * recall / (recall + 9 * precision)) if (recall + 9 * precision) else 0.0)


def compute_text_generation_metrics(predictions_df: pd.DataFrame, bert_batch_size: int, use_bertscore: bool) -> Dict[str, float]:
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
    if not use_bertscore:
        return scores
    try:
        from bert_score import score as bert_score

        precision, recall, f1 = bert_score(candidates, references, lang="en", verbose=False, batch_size=bert_batch_size)
        scores["bertscore_precision"] = float(precision.mean().item())
        scores["bertscore_recall"] = float(recall.mean().item())
        scores["bertscore_f1"] = float(f1.mean().item())
    except Exception as exc:
        warnings.warn(f"BERTScore skipped: {exc}")
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
    use_bertscore: bool,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    y_true, y_pred = flatten_labels(predictions_df, label_to_id)
    label_ids = list(id_to_label.keys())
    class_scores = classification_metrics(y_true, y_pred, label_ids)
    text_scores = compute_text_generation_metrics(predictions_df, bert_batch_size, use_bertscore)
    confusion_df = pd.DataFrame(
        class_scores["confusion_matrix"],
        index=[f"TRUE_{id_to_label[label_id]}" for label_id in label_ids],
        columns=[f"PRED_{id_to_label[label_id]}" for label_id in label_ids],
    )
    return {"precision": class_scores["precision"], "recall": class_scores["recall"], "f1": class_scores["f1"], "accuracy": class_scores["accuracy"], **text_scores}, confusion_df


def create_data_loaders(
    splits: Dict[str, pd.DataFrame],
    char_vocab: Dict[str, int],
    label_scheme: str,
    label_to_id: Dict[object, int],
    config: Config,
) -> Dict[str, DataLoader]:
    return {
        split_name: DataLoader(
            LineWiseDataset(dataframe, char_vocab, label_scheme, config.window_size, label_to_id),
            batch_size=config.batch_size,
            shuffle=(split_name == "train"),
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_batch,
        )
        for split_name, dataframe in splits.items()
    }


def fit_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    config: Config,
    label_to_id: Dict[object, int],
    id_to_label: Dict[int, object],
    label_scheme: str,
    model_display_name: str,
) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = None
    best_val_f1 = -1.0
    patience_counter = 0
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, device)
        val_loss, val_predictions = predict_batch(model, loaders["val"], device, id_to_label, label_scheme)
        val_f1 = quick_label_f1(val_predictions, label_to_id, id_to_label)
        print(
            f"[{label_scheme.upper()}][{model_display_name}] "
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}",
            flush=True,
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


def summary_row(
    model_display_name: str,
    label_scheme: str,
    split_name: str,
    embedding_dim: int,
    window_size: int,
    summary: Dict[str, float],
) -> Dict[str, object]:
    return {
        "MODEL": model_display_name,
        "LABEL_SCHEME": label_scheme.upper(),
        "SPLIT": split_name.upper(),
        "WINDOW_SIZE": window_size,
        "EMBEDDING": f"trainable_char_embedding_{embedding_dim}d",
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


def add_prediction_metadata(
    predictions_df: pd.DataFrame,
    model_display_name: str,
    label_scheme: str,
    split_name: str,
    mode: str,
) -> pd.DataFrame:
    predictions_df = predictions_df.copy()
    predictions_df.insert(0, "MODEL", model_display_name)
    predictions_df.insert(1, "LABEL_SCHEME", label_scheme.upper())
    predictions_df.insert(2, "SPLIT", split_name.upper())
    predictions_df.insert(3, "MODE", mode)
    return predictions_df


def run_experiment(config: Config) -> None:
    set_seed(RANDOM_SEED)
    device = select_device(config.require_gpu)
    print(f"Using device: {device}", flush=True)
    print(
        f"Line-wise mode: previous predicted lines + current scriptio continua line, window_size={config.window_size}",
        flush=True,
    )

    splits = limit_split_rows(load_split_data(config.excel_path), config)
    char_vocab = build_char_vocab(splits["train"])
    embedding_weights = create_embedding_weights(len(char_vocab), config.embedding_dim, RANDOM_SEED)

    label_scheme = "state4"
    label_to_id = {"B": 0, "I": 1, "E": 2, "S": 3}
    id_to_label = {0: "B", 1: "I", 2: "E", 3: "S"}
    print("Training STATE_4 plain models and STATE_4+CRF models.", flush=True)
    print("CRF constraints: start=B/S, end=E/S, B->I/E, I->I/E, E->B/S, S->B/S", flush=True)

    summary_frames: Dict[str, pd.DataFrame] = {}
    prediction_frames: Dict[str, pd.DataFrame] = {}
    confusion_frames: Dict[str, pd.DataFrame] = {}

    loaders = create_data_loaders(splits, char_vocab, label_scheme, label_to_id, config)
    summary_rows: List[Dict[str, object]] = []
    all_predictions: List[pd.DataFrame] = []
    model_variants = [
        (model_name, model_display_name, False)
        for model_name, model_display_name in MODEL_DISPLAY_NAMES.items()
    ]
    model_variants.extend(
        (model_name, f"{model_display_name}+CRF", True)
        for model_name, model_display_name in MODEL_DISPLAY_NAMES.items()
    )

    for model_name, model_display_name, use_crf in model_variants:
        print(f"\nTraining {model_display_name} on {label_scheme.upper()} ...", flush=True)
        model = build_model(
            model_name,
            embedding_weights,
            config,
            len(label_to_id),
            use_crf=use_crf,
            label_to_id=label_to_id if use_crf else None,
        ).to(device)
        model = fit_model(model, loaders, device, config, label_to_id, id_to_label, label_scheme, model_display_name)

        for split_name in ("train", "val", "test"):
            if split_name == "train":
                _, predictions_df = predict_batch(model, loaders[split_name], device, id_to_label, label_scheme)
                mode = "TEACHER_FORCED_CONTEXT"
            else:
                predictions_df = predict_linewise_cascade(
                    model,
                    splits[split_name],
                    char_vocab,
                    device,
                    id_to_label,
                    label_scheme,
                    config.window_size,
                )
                mode = "CASCADED_PREDICTED_CONTEXT"

            summary, confusion_df = evaluate_predictions(
                predictions_df,
                label_to_id,
                id_to_label,
                config.bert_score_batch_size,
                config.use_bertscore,
            )
            summary_rows.append(summary_row(model_display_name, label_scheme, split_name, config.embedding_dim, config.window_size, summary))
            all_predictions.append(add_prediction_metadata(predictions_df, model_display_name, label_scheme, split_name, mode))
            crf_suffix = "_crf" if use_crf else ""
            confusion_frames[f"cm_{label_scheme}_{model_name}{crf_suffix}_{split_name}"] = confusion_df

    summary_frames[f"summary_{label_scheme}"] = pd.DataFrame(summary_rows)
    prediction_frames[f"pred_{label_scheme}"] = pd.concat(all_predictions, ignore_index=True)

    summary_frames["run_config"] = pd.DataFrame([config.__dict__ | {"random_seed": RANDOM_SEED}])
    write_results_to_excel(config.output_path, summary_frames, prediction_frames, confusion_frames)
    print(f"\nFinished. Output written to: {config.output_path.resolve()}", flush=True)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train line-wise 4-state BIES models with and without CRF.")
    parser.add_argument("--excel-path", type=Path, default=DEFAULT_EXCEL_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--window-size", type=int, default=Config.window_size)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--embedding-dim", type=int, default=Config.embedding_dim)
    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--cnn-channels", type=int, default=Config.cnn_channels)
    parser.add_argument("--use-bertscore", action="store_true")
    parser.add_argument("--invalid-transition-penalty-weight", type=float, default=Config.invalid_transition_penalty_weight)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--train-rows", type=int, default=Config.train_rows)
    parser.add_argument("--val-rows", type=int, default=Config.val_rows)
    parser.add_argument("--test-rows", type=int, default=Config.test_rows)
    args = parser.parse_args()
    return Config(
        excel_path=args.excel_path,
        output_path=args.output_path,
        window_size=args.window_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        cnn_channels=args.cnn_channels,
        use_bertscore=args.use_bertscore,
        invalid_transition_penalty_weight=args.invalid_transition_penalty_weight,
        require_gpu=args.require_gpu,
        train_rows=args.train_rows,
        val_rows=args.val_rows,
        test_rows=args.test_rows,
    )


if __name__ == "__main__":
    run_experiment(parse_args())
