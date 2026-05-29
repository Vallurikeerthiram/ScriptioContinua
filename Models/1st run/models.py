import argparse
import json
import math
import random
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

try:
    import sklearn_crfsuite

    HAS_CRF = True
except Exception:
    HAS_CRF = False

try:
    from bert_score import BERTScorer

    HAS_BERTSCORE = True
except Exception:
    HAS_BERTSCORE = False
    BERTScorer = None  # type: ignore[assignment]

_BERT_SCORER: Optional["BERTScorer"] = None

try:
    from transformers import logging as hf_logging
except Exception:
    hf_logging = None


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_state2(value: str, n_chars: int) -> List[str]:
    flags = re.findall(r"[01]", str(value))
    if len(flags) < n_chars:
        flags += ["0"] * (n_chars - len(flags))
    return flags[:n_chars]


def parse_state4(value: str, n_chars: int) -> List[str]:
    tags = re.findall(r"[SBEMI]", str(value).upper())
    tags = ["M" if t == "I" else t for t in tags]
    if len(tags) < n_chars:
        tags += ["M"] * (n_chars - len(tags))
    return tags[:n_chars]


def segment_with_state2(script: str, labels: List[str]) -> str:
    script = str(script)
    pieces = []
    n = min(len(script), len(labels))
    for i in range(n):
        pieces.append(script[i])
        if labels[i] == "1" and i < n - 1:
            pieces.append(" ")
    return re.sub(r"\s+", " ", "".join(pieces)).strip()


def segment_with_state4(script: str, labels: List[str]) -> str:
    script = str(script)
    n = min(len(script), len(labels))
    words = []
    current = []
    for i in range(n):
        ch = script[i]
        tag = "M" if labels[i] == "I" else labels[i]
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


def lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    return int(dp[len(a), len(b)])


def corpus_bleu_simple(references: List[str], candidates: List[str], max_n: int = 4) -> float:
    eps = 1e-9
    if not references:
        return 0.0
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    ref_len = 0
    cand_len = 0
    for ref, cand in zip(references, candidates):
        ref_toks = ref.split()
        cand_toks = cand.split()
        ref_len += len(ref_toks)
        cand_len += len(cand_toks)
        for n in range(1, max_n + 1):
            ref_ngrams = Counter(tuple(ref_toks[i : i + n]) for i in range(max(0, len(ref_toks) - n + 1)))
            cand_ngrams = Counter(tuple(cand_toks[i : i + n]) for i in range(max(0, len(cand_toks) - n + 1)))
            total_counts[n - 1] += sum(cand_ngrams.values())
            clipped_counts[n - 1] += sum(min(count, ref_ngrams[ng]) for ng, count in cand_ngrams.items())
    precisions = [(clipped_counts[i] + eps) / (total_counts[i] + eps) for i in range(max_n)]
    if cand_len == 0:
        return 0.0
    bp = 1.0 if cand_len > ref_len else math.exp(1.0 - (ref_len / max(cand_len, 1)))
    bleu = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return float(bleu)


def rouge_l_f1_avg(references: List[str], candidates: List[str]) -> float:
    scores = []
    for ref, cand in zip(references, candidates):
        ref_toks = ref.split()
        cand_toks = cand.split()
        if len(ref_toks) == 0 and len(cand_toks) == 0:
            scores.append(1.0)
            continue
        if len(ref_toks) == 0 or len(cand_toks) == 0:
            scores.append(0.0)
            continue
        lcs = lcs_length(ref_toks, cand_toks)
        prec = lcs / len(cand_toks)
        rec = lcs / len(ref_toks)
        denom = prec + rec
        scores.append(0.0 if denom == 0 else (2 * prec * rec / denom))
    return float(np.mean(scores) if scores else 0.0)


def meteor_like_avg(references: List[str], candidates: List[str]) -> float:
    scores = []
    for ref, cand in zip(references, candidates):
        ref_toks = ref.split()
        cand_toks = cand.split()
        if len(ref_toks) == 0 and len(cand_toks) == 0:
            scores.append(1.0)
            continue
        if len(ref_toks) == 0 or len(cand_toks) == 0:
            scores.append(0.0)
            continue
        ref_counts = Counter(ref_toks)
        cand_counts = Counter(cand_toks)
        matches = sum(min(ref_counts[w], cand_counts[w]) for w in cand_counts)
        p = matches / max(len(cand_toks), 1)
        r = matches / max(len(ref_toks), 1)
        denom = r + 9 * p
        scores.append(0.0 if denom == 0 else (10 * p * r / denom))
    return float(np.mean(scores) if scores else 0.0)


def get_bertscorer() -> Optional["BERTScorer"]:
    global _BERT_SCORER
    if not HAS_BERTSCORE:
        return None
    if _BERT_SCORER is None:
        _BERT_SCORER = BERTScorer(lang="en", model_type="roberta-large")
    return _BERT_SCORER


def compute_text_metrics(references: List[str], candidates: List[str], use_bertscore: bool = True) -> Dict[str, float]:
    bleu = corpus_bleu_simple(references, candidates)
    rouge_l = rouge_l_f1_avg(references, candidates)
    meteor = meteor_like_avg(references, candidates)
    bert_f1 = float("nan")
    if HAS_BERTSCORE and use_bertscore:
        try:
            scorer = get_bertscorer()
            if scorer is None:
                raise RuntimeError("BERTScorer unavailable")
            _, _, f1 = scorer.score(candidates, references)
            bert_f1 = float(f1.mean().item())
        except Exception:
            bert_f1 = float("nan")
    return {
        "BLEU": bleu,
        "ROUGE_L": rouge_l,
        "METEOR": meteor,
        "BERT_F1": bert_f1,
    }


@dataclass
class CharVocab:
    stoi: Dict[str, int]
    itos: Dict[int, str]


def build_char_vocab(texts: List[str]) -> CharVocab:
    chars = sorted(set("".join(texts)))
    stoi = {"<PAD>": 0, "<UNK>": 1}
    for c in chars:
        if c not in stoi:
            stoi[c] = len(stoi)
    itos = {i: s for s, i in stoi.items()}
    return CharVocab(stoi=stoi, itos=itos)


class SeqDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[List[str]], char_stoi: Dict[str, int], label_stoi: Dict[str, int]):
        self.texts = texts
        self.labels = labels
        self.char_stoi = char_stoi
        self.label_stoi = label_stoi

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        lbls = self.labels[idx]
        x = [self.char_stoi.get(ch, self.char_stoi["<UNK>"]) for ch in text]
        y = [self.label_stoi[t] for t in lbls]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long), len(x)


def collate_batch(batch):
    xs, ys, lens = zip(*batch)
    max_len = max(lens)
    x_pad = torch.zeros((len(batch), max_len), dtype=torch.long)
    y_pad = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, (x, y, l) in enumerate(zip(xs, ys, lens)):
        x_pad[i, :l] = x
        y_pad[i, :l] = y
    return x_pad, y_pad, torch.tensor(lens, dtype=torch.long)


class RNNTagger(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int, emb_dim: int, hid_dim: int, kind: str = "RNN"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.2)
        kind = kind.upper()
        if kind == "GRU":
            self.encoder = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=False)
            out_dim = hid_dim
        elif kind == "BILSTM":
            self.encoder = nn.LSTM(emb_dim, hid_dim, batch_first=True, bidirectional=True)
            out_dim = hid_dim * 2
        else:
            self.encoder = nn.RNN(emb_dim, hid_dim, batch_first=True, bidirectional=False)
            out_dim = hid_dim
        self.classifier = nn.Linear(out_dim, num_labels)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, _ = self.encoder(emb)
        out = self.dropout(out)
        return self.classifier(out)


class CNNTagger(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int, emb_dim: int, hid_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv = nn.Conv1d(emb_dim, hid_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Conv1d(hid_dim, num_labels, kernel_size=1)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)
        z = torch.relu(self.conv(emb))
        z = self.dropout(z)
        logits = self.classifier(z).transpose(1, 2)
        return logits


def seq_to_char_features(text: str, i: int) -> Dict[str, object]:
    def safe_char(pos):
        if pos < 0 or pos >= len(text):
            return "<PAD>"
        return text[pos]

    c0 = safe_char(i)
    feat = {
        "c0": c0,
        "c-1": safe_char(i - 1),
        "c+1": safe_char(i + 1),
        "c-2": safe_char(i - 2),
        "c+2": safe_char(i + 2),
        "is_digit": c0.isdigit(),
        "is_alpha": c0.isalpha(),
    }
    return feat


class FeatureSequenceModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.vectorizer = DictVectorizer(sparse=True)
        self.model = None

    def fit(self, texts: List[str], labels: List[List[str]]):
        X_dict = []
        y = []
        for txt, seq in zip(texts, labels):
            for i in range(min(len(txt), len(seq))):
                X_dict.append(seq_to_char_features(txt, i))
                y.append(seq[i])
        X = self.vectorizer.fit_transform(X_dict)
        if self.model_name == "HRF":
            self.model = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
            self.model.fit(X, y)
        elif self.model_name == "CRF" and HAS_CRF:
            crf = sklearn_crfsuite.CRF(
                algorithm="lbfgs",
                c1=0.1,
                c2=0.1,
                max_iterations=100,
                all_possible_transitions=True,
            )
            seq_X = [[seq_to_char_features(txt, i) for i in range(len(txt))] for txt in texts]
            seq_y = [seq[: len(txt)] for txt, seq in zip(texts, labels)]
            crf.fit(seq_X, seq_y)
            self.model = crf
        else:
            if self.model_name == "CRF" and not HAS_CRF:
                warnings.warn("sklearn_crfsuite not installed. Using LogisticRegression fallback for CRF.")
            self.model = LogisticRegression(max_iter=400, n_jobs=None, multi_class="auto")
            self.model.fit(X, y)

    def predict(self, texts: List[str]) -> List[List[str]]:
        preds = []
        if self.model_name == "CRF" and HAS_CRF and isinstance(self.model, sklearn_crfsuite.CRF):
            seq_X = [[seq_to_char_features(txt, i) for i in range(len(txt))] for txt in texts]
            return self.model.predict(seq_X)
        for txt in texts:
            X_dict = [seq_to_char_features(txt, i) for i in range(len(txt))]
            X = self.vectorizer.transform(X_dict)
            p = self.model.predict(X).tolist()
            preds.append(p)
        return preds


def flatten_labels(labels: List[List[str]]) -> List[str]:
    out = []
    for seq in labels:
        out.extend(seq)
    return out


def build_segmentation(script: str, seq: List[str], state_type: str) -> str:
    if state_type == "STATE_2":
        return segment_with_state2(script, seq)
    return segment_with_state4(script, seq)


def evaluate_predictions(
    scripts: List[str],
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
    state_type: str,
    label_order: List[str],
    use_bertscore: bool = True,
) -> Dict[str, object]:
    aligned_true = []
    aligned_pred = []
    references = []
    candidates = []
    for script, t_seq, p_seq in zip(scripts, true_labels, pred_labels):
        n = min(len(script), len(t_seq), len(p_seq))
        t = t_seq[:n]
        p = p_seq[:n]
        aligned_true.append(t)
        aligned_pred.append(p)
        references.append(build_segmentation(script[:n], t, state_type))
        candidates.append(build_segmentation(script[:n], p, state_type))

    y_true = flatten_labels(aligned_true)
    y_pred = flatten_labels(aligned_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    text_scores = compute_text_metrics(references, candidates, use_bertscore=use_bertscore)

    result = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
    }
    result.update(text_scores)
    return result


def parse_labels_for_split(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[List[str]]]:
    scripts = df["SCRIPT_CONTIN"].astype(str).tolist()
    labels = []
    for s, raw_lbl in zip(scripts, df[target_col].astype(str).tolist()):
        n = len(s)
        if target_col == "STATE_2":
            labels.append(parse_state2(raw_lbl, n))
        else:
            labels.append(parse_state4(raw_lbl, n))
    return scripts, labels


def train_neural_model(
    model_name: str,
    train_texts: List[str],
    train_labels: List[List[str]],
    val_texts: List[str],
    val_labels: List[List[str]],
    label_stoi: Dict[str, int],
    label_itos: Dict[int, str],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    use_bertscore: bool,
):
    vocab = build_char_vocab(train_texts)
    train_ds = SeqDataset(train_texts, train_labels, vocab.stoi, label_stoi)
    val_ds = SeqDataset(val_texts, val_labels, vocab.stoi, label_stoi)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    num_labels = len(label_stoi)
    if model_name == "CNN":
        model = CNNTagger(len(vocab.stoi), num_labels, emb_dim=64, hid_dim=128).to(device)
    elif model_name == "GRU":
        model = RNNTagger(len(vocab.stoi), num_labels, emb_dim=64, hid_dim=128, kind="GRU").to(device)
    elif model_name == "BiLSTM":
        model = RNNTagger(len(vocab.stoi), num_labels, emb_dim=64, hid_dim=128, kind="BiLSTM").to(device)
    else:
        model = RNNTagger(len(vocab.stoi), num_labels, emb_dim=64, hid_dim=128, kind="RNN").to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    best_val_f1 = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for x, y, _lens in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_preds = predict_neural(model, val_loader, label_itos, device)
        val_true = []
        for _, y, lens in val_loader:
            for seq, ln in zip(y, lens):
                val_true.append([label_itos[int(idx)] for idx in seq[: ln].tolist()])
        tmp = evaluate_predictions(
            val_texts,
            val_true,
            val_preds,
            "STATE_2" if "1" in label_stoi else "STATE_4",
            [lbl for lbl in label_stoi if lbl != "<PAD>"],
            use_bertscore=use_bertscore,
        )
        if tmp["f1"] > best_val_f1:
            best_val_f1 = tmp["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, vocab


def predict_neural(model: nn.Module, loader: DataLoader, label_itos: Dict[int, str], device: torch.device):
    model.eval()
    out_preds = []
    with torch.no_grad():
        for x, _y, lens in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=-1).cpu()
            for seq, ln in zip(pred, lens):
                seq_lbl = [label_itos[int(idx)] for idx in seq[: ln].tolist()]
                out_preds.append(seq_lbl)
    return out_preds


def build_loader_for_prediction(texts: List[str], labels: List[List[str]], vocab: CharVocab, label_stoi: Dict[str, int], batch_size: int):
    ds = SeqDataset(texts, labels, vocab.stoi, label_stoi)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)


def run_task(
    task_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    use_bertscore: bool,
) -> pd.DataFrame:
    print(f"\n[INFO] Starting task: {task_name}", flush=True)
    train_texts, train_labels = parse_labels_for_split(train_df, task_name)
    val_texts, val_labels = parse_labels_for_split(val_df, task_name)
    test_texts, test_labels = parse_labels_for_split(test_df, task_name)

    if task_name == "STATE_2":
        label_order = ["0", "1"]
    else:
        label_order = ["S", "B", "M", "E"]

    label_stoi = {"<PAD>": 0}
    for lbl in label_order:
        label_stoi[lbl] = len(label_stoi)
    label_itos = {i: s for s, i in label_stoi.items()}

    rows = []
    model_names = ["BiLSTM", "RNN", "CNN", "GRU", "HRF", "CRF"]

    for model_name in model_names:
        print(f"[INFO] Training/Evaluating model: {model_name} ({task_name})", flush=True)
        if model_name in {"BiLSTM", "RNN", "CNN", "GRU"}:
            model, vocab = train_neural_model(
                model_name=model_name,
                train_texts=train_texts,
                train_labels=train_labels,
                val_texts=val_texts,
                val_labels=val_labels,
                label_stoi=label_stoi,
                label_itos=label_itos,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                use_bertscore=use_bertscore,
            )
            val_loader = build_loader_for_prediction(val_texts, val_labels, vocab, label_stoi, batch_size)
            test_loader = build_loader_for_prediction(test_texts, test_labels, vocab, label_stoi, batch_size)
            pred_val = predict_neural(model, val_loader, label_itos, device)
            pred_test = predict_neural(model, test_loader, label_itos, device)
        else:
            classic = FeatureSequenceModel(model_name)
            classic.fit(train_texts, train_labels)
            pred_val = classic.predict(val_texts)
            pred_test = classic.predict(test_texts)

        eval_val = evaluate_predictions(
            val_texts, val_labels, pred_val, task_name, label_order, use_bertscore=use_bertscore
        )
        eval_test = evaluate_predictions(
            test_texts, test_labels, pred_test, task_name, label_order, use_bertscore=use_bertscore
        )

        for split_name, metrics in [("VAL", eval_val), ("TEST", eval_test)]:
            rows.append(
                {
                    "Task": f"SCRIPT_CONTIN -> {task_name}",
                    "Model": model_name,
                    "Split": split_name,
                    "BLEU": metrics["BLEU"],
                    "ROUGE": metrics["ROUGE_L"],
                    "METEOR": metrics["METEOR"],
                    "BERT": metrics["BERT_F1"],
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "accuracy": metrics["accuracy"],
                    "recall": metrics["recall"],
                    "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
                }
            )
        print(f"[INFO] Completed model: {model_name} ({task_name})", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel_path", type=str, default="Sentence_First_Layer/SENT_based_split.xlsx")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=str, default="Sentence_First_Layer/evaluation_metrics.csv")
    parser.add_argument("--disable_bertscore", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings(
        "ignore",
        message=r"Some weights of RobertaModel were not initialized.*",
    )
    if hf_logging is not None:
        hf_logging.set_verbosity_error()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bertscore = HAS_BERTSCORE and not args.disable_bertscore
    print(f"[INFO] Device: {device}", flush=True)
    print(f"[INFO] BERTScore enabled: {use_bertscore}", flush=True)

    df_train = pd.read_excel(args.excel_path, sheet_name="TRAIN")
    df_val = pd.read_excel(args.excel_path, sheet_name="VAL")
    df_test = pd.read_excel(args.excel_path, sheet_name="TEST")

    all_results = []
    for task in ["STATE_2", "STATE_4"]:
        res = run_task(
            task_name=task,
            train_df=df_train,
            val_df=df_val,
            test_df=df_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            use_bertscore=use_bertscore,
        )
        all_results.append(res)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(args.output_csv, index=False)
    pd.set_option("display.max_colwidth", None)
    print("\nEvaluation Metrics Table:\n")
    print(final_df.to_string(index=False))
    print(f"\nSaved metrics to: {args.output_csv}")
    if not HAS_BERTSCORE:
        print("Note: bert_score is not installed, so BERT column is NaN.")
    elif args.disable_bertscore:
        print("Note: BERTScore was disabled by --disable_bertscore, so BERT column is NaN.")
    if not HAS_CRF:
        print("Note: sklearn_crfsuite is not installed, so CRF uses LogisticRegression fallback.")


if __name__ == "__main__":
    main()
