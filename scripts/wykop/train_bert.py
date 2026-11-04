"""Stage 2 BERT classifier for Wykop: train on the adjudicated labels, then label the corpus.

    uv sync --group bert
    uv run --group bert python scripts/wykop/train_bert.py train
    uv run --group bert python scripts/wykop/train_bert.py predict

`train` fine-tunes HerBERT (one encoder, a head per stage 2 label) on the work texts of
data/wykop/agents/gold.csv, keeps the epoch with the best validation macro-F1 and writes
the model and test metrics to data/wykop/bert/. The texts of the human reference sample
(wykop_sample_2) are left out. `predict` labels every stage 1 work text of the cleaned
collection, with the stage 1 ai label for the rules (amc/wykop/bert.py), into
data/wykop/wykop_stage2_bert.csv. Written for a CUDA GPU (e.g. RTX 4070 Super, 12 GB, bf16);
on CPU it works but is slow.
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from amc.classification import ai_work  # noqa: E402
from amc.csv_files import read_rows  # noqa: E402
from amc.wykop import bert, records  # noqa: E402
from amc.wykop.classification import COLUMNS as STAGE1_COLUMNS  # noqa: E402
from amc.wykop.classification import model_text, text_hash  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/wykop"
OUT = DATA / "bert"
BASE = "allegro/herbert-base-cased"
IGNORE = -100
PREDICTION_COLUMNS = ("type", "entry_id", "text_hash", *bert.LABELS, "model", "classified_at")


class Classifier(nn.Module):
    """The encoder's [CLS] vector, dropout, and a linear head per label."""

    def __init__(self, base: str, classes: dict[str, tuple[str, ...]], dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({label: nn.Linear(size, len(c)) for label, c in classes.items()})

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(hidden.last_hidden_state[:, 0])
        return {label: head(cls) for label, head in self.heads.items()}


def targets(rows: list[dict[str, str]]) -> dict[str, torch.Tensor]:
    """Class index per text and head; IGNORE where the label does not apply."""
    return {
        label: torch.tensor(
            [
                bert.CLASSES[label].index(row[label]) if bert.applies(label, row) else IGNORE
                for row in rows
            ]
        )
        for label in bert.LABELS
    }


def class_weights(rows: list[dict[str, str]], device) -> dict[str, torch.Tensor]:
    """Inverse square root of class frequency in training, so rare classes count more
    without dominating; classes absent from training get the largest weight."""
    weights = {}
    for label, classes in bert.CLASSES.items():
        counts = [
            sum(bert.applies(label, row) and row[label] == value for row in rows)
            for value in classes
        ]
        top = max(counts) or 1
        weights[label] = torch.tensor(
            [math.sqrt(top / max(count, 1)) for count in counts], device=device
        )
    return weights


def batches(rows, tokenizer, max_length, size, shuffle, seed=0):
    order = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(order)
    else:  # similar lengths together: less padding when predicting
        order.sort(key=lambda i: len(rows[i]["text"]))
    for start in range(0, len(order), size):
        chunk = [rows[i] for i in order[start : start + size]]
        encoded = tokenizer(
            [row["text"] for row in chunk],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        yield [order[i] for i in range(start, min(start + size, len(order)))], chunk, encoded


def predict_heads(model, rows, tokenizer, max_length, size, device):
    """Predicted class of every head for every row (before the rules)."""
    model.eval()
    predictions: list[dict[str, str] | None] = [None] * len(rows)
    with torch.no_grad(), autocast(device):
        for indices, _, encoded in batches(rows, tokenizer, max_length, size, shuffle=False):
            logits = model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
            best = {label: values.argmax(-1).tolist() for label, values in logits.items()}
            for n, index in enumerate(indices):
                predictions[index] = {
                    label: bert.CLASSES[label][best[label][n]] for label in bert.LABELS
                }
    return predictions


def autocast(device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def evaluate(rows, predictions) -> dict[str, dict[str, float]]:
    """Per head on the texts where the gold label applies (head quality), and after the
    rules on every work text (what the corpus labels will look like)."""
    result = {}
    for label in bert.LABELS:
        applied = [
            (row, p) for row, p in zip(rows, predictions, strict=True) if bert.applies(label, row)
        ]
        gold = [row[label] for row, _ in applied]
        head = [p[label] for _, p in applied]
        gated = [bert.gate(row["ai"], p)[label] for row, p in zip(rows, predictions, strict=True)]
        everything = [row[label] for row in rows]
        result[label] = {
            "texts": len(applied),
            "macro_f1": round(bert.macro_f1(gold, head, bert.CLASSES[label]), 3),
            "accuracy": round(
                sum(g == h for g, h in zip(gold, head, strict=True)) / max(len(gold), 1), 3
            ),
            "gated_accuracy": round(
                sum(g == h for g, h in zip(everything, gated, strict=True)) / max(len(rows), 1), 3
            ),
        }
    return result


def held_out() -> set[tuple[str, str]]:
    keys = set()
    for path in DATA.glob("wykop_sample_2_*.xlsx"):
        sheet = pd.read_excel(path, sheet_name="sample", dtype=str).dropna(subset=["entry_id"])
        keys |= set(zip(sheet["type"], sheet["entry_id"], strict=True))
    return keys


def train(options, device) -> int:
    rows = list(csv.DictReader(open(options.gold, encoding="utf-8-sig", newline="")))
    parts = bert.split(rows, held_out(), seed=options.seed)
    print({name: len(part) for name, part in parts.items()}, "work texts")
    torch.manual_seed(options.seed)
    tokenizer = AutoTokenizer.from_pretrained(options.base)
    model = Classifier(options.base, bert.CLASSES).to(device)
    weights = class_weights(parts["train"], device)
    losses = {
        label: nn.CrossEntropyLoss(weight=weights[label], ignore_index=IGNORE)
        for label in bert.LABELS
    }
    labels = targets(parts["train"])
    steps = math.ceil(len(parts["train"]) / options.batch_size) * options.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.lr, weight_decay=0.01)
    schedule = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    best, history = -1.0, []
    options.out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, options.epochs + 1):
        model.train()
        started, total = time.time(), 0.0
        for indices, _, encoded in batches(
            parts["train"],
            tokenizer,
            options.max_length,
            options.batch_size,
            True,
            options.seed + epoch,
        ):
            with autocast(device):
                logits = model(
                    encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
                )
            loss = 0.0
            for label in bert.LABELS:
                target = labels[label][indices].to(device)
                if (target != IGNORE).any():
                    loss = loss + losses[label](logits[label].float(), target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            total += loss.item()
        scores = evaluate(
            parts["validation"],
            predict_heads(
                model,
                parts["validation"],
                tokenizer,
                options.max_length,
                options.batch_size * 2,
                device,
            ),
        )
        mean = sum(s["macro_f1"] for s in scores.values()) / len(scores)
        history.append({"epoch": epoch, "loss": round(total, 2), "validation_macro_f1": mean})
        print(
            f"epoch {epoch}: loss {total:.1f}, validation mean macro-F1 {mean:.3f}, "
            f"{time.time() - started:.0f} s"
        )
        if mean > best:
            best = mean
            torch.save(model.state_dict(), options.out / "model.pt")
    model.load_state_dict(torch.load(options.out / "model.pt", map_location=device))
    test = evaluate(
        parts["test"],
        predict_heads(
            model, parts["test"], tokenizer, options.max_length, options.batch_size * 2, device
        ),
    )
    config = {
        "base": options.base,
        "max_length": options.max_length,
        "classes": bert.CLASSES,
        "seed": options.seed,
        "epochs": options.epochs,
        "lr": options.lr,
        "batch_size": options.batch_size,
        "sizes": {name: len(part) for name, part in parts.items()},
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (options.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (options.out / "metrics.json").write_text(
        json.dumps({"history": history, "test": test}, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(test).T.to_string())
    print(f"Saved the model and metrics to {options.out}")
    return 0


def predict(options, device) -> int:
    config = json.loads((options.out / "config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(config["base"])
    model = Classifier(config["base"], {k: tuple(v) for k, v in config["classes"].items()})
    model.load_state_dict(torch.load(options.out / "model.pt", map_location=device))
    model.to(device)
    stage1 = {
        (row["type"], row["entry_id"]): row
        for row in read_rows(options.labels, STAGE1_COLUMNS)
        if row["model"] == options.stage1_model and row["prompt_hash"] == ai_work.PROMPT_HASH
    }
    rows = []
    for row in read_rows(options.entries, records.COLUMNS):
        label = stage1.get((row["type"], row["entry_id"]))
        text = model_text(row["text"])
        if label and label["work"] == "yes" and label["text_hash"] == text_hash(text):
            rows.append(
                {"type": row["type"], "entry_id": row["entry_id"], "text": text, "ai": label["ai"]}
            )
    print(f"{len(rows)} work texts to label")
    predictions = predict_heads(
        model, rows, tokenizer, config["max_length"], options.batch_size, device
    )
    now = datetime.now(timezone.utc).isoformat()
    with options.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        for row, predicted in zip(rows, predictions, strict=True):
            writer.writerow(
                {
                    "type": row["type"],
                    "entry_id": row["entry_id"],
                    "text_hash": text_hash(row["text"]),
                    **bert.gate(row["ai"], predicted),
                    "model": f"herbert-stage2:{config['trained_at'][:10]}",
                    "classified_at": now,
                }
            )
    print(f"Saved {len(rows)} predictions to {options.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("train", "predict"))
    parser.add_argument("--gold", type=Path, default=DATA / "agents/gold.csv")
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384, help="tokens per text")
    parser.add_argument("--seed", type=int, default=bert.SEED)
    parser.add_argument("--out", type=Path, default=OUT, help="model folder")
    parser.add_argument("--entries", type=Path, default=DATA / "wykop_entries_clean.csv")
    parser.add_argument("--labels", type=Path, default=DATA / "wykop_stage1_azure.csv")
    parser.add_argument("--stage1-model", default="gpt-4.1-mini")
    parser.add_argument("--output", type=Path, default=DATA / "wykop_stage2_bert.csv")
    options = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU (slow)"
    print(f"Device: {name}")
    return train(options, device) if options.command == "train" else predict(options, device)


if __name__ == "__main__":
    raise SystemExit(main())
