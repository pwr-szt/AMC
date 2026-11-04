"""Data for the stage 2 BERT classifier of Wykop texts, trained on the adjudicated labels.

Stage 1 (ai, work) is already labelled for the whole collection, so the classifier only
assigns the seven stage 2 labels, and only to texts with work = yes. One encoder (HerBERT)
with a classification head per label; a head is trained only on the texts where its label
applies, so that the "none by rule" values do not swamp it:

- `job_insecurity`, `sector`: every work text;
- `insecurity_cause`: job_insecurity = yes;
- `stance`, `attitude`, `competence_claim`: texts with ai = yes;
- `threat_type`: stance is threat or mixed.

`ai` comes from stage 1 (gold labels for training, the stage 1 run for prediction). `gate`
applies the same rules at prediction time. This module does not need torch; the training
itself is in scripts/wykop/train_bert.py.
"""

import random
from collections import defaultdict

from amc.classification import job_concern_details
from amc.wykop import agent_annotation as agents

NONE = agents.NONE
LABELS = agents.DETAILS
SEED = 20261105
# Classes of each head; `none` only where it is a real answer (no view, no claim).
CLASSES = {
    "job_insecurity": ("yes", "no"),
    "insecurity_cause": job_concern_details.CAUSE,
    "sector": job_concern_details.SECTOR,
    "stance": job_concern_details.STANCE,
    "attitude": job_concern_details.ATTITUDE,
    "threat_type": job_concern_details.THREAT_TYPE,
    "competence_claim": job_concern_details.COMPETENCE_CLAIM,
}


def work_texts(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """The texts stage 2 applies to: work = yes."""
    return [row for row in rows if row["work"] == "yes"]


def applies(label: str, row: dict[str, str]) -> bool:
    """Whether `label` is a real decision for a work text, given its ai label and the
    labels decided before it."""
    if label in ("job_insecurity", "sector"):
        return True
    if label == "insecurity_cause":
        return row["job_insecurity"] == "yes"
    if row["ai"] != "yes":
        return False
    if label == "threat_type":
        return row["stance"] in job_concern_details.WITH_THREAT
    return True


def gate(ai: str, predicted: dict[str, str]) -> dict[str, str]:
    """Final stage 2 labels of a work text from the heads' predictions and its stage 1 ai
    label, in the order of the rules: a label that does not apply becomes none."""
    labels: dict[str, str] = {}
    for label in LABELS:
        labels[label] = predicted[label]
        if not applies(label, {"ai": ai, **predicted, **labels}):
            labels[label] = NONE
    return labels


def split(
    rows: list[dict[str, str]],
    held_out: set[tuple[str, str]] = frozenset(),
    shares: tuple[float, float] = (0.7, 0.15),
    seed: int = SEED,
) -> dict[str, list[dict[str, str]]]:
    """Train, validation and test sets of the work texts, stratified by period and
    screening stratum. Texts in `held_out` (type, entry_id), such as the human reference
    sample, are left out of all three."""
    groups = defaultdict(list)
    for row in sorted(work_texts(rows), key=lambda row: row["sample_id"]):
        if (row["type"], row["entry_id"]) not in held_out:
            groups[row["period"], row["stratum"]].append(row)
    rng = random.Random(seed)
    parts: dict[str, list[dict[str, str]]] = {"train": [], "validation": [], "test": []}
    for key in sorted(groups):
        group = groups[key]
        rng.shuffle(group)
        train = round(len(group) * shares[0])
        validation = round(len(group) * shares[1])
        parts["train"] += group[:train]
        parts["validation"] += group[train : train + validation]
        parts["test"] += group[train + validation :]
    return parts


def macro_f1(gold: list[str], predicted: list[str], classes: tuple[str, ...]) -> float:
    """Mean F1 over the classes present in gold or predictions."""
    scores = []
    for value in classes:
        tp = sum(g == p == value for g, p in zip(gold, predicted, strict=True))
        fp = sum(p == value != g for g, p in zip(gold, predicted, strict=True))
        fn = sum(g == value != p for g, p in zip(gold, predicted, strict=True))
        if tp + fp + fn:
            scores.append(2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores) if scores else float("nan")
