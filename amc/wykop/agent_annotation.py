"""Stage 2 on Wykop with two coding agents (Claude and Codex), as for YouTube.

1. `sample`: a stratified sample of the cleaned collection (period x screening labels from
   stage 1), with the size and weight of each stratum, split into batches of 40 texts.
   The batches contain only `sample_id` and the text: no screening labels, dates or period.
2. Each agent answers every batch on its own, following `INSTRUCTIONS` (the stage 1 and
   stage 2 prompts), into `answers/<agent>/batch_NNN.csv`.
3. `check`: every answer file has the batch's ids in order and valid, consistent labels.
4. `compare`: agreement between the agents per label and a sheet of disagreements for
   human adjudication.
5. `gold`: agreed labels plus adjudicated ones, the training data for the BERT classifier.
"""

import csv
import json
import random
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from amc.classification import ai_work, job_concern_details
from amc.csv_files import read_rows
from amc.wykop import records
from amc.wykop.classification import COLUMNS as STAGE1_COLUMNS
from amc.wykop.classification import model_text, text_hash

SEED = 20261104
CHATGPT = "2022-11-30"
BATCH_SIZE = 40
NONE = job_concern_details.NONE
DETAILS = job_concern_details.LABELS
LABELS = ("ai", "work", *DETAILS)
# Detailed labels that apply only to texts about AI (none by rule otherwise).
AI_DETAILS = ("stance", "attitude", "threat_type", "competence_claim")
ANSWER_COLUMNS = ("sample_id", *LABELS)
KEY_COLUMNS = (
    "sample_id",
    "batch",
    "type",
    "entry_id",
    "text_hash",
    "period",
    "stratum",
    "stratum_size",
    "stratum_sample",
    "weight",
)
# Texts per period and screening stratum; a stratum with fewer texts gives the rest to `work`.
ALLOCATION = {"ai_work": 1000, "work": 1000, "other": 500}
AGENTS = ("claude", "codex")

RULES = f"""## Step 1: ai and work

{ai_work.INSTRUCTION}

## Step 2: detailed labels

If work is "no", write "none" in all seven detailed columns. If work is "yes":

{job_concern_details.INSTRUCTION}

## Consistency

- insecurity_cause is "none" exactly when job_insecurity is "no".
- threat_type is "none" unless stance is "threat" or "mixed".
- attitude is "none" exactly when ai is "no"; then stance and competence_claim are "none".
"""
INSTRUCTIONS = f"""# Wykop stage 2 annotation (agent instructions)

You annotate posts and comments from the Polish forum Wykop. Each batch file
`batches/batch_NNN.csv` has the columns `sample_id` and `text`. Write your answers to
`answers/<agent>/batch_NNN.csv` with exactly these columns, in this order:

    {",".join(ANSWER_COLUMNS)}

one row per text, in the same order as the batch, using only the values listed below.
Judge every text on its own. Do not look at other agents' answers, at the screening labels or
at any other file of the project. Do not write code or call a model to label the texts: read
each text and decide yourself.

{RULES}"""
# The same rules for a model called through an API, one text per request.
MODEL_INSTRUCTION = f"""You annotate one post or comment from the Polish forum Wykop. Return
only JSON with {", ".join(LABELS)}, using only the values listed below.

{RULES}"""
SCHEMA = {
    "type": "object",
    "properties": {
        "ai": {"type": "string", "enum": list(ai_work.YES_NO)},
        "work": {"type": "string", "enum": list(ai_work.YES_NO)},
        **{
            label: {
                "type": "string",
                "enum": sorted({*job_concern_details.SCHEMA["properties"][label]["enum"], NONE}),
            }
            for label in DETAILS
        },
    },
    "required": list(LABELS),
    "additionalProperties": False,
}


def period(created_at: str) -> str:
    return "after" if created_at[:10] >= CHATGPT else "before"


def stratum(labels: dict[str, str]) -> str:
    if labels["work"] != "yes":
        return "other"
    return "ai_work" if labels["ai"] == "yes" else "work"


def screened(entries: Path, labels: Path, model: str) -> list[dict[str, str]]:
    """Cleaned texts with their current stage 1 labels (texts changed since are left out)."""
    current = {
        (row["type"], row["entry_id"]): row
        for row in read_rows(labels, STAGE1_COLUMNS)
        if row["model"] == model and row["prompt_hash"] == ai_work.PROMPT_HASH
    }
    rows = []
    for row in read_rows(entries, records.COLUMNS):
        label = current.get((row["type"], row["entry_id"]))
        text = model_text(row["text"])
        if label is None or not text or label["text_hash"] != text_hash(text):
            continue
        rows.append(
            {
                "type": row["type"],
                "entry_id": row["entry_id"],
                "text": text,
                "text_hash": label["text_hash"],
                "period": period(row["created_at"]),
                "stratum": stratum(label),
            }
        )
    return rows


def draw(
    rows: list[dict[str, str]], allocation: dict[str, int] = ALLOCATION, seed: int = SEED
) -> list[dict[str, str]]:
    """The stratified sample in a random order, with stratum sizes and weights."""
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda row: (row["type"], row["entry_id"])):
        groups[row["period"], row["stratum"]].append(row)
    rng = random.Random(seed)
    sample = []
    for when in ("before", "after"):
        wanted = dict(allocation)
        for name in allocation:
            short = wanted[name] - len(groups[when, name])
            if short > 0 and name != "work":
                wanted[name] -= short
                wanted["work"] += short
        for name in allocation:
            group = groups[when, name]
            chosen = rng.sample(group, min(wanted[name], len(group)))
            for row in chosen:
                sample.append(
                    {
                        **row,
                        "stratum_size": str(len(group)),
                        "stratum_sample": str(len(chosen)),
                        "weight": f"{len(group) / len(chosen):.4f}",
                    }
                )
    rng.shuffle(sample)
    for n, row in enumerate(sample):
        row["sample_id"] = f"W{n + 1:05d}"
        row["batch"] = f"{n // BATCH_SIZE + 1:03d}"
    return sample


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_sample(sample: list[dict[str, str]], folder: Path) -> None:
    """The key (kept away from the agents), the batches and the instructions."""
    if (folder / "sample_key.csv").exists():
        raise ValueError(f"{folder} already has a sample; choose another folder.")
    write_csv(folder / "sample_key.csv", KEY_COLUMNS, sample)
    batches = defaultdict(list)
    for row in sample:
        batches[row["batch"]].append(row)
    for batch, rows in batches.items():
        write_csv(folder / "batches" / f"batch_{batch}.csv", ("sample_id", "text"), rows)
    (folder / "INSTRUCTIONS.md").write_text(INSTRUCTIONS, encoding="utf-8")


def validate(row: dict[str, str]) -> dict[str, str]:
    """Labels of one answer; raises ValueError when a value or a combination is not allowed."""
    labels = {label: (row.get(label) or "").strip().lower() for label in LABELS}
    for label in ("ai", "work"):
        if labels[label] not in ai_work.YES_NO:
            raise ValueError(f"{label} must be yes or no")
    details = {label: labels[label] for label in DETAILS}
    if labels["work"] == "no":
        if any(value != NONE for value in details.values()):
            raise ValueError("work=no needs none in all detailed labels")
        return labels
    job_concern_details.validate(details)
    if (labels["ai"] == "yes") != (details["attitude"] != NONE):
        raise ValueError("attitude is none exactly when ai is no")
    return labels


def check(folder: Path, agent: str) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Valid answers of one agent by sample_id, and a list of problems."""
    answers, problems = {}, []
    for batch in sorted((folder / "batches").glob("batch_*.csv")):
        ids = [row["sample_id"] for row in read_rows(batch, ("sample_id", "text"))]
        path = folder / "answers" / agent / batch.name
        if not path.exists():
            problems.append(f"{batch.name}: no answer file")
            continue
        try:
            rows = read_rows(path, ANSWER_COLUMNS)
        except ValueError as error:
            problems.append(f"{batch.name}: {error}")
            continue
        if [row["sample_id"] for row in rows] != ids:
            problems.append(f"{batch.name}: ids differ from the batch or are out of order")
            continue
        for row in rows:
            try:
                answers[row["sample_id"]] = validate(row)
            except ValueError as error:
                problems.append(f"{batch.name} {row['sample_id']}: {error}")
    return answers, problems


def kappa(first: list[str], second: list[str]) -> float:
    n = len(first)
    if not n:
        return float("nan")
    observed = sum(a == b for a, b in zip(first, second, strict=True)) / n
    expected = sum(
        first.count(value) / n * second.count(value) / n for value in set(first) | set(second)
    )
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def compare(
    folder: Path, agents: tuple[str, str] = AGENTS
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Agreement per label (on texts both agents judge as work for the detailed labels) and
    the disagreements, one row per text and label."""
    first, second = (check(folder, agent)[0] for agent in agents)
    texts = {
        row["sample_id"]: row["text"]
        for batch in sorted((folder / "batches").glob("batch_*.csv"))
        for row in read_rows(batch, ("sample_id", "text"))
    }
    both = [sample_id for sample_id in texts if sample_id in first and sample_id in second]
    summary, disputes = [], []
    for label in LABELS:
        ids = both
        if label in DETAILS:
            ids = [i for i in both if first[i]["work"] == second[i]["work"] == "yes"]
        a = [first[i][label] for i in ids]
        b = [second[i][label] for i in ids]
        agree = sum(x == y for x, y in zip(a, b, strict=True))
        summary.append(
            {
                "label": label,
                "texts": str(len(ids)),
                "agreement": f"{agree / len(ids):.3f}" if ids else "",
                "kappa": f"{kappa(a, b):.3f}",
            }
        )
        # Where the agents disagree on work (or on ai, for the labels about AI), the agent who
        # said no wrote none by rule; these are not disputes of their own, `gold` takes the
        # labels of the agent who said yes once work (or ai) is adjudicated yes.
        if label in AI_DETAILS:
            ids = [i for i in ids if first[i]["ai"] == second[i]["ai"]]
        for i in ids:
            x, y = first[i][label], second[i][label]
            if x != y:
                disputes.append(
                    {
                        "sample_id": i,
                        "label": label,
                        agents[0]: x,
                        agents[1]: y,
                        "final": "",
                        "note": "",
                        "text": texts[i],
                    }
                )
    return summary, disputes


def gold(
    folder: Path, adjudicated: list[dict[str, str]], agents: tuple[str, str] = AGENTS
) -> list[dict[str, str]]:
    """Final labels: agreed values, and `final` from the adjudication sheet where the agents
    disagreed. When only one agent said work=yes (or ai=yes) and the adjudication agrees, the
    detailed labels (or the labels about AI) are that agent's; when it says no, they are none.
    Texts with a dispute left undecided, or with an inconsistent result, are left out."""
    first, second = (check(folder, agent)[0] for agent in agents)
    decided = defaultdict(dict)
    for row in adjudicated:
        if (row.get("final") or "").strip():
            decided[row["sample_id"]][row["label"]] = row["final"].strip().lower()
    key = {row["sample_id"]: row for row in read_rows(folder / "sample_key.csv", KEY_COLUMNS)}
    texts = {
        row["sample_id"]: row["text"]
        for batch in sorted((folder / "batches").glob("batch_*.csv"))
        for row in read_rows(batch, ("sample_id", "text"))
    }
    rows = []
    for i in key:
        if i not in first or i not in second:
            continue
        pair = (first[i], second[i])
        labels = {}
        for label in ("ai", "work"):
            if pair[0][label] == pair[1][label]:
                labels[label] = pair[0][label]
            elif label in decided[i]:
                labels[label] = decided[i][label]
        if len(labels) < 2:
            continue
        open_dispute = False
        for label in DETAILS:
            about_ai = label in AI_DETAILS
            # Agents whose answer for this label was not none by rule.
            judged = [
                answer
                for answer in pair
                if answer["work"] == "yes" and (not about_ai or answer["ai"] == "yes")
            ]
            if labels["work"] == "no" or (about_ai and labels["ai"] == "no"):
                labels[label] = NONE
            elif label in decided[i]:
                labels[label] = decided[i][label]
            elif len(judged) == 2 and judged[0][label] == judged[1][label]:
                labels[label] = judged[0][label]
            elif len(judged) == 1:
                labels[label] = judged[0][label]
            else:
                open_dispute = True
        if open_dispute:
            continue
        try:
            rows.append({**key[i], **validate(labels), "text": texts[i]})
        except ValueError:
            continue
    return rows


def annotate(
    folder: Path,
    agent: str,
    classify: Callable[[str], dict[str, str]],
    *,
    workers: int = 16,
    attempts: int = 3,
    last: int | None = None,
    log: Callable[[str], None] = print,
) -> int:
    """Answer the batches with a model, one text per request, as a further annotator.

    `classify` returns the labels of one text. A batch file is written only when all of its
    texts have valid answers, so a rerun repeats just the unfinished batches. With `last`, only
    batches up to that number. Returns the number of batches written.
    """
    out = folder / "answers" / agent
    batches = {
        path.name: read_rows(path, ("sample_id", "text"))
        for path in sorted((folder / "batches").glob("batch_*.csv"))
        if not (out / path.name).exists() and (last is None or int(path.stem[6:]) <= last)
    }
    todo = [(name, row) for name, rows in batches.items() for row in rows]
    log(f"{agent}: {len(batches)} batches ({len(todo)} texts) to annotate.")
    answers: dict[str, dict[str, str]] = {}
    left = {name: len(rows) for name, rows in batches.items()}
    written = failed = 0

    def task(row: dict[str, str]) -> dict[str, str] | None:
        for _ in range(attempts):
            try:
                return validate(classify(row["text"]))
            except ValueError:
                continue
        return None

    pool = ThreadPoolExecutor(workers)
    try:
        jobs = {pool.submit(task, row): (name, row) for name, row in todo}
        for n, job in enumerate(as_completed(jobs), 1):
            name, row = jobs[job]
            labels = job.result()
            if labels is None:
                failed += 1
            else:
                answers[row["sample_id"]] = labels
            left[name] -= 1
            if left[name] == 0 and all(r["sample_id"] in answers for r in batches[name]):
                rows = [
                    {"sample_id": r["sample_id"], **answers[r["sample_id"]]} for r in batches[name]
                ]
                write_csv(out / name, ANSWER_COLUMNS, rows)
                written += 1
            if n % 200 == 0:
                log(f"{n}/{len(todo)} texts, {written} batches written, {failed} failed")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    log(f"{agent}: wrote {written} batches; {failed} texts failed (their batches will be redone).")
    return written


class AzureAnnotator:
    """Labels of one text from a model on Azure OpenAI, with `MODEL_INSTRUCTION`."""

    def __init__(self, client, model: str) -> None:
        self._client = client
        self._model = model

    def __call__(self, text: str) -> dict[str, str]:
        answer = json.loads(self._client.complete(self._model, MODEL_INSTRUCTION, text, SCHEMA))
        if not isinstance(answer, dict):
            raise ValueError("The model did not return a JSON object.")
        return answer
