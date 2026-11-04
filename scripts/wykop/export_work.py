"""Export Wykop entries and comments that stage 1 labelled work=yes, the input of stage 2.

Texts with and without AI are both kept, so that stage 2 can compare worry about work
before and after ChatGPT; `--ai-only` keeps only ai=yes as before. By default the labels
are those of the Azure run (`classify_stage1_azure.py`); `--stage1 ollama` reads the older
local run (`classify_entries.py`).
"""

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dotenv import load_dotenv  # noqa: E402

from amc.classification import ai_work, job_concern  # noqa: E402
from amc.csv_files import read_rows  # noqa: E402
from amc.wykop import classification, records  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/wykop"
LABELS = ("ai", "work", "model", "prompt_hash")
STAGE1 = {
    "azure": {
        "entries": DATA / "wykop_entries_clean.csv",
        "labels": DATA / "wykop_stage1_azure.csv",
        "model": lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-4o",
        "prompt_hash": ai_work.PROMPT_HASH,
    },
    "ollama": {
        "entries": DATA / "wykop_entries.csv",
        "labels": DATA / "wykop_stage1.csv",
        "model": lambda: "qwen3:8b",
        "prompt_hash": job_concern.PROMPT_HASH,
    },
}


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", choices=sorted(STAGE1), default="azure")
    parser.add_argument("--entries", type=Path, help="texts (default depends on --stage1)")
    parser.add_argument("--labels", type=Path, help="stage 1 labels (default by --stage1)")
    parser.add_argument("--model", help="stage 1 model (default depends on --stage1)")
    parser.add_argument("--output", type=Path, default=DATA / "wykop_work.csv")
    parser.add_argument("--ai-only", action="store_true", help="keep only ai=yes as well")
    options = parser.parse_args(argv)
    stage1 = STAGE1[options.stage1]
    model = options.model or stage1["model"]()
    # Only labels from the current instruction and model; older rows stay in the file.
    labels = {
        (row["type"], row["entry_id"]): row
        for row in read_rows(options.labels or stage1["labels"], classification.COLUMNS)
        if row["prompt_hash"] == stage1["prompt_hash"]
        and row["model"] == model
        and row["work"] == "yes"
        and (row["ai"] == "yes" or not options.ai_only)
    }
    rows = [
        {**row, **{name: label[name] for name in LABELS}}
        for row in read_rows(options.entries or stage1["entries"], records.COLUMNS)
        if (label := labels.get((row["type"], row["entry_id"])))
    ]
    rows.sort(key=lambda row: row["created_at"])
    with options.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(*records.COLUMNS, *LABELS))
        writer.writeheader()
        writer.writerows(rows)
    with_ai = sum(row["ai"] == "yes" for row in rows)
    print(f"Saved {len(rows)} texts with work=yes ({with_ai} with ai=yes) to {options.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
