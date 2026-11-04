"""Texts of the stage 1 reference sample on which the two annotators disagree.

Writes a sheet for a joint adjudication: both labels side by side and an empty `final`
column per label. The original annotation files are not changed; report the kappa of the
independent annotation, and the adjudicated labels as the gold standard.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/wykop"
KEY = ["type", "entry_id"]
LABELS = ["ai", "work"]


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="sample", dtype=str).dropna(subset=["entry_id"])
    for label in LABELS:
        frame[label] = frame[label].fillna("").str.strip().str.lower()
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, default=DATA / "wykop_sample_2_Zuza.xlsx")
    parser.add_argument("--second", type=Path, default=DATA / "wykop_sample_2_mikolaj.xlsx")
    parser.add_argument("--names", nargs=2, default=["zuza", "mikolaj"])
    parser.add_argument(
        "--output", type=Path, default=DATA / "agreement/wykop_sample_2_disputes.xlsx"
    )
    options = parser.parse_args(argv)
    if options.output.exists():
        print(f"{options.output} exists and may hold decisions; it was not replaced.")
        return 1
    a, b = (load(path) for path in (options.first, options.second))
    joined = a[[*KEY, "url", "text", *LABELS]].merge(
        b[[*KEY, *LABELS]], on=KEY, suffixes=[f"_{n}" for n in options.names], validate="1:1"
    )
    first, second = options.names
    disputed = joined[
        (joined[f"ai_{first}"] != joined[f"ai_{second}"])
        | (joined[f"work_{first}"] != joined[f"work_{second}"])
    ].copy()
    columns = [*KEY, "url"]
    for label in LABELS:
        columns += [f"{label}_{first}", f"{label}_{second}", f"{label}_final"]
        disputed[f"{label}_final"] = ""
    disputed["note"] = ""
    options.output.parent.mkdir(parents=True, exist_ok=True)
    disputed[[*columns, "note", "text"]].to_excel(options.output, index=False)
    for label in LABELS:
        n = (disputed[f"{label}_{first}"] != disputed[f"{label}_{second}"]).sum()
        print(f"{label}: {n} disagreements")
    print(f"Saved {len(disputed)} texts to {options.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
