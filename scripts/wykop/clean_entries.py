"""Remove empty, hashtag-only and duplicate texts from a Wykop CSV export."""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from amc.wykop.cleaning import clean_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/wykop/wykop_entries.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/wykop/wykop_entries_clean.csv")
    options = parser.parse_args(argv)

    try:
        stats = clean_csv(options.input, options.output)
    except (OSError, ValueError, csv.Error) as error:
        print(f"Could not clean CSV: {error}", file=sys.stderr)
        return 1

    print(
        f"Wrote {stats.written} of {stats.read} rows to {options.output}; "
        f"removed {stats.empty} empty, {stats.hashtags_only} hashtag-only and "
        f"{stats.duplicates} duplicate rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
