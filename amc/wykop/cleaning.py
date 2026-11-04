"""Filters for cleaning exported Wykop entries."""

import csv
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

HASHTAG = re.compile(r"#[\w-]+$", re.UNICODE)


@dataclass(frozen=True)
class CleaningStats:
    """Counts of rows read, written and removed by the cleaner."""

    read: int
    written: int
    empty: int
    hashtags_only: int
    duplicates: int


def _normalized_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def _contains_only_hashtags(text: str) -> bool:
    tokens = text.split()
    return bool(tokens) and all(HASHTAG.fullmatch(token) for token in tokens)


def clean_csv(input_path: Path, output_path: Path) -> CleaningStats:
    """Write a cleaned copy, keeping the first occurrence of each text."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output must be separate from the input file.")

    csv.field_size_limit(2**31 - 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts = Counter(read=0, written=0, empty=0, hashtags_only=0, duplicates=0)
    temporary_path: Path | None = None

    try:
        with input_path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            fieldnames = reader.fieldnames
            if not fieldnames or "text" not in fieldnames:
                raise ValueError("Input CSV must contain a 'text' column.")

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8-sig",
                newline="",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                writer = csv.DictWriter(temporary, fieldnames=fieldnames)
                writer.writeheader()
                for row in reader:
                    counts["read"] += 1
                    text = row["text"] or ""
                    if not text.strip():
                        counts["empty"] += 1
                        continue
                    if _contains_only_hashtags(text):
                        counts["hashtags_only"] += 1
                        continue
                    key = _normalized_text(text)
                    if key in seen:
                        counts["duplicates"] += 1
                        continue
                    seen.add(key)
                    writer.writerow(row)
                    counts["written"] += 1
        temporary_path.replace(output_path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()

    return CleaningStats(**counts)
