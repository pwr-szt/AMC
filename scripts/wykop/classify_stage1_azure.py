"""Stage 1 labels (ai, work) again, on Azure OpenAI, for the cleaned collection.

Input: wykop_entries_clean.csv from clean_entries.py. Output: wykop_stage1_azure.csv,
resumable like the other stages (same key, text hash, model and prompt hash are skipped).
Try a small sample first with --limit 50.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dotenv import load_dotenv  # noqa: E402

from amc.classification import ai_work  # noqa: E402
from amc.classification.azure import AzureClient, AzureError  # noqa: E402
from amc.wykop.classification import run  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "gpt-4o"


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/wykop/wykop_entries_clean.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/wykop/wykop_stage1_azure.csv")
    parser.add_argument(
        "--model",
        help=f"Azure deployment name (default: AZURE_OPENAI_DEPLOYMENT or {DEFAULT_MODEL})",
    )
    parser.add_argument("--workers", type=int, default=16, help="parallel requests")
    parser.add_argument("--limit", type=int, help="classify at most this many texts")
    parser.add_argument(
        "--reasoning-effort",
        default=os.getenv("AZURE_OPENAI_REASONING_EFFORT", ""),
        help="for example low; empty sends none (the deployment default)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ["AZURE_OPENAI_TEMPERATURE"])
        if os.getenv("AZURE_OPENAI_TEMPERATURE")
        else None,
        help="for example 0 for gpt-4o; reasoning models accept none (the default)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("AZURE_OPENAI_MAX_TOKENS") or 4000),
        help="answer limit; Azure counts it towards the tokens-per-minute quota, so use about "
        "50 for models without reasoning (the answer has about 10 tokens)",
    )
    options = parser.parse_args(argv)
    try:
        client = AzureClient(
            os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            os.getenv("AZURE_OPENAI_API_KEY", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", ""),
            max_tokens=options.max_tokens,
            reasoning_effort=options.reasoning_effort,
            temperature=options.temperature,
        )
        model = options.model or os.getenv("AZURE_OPENAI_DEPLOYMENT") or DEFAULT_MODEL
        run(
            ai_work.AzureClassifier(client),
            options.input,
            options.output,
            model,
            prompt_hash=ai_work.PROMPT_HASH,
            workers=options.workers,
            limit=options.limit,
        )
    except (AzureError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Saved predictions are retained; rerun to resume.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
