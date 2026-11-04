"""Stage 2 on Wykop with two coding agents: sample, check, compare, gold.

    sample   stratified sample of 5000 texts in batches of 40 (data/wykop/agents/)
    check    are the answers of one agent complete and valid?
    compare  agreement of the agents and the sheet of disagreements to adjudicate
    gold     agreed plus adjudicated labels, the training data for BERT
    model    answer the batches with a model on Azure OpenAI as a further annotator
             (answers/<deployment>/), e.g. model --deployment gpt-6.1-sol --last 65

See amc/wykop/agent_annotation.py for the procedure.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from amc.wykop import agent_annotation as agents  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/wykop"


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("sample", "check", "compare", "gold", "model"))
    parser.add_argument("--folder", type=Path, default=DATA / "agents")
    parser.add_argument("--entries", type=Path, default=DATA / "wykop_entries_clean.csv")
    parser.add_argument("--labels", type=Path, default=DATA / "wykop_stage1_azure.csv")
    parser.add_argument("--model", default=os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-4.1-mini")
    parser.add_argument("--agent", help="for check: the answers folder (default: both --agents)")
    parser.add_argument(
        "--agents",
        nargs=2,
        default=["claude", "gpt-6.1-sol"],
        help="for check, compare and gold: the two annotators (answers folders)",
    )
    parser.add_argument("--deployment", default="gpt-6.1-sol", help="for model")
    parser.add_argument("--last", type=int, help="for model: only batches up to this number")
    parser.add_argument("--workers", type=int, default=16, help="for model")
    parser.add_argument("--reasoning-effort", default="", help="for model, e.g. low")
    parser.add_argument("--seed", type=int, default=agents.SEED)
    options = parser.parse_args(argv)
    folder = options.folder
    disputes = folder / "disputes.xlsx"

    if options.command == "model":
        from amc.classification.azure import AzureClient, AzureError

        try:
            client = AzureClient(
                os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                os.getenv("AZURE_OPENAI_API_KEY", ""),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", ""),
                reasoning_effort=options.reasoning_effort,
            )
            agents.annotate(
                folder,
                options.deployment,
                agents.AzureAnnotator(client, options.deployment),
                workers=options.workers,
                last=options.last,
            )
        except AzureError as error:
            print(str(error), file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("Interrupted. Finished batches are kept; rerun to resume.")
            return 130

    elif options.command == "sample":
        sample = agents.draw(
            agents.screened(options.entries, options.labels, options.model), seed=options.seed
        )
        agents.write_sample(sample, folder)
        counts = pd.DataFrame(sample).groupby(["period", "stratum"]).size()
        print(counts.to_string())
        batches = -(-len(sample) // agents.BATCH_SIZE)
        print(f"Saved {len(sample)} texts in {batches} batches to {folder}")

    elif options.command == "check":
        for agent in [options.agent] if options.agent else options.agents:
            answers, problems = agents.check(folder, agent)
            print(f"{agent}: {len(answers)} valid answers, {len(problems)} problems")
            for problem in problems[:30]:
                print("  " + problem)

    elif options.command == "compare":
        summary, rows = agents.compare(folder, tuple(options.agents))
        # One text's disputes together, in the order of the labels.
        rows.sort(key=lambda row: (row["sample_id"], agents.LABELS.index(row["label"])))
        print(pd.DataFrame(summary).to_string(index=False))
        if disputes.exists():
            print(f"{disputes} exists and may hold adjudications; it was not replaced.")
            return 1
        pd.DataFrame(
            rows, columns=["sample_id", "label", *options.agents, "final", "note", "text"]
        ).to_excel(disputes, index=False)
        agents.write_csv(
            folder / "agreement.csv", ("label", "texts", "agreement", "kappa"), summary
        )
        texts = len({row["sample_id"] for row in rows})
        print(f"{len(rows)} disagreements in {texts} texts -> {disputes}")

    else:
        adjudicated = (
            pd.read_excel(disputes, dtype=str).fillna("").to_dict("records")
            if disputes.exists()
            else []
        )
        rows = agents.gold(folder, adjudicated, tuple(options.agents))
        columns = (*agents.KEY_COLUMNS, *agents.LABELS, "text")
        agents.write_csv(folder / "gold.csv", columns, rows)
        print(f"Saved {len(rows)} texts with final labels to {folder / 'gold.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
