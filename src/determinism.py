"""Run-to-run stability: re-run one system on the first N questions of every dataset with the loaded
generator and compare the answers with those stored from the main run (temperature 0, parallel slots).

    python src/determinism.py --model qwen38-27b --n 100   # -> results/determinism_naive_qwen38-27b.csv
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from clients import Usage, model_name
from config import DATASETS, MODELS, RESULTS_DIR, data_dir
from judge import is_abstain
from retriever import Retriever
from systems import SYSTEMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--system", default="naive", choices=SYSTEMS)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    if MODELS[args.model] not in model_name():
        raise SystemExit(f"Load {MODELS[args.model]} on the LLM server first.")
    rows = []
    for ds in DATASETS:
        path = RESULTS_DIR / ds / args.model / f"{args.system}.jsonl"
        stored = {r["id"]: r["answer"] for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
                  if r["error"] is None}
        questions = json.loads((data_dir(ds) / "questions.json").read_text(encoding="utf-8"))[:args.n]
        retriever = Retriever(ds)
        with ThreadPoolExecutor(4) as pool:
            answers = list(pool.map(lambda q: SYSTEMS[args.system](q["question"], retriever, Usage())["answer"],
                                    questions))
        for q, a in zip(questions, answers):
            old = stored[q["id"]]
            rows.append({"dataset": ds, "id": q["id"], "stored": old, "rerun": a, "same": old == a,
                         "same_abstention": is_abstain(old) == is_abstain(a)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"determinism_{args.system}_{args.model}.csv", index=False)
    print(df.groupby("dataset")[["same", "same_abstention"]].mean().round(3).to_string())
    print(f"overall: identical answers {df.same.mean():.3f}, same abstention decision {df.same_abstention.mean():.3f}")


if __name__ == "__main__":
    main()
