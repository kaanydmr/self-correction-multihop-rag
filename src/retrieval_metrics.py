"""Ranking quality of the two retrieval stages (dense embedder, then listwise reranker) on the
evaluated questions, with paragraph-level gold labels.

Ranking metrics only apply to these two stages: the self-correcting systems output a filtered
set rather than a ranking, which is measured by context precision/recall in analyze.py.

    python src/retrieval_metrics.py   # -> results/retrieval_metrics.csv
"""
import json

import numpy as np
import pandas as pd

from config import DATASETS, EMBEDDER, RERANKER, RESULTS_DIR, data_dir
from contexts import load_gold_pids

N_EVAL = 500  # questions per dataset in the main runs


def metrics(ranking, gold):
    hits = [p in gold for p in ranking]
    first = next((i for i, h in enumerate(hits) if h), None)
    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits[:10]))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gold), 10)))
    return {
        "recall@5": sum(hits[:5]) / len(gold),
        "recall@20": sum(hits[:20]) / len(gold),
        "all_gold@5": float(sum(hits[:5]) == len(gold)),  # complete evidence for the whole chain
        "mrr@20": 0.0 if first is None else 1 / (first + 1),
        "ndcg@10": dcg / idcg,
    }


def main():
    gold = load_gold_pids()
    rows = []
    for ds in DATASETS:
        questions = json.loads((data_dir(ds) / "questions.json").read_text(encoding="utf-8"))[:N_EVAL]
        cache = json.loads((data_dir(ds) / f"rerank_{EMBEDDER}_{RERANKER}.json").read_text(encoding="utf-8"))
        for q in questions:
            if not q["answerable"]:
                continue
            c = cache[q["id"]]
            for stage, ranking in [("dense", c["first_stage_pids"]), ("dense+rerank", c["pids"])]:
                rows.append({"dataset": ds, "stage": stage, **metrics(ranking, gold[ds][q["id"]])})
    out = pd.DataFrame(rows).groupby(["dataset", "stage"]).mean().reset_index()
    out.to_csv(RESULTS_DIR / "retrieval_metrics.csv", index=False, float_format="%.4f")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
