"""Compare embedders and rerankers on gold-passage recall (answerable questions), to pick the
retrieval components before the main runs. Each component is served on its own port.

    python src/compare_retrieval.py --embedders bge-base=8081 qwen3-emb-0.6b=8083 \
        --rerankers bge-reranker-v2-m3=8082 qwen3-reranker-0.6b=8084 --rerank-base qwen3-emb-0.6b
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from clients import rerank
from config import DATASETS, RERANK_CANDIDATES, RESULTS_DIR, TOP_K, data_dir
from retriever import Retriever


def recall(gold, titles):
    return len(set(gold) & set(titles)) / len(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedders", nargs="+", required=True, help="name=port")
    ap.add_argument("--rerankers", nargs="*", default=[], help="name=port")
    ap.add_argument("--rerank-base", help="embedder whose top-20 the rerankers reorder")
    args = ap.parse_args()
    embedders = dict(e.split("=") for e in args.embedders)
    rerankers = dict(r.split("=") for r in args.rerankers)

    rows = []
    for ds in DATASETS:
        qs = [q for q in json.loads((data_dir(ds) / "questions.json").read_text(encoding="utf-8")) if q["answerable"]]
        for name, port in embedders.items():
            ret = Retriever(ds, embedder=name, url=f"http://127.0.0.1:{port}")
            with ThreadPoolExecutor(8) as pool:
                hits = list(pool.map(lambda q: ret.search(q["question"], RERANK_CANDIDATES), qs))
            for q, h in zip(qs, hits):
                titles = [p["title"] for p in h]
                rows.append({"dataset": ds, "type": q["type"], "component": f"emb:{name}",
                             "r@5": recall(q["gold_titles"], titles[:TOP_K]),
                             "r@20": recall(q["gold_titles"], titles)})
                if name == args.rerank_base:
                    for rname, rport in rerankers.items():
                        scores = rerank(q["question"], [f"{p['title']}: {p['text']}" for p in h],
                                        url=f"http://127.0.0.1:{rport}")
                        order = [t for _, t in sorted(zip(scores, titles), key=lambda x: -x[0])]
                        rows.append({"dataset": ds, "type": q["type"], "component": f"{name}+{rname}",
                                     "r@5": recall(q["gold_titles"], order[:TOP_K]), "r@20": np.nan})
            print(f"done {ds} {name}")

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "retrieval_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table = df.groupby(["dataset", "component"])[["r@5", "r@20"]].mean().round(3)
    by_type = df.pivot_table(index=["dataset", "component"], columns="type", values="r@5").round(3)
    table.join(by_type).to_csv(out)
    pd.set_option("display.width", 250)
    print(table.join(by_type).to_string())


if __name__ == "__main__":
    main()
