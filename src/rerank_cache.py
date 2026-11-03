"""Precompute first-stage candidates and reranker scores for every question.

Reranking the original question does not depend on the generator LLM, so it is done once per
(dataset, embedder, reranker) before the main runs and cached in
data/<dataset>/rerank_<embedder>_<reranker>.json. This keeps the reranker off the GPU while
the LLM runs, and lets us use rerankers that llama.cpp cannot serve (jina-reranker-v3.5).

    python src/rerank_cache.py --dataset hotpotqa --embedder harrier-0.6b --embed-url http://127.0.0.1:8086 \
        --reranker jina-v3.5
    python src/rerank_cache.py ... --reranker qwen3-reranker-0.6b --rerank-url http://127.0.0.1:8084
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from clients import rerank
from config import DATASETS, EMBEDDERS, RERANK_CANDIDATES, ROOT, TOP_K, data_dir
from retriever import Retriever

JINA_DIR = ROOT / "models" / "jina-reranker-v3.5"


def cache_path(dataset, embedder, reranker):
    return data_dir(dataset) / f"rerank_{embedder}_{reranker}.json"


def load_cache(dataset, embedder, reranker):
    p = cache_path(dataset, embedder, reranker)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def jina_scorer():
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained(JINA_DIR, dtype="auto", trust_remote_code=True).to("cuda").eval()

    def score(query, docs):
        out = model.rerank(query, docs)
        scores = [0.0] * len(docs)
        for r in out:
            scores[int(r["index"])] = float(r["relevance_score"])
        return scores
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--embedder", required=True, choices=EMBEDDERS)
    ap.add_argument("--embed-url", required=True)
    ap.add_argument("--reranker", required=True)
    ap.add_argument("--rerank-url", help="llama-server /v1/rerank endpoint; omit for jina-v3.5 (torch)")
    args = ap.parse_args()

    qs = json.loads((data_dir(args.dataset) / "questions.json").read_text(encoding="utf-8"))
    ret = Retriever(args.dataset, embedder=args.embedder, url=args.embed_url)
    with ThreadPoolExecutor(8) as pool:
        cands = list(pool.map(lambda q: ret.search(q["question"], RERANK_CANDIDATES), qs))

    if args.rerank_url:
        def score(query, docs):
            return rerank(query, docs, url=args.rerank_url)
    else:
        score = jina_scorer()

    cache, recalls = {}, []
    for q, c in tqdm(zip(qs, cands), total=len(qs), desc=f"{args.dataset} {args.reranker}"):
        s = score(q["question"], [f"{p['title']}: {p['text']}" for p in c])
        order = sorted(range(len(c)), key=lambda i: -s[i])
        cache[q["id"]] = {"pids": [c[i]["pid"] for i in order], "scores": [s[i] for i in order],
                          "first_stage_pids": [p["pid"] for p in c]}
        if q["answerable"]:
            titles = [c[i]["title"] for i in order[:TOP_K]]
            recalls.append(len(set(q["gold_titles"]) & set(titles)) / len(q["gold_titles"]))
    cache_path(args.dataset, args.embedder, args.reranker).write_text(json.dumps(cache), encoding="utf-8")
    print(f"[{args.dataset}] {args.embedder} + {args.reranker}: recall@{TOP_K} = {np.mean(recalls):.3f}")


if __name__ == "__main__":
    main()
