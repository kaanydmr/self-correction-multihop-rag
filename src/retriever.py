"""Dense retriever over a dataset's corpus (embedding model via llama-server, cosine similarity)."""
import json
import threading

import numpy as np
import requests

from clients import embed, rerank
from config import EMBED_URL, EMBEDDER, EMBEDDERS, RERANK_CANDIDATES, RERANKER, ROOT, data_dir

JINA_DIR = ROOT / "models" / "jina-reranker-v3.5"


class Retriever:
    def __init__(self, dataset, embedder=EMBEDDER, url=EMBED_URL):
        self.cfg = EMBEDDERS[embedder]
        self.url = url
        d = data_dir(dataset)
        self.passages = json.loads((d / "corpus.json").read_text(encoding="utf-8"))
        self.by_pid = {p["pid"]: p for p in self.passages}
        # Reranked candidate lists precomputed for the original questions (src/rerank_cache.py)
        cache = d / f"rerank_{embedder}_{RERANKER}.json"
        self.rerank_cache = {}
        if cache.exists():
            text_of = {q["id"]: q["question"] for q in json.loads((d / "questions.json").read_text(encoding="utf-8"))}
            self.rerank_cache = {text_of[qid]: v["pids"] for qid, v in json.loads(cache.read_text(encoding="utf-8")).items()
                                 if qid in text_of}
        index_path = d / f"corpus_emb_{embedder}.npy"
        if index_path.exists():
            self.emb = np.load(index_path)
        else:
            self.emb = self._build()
            np.save(index_path, self.emb)

    def _build(self, batch=32):
        # Passages longer than the embedder's window are truncated for embedding only
        # (the LLM always sees the full passage)
        texts = [f"{p['title']}: {p['text']}"[:self.cfg["max_chars"]] for p in self.passages]
        vecs = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            try:
                vecs += embed(chunk, self.url)
            except requests.HTTPError:
                # a passage in this batch still exceeds the window (dense non-Latin text): shorten it
                for t in chunk:
                    while True:
                        try:
                            vecs += embed([t], self.url)
                            break
                        except requests.HTTPError:
                            t = t[:len(t) * 2 // 3]
        return _normalize(np.array(vecs, dtype=np.float32))

    def search(self, query: str, k: int) -> list[dict]:
        q = _normalize(np.array(embed([self.cfg["query_prefix"] + query], self.url), dtype=np.float32))[0]
        scores = self.emb @ q
        top = np.argsort(-scores)[:k]
        return [{**self.passages[i], "score": float(scores[i])} for i in top]

    def rerank_top(self, query: str, k: int) -> list[dict]:
        """Top-k after reranking the first-stage top-RERANK_CANDIDATES (cached when available)."""
        if query in self.rerank_cache:
            return [self.by_pid[pid] for pid in self.rerank_cache[query][:k]]
        cands = self.search(query, RERANK_CANDIDATES)
        scores = rerank(query, [f"{p['title']}: {p['text']}" for p in cands])
        return [p for _, p in sorted(zip(scores, cands), key=lambda x: -x[0])][:k]

    def rerank_live(self, query: str, n_candidates: int) -> list[dict]:
        """The dense top-n reranked, for a new query (EA-CRAG's missing-link search): jina on the CPU."""
        cands = self.search(query, n_candidates)
        scores = jina_scores(query, [f"{p['title']}: {p['text']}" for p in cands])
        return [p for _, p in sorted(zip(scores, cands), key=lambda x: -x[0])]


_JINA, _JINA_LOCK = None, threading.Lock()


def jina_scores(query: str, docs: list[str]) -> list[float]:
    """jina-reranker-v3.5 on the CPU for queries that are not in the precomputed cache (the GPU holds
    the generator). One model shared by all threads, one query at a time."""
    global _JINA
    with _JINA_LOCK:
        if _JINA is None:
            import torch
            from transformers import AutoModel
            torch.set_num_threads(8)
            _JINA = AutoModel.from_pretrained(JINA_DIR, dtype=torch.float32, trust_remote_code=True).eval()
        out = _JINA.rerank(query, docs)
    scores = [0.0] * len(docs)
    for r in out:
        scores[int(r["index"])] = float(r["relevance_score"])
    return scores


def _normalize(m):
    return m / np.linalg.norm(m, axis=1, keepdims=True)
