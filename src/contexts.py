"""Exact passages each system handed to the generator, and the gold passages, as passage ids.

Result records store passage titles only, and in MuSiQue one title can have many paragraphs
(e.g. 42 for "New York City"), so titles cannot tell which paragraph was used. Retrieval is
deterministic (exact embedding search, cached reranker scores) and the traces store every
grading decision, so replaying a record's retrieval recovers its passage ids. Every replay is
checked against the stored titles, ignoring order: two near-equal retrieval scores can swap
places between runs, which changes neither the passage set nor any metric. Output:
results/contexts.jsonl (incremental; replays that did not match are retried).

    python src/contexts.py
"""
import json

from tqdm import tqdm

from config import DATASETS, RESULTS_DIR, TOP_K, data_dir
from judge import load_questions, load_results
from retriever import Retriever
from systems import KEEP_FLOOR

CONTEXTS = RESULTS_DIR / "contexts.jsonl"


def replay(system, question, rec, R):
    """Passages the system gave the generator for this record (the systems.py logic, minus LLM calls)."""
    trace = rec["trace"]
    if system == "naive":
        return R.search(question, TOP_K)
    if system == "rerank":
        return R.rerank_top(question, TOP_K)
    if system in ("rerank_k2", "rerank_k3"):
        return R.rerank_top(question, int(system[-1]))
    if system == "rerank_search":
        return R.rerank_top(question, TOP_K) + [R.r.by_pid[pid] for pid in trace["new_pids"]]
    if system in ("ecrag_hop", "ecrag_hop_dense", "ecrag_nohop", "ecrag_hop_rerun"):
        labels = trace["labels"]
        candidates = R.rerank_top(question, TOP_K)
        kept = [] if all(lab == "DROP" for lab in labels) else \
            [p for i, (p, lab) in enumerate(zip(candidates, labels)) if lab != "DROP" or i < KEEP_FLOOR]
        for hop in trace["hops"]:  # passages of the missing-link searches, by id
            kept += [R.r.by_pid[pid] for pid, lab in zip(hop["new_pids"], hop["labels"]) if lab != "DROP"]
        return kept
    if system.startswith("ecrag"):
        labels = trace["labels"]
        if all(lab == "DROP" for lab in labels):
            return []
        candidates = R.search(question, TOP_K) if system == "ecrag_dense" else R.rerank_top(question, TOP_K)
        floor = 0 if system == "ecrag_nofloor" else KEEP_FLOOR
        return [p for i, (p, lab) in enumerate(zip(candidates, labels)) if lab != "DROP" or i < floor]
    if system == "crag":
        passages = R.search(question, TOP_K)
        kept = [p for p, g in zip(passages, trace["grades"]) if g != "INCORRECT"]
        if "query" in trace:
            seen = {p["pid"] for p in passages}
            extra = [p for p in R.search(trace["query"], TOP_K) if p["pid"] not in seen]
            kept += [p for p, g in zip(extra, trace["extra_grades"]) if g != "INCORRECT"]
        return kept
    if system in ("selfrag", "selfrag_open", "selfrag_joint", "selfrag_open_joint"):
        # official-style Self-RAG: the answer comes from the passage of the best candidate, or from
        # all five for the joint candidate (index TOP_K)
        best, passages = trace["best"], R.search(question, TOP_K)
        return [] if best is None else passages if best == len(passages) else [passages[best]]
    if system == "selfrag_cons":
        passages = R.search(question, TOP_K)
        relevant = [p for p, r in zip(passages, trace["isrel"]) if r == "RELEVANT"]
        # the final answer came either from the relevant passages or from the full retrieved set
        return relevant if [p["title"] for p in relevant] == rec["context_titles"] else passages
    raise ValueError(system)


class CachedRetriever:
    """Each query is embedded once (the same questions recur across systems and models)."""

    def __init__(self, dataset):
        self.r = Retriever(dataset)
        self.memo = {}

    def search(self, query, k):
        if ("s", query, k) not in self.memo:
            self.memo["s", query, k] = self.r.search(query, k)
        return self.memo["s", query, k]

    def rerank_top(self, query, k):
        if ("r", query, k) not in self.memo:
            self.memo["r", query, k] = self.r.rerank_top(query, k)
        return self.memo["r", query, k]


def load_contexts():
    """(dataset, model, system, id) -> list of passage ids, for replays that matched the stored titles."""
    out = {}
    if CONTEXTS.exists():
        for line in CONTEXTS.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["match"]:
                out[(r["dataset"], r["model"], r["system"], r["id"])] = r["pids"]
    return out


def load_gold_pids():
    """dataset -> question id -> set of gold passage ids (answerable questions only)."""
    out = {}
    for ds in DATASETS:
        corpus = json.loads((data_dir(ds) / "corpus.json").read_text(encoding="utf-8"))
        questions = json.loads((data_dir(ds) / "questions.json").read_text(encoding="utf-8"))
        if ds == "musique":
            # several paragraphs per title: match the supporting paragraphs by (title, text)
            pid = {(p["title"], p["text"]): p["pid"] for p in corpus}
            wanted = {q["id"] for q in questions if q["answerable"]}
            out[ds] = {}
            for line in (data_dir(ds) / "dev.jsonl").read_text(encoding="utf-8").splitlines():
                r = json.loads(line) if line else None
                if r and r["id"] in wanted:
                    out[ds][r["id"]] = {pid[(p["title"], p["paragraph_text"].strip())]
                                        for p in r["paragraphs"] if p["is_supporting"]}
        else:
            pid = {p["title"]: p["pid"] for p in corpus}  # one paragraph per title
            out[ds] = {q["id"]: {pid[t] for t in q["gold_titles"]} for q in questions if q["answerable"]}
    return out


def main():
    questions = load_questions()
    done = set()
    if CONTEXTS.exists():
        done = {(r["dataset"], r["model"], r["system"], r["id"])
                for r in map(json.loads, CONTEXTS.read_text(encoding="utf-8").splitlines()) if r["match"]}
    results = load_results()
    todo = sorted(k for k in results if k not in done)
    print(f"{len(todo)} records to replay")
    retrievers, mismatches = {}, 0
    with CONTEXTS.open("a", encoding="utf-8") as f:
        for key in tqdm(todo):
            ds, model, system, qid = key
            if ds not in retrievers:
                retrievers[ds] = CachedRetriever(ds)
            rec = results[key]
            passages = replay(system, questions[ds][qid]["question"], rec, retrievers[ds])
            match = sorted(p["title"] for p in passages) == sorted(rec["context_titles"])
            mismatches += not match
            f.write(json.dumps({"dataset": ds, "model": model, "system": system, "id": qid,
                                "pids": [p["pid"] for p in passages], "match": match}) + "\n")
    print(f"replays not matching the stored titles: {mismatches}")


if __name__ == "__main__":
    main()
