"""LLM judge for faithfulness: is an answer grounded in the passages its system gave the generator?

Complements the correctness judge: an answer can be correct but ungrounded (parametric knowledge)
or grounded but wrong (a reasoning error), and an answer to an unanswerable question is a
hallucination only if it is also ungrounded. Every answered question is judged, answerable or not,
with the exact passages recovered by src/contexts.py (run it first). Verdicts are cached by
(dataset, question id, answer, passage ids) in results/faithfulness.jsonl (primary judge) or
results/faithfulness_<model>.jsonl (cross-judge).

    python src/faithfulness.py                                     # primary judge (Qwen), all answers
    python src/faithfulness.py --judge gemma4-12b --fraction 0.15  # cross-judge on a fixed random 15%
"""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

import prompts
from clients import Usage, chat, expect_model, model_name
from config import DATASETS, MODELS, RESULTS_DIR, data_dir
from contexts import load_contexts
from judge import JUDGE_MODEL, is_abstain, load_questions, load_results


def cache_path(judge=JUDGE_MODEL):
    return RESULTS_DIR / ("faithfulness.jsonl" if judge == JUDGE_MODEL else f"faithfulness_{judge}.jsonl")


def load_faithfulness(judge=JUDGE_MODEL):
    """(dataset, id, answer, passage ids) -> grounded?"""
    verdicts = {}
    if cache_path(judge).exists():
        for line in cache_path(judge).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            verdicts[(r["dataset"], r["id"], r["pred"], tuple(r["pids"]))] = r["verdict"]
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=JUDGE_MODEL, choices=MODELS)
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="judge a fixed pseudo-random share of the answers (hash-based, stable across runs)")
    ap.add_argument("--limit", type=int, help="judge only this many pairs (for testing)")
    args = ap.parse_args()
    if MODELS[args.judge] not in model_name():
        raise SystemExit(f"Load {MODELS[args.judge]} on the LLM server to run this judge.")
    expect_model(MODELS[args.judge])
    questions = load_questions()
    contexts = load_contexts()
    results = load_results()
    answered = {k: rec for k, rec in results.items() if not is_abstain(rec["answer"])}
    missing = sum(k not in contexts for k in answered)
    if missing:
        print(f"skipping {missing} answered records without recovered passages: run src/contexts.py first")
    todo = {(ds, qid, rec["answer"], tuple(contexts[(ds, m, s, qid)]))
            for (ds, m, s, qid), rec in answered.items() if (ds, m, s, qid) in contexts}
    todo = [k for k in todo if int(hashlib.md5(json.dumps(k).encode()).hexdigest(), 16) % 10_000 < args.fraction * 10_000]
    todo = sorted(set(todo) - load_faithfulness(args.judge).keys())
    print(f"{len(todo)} new (question, answer, passages) triples to judge"
          + (f", judging the first {args.limit}" if args.limit is not None else ""))
    todo = todo[:args.limit]

    passages = {ds: {p["pid"]: p for p in json.loads((data_dir(ds) / "corpus.json").read_text(encoding="utf-8"))}
                for ds in DATASETS}

    def grade(key):
        ds, qid, pred, pids = key
        if not pids:  # answered without any passage: nothing to ground it in
            return key, False
        text = chat(prompts.faithfulness(questions[ds][qid]["question"], [passages[ds][p] for p in pids], pred),
                    Usage(), "faithfulness", max_tokens=120)
        return key, prompts.parse_verdict(text)

    with ThreadPoolExecutor(4) as pool, cache_path(args.judge).open("a", encoding="utf-8") as f:
        for (ds, qid, pred, pids), verdict in tqdm(pool.map(grade, todo), total=len(todo)):
            f.write(json.dumps({"dataset": ds, "id": qid, "pred": pred, "pids": list(pids), "verdict": verdict},
                               ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
