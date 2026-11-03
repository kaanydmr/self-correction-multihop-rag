"""LLM-as-judge for answer correctness (answerable questions only).

The primary judge (Qwen3.8-27B) grades the outputs of every generator; a second judge
(Gemma-4-12B) re-grades them to measure judge agreement, since Qwen also grades its own answers.
The judge must be the model loaded on the LLM server. Verdicts are cached by
(dataset, question id, prediction) in results/judge.jsonl (primary) or
results/judge_<model>.jsonl, so identical answers are graded once and reruns are free.

    python src/judge.py                      # primary judge
    python src/judge.py --judge gemma4-12b   # cross-judge
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

import prompts
from clients import Usage, chat, expect_model, model_name
from config import ABSTAIN, DATASETS, MODELS, RESULTS_DIR, data_dir

JUDGE_MODEL = "qwen38-27b"


def cache_path(judge=JUDGE_MODEL):
    return RESULTS_DIR / ("judge.jsonl" if judge == JUDGE_MODEL else f"judge_{judge}.jsonl")


def load_questions():
    return {ds: {q["id"]: q for q in json.loads((data_dir(ds) / "questions.json").read_text(encoding="utf-8"))}
            for ds in DATASETS}


def load_results():
    """Last error-free record per (dataset, model, system, question)."""
    out = {}
    for path in RESULTS_DIR.glob("*/*/*.jsonl"):
        dataset, model = path.parent.parent.name, path.parent.name
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["error"] is None:
                out[(dataset, model, rec["system"], rec["id"])] = rec
    return out


def load_verdicts(judge=JUDGE_MODEL):
    verdicts = {}
    if cache_path(judge).exists():
        for line in cache_path(judge).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            verdicts[(r["dataset"], r["id"], r["pred"])] = r["verdict"]
    return verdicts


def is_abstain(ans):
    return ans is None or ABSTAIN.lower() in ans.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=JUDGE_MODEL, choices=MODELS)
    judge = ap.parse_args().judge
    if MODELS[judge] not in model_name():
        raise SystemExit(f"Load {MODELS[judge]} on the LLM server to run this judge.")
    expect_model(MODELS[judge])
    questions = load_questions()
    todo = {(ds, qid, rec["answer"]) for (ds, _, _, qid), rec in load_results().items()
            if questions[ds][qid]["answerable"] and not is_abstain(rec["answer"])} - load_verdicts(judge).keys()
    print(f"{len(todo)} new (question, answer) pairs to judge")

    def grade(key):
        ds, qid, pred = key
        q = questions[ds][qid]
        text = chat(prompts.judge(q["question"], q["answer"], pred, q.get("aliases", [])), Usage(), "judge",
                    max_tokens=8)
        return ds, qid, pred, ("INCORRECT" not in text.upper() and "CORRECT" in text.upper())

    with ThreadPoolExecutor(4) as pool, cache_path(judge).open("a", encoding="utf-8") as f:
        for ds, qid, pred, verdict in tqdm(pool.map(grade, sorted(todo)), total=len(todo)):
            f.write(json.dumps({"dataset": ds, "id": qid, "pred": pred, "verdict": verdict},
                               ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
