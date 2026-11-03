"""Run one or more systems over the question set with the currently loaded LLM.

Results are appended to results/<dataset>/<model>/<system>.jsonl one line per question, so an
interrupted run resumes where it stopped.

    python src/run_experiment.py --dataset hotpotqa --model qwen38-27b --workers 4
    python src/run_experiment.py --dataset 2wiki --model qwen38-27b --systems naive --limit 10   # pilot
"""
import argparse
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from clients import Usage, expect_model, model_name
from config import DATASETS, MODELS, RESULTS_DIR, data_dir
from retriever import Retriever
from systems import MAIN_SYSTEMS, SYSTEMS


def run_one(system, q, retriever):
    usage = Usage()
    t0 = time.perf_counter()
    try:
        out = SYSTEMS[system](q["question"], retriever, usage)
        err = None
    except Exception:
        out, err = {"answer": None, "context_titles": [], "trace": {}}, traceback.format_exc()
    return {"id": q["id"], "system": system, **out, "usage": usage.as_dict(),
            "latency_s": round(time.perf_counter() - t0, 3), "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--systems", nargs="+", default=MAIN_SYSTEMS, choices=SYSTEMS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    loaded = model_name()
    if MODELS[args.model] not in loaded:
        raise SystemExit(f"LLM server has {loaded!r} loaded, expected {MODELS[args.model]!r}")
    expect_model(MODELS[args.model])

    questions = json.loads((data_dir(args.dataset) / "questions.json").read_text(encoding="utf-8"))[:args.limit]
    retriever = Retriever(args.dataset)
    out_dir = RESULTS_DIR / args.dataset / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    for system in args.systems:
        path = out_dir / f"{system}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()
                    if l and json.loads(l)["error"] is None}
        todo = [q for q in questions if q["id"] not in done]
        print(f"[{args.dataset}/{args.model}] {system}: {len(done)} done, {len(todo)} to run")
        with ThreadPoolExecutor(args.workers) as pool, path.open("a", encoding="utf-8") as f:
            futures = [pool.submit(run_one, system, q, retriever) for q in todo]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=system):
                rec = fut.result()
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if rec["error"]:
                    print(f"\nERROR {rec['id']}:\n{rec['error']}")


if __name__ == "__main__":
    main()
