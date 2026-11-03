"""Build the evaluation set and retrieval corpus for one dataset (dev split, distractor setting).

    python src/prepare_data.py --dataset hotpotqa
    python src/prepare_data.py --dataset 2wiki

- Samples N_QUESTIONS questions, balanced over the dataset's question types.
- Corpus = every paragraph (gold + distractor) of the full dev set, deduplicated by title,
  so the sampled questions compete with tens of thousands of distractor passages.
- For N_UNANSWERABLE questions, all their gold paragraphs are removed from the corpus,
  so the correct behaviour is to abstain. Gold titles shared with answerable questions
  are never removed; such questions are skipped when picking unanswerable ones.
"""
import argparse
import json
import random

import pandas as pd

from config import DATASETS, N_QUESTIONS, N_UNANSWERABLE, SEED, data_dir


def load_hotpotqa():
    """HotpotQA dev (distractor) from the Hugging Face parquet. Sentences carry their own spacing."""
    df = pd.read_parquet(data_dir("hotpotqa") / "dev.parquet")
    return [{
        "_id": r["id"], "question": r["question"], "answer": r["answer"], "type": r["type"],
        "supporting_facts": list(zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"])),
        "context": [(t, "".join(s).strip()) for t, s in zip(r["context"]["title"], r["context"]["sentences"])],
    } for r in df.to_dict("records")]


def load_2wiki():
    """2WikiMultiHopQA dev from the authors' Hugging Face parquet (JSON-encoded fields)."""
    df = pd.read_parquet(data_dir("2wiki") / "dev.parquet")
    return [{
        "_id": r["_id"], "question": r["question"], "answer": r["answer"], "type": r["type"],
        "supporting_facts": json.loads(r["supporting_facts"]),
        "context": [(t, " ".join(s).strip()) for t, s in json.loads(r["context"])],
    } for r in df.to_dict("records")]


def load_musique():
    """MuSiQue-Ans v1.0 dev (Trivedi et al., 2022). Question type = number of hops (2/3/4).
    Some titles have several distinct paragraphs, so the corpus is deduplicated by (title, text)."""
    rows = [json.loads(l) for l in (data_dir("musique") / "dev.jsonl").read_text(encoding="utf-8").splitlines() if l]
    return [{
        "_id": r["id"], "question": r["question"], "answer": r["answer"], "aliases": r.get("answer_aliases", []),
        "type": r["id"].split("__")[0][:4],
        "supporting_facts": [(p["title"], 0) for p in r["paragraphs"] if p["is_supporting"]],
        "context": [(p["title"], p["paragraph_text"].strip()) for p in r["paragraphs"]],
        # the paragraph supporting the last decomposition step contains the final answer
        "answer_title": r["paragraphs"][r["question_decomposition"][-1]["paragraph_support_idx"]]["title"],
    } for r in rows]


LOADERS = {"hotpotqa": load_hotpotqa, "2wiki": load_2wiki, "musique": load_musique}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    args = ap.parse_args()

    raw = LOADERS[args.dataset]()
    rng = random.Random(SEED)

    by_type = {}
    for ex in raw:
        by_type.setdefault(ex["type"], []).append(ex)
    types = sorted(by_type)
    per_type = {t: N_QUESTIONS // len(types) + (i < N_QUESTIONS % len(types)) for i, t in enumerate(types)}
    sample = []
    for t in types:
        sample += rng.sample(by_type[t], per_type[t])
    rng.shuffle(sample)

    gold_titles = {ex["_id"]: sorted({t for t, _ in ex["supporting_facts"]}) for ex in sample}

    # Paragraphs removed to make a question unanswerable: all gold paragraphs (HotpotQA, 2Wiki), or
    # only the answer-bearing one (MuSiQue, following how MuSiQue-Full builds its unanswerable
    # contrast questions; MuSiQue questions share too many gold paragraphs to remove them all).
    # A question is a candidate only if no other sampled question needs the removed paragraphs.
    removal = {ex["_id"]: [ex["answer_title"]] if "answer_title" in ex else gold_titles[ex["_id"]]
               for ex in sample}
    title_owner_count = {}
    for titles in gold_titles.values():
        for t in titles:
            title_owner_count[t] = title_owner_count.get(t, 0) + 1
    candidates = [ex for ex in sample if all(title_owner_count[t] == 1 for t in removal[ex["_id"]])]
    if len(candidates) >= N_UNANSWERABLE:
        unanswerable_ids = {ex["_id"] for ex in rng.sample(candidates, N_UNANSWERABLE)}
    else:
        # MuSiQue: too few questions in the sample have exclusive answer paragraphs. Keep the
        # answerable questions type-balanced and draw the unanswerable ones (also type-balanced)
        # from the rest of the dev set, among questions whose answer paragraph no answerable
        # question needs.
        n_ans = N_QUESTIONS - N_UNANSWERABLE
        ans_per = {t: n_ans // len(types) + (i < n_ans % len(types)) for i, t in enumerate(types)}
        answerable = []
        for t in types:
            answerable += [ex for ex in sample if ex["type"] == t][:ans_per[t]]
        ans_ids = {ex["_id"] for ex in answerable}
        needed = {t for ex in answerable for t, _ in ex["supporting_facts"]}
        pools = {t: [ex for ex in by_type[t] if ex["_id"] not in ans_ids and ex["answer_title"] not in needed]
                 for t in types}
        # 3- and 4-hop answer paragraphs are widely shared, so few of them qualify: take as many as
        # possible up to an equal share, and fill the rest with 2-hop questions
        una_per = {t: min(len(pools[t]), N_UNANSWERABLE // len(types)) for t in types if t != types[0]}
        una_per[types[0]] = N_UNANSWERABLE - sum(una_per.values())
        unanswerable = []
        for t in types:
            unanswerable += rng.sample(pools[t], una_per[t])
        print(f"unanswerable per type: {una_per}")
        sample = answerable + unanswerable
        rng.shuffle(sample)
        gold_titles = {ex["_id"]: sorted({t for t, _ in ex["supporting_facts"]}) for ex in sample}
        removal = {ex["_id"]: [ex["answer_title"]] for ex in sample}
        unanswerable_ids = {ex["_id"] for ex in unanswerable}
    removed_titles = {t for qid in unanswerable_ids for t in removal[qid]}

    corpus = {}  # (title, text) -> None, insertion-ordered
    seen_titles = set()
    for ex in raw:
        for title, text in ex["context"]:
            if title in removed_titles or not text:
                continue
            # HotpotQA/2Wiki: one paragraph per title. MuSiQue: keep distinct paragraphs of a title
            if args.dataset != "musique" and title in seen_titles:
                continue
            seen_titles.add(title)
            corpus[(title, text)] = None

    questions = [{
        "id": ex["_id"],
        "question": ex["question"],
        "answer": ex["answer"],
        "aliases": ex.get("aliases", []),
        "type": ex["type"],
        "gold_titles": gold_titles[ex["_id"]],
        "answerable": ex["_id"] not in unanswerable_ids,
    } for ex in sample]
    missing = sum(1 for q in questions if q["answerable"] and not set(q["gold_titles"]) <= seen_titles)

    passages = [{"pid": i, "title": t, "text": txt} for i, (t, txt) in enumerate(corpus)]

    out = data_dir(args.dataset)
    (out / "questions.json").write_text(json.dumps(questions, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "corpus.json").write_text(json.dumps(passages, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[{args.dataset}] types: { {t: len(v) for t, v in by_type.items()} } -> sampled {per_type}")
    print(f"questions: {len(questions)} ({sum(q['answerable'] for q in questions)} answerable, "
          f"{missing} answerable with a gold title missing from corpus)")
    print(f"corpus passages: {len(passages)} (removed {len(removed_titles)} gold titles)")


if __name__ == "__main__":
    main()
