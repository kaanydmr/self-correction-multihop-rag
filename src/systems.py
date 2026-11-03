"""The five RAG systems compared in the paper (four baselines and our Evidence-aware CRAG).

Every system returns {"answer", "context_titles", "trace"} and fills the Usage object.
CRAG and Self-RAG are training-free, prompt-based adaptations:
- CRAG (Yan et al., 2024): the T5 retrieval evaluator is replaced by LLM grading, and the
  web-search fallback by query rewriting + a second retrieval over the same corpus.
- Self-RAG (Asai et al., 2024): the reflection tokens ISREL / ISSUP / ISUSE are produced by
  prompting instead of a fine-tuned critic; retrieval is always on (every question needs it).
"""
import re

import prompts
from clients import Usage, _complete, chat, chat_label
from config import ABSTAIN, TOP_K

# room for the at most three sentences of reasoning before the answer line (200 cut off ~5% of
# Qwen's answers before "Answer:" in the first runs)
ANSWER_MAX_TOKENS = 512


def _label(text, labels, default):
    up = text.upper()
    for lab in labels:
        if re.search(rf"\b{lab}\b", up):
            return lab
    return default


def _is_abstain(ans):
    return ans.strip().rstrip(".").lower() == ABSTAIN.lower()


def answer_call(messages, usage, step="generate"):
    """One answer call; if the reasoning hits the token limit before the answer line, one short
    follow-up call asks for the answer line only."""
    choice = _complete(messages, usage, step, ANSWER_MAX_TOKENS)
    text = choice["message"]["content"].strip()
    if choice.get("finish_reason") == "length":
        text = chat(messages + [{"role": "assistant", "content": text}, {"role": "user", "content": prompts.FINISH}],
                    usage, step + "_finish", max_tokens=48)
    return prompts.parse_answer(text)


def _generate(question, passages, usage, step="generate"):
    return answer_call(prompts.answer(question, passages), usage, step)


def _result(answer, passages, trace=None):
    return {"answer": answer, "context_titles": [p["title"] for p in passages], "trace": trace or {}}


# ---------------------------------------------------------------- 1. Naive RAG
def naive(question, retriever, usage: Usage):
    passages = retriever.search(question, TOP_K)
    return _result(_generate(question, passages, usage), passages)


# ---------------------------------------------------------------- 2. RAG + reranker
def reranked(question, retriever, usage: Usage):
    passages = retriever.rerank_top(question, TOP_K)
    return _result(_generate(question, passages, usage), passages)


# ---------------------------------------------------------------- 3. CRAG
def _crag_grade(question, passages, usage):
    return [_label(chat(prompts.crag_grade(question, p), usage, "grade", max_tokens=8),
                   ["CORRECT", "AMBIGUOUS", "INCORRECT"], "AMBIGUOUS") for p in passages]


def crag(question, retriever, usage: Usage):
    passages = retriever.search(question, TOP_K)
    grades = _crag_grade(question, passages, usage)
    kept = [p for p, g in zip(passages, grades) if g != "INCORRECT"]
    trace = {"grades": grades}

    if "CORRECT" in grades:
        action = "correct"
    else:
        # Ambiguous: keep what we have and add a corrective search.
        # Incorrect: discard everything and rely on the corrective search only.
        action = "ambiguous" if kept else "incorrect"
        query = chat(prompts.crag_rewrite(question), usage, "rewrite", max_tokens=48)
        seen = {p["pid"] for p in passages}
        extra = [p for p in retriever.search(query, TOP_K) if p["pid"] not in seen]
        extra_grades = _crag_grade(question, extra, usage)
        kept += [p for p, g in zip(extra, extra_grades) if g != "INCORRECT"]
        trace.update(query=query, extra_grades=extra_grades)
    trace["action"] = action

    if not kept:
        return _result(ABSTAIN, [], trace)  # nothing trustworthy retrieved: abstain without generating
    return _result(_generate(question, kept, usage), kept, trace)


# ---------------------------------------------------------------- 4. Self-RAG
# Follows the short-form inference of the official implementation (retrieval_lm/run_short_form.py,
# always-retrieve mode): one candidate answer per retrieved passage, each scored by
# w_rel * ISREL + w_sup * ISSUP + w_use * ISUSE with the paper's default weights 1.0 / 1.0 / 0.5,
# where each critique is the label probability (Fully + 0.5 * Partially; utility 1..5 -> -1..1).
# The critiques come from the prompted generator instead of trained reflection tokens. Nothing is
# filtered out; the answer is "I don't know" only if every candidate is.
W_REL, W_SUP, W_USE = 1.0, 1.0, 0.5
UTILITY = {"1": -1.0, "2": -0.5, "3": 0.0, "4": 0.5, "5": 1.0}


def _selfrag(question, retriever, usage: Usage, open_gen=False, joint=False):
    """open_gen: candidates are generated as by the original Self-RAG generator, without the
    instruction to abstain (answer_open); joint: one extra candidate from all five passages, which a
    multi-hop question may need (its relevance score is the highest of the five)."""
    passages = retriever.search(question, TOP_K)
    contexts = [[p] for p in passages] + ([passages] if joint else [])
    candidates = []
    for i, ctx in enumerate(contexts):
        if i < len(passages):
            rel_label, rel = chat_label(prompts.selfrag_isrel(question, ctx[0]), ["RELEVANT", "IRRELEVANT"],
                                        usage, "isrel")
            p_rel = rel["RELEVANT"]
        else:
            rel_label, p_rel = "JOINT", max(c["p_rel"] for c in candidates)
        msgs = prompts.answer_open(question, ctx) if open_gen else prompts.answer(question, ctx)
        ans = answer_call(msgs, usage)
        cand = {"answer": ans, "isrel": rel_label, "p_rel": round(p_rel, 4)}
        if not _is_abstain(ans):
            sup_label, sup = chat_label(prompts.selfrag_issup(question, ctx, ans), ["FULLY", "PARTIALLY", "NO"],
                                        usage, "issup")
            use_label, use = chat_label(prompts.selfrag_isuse(question, ans), list(UTILITY), usage, "isuse")
            s_sup = sup["FULLY"] + 0.5 * sup["PARTIALLY"]
            s_use = sum(UTILITY[k] * v for k, v in use.items())
            cand.update(issup=sup_label, isuse=use_label,
                        score=round(W_REL * p_rel + W_SUP * s_sup + W_USE * s_use, 4))
        candidates.append(cand)
    scored = [i for i, c in enumerate(candidates) if "score" in c]
    trace = {"candidates": candidates, "best": None}
    if not scored:
        return _result(ABSTAIN, [], trace)
    best = max(scored, key=lambda i: candidates[i]["score"])  # ties: the higher-ranked passage
    trace["best"] = best
    return _result(candidates[best]["answer"], contexts[best], trace)


def selfrag(question, retriever, usage: Usage):
    return _selfrag(question, retriever, usage)


def selfrag_open(question, retriever, usage: Usage):
    return _selfrag(question, retriever, usage, open_gen=True)


def selfrag_joint(question, retriever, usage: Usage):
    return _selfrag(question, retriever, usage, joint=True)


def selfrag_open_joint(question, retriever, usage: Usage):
    return _selfrag(question, retriever, usage, open_gen=True, joint=True)


# The first, conservative adaptation (kept as a variant): ISREL as a hard filter, one answer from the
# relevant passages, at most one regeneration from all passages, and an abstention when the best
# candidate is unsupported. Labels now come from chat_label (the first runs parsed free text cut at
# 8 tokens, so a reply that began with reasoning defaulted to "NO").
SUP_SCORE = {"FULLY": 1.0, "PARTIALLY": 0.5, "NO": 0.0}


def _critique(question, passages, ans, usage):
    sup, _ = chat_label(prompts.selfrag_issup(question, passages, ans), ["FULLY", "PARTIALLY", "NO"], usage, "issup")
    use, _ = chat_label(prompts.selfrag_isuse(question, ans), list(UTILITY), usage, "isuse")
    return sup, int(use), SUP_SCORE[sup] + 0.5 * int(use) / 5


def selfrag_cons(question, retriever, usage: Usage):
    passages = retriever.search(question, TOP_K)
    rel = [chat_label(prompts.selfrag_isrel(question, p), ["RELEVANT", "IRRELEVANT"], usage, "isrel")[0]
           for p in passages]
    relevant = [p for p, r in zip(passages, rel) if r == "RELEVANT"]
    trace = {"isrel": rel, "candidates": []}

    # Candidate 1: answer from the passages judged relevant
    ans = _generate(question, relevant, usage)
    if _is_abstain(ans):
        return _result(ABSTAIN, relevant, trace)
    sup, use, score = _critique(question, relevant, ans, usage)
    best = (score, ans, relevant, sup)
    trace["candidates"].append({"answer": ans, "issup": sup, "isuse": use})

    # Not fully supported or not useful: regenerate once over the full retrieved set and keep the
    # higher-scoring candidate (critique-guided selection, as in Self-RAG's segment ranking)
    if not (sup == "FULLY" and use >= 4) and len(relevant) < len(passages):
        ans2 = _generate(question, passages, usage, step="regenerate")
        if not _is_abstain(ans2):
            sup2, use2, score2 = _critique(question, passages, ans2, usage)
            trace["candidates"].append({"answer": ans2, "issup": sup2, "isuse": use2})
            if score2 > best[0]:
                best = (score2, ans2, passages, sup2)

    _, ans, ctx, sup = best
    if sup == "NO":
        return _result(ABSTAIN, ctx, trace)  # refuse to output an unsupported answer
    return _result(ans, ctx, trace)


# ---------------------------------------------------------------- 5. Evidence-aware CRAG (ours)
# Addresses the two failure modes seen in the pilot: per-passage grading drops bridge passages of
# multi-hop questions, and critique loops cost 6-8 LLM calls. Passages come from the reranker,
# are graded jointly with a chain-aware prompt, and the reranker's top passages act as a floor.
KEEP_FLOOR = 2


def _evidence_labels(text, n):
    labels = ["MAYBE"] * n  # unparsed passages are kept
    for m in re.finditer(r"\[?(\d+)\]?\s*[:\-–]\s*\**\s*(NEEDED|MAYBE|DROP)", text.upper()):
        i = int(m.group(1)) - 1
        if 0 <= i < n:
            labels[i] = m.group(2)
    return labels


def _filter(question, passages, labels, usage, trace, floor=KEEP_FLOOR):
    if all(lab == "DROP" for lab in labels):
        return _result(ABSTAIN, [], trace)  # no evidence for any step of the chain: abstain
    kept = [p for i, (p, lab) in enumerate(zip(passages, labels)) if lab != "DROP" or i < floor]
    return _result(_generate(question, kept, usage), kept, trace)


def _joint_grade(question, passages, usage):
    return _evidence_labels(grade_call(prompts.evidence_grade(question, passages), usage, finish=prompts.FINISH_LABELS),
                            len(passages))


def ecrag(question, retriever, usage: Usage):
    passages = retriever.rerank_top(question, TOP_K)
    labels = _joint_grade(question, passages, usage)
    return _filter(question, passages, labels, usage, {"labels": labels})


# ---------------------------------------------------------------- EA-CRAG with missing-link search
# On multi-hop questions the reranked top-5 often lacks a hop entirely (all gold paragraphs are in
# the top-5 for only 25% of MuSiQue questions), which no filter can repair. The chain-aware grader
# already reasons about the chain, so it also names the first missing fact as a search query; the
# query is retrieved and reranked, the new passages are graded, and the non-DROP ones are added.
MAX_HOPS, HOP_NEW, HOP_CANDIDATES = 2, 3, 10


def _missing_query(text):
    found = re.findall(r"MISSING:\s*(.+)", text, flags=re.IGNORECASE)
    query = found[-1].strip().strip("'\"*`<> ").rstrip(".") if found else ""
    return None if not query or query.lower().startswith("none") else query


def grade_call(messages, usage, step="grade", finish=prompts.FINISH_GRADE):
    """The chain-aware grading call; if its reasoning reaches the token limit before the labels
    (8 of 60 calls in a development test; 4-12% of Qwen's plain EA-CRAG grades in the first runs,
    whose unlabelled passages then defaulted to MAYBE), one short follow-up asks for the labels only."""
    choice = _complete(messages, usage, step, 400)
    text = choice["message"]["content"].strip()
    if choice.get("finish_reason") == "length":
        text = chat(messages + [{"role": "assistant", "content": text}, {"role": "user", "content": finish}],
                    usage, step + "_finish", max_tokens=120)
    return text


def ecrag_hop(question, retriever, usage: Usage, rerank_hops=True, max_hops=MAX_HOPS):
    """rerank_hops: the missing-link query's dense top-10 is reranked by jina on the CPU (about 4 s);
    otherwise its dense top results are graded directly. max_hops=0 is the same system without the
    missing-link search (the ablation)."""
    passages = retriever.rerank_top(question, TOP_K)
    text = grade_call(prompts.evidence_grade_hop(question, passages), usage)
    labels = _evidence_labels(text, len(passages))
    kept = [] if all(lab == "DROP" for lab in labels) else \
        [p for i, (p, lab) in enumerate(zip(passages, labels)) if lab != "DROP" or i < KEEP_FLOOR]
    trace = {"labels": labels, "hops": []}
    seen = {p["pid"] for p in passages}
    query = _missing_query(text)
    for _ in range(max_hops):
        if not query:
            break
        found = retriever.rerank_live(query, HOP_CANDIDATES) if rerank_hops else retriever.search(query, HOP_CANDIDATES)
        new = [p for p in found if p["pid"] not in seen][:HOP_NEW]
        if not new:
            break
        seen |= {p["pid"] for p in new}
        text = grade_call(prompts.evidence_grade_hop(question, kept + new), usage, "grade_hop")
        new_labels = _evidence_labels(text, len(kept) + len(new))[len(kept):]
        kept += [p for p, lab in zip(new, new_labels) if lab != "DROP"]
        trace["hops"].append({"query": query, "new_pids": [p["pid"] for p in new], "labels": new_labels})
        query = _missing_query(text)
    if not kept:
        return _result(ABSTAIN, [], trace)  # no evidence for any step of the chain: abstain
    return _result(_generate(question, kept, usage), kept, trace)


def ecrag_hop_dense(question, retriever, usage: Usage):
    return ecrag_hop(question, retriever, usage, rerank_hops=False)


def ecrag_nohop(question, retriever, usage: Usage):
    return ecrag_hop(question, retriever, usage, max_hops=0)


# ---------------------------------------------------------------- EA-CRAG ablations
# Each variant undoes exactly one of EA-CRAG's three changes to CRAG.
def ecrag_dense(question, retriever, usage: Usage):
    """Without reranked candidates: joint grading of the dense top-5."""
    passages = retriever.search(question, TOP_K)
    labels = _joint_grade(question, passages, usage)
    return _filter(question, passages, labels, usage, {"labels": labels})


def ecrag_perpassage(question, retriever, usage: Usage):
    """Without joint chain-aware grading: CRAG's per-passage grades (INCORRECT counts as DROP)."""
    passages = retriever.rerank_top(question, TOP_K)
    grades = _crag_grade(question, passages, usage)
    labels = ["DROP" if g == "INCORRECT" else "MAYBE" for g in grades]
    return _filter(question, passages, labels, usage, {"grades": grades, "labels": labels})


def ecrag_nofloor(question, retriever, usage: Usage):
    """Without the top-2 floor: every passage graded DROP is removed."""
    passages = retriever.rerank_top(question, TOP_K)
    labels = _joint_grade(question, passages, usage)
    return _filter(question, passages, labels, usage, {"labels": labels}, floor=0)


# ---------------------------------------------------------------- fixed truncation baselines
# Cleaner context without an LLM: the reranker's top-2 or top-3 only (EA-CRAG keeps 2.3 on average).
def reranked_k2(question, retriever, usage: Usage):
    passages = retriever.rerank_top(question, 2)
    return _result(_generate(question, passages, usage), passages)


def reranked_k3(question, retriever, usage: Usage):
    passages = retriever.rerank_top(question, 3)
    return _result(_generate(question, passages, usage), passages)


# ---------------------------------------------------------------- a second search without the grader (control)
# Is it the grader's diagnosis that finds the missing hop, or would any second search do? Here the
# question is rewritten as a search query with CRAG's rewrite prompt, which sees no passages, and the
# three best new passages for it (dense top-10, reranked as in the missing-link search) are added to
# the reranked top-5, without any grading.
def rerank_search(question, retriever, usage: Usage):
    passages = retriever.rerank_top(question, TOP_K)
    query = chat(prompts.crag_rewrite(question), usage, "rewrite", max_tokens=48)
    seen = {p["pid"] for p in passages}
    new = [p for p in retriever.rerank_live(query, HOP_CANDIDATES) if p["pid"] not in seen][:HOP_NEW]
    kept = passages + new
    return _result(_generate(question, kept, usage), kept, {"query": query, "new_pids": [p["pid"] for p in new]})


MAIN_SYSTEMS = ["naive", "rerank", "crag", "selfrag_joint", "ecrag", "ecrag_hop", "ecrag_nohop",
                "rerank_search"]  # the systems of the paper; ablations and the rerun run on request
SYSTEMS = {"naive": naive, "rerank": reranked, "crag": crag, "selfrag": selfrag, "ecrag": ecrag,
           "ecrag_dense": ecrag_dense, "ecrag_perpassage": ecrag_perpassage, "ecrag_nofloor": ecrag_nofloor,
           "rerank_k2": reranked_k2, "rerank_k3": reranked_k3, "selfrag_cons": selfrag_cons,
           "selfrag_open": selfrag_open, "selfrag_joint": selfrag_joint, "selfrag_open_joint": selfrag_open_joint,
           "ecrag_hop": ecrag_hop, "ecrag_hop_dense": ecrag_hop_dense, "ecrag_nohop": ecrag_nohop,
           "rerank_search": rerank_search,
           # a second, independent run of the missing-link search on part of the questions (run-to-run check)
           "ecrag_hop_rerun": ecrag_hop}
