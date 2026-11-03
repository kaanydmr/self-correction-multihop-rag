"""Compute all metrics, significance tests and figures from results/.

    python src/analyze.py

Outputs results/summary.csv, results/significance.csv and figures in results/figures/.
"""
import collections
import json
import re
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from config import RESULTS_DIR, SEED
from contexts import load_contexts, load_gold_pids
from faithfulness import load_faithfulness
from judge import is_abstain, load_questions, load_results, load_verdicts

# Self-RAG follows the official per-passage inference plus a candidate from all five passages;
# HOP is EA-CRAG with the missing-link search.
HOP = "ecrag_hop"
MAIN = ["naive", "rerank", "crag", "selfrag_joint", "ecrag", HOP]
# both generators: HOP without its search, and a second search whose query does not come from the
# grader; HOP run again on part of the questions (Qwen) to measure run-to-run variation
VARIANTS = ["ecrag_nohop", "rerank_search", "ecrag_hop_rerun"]
# EA-CRAG minus one component, and the reranker's top-2/top-3 without grading (Qwen only)
ABLATIONS = ["ecrag_dense", "ecrag_perpassage", "ecrag_nofloor", "rerank_k2", "rerank_k3"]
SYSTEM_ORDER = MAIN + VARIANTS + ABLATIONS
SYSTEM_LABEL = {"naive": "Naive RAG", "rerank": "RAG + Reranker", "crag": "CRAG", "selfrag_joint": "Self-RAG",
                "ecrag": "EA-CRAG", HOP: "EA-CRAG + missing-link search",
                "selfrag_cons": "Self-RAG (conservative)", "ecrag_nohop": "EA-CRAG + missing-link prompt, no search",
                "rerank_search": "RAG + Reranker + search without the grader",
                "ecrag_hop_rerun": "EA-CRAG + missing-link search, second run",
                "ecrag_dense": "EA-CRAG, dense candidates",
                "ecrag_perpassage": "EA-CRAG, per-passage grading", "ecrag_nofloor": "EA-CRAG, no top-2 floor",
                "rerank_k2": "RAG + Reranker, top-2", "rerank_k3": "RAG + Reranker, top-3"}


# ---- HotpotQA official answer normalisation / EM / F1
def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, gold):
    p, g = normalize(pred), normalize(gold)
    if p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer"):
        return float(p == g)
    common = collections.Counter(p.split()) & collections.Counter(g.split())
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p.split()), same / len(g.split())
    return 2 * prec * rec / (prec + rec)


def build_table():
    questions = load_questions()
    verdicts = load_verdicts()
    contexts, gold_pids, faithful = load_contexts(), load_gold_pids(), load_faithfulness()

    rows = []
    for (dataset, model, system, qid), rec in load_results().items():
        q = questions[dataset][qid]
        ans = rec["answer"]
        abst = is_abstain(ans)
        golds = [q["answer"]] + q.get("aliases", [])
        # Gold vs. given passages by passage id (MuSiQue has several paragraphs per title);
        # by title only if the record's passages have not been recovered (src/contexts.py)
        pids = contexts.get((dataset, model, system, qid))
        if pids is not None and q["answerable"]:
            gold, ctx = gold_pids[dataset][qid], pids
        else:
            gold, ctx = set(q["gold_titles"]), rec["context_titles"]
        grounded = faithful.get((dataset, qid, ans, tuple(pids))) if pids is not None and not abst else None
        u = rec["usage"]
        rows.append({
            "dataset": dataset, "model": model, "system": system, "id": qid, "answerable": q["answerable"], "type": q["type"],
            "abstain": abst,
            "em": float(not abst and any(normalize(ans) == normalize(g) for g in golds)),
            "f1": 0.0 if abst else max(f1(ans, g) for g in golds),
            "judge": float(not abst and verdicts.get((dataset, qid, ans), False)),
            "grounded": np.nan if grounded is None else float(grounded),  # answered questions only
            "ctx_recall": len(gold & set(ctx)) / len(gold) if q["answerable"] else np.nan,
            "ctx_precision": (len(gold & set(ctx)) / len(ctx) if ctx else np.nan) if q["answerable"] else np.nan,
            "ctx_size": len(ctx),
            "llm_calls": u["llm_calls"], "prompt_tokens": u["prompt_tokens"],
            "completion_tokens": u["completion_tokens"], "latency_s": rec["latency_s"],
        })
    return pd.DataFrame(rows)


def bootstrap_ci(x, n=2000, seed=SEED):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def mcnemar(a, b):
    """Exact McNemar test on paired binary outcomes."""
    b01 = int(((a == 0) & (b == 1)).sum())
    b10 = int(((a == 1) & (b == 0)).sum())
    if b01 + b10 == 0:
        return 1.0
    return binomtest(b01, b01 + b10, 0.5).pvalue


def faithfulness_metrics(g, ans, una):
    """Grounding of answers in the given passages; NaN until every answer has a faithfulness verdict."""
    if g.loc[~g.abstain, "grounded"].isna().any():
        return {"grounded": np.nan, "correct_ungrounded": np.nan, "wrong_grounded": np.nan,
                "halluc_ungrounded": np.nan}
    correct, wrong = ans[ans.judge == 1], ans[(ans.judge == 0) & ~ans.abstain]
    return {
        "grounded": g[~g.abstain].grounded.mean(),                     # share of answers grounded in passages
        "correct_ungrounded": (correct.grounded == 0).mean(),          # correct but from parametric knowledge
        "wrong_grounded": (wrong.grounded == 1).mean(),                # evidence was there: reasoning error
        "halluc_ungrounded": (~una.abstain & (una.grounded == 0)).mean(),  # unanswerable, answered without support
    }


def summarize(df):
    rows = []
    for (dataset, model, system), g in df.groupby(["dataset", "model", "system"]):
        ans, una = g[g.answerable], g[~g.answerable]
        lo, hi = bootstrap_ci(ans.judge)
        rows.append({
            "dataset": dataset, "model": model, "system": system,
            "acc_judge": ans.judge.mean(), "acc_ci_lo": lo, "acc_ci_hi": hi,
            "em": ans.em.mean(), "f1": ans.f1.mean(),
            "false_abstain": ans.abstain.mean(),
            # accuracy among answered questions (Mala et al., 2026 "adjusted accuracy")
            "adj_acc": ans[~ans.abstain].judge.mean() if (~ans.abstain).any() else np.nan,
            "halluc_rate": 1 - una.abstain.mean(),          # answered an unanswerable question
            **faithfulness_metrics(g, ans, una),
            "ctx_recall": ans.ctx_recall.mean(), "ctx_precision": ans.ctx_precision.mean(),
            "ctx_size": g.ctx_size.mean(),
            "llm_calls": g.llm_calls.mean(), "prompt_tokens": g.prompt_tokens.mean(),
            "completion_tokens": g.completion_tokens.mean(), "latency_s": g.latency_s.mean(),
            "n_answerable": len(ans), "n_unanswerable": len(una),
        })
    s = pd.DataFrame(rows)
    s["system"] = pd.Categorical(s.system, SYSTEM_ORDER, ordered=True)
    return s.sort_values(["dataset", "model", "system"])


# Paired comparisons: every system against naive RAG, EA-CRAG against the systems it builds on
# (same reranked candidates as RAG + Reranker; CRAG-style grading), the missing-link search against
# the reranker, EA-CRAG and its own no-search control, and the conservative Self-RAG against naive RAG
PAIRS = [("rerank", "naive"), ("crag", "naive"), ("selfrag_joint", "naive"), ("ecrag", "naive"), (HOP, "naive"),
         ("ecrag", "rerank"), ("ecrag", "crag"), (HOP, "rerank"), (HOP, "ecrag"), (HOP, "ecrag_nohop"),
         ]
# ablation family, corrected separately: each variant against the full EA-CRAG
ABLATION_PAIRS = [(v, "ecrag") for v in ABLATIONS]
# search family, corrected separately (added after the main runs): does a second search need the grader?
SEARCH_PAIRS = [(HOP, "rerank_search"), ("rerank_search", "rerank")]


def paired_diff_ci(a, b, n=2000, seed=SEED):
    """95% bootstrap CI of mean(b) - mean(a) over paired questions."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    return np.percentile(rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1), [2.5, 97.5])


def significance(df):
    rows = []
    for (dataset, model), g in df.groupby(["dataset", "model"]):
        for family, system, vs in ([("main", a, b) for a, b in PAIRS] + [("ablation", a, b) for a, b in ABLATION_PAIRS]
                                   + [("search", a, b) for a, b in SEARCH_PAIRS]):
            base = g[g.system == vs].set_index("id")
            other = g[g.system == system].set_index("id")
            ids = base.index.intersection(other.index)
            if len(ids) == 0:
                continue
            ans_ids = [i for i in ids if base.loc[i, "answerable"]]
            una_ids = [i for i in ids if not base.loc[i, "answerable"]]
            a, b = base.loc[ans_ids, "judge"].values, other.loc[ans_ids, "judge"].values
            lo, hi = paired_diff_ci(a, b)
            rows.append({
                "family": family, "dataset": dataset, "model": model, "system": system, "vs": vs,
                "acc_diff": b.mean() - a.mean(), "acc_diff_lo": lo, "acc_diff_hi": hi,
                "p_accuracy": mcnemar(a, b),
                "p_abstention_unanswerable": mcnemar(base.loc[una_ids, "abstain"].astype(float).values,
                                                     other.loc[una_ids, "abstain"].astype(float).values),
            })
    sig = pd.DataFrame(rows)
    # Holm correction within each family of tests (main comparisons, ablations)
    for col in ["p_accuracy", "p_abstention_unanswerable"]:
        sig[col + "_holm"] = sig.groupby("family")[col].transform(lambda p: holm(p.values))
    return sig


def by_type(df):
    """Accuracy and abstention per question type (answerable questions)."""
    ans = df[df.answerable]
    t = ans.groupby(["dataset", "model", "system", "type"]).agg(
        n=("judge", "size"), acc_judge=("judge", "mean"), false_abstain=("abstain", "mean"),
        ctx_recall=("ctx_recall", "mean")).reset_index()
    t["system"] = pd.Categorical(t.system, SYSTEM_ORDER, ordered=True)
    return t.sort_values(["dataset", "model", "type", "system"])


def holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def figures(s, out):
    out.mkdir(parents=True, exist_ok=True)
    models = list(s.model.unique())
    x = np.arange(len(MAIN))
    w = 0.8 / max(len(models), 1)

    for metric, ylabel, fname in [("acc_judge", "Accuracy (LLM judge), answerable", "accuracy.png"),
                                  ("halluc_rate", "Hallucination rate, unanswerable", "hallucination.png"),
                                  ("ctx_precision", "Context precision (gold passages)", "ctx_precision.png")]:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for i, m in enumerate(models):
            sub = s[s.model == m].set_index("system").reindex(MAIN)
            bars = ax.bar(x + i * w - 0.4 + w / 2, sub[metric], w, label=m)
            if metric == "acc_judge":
                ax.errorbar(x + i * w - 0.4 + w / 2, sub[metric],
                            yerr=[sub[metric] - sub.acc_ci_lo, sub.acc_ci_hi - sub[metric]],
                            fmt="none", ecolor="black", capsize=3, lw=1)
            ax.bar_label(bars, fmt="%.2f", fontsize=7)
        ax.set_xticks(x, [SYSTEM_LABEL[k] for k in MAIN])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=200)
        plt.close(fig)

    # Accuracy vs. cost trade-off
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for m, marker in zip(models, ["o", "s", "^"]):
        sub = s[(s.model == m) & s.system.isin(MAIN)]
        ax.scatter(sub.prompt_tokens + sub.completion_tokens, sub.acc_judge, marker=marker, label=m)
        for _, r in sub.iterrows():
            ax.annotate(SYSTEM_LABEL[r.system], (r.prompt_tokens + r.completion_tokens, r.acc_judge),
                        fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("LLM tokens per question")
    ax.set_ylabel("Accuracy (LLM judge)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_cost.png", dpi=200)
    plt.close(fig)


def judge_agreement():
    """Agreement and Cohen's kappa between the primary (Qwen) and cross (Gemma) judges,
    for the correctness and the faithfulness judge."""
    rows = []
    for task, load in [("correctness", load_verdicts), ("faithfulness", load_faithfulness)]:
        a, b = load(), load("gemma4-12b")
        keys = sorted(a.keys() & b.keys())
        if not keys:
            continue
        x = np.array([a[k] for k in keys], dtype=float)
        y = np.array([b[k] for k in keys], dtype=float)
        po = (x == y).mean()
        pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
        rows.append({"judge": task, "n": len(keys), "agreement": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else 1.0})
    return rows


def main():
    df = build_table()
    df.to_csv(RESULTS_DIR / "per_question.csv", index=False)
    s = summarize(df)
    s.to_csv(RESULTS_DIR / "summary.csv", index=False, float_format="%.4f")
    sig = significance(df)
    sig.to_csv(RESULTS_DIR / "significance.csv", index=False, float_format="%.4g")
    by_type(df).to_csv(RESULTS_DIR / "by_type.csv", index=False, float_format="%.4f")
    for dataset in s.dataset.unique():
        figures(s[s.dataset == dataset], RESULTS_DIR / "figures" / dataset)
    pd.set_option("display.width", 250)
    print(s.round(3).to_string(index=False))
    print()
    print(sig.round(4).to_string(index=False))
    agr = judge_agreement()
    for a in agr:
        print(f"\n{a['judge']} judge agreement (Qwen vs Gemma): n={a['n']}, agreement={a['agreement']:.3f}, "
              f"kappa={a['kappa']:.3f}")
    if agr:
        pd.DataFrame(agr).to_csv(RESULTS_DIR / "judge_agreement.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
