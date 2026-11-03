"""Thin HTTP clients for the three llama-server instances, with usage accounting."""
import math
import time
from dataclasses import dataclass, field

import requests

from config import EMBED_URL, LLM_URL, RERANK_URL, TEMPERATURE

_session = requests.Session()
_expected_model = None


def expect_model(name):
    """Check every response against this model file: a server swap during a run then fails the
    record (it is retried) instead of storing the other model's answer."""
    global _expected_model
    _expected_model = name


@dataclass
class Usage:
    """Per-question cost counters, filled in by every LLM call."""
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_seconds: float = 0.0
    calls: list = field(default_factory=list)  # (step, prompt_tokens, completion_tokens, hit max_tokens)

    def as_dict(self):
        return {"llm_calls": self.llm_calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "llm_seconds": round(self.llm_seconds, 3),
                "steps": [c[0] for c in self.calls], "truncated": [c[0] for c in self.calls if c[3]]}


def _complete(messages, usage: Usage, step: str, max_tokens: int, **extra) -> dict:
    t0 = time.perf_counter()
    r = _session.post(f"{LLM_URL}/v1/chat/completions", json={
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        **extra,
    }, timeout=600)
    r.raise_for_status()
    data = r.json()
    if _expected_model and _expected_model not in data.get("model", ""):
        raise RuntimeError(f"answered by {data.get('model')!r}, expected {_expected_model!r}")
    dt = time.perf_counter() - t0
    u = data.get("usage", {})
    usage.llm_calls += 1
    usage.prompt_tokens += u.get("prompt_tokens", 0)
    usage.completion_tokens += u.get("completion_tokens", 0)
    usage.llm_seconds += dt
    usage.calls.append((step, u.get("prompt_tokens", 0), u.get("completion_tokens", 0),
                        data["choices"][0].get("finish_reason") == "length"))
    return data["choices"][0]


def chat(messages, usage: Usage, step: str, max_tokens=256) -> str:
    return _complete(messages, usage, step, max_tokens)["message"]["content"].strip()


def chat_label(messages, labels, usage: Usage, step: str):
    """A reply restricted by a grammar to exactly one of `labels`, so it can never be cut off or
    unparseable, plus a probability for each label: the first-token probabilities of the labels,
    renormalised over the labels (as Self-RAG scores its reflection tokens)."""
    grammar = "root ::= " + " | ".join(f'"{lab}"' for lab in labels)
    choice = _complete(messages, usage, step, max_tokens=8, grammar=grammar, logprobs=True, top_logprobs=20)
    label = choice["message"]["content"].strip()
    mass = dict.fromkeys(labels, 0.0)
    for t in (choice.get("logprobs") or {}).get("content", [{}])[0].get("top_logprobs", []):
        owners = [lab for lab in labels if t["token"] and lab.startswith(t["token"])]
        if len(owners) == 1:  # a token that starts only one label
            mass[owners[0]] += math.exp(t["logprob"])
    total = sum(mass.values())
    probs = {lab: m / total for lab, m in mass.items()} if total > 0 else {lab: float(lab == label) for lab in labels}
    return label, probs


def embed(texts: list[str], url=EMBED_URL) -> list[list[float]]:
    r = _session.post(f"{url}/v1/embeddings", json={"input": texts}, timeout=600)
    r.raise_for_status()
    return [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]


def rerank(query: str, docs: list[str], url=RERANK_URL) -> list[float]:
    """Returns one relevance score per doc, in input order."""
    r = _session.post(f"{url}/v1/rerank", json={"query": query, "documents": docs}, timeout=600)
    r.raise_for_status()
    scores = [0.0] * len(docs)
    for item in r.json()["results"]:
        scores[item["index"]] = item["relevance_score"]
    return scores


def model_name() -> str:
    r = _session.get(f"{LLM_URL}/v1/models", timeout=30)
    r.raise_for_status()
    return r.json()["data"][0]["id"]
