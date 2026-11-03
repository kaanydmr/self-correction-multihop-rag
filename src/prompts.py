"""Prompt templates. All systems share the same answer prompt so differences come from
retrieval control, not wording. Critique prompts follow the Self-RAG reflection tokens
(ISREL / ISSUP / ISUSE) and the CRAG retrieval-evaluator labels."""
from config import ABSTAIN

SYSTEM = "You are a precise question-answering assistant."


def format_passages(passages):
    return "\n\n".join(f"[{i + 1}] {p['title']}: {p['text']}" for i, p in enumerate(passages))


def answer(question, passages):
    ctx = format_passages(passages) if passages else "(no passages)"
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"Answer the question based on the passages below. The needed facts may be spread over several "
            f"passages, so combine them, and tolerate small spelling differences in names. "
            f"Do not use knowledge that is not in the passages; if the passages do not contain the needed "
            f"information, the final answer must be: {ABSTAIN}\n\n"
            f"Passages:\n{ctx}\n\nQuestion: {question}\n\n"
            f"First reason in at most three short sentences, then write the final line as\n"
            f"Answer: <short answer: a few words, a name, a date, or yes/no>"},
    ]


def answer_open(question, passages):
    """Self-RAG-style generation: the original Self-RAG generator is not told to abstain and may draw
    on its own knowledge; whether the answer is supported is left to its ISSUP critique."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"Answer the question using the passages below. The needed facts may be spread over several "
            f"passages, so combine them; if the passages are not enough, you may also use your own knowledge.\n\n"
            f"Passages:\n{format_passages(passages)}\n\nQuestion: {question}\n\n"
            f"First reason in at most three short sentences, then write the final line as\n"
            f"Answer: <short answer: a few words, a name, a date, or yes/no>"},
    ]


# Sent after an answer call that ran out of tokens before its final line (Qwen occasionally keeps
# reasoning in circles at temperature 0), so that every question still gets a parseable answer.
FINISH = "Stop reasoning now. Write only the final line, in the form: Answer: <short answer>"
FINISH_GRADE = ("Stop reasoning now. Write only the label lines, one per passage in the form '[number]: LABEL', "
                "and then the MISSING line.")
FINISH_LABELS = "Stop reasoning now. Write only the label lines, one per passage in the form '[number]: LABEL'."


def parse_answer(text):
    """Extract the short answer after the last 'Answer:' marker."""
    idx = text.rfind("Answer:")
    ans = text[idx + len("Answer:"):] if idx >= 0 else text.strip().splitlines()[-1]
    return ans.strip().strip("*").strip().rstrip(".").strip()


def crag_grade(question, passage):
    """CRAG retrieval evaluator: Correct / Ambiguous / Incorrect."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"Evaluate whether the passage is useful for answering the question. A multi-hop question may need "
            f"several passages, so a passage that provides one necessary piece of information counts as useful.\n\n"
            f"Question: {question}\n\nPassage: {passage['title']}: {passage['text']}\n\n"
            f"Reply with exactly one word: CORRECT (clearly useful), AMBIGUOUS (possibly useful), "
            f"or INCORRECT (not useful)."},
    ]


def crag_rewrite(question):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"The search for the question below returned no useful passages. Rewrite it as a short search query "
            f"that names the key entities and the information needed. Reply with the query only.\n\n"
            f"Question: {question}"},
    ]


def evidence_grade(question, passages):
    """Joint, chain-aware grading of all passages in one call (Evidence-aware CRAG)."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"The question below may need a chain of facts across several passages. First think about which "
            f"facts and entities the answer depends on, including intermediate (bridge) entities that the question "
            f"does not name. Then label every passage:\n"
            f"- NEEDED: states a fact on that chain, including a fact about an intermediate entity\n"
            f"- MAYBE: about an entity on the chain but not clearly useful\n"
            f"- DROP: unrelated to every entity and fact the answer depends on\n\n"
            f"Question: {question}\n\nPassages:\n{format_passages(passages)}\n\n"
            f"Reason in at most three short sentences, then give one line per passage in the form "
            f"'[number]: LABEL', e.g. '[1]: NEEDED'."},
    ]


def evidence_grade_hop(question, passages):
    """EA-CRAG grading that also names the missing link of the chain as a search query."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"The question below may need a chain of facts across several passages. First think about which "
            f"facts and entities the answer depends on, including intermediate (bridge) entities that the question "
            f"does not name. Then label every passage:\n"
            f"- NEEDED: states a fact on that chain, including a fact about an intermediate entity\n"
            f"- MAYBE: about an entity on the chain but not clearly useful\n"
            f"- DROP: unrelated to every entity and fact the answer depends on\n\n"
            f"Question: {question}\n\nPassages:\n{format_passages(passages)}\n\n"
            f"Reason in at most three short sentences, then give one line per passage in the form "
            f"'[number]: LABEL', e.g. '[1]: NEEDED'. Finally, if the NEEDED and MAYBE passages together do not "
            f"yet state every fact the answer depends on, add a last line 'MISSING: <short search query>' asking "
            f"for the first missing fact and naming the entity it is about (a bridge entity found in the passages "
            f"if there is one); otherwise add the line 'MISSING: none'."},
    ]


def selfrag_isrel(question, passage):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"Does the passage provide information relevant to answering the question? A multi-hop question may "
            f"need several passages, so partial but necessary information counts as relevant.\n\n"
            f"Question: {question}\n\nPassage: {passage['title']}: {passage['text']}\n\n"
            f"Reply with exactly one word: RELEVANT or IRRELEVANT."},
    ]


def selfrag_issup(question, passages, ans):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"Check whether the answer is supported by the passages.\n\n"
            f"Passages:\n{format_passages(passages)}\n\nQuestion: {question}\nAnswer: {ans}\n\n"
            f"Reply with exactly one label: FULLY (every part of the answer is stated in or directly follows from "
            f"the passages), PARTIALLY (only some of it is supported), or NO (not supported)."},
    ]


def selfrag_isuse(question, ans):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"Rate how well the answer responds to the question, regardless of whether it is correct. "
            f"Answers are meant to be short (a name, date, number or yes/no); a short answer that gives "
            f"exactly what was asked is fully useful.\n\n"
            f"Question: {question}\nAnswer: {ans}\n\n"
            f"Reply with a single digit from 1 (does not answer it) to 5 (fully and directly answers it)."},
    ]


def faithfulness(question, passages, pred):
    """Evaluation only: is the answer grounded in the passages the generator saw? Worded independently
    of Self-RAG's ISSUP prompt so that the system's own critique is not re-used as the measure."""
    return [
        {"role": "system", "content": "You are a strict grader."},
        {"role": "user", "content":
            f"Passages:\n{format_passages(passages)}\n\nQuestion: {question}\nAnswer: {pred}\n\n"
            f"Is the answer grounded in the passages? It is grounded only if the passages state it or it follows "
            f"from combining facts stated in them, including comparing or calculating with stated dates and numbers. "
            f"Ignore what you know yourself, and ignore whether the answer is actually true.\n"
            f"First, in one short sentence, name the facts from the passages the answer relies on (if any) and do any "
            f"comparison or calculation needed. Then write the final line as\n"
            f"Verdict: SUPPORTED or Verdict: UNSUPPORTED"},
    ]


def parse_verdict(text):
    """Grounded? Read from the last 'Verdict:' line (whole text if missing)."""
    idx = text.upper().rfind("VERDICT:")
    v = text.upper()[idx:] if idx >= 0 else text.upper()
    return "UNSUPPORTED" not in v and "SUPPORTED" in v


def judge(question, gold, pred, aliases=()):
    ref = gold + (f" (also accepted: {'; '.join(aliases)})" if aliases else "")
    return [
        {"role": "system", "content": "You are a strict grader."},
        {"role": "user", "content":
            f"Question: {question}\nReference answer: {ref}\nPredicted answer: {pred}\n\n"
            f"Is the predicted answer semantically equivalent to the reference answer? Minor wording, formatting "
            f"or extra harmless detail is fine; a different entity, number or yes/no is not.\n"
            f"Reply with exactly one word: CORRECT or INCORRECT."},
    ]
