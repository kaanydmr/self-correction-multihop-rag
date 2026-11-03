# Prompt templates

The exact prompts of all systems and judges, rendered from `src/prompts.py`. Placeholders are in braces; `{numbered passages ...}` stands for the passages, numbered from [1], each as `title: text`.

**Answer generation (all systems)**

```
[system] You are a precise question-answering assistant.

[user] Answer the question based on the passages below. The needed facts
may be spread over several passages, so combine them, and tolerate small
spelling differences in names. Do not use knowledge that is not in the
passages; if the passages do not contain the needed information, the
final answer must be: I don't know

Passages:
{numbered passages: [1] title: text ...}

Question: {question}

First reason in at most three short sentences, then write the final line
as
Answer: <short answer: a few words, a name, a date, or yes/no>
```

**CRAG retrieval evaluator (per passage)**

```
[system] You are a precise question-answering assistant.

[user] Evaluate whether the passage is useful for answering the
question. A multi-hop question may need several passages, so a passage
that provides one necessary piece of information counts as useful.

Question: {question}

Passage: {title}: {text}

Reply with exactly one word: CORRECT (clearly useful), AMBIGUOUS
(possibly useful), or INCORRECT (not useful).
```

**CRAG query rewriting (also the search without grader)**

```
[system] You are a precise question-answering assistant.

[user] The search for the question below returned no useful passages.
Rewrite it as a short search query that names the key entities and the
information needed. Reply with the query only.

Question: {question}
```

**Self-RAG ISREL (per passage; label constrained to RELEVANT or IRRELEVANT)**

```
[system] You are a precise question-answering assistant.

[user] Does the passage provide information relevant to answering the
question? A multi-hop question may need several passages, so partial but
necessary information counts as relevant.

Question: {question}

Passage: {title}: {text}

Reply with exactly one word: RELEVANT or IRRELEVANT.
```

**Self-RAG ISSUP (label constrained to FULLY, PARTIALLY or NO)**

```
[system] You are a precise question-answering assistant.

[user] Check whether the answer is supported by the passages.

Passages:
{numbered passages: [1] title: text ...}

Question: {question}
Answer: {answer}

Reply with exactly one label: FULLY (every part of the answer is stated
in or directly follows from the passages), PARTIALLY (only some of it is
supported), or NO (not supported).
```

**Self-RAG ISUSE (label constrained to 1-5)**

```
[system] You are a precise question-answering assistant.

[user] Rate how well the answer responds to the question, regardless of
whether it is correct. Answers are meant to be short (a name, date,
number or yes/no); a short answer that gives exactly what was asked is
fully useful.

Question: {question}
Answer: {answer}

Reply with a single digit from 1 (does not answer it) to 5 (fully and
directly answers it).
```

**EA-CRAG chain-aware joint grading**

```
[system] You are a precise question-answering assistant.

[user] The question below may need a chain of facts across several
passages. First think about which facts and entities the answer depends
on, including intermediate (bridge) entities that the question does not
name. Then label every passage:
- NEEDED: states a fact on that chain, including a fact about an
  intermediate entity
- MAYBE: about an entity on the chain but not clearly useful
- DROP: unrelated to every entity and fact the answer depends on

Question: {question}

Passages:
{numbered passages: [1] title: text ...}

Reason in at most three short sentences, then give one line per passage
in the form '[number]: LABEL', e.g. '[1]: NEEDED'.
```

**EA-CRAG + MLS grading with missing-link query (also the no-search control)**

```
[system] You are a precise question-answering assistant.

[user] The question below may need a chain of facts across several
passages. First think about which facts and entities the answer depends
on, including intermediate (bridge) entities that the question does not
name. Then label every passage:
- NEEDED: states a fact on that chain, including a fact about an
  intermediate entity
- MAYBE: about an entity on the chain but not clearly useful
- DROP: unrelated to every entity and fact the answer depends on

Question: {question}

Passages:
{numbered passages: [1] title: text ...}

Reason in at most three short sentences, then give one line per passage
in the form '[number]: LABEL', e.g. '[1]: NEEDED'. Finally, if the
NEEDED and MAYBE passages together do not yet state every fact the
answer depends on, add a last line 'MISSING: <short search query>'
asking for the first missing fact and naming the entity it is about (a
bridge entity found in the passages if there is one); otherwise add the
line 'MISSING: none'.
```

**Follow-up when the answer call reaches its token limit**

```
[user] Stop reasoning now. Write only the final line, in the form:
Answer: <short answer>
```

**Follow-up when a grading call reaches its token limit (EA-CRAG; EA-CRAG + MLS)**

```
[user] Stop reasoning now. Write only the label lines, one per passage
in the form '[number]: LABEL'.

[user] Stop reasoning now. Write only the label lines, one per passage
in the form '[number]: LABEL', and then the MISSING line.
```

**Correctness judge**

```
[system] You are a strict grader.

[user] Question: {question}
Reference answer: {reference} (also accepted: {alias})
Predicted answer: {answer}

Is the predicted answer semantically equivalent to the reference answer?
Minor wording, formatting or extra harmless detail is fine; a different
entity, number or yes/no is not.
Reply with exactly one word: CORRECT or INCORRECT.
```

**Groundedness judge**

```
[system] You are a strict grader.

[user] Passages:
{numbered passages: [1] title: text ...}

Question: {question}
Answer: {answer}

Is the answer grounded in the passages? It is grounded only if the
passages state it or it follows from combining facts stated in them,
including comparing or calculating with stated dates and numbers. Ignore
what you know yourself, and ignore whether the answer is actually true.
First, in one short sentence, name the facts from the passages the
answer relies on (if any) and do any comparison or calculation needed.
Then write the final line as
Verdict: SUPPORTED or Verdict: UNSUPPORTED
```
