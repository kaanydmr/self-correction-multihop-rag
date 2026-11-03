# Self-Correction with CRAG and Self-RAG in Multi-Hop RAG

Code, prompts and results for the paper

> K. Aydemir, "Self-Correction with CRAG and Self-RAG in Multi-Hop RAG: Retrieval Quality Control and Answer Accuracy," Department of Artificial Intelligence and Data Engineering, Ankara University, 2026.

The paper compares training-free (prompted) CRAG and Self-RAG with naive RAG and a reranker on HotpotQA, 2WikiMultiHopQA and MuSiQue, with two open generators (Qwen3.8-27B and Gemma-4-12B) and questions whose evidence was removed from the corpus. It also proposes EA-CRAG, which grades the reranked passages jointly with a chain-aware prompt, and EA-CRAG + MLS, which lets the grader name the first missing fact as a search query (missing-link search).

## Contents

| Path | What it holds |
|---|---|
| `src/` | Data preparation, retrieval, the compared systems, the judges and the analysis |
| `scripts/start_servers.ps1` | Starts the llama.cpp servers (generator on port 8080, embedder on 8081) |
| `scripts/run_all.sh` | The full pipeline in order |
| `PROMPTS.md` | The exact prompt templates of all systems and judges |
| `supplementary/Supplementary_Material.docx` | Selection of the retrieval components, full results per dataset, generator and system, and the prompts |
| `data/<dataset>/sample.json` | IDs of the 1,000 sampled questions per dataset (the first 500 are evaluated) |
| `data/<dataset>/rerank_harrier-0.6b_jina-v3.5.json` | Cached reranker candidates for the original questions |
| `results/<dataset>/<generator>/<system>.jsonl` | Every answer, with the passages given to the generator, the grading trace and the cost |
| `results/judge*.jsonl`, `results/faithfulness*.jsonl` | Correctness and groundedness verdicts of both judges |
| `results/*.csv`, `results/figures/` | Metrics, significance tests and figures computed by `src/analyze.py` |

## Systems

| Name in the code | Name in the paper |
|---|---|
| `naive` | Naive RAG |
| `rerank` | RAG + Reranker |
| `crag` | CRAG (prompted) |
| `selfrag_joint` | Self-RAG (prompted) |
| `ecrag` | EA-CRAG |
| `ecrag_hop` | EA-CRAG + MLS |
| `ecrag_nohop` | No-search control |
| `rerank_search` | Search without grader |
| `ecrag_hop_rerun` | Second run of EA-CRAG + MLS (first 100 questions per dataset, Qwen) |
| `ecrag_dense`, `ecrag_perpassage`, `ecrag_nofloor` | Ablation of EA-CRAG (dense candidates, per-passage grading, no top-2 floor) |
| `rerank_k2`, `rerank_k3` | The reranker's top-2 or top-3 without grading |

`src/systems.py` also contains variants that are not part of the paper (`selfrag`, `selfrag_open`, `selfrag_open_joint`, `selfrag_cons`, `ecrag_hop_dense`).

## Setup

- Python (tested with 3.14) and the packages in `requirements.txt`.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (build b11982): put `llama-server` in `tools/llama.cpp/`.
- Models in `models/`:
  - [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), GGUF `Qwen3.8-27B-UD-Q3_K_XL.gguf` from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
  - [gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it), GGUF `gemma-4-12b-it-Q8_0.gguf`
  - [harrier-oss-v1-0.6b](https://huggingface.co/microsoft/harrier-oss-v1-0.6b), GGUF `harrier-oss-v1-0.6b.Q8_0.gguf`
  - jina-reranker-v3.5 (Hugging Face weights, non-commercial licence) in `models/jina-reranker-v3.5/`
- Datasets (development sets) in `data/`:
  - `data/hotpotqa/dev.parquet`: [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa), distractor setting, validation split
  - `data/2wiki/dev.parquet`: [2WikiMultiHopQA](https://huggingface.co/datasets/xanhho/2WikiMultihopQA), dev split
  - `data/musique/dev.jsonl`: [MuSiQue](https://github.com/StonyBrookNLP/musique), `musique_ans_v1.0_dev.jsonl`

`src/config.py` holds the server addresses, model files and experiment constants (seed 42, 1,000 questions per dataset of which 250 unanswerable, top-5 passages, top-20 reranker candidates, temperature 0).

## Reproducing the results

`scripts/run_all.sh` runs every step in order; the comments in it say which generator must be loaded on the LLM server for each step. In short:

1. `src/prepare_data.py` samples the questions, builds the corpus and removes the evidence of the unanswerable questions; `src/rerank_cache.py` caches the reranker candidates.
2. `src/run_experiment.py` runs the systems with each generator.
3. `src/contexts.py` recovers the exact passages of every answer; `src/judge.py` and `src/faithfulness.py` run the correctness and groundedness judges (Gemma as cross-judge, Qwen as primary judge).
4. `src/analyze.py` computes the metrics, the McNemar tests with Holm correction, the bootstrap intervals and the figures.

All experiments ran on a single 16 GB GPU; the 31,800 answers and both judges took about 34 hours. GPU inference is not bit-exact, so a new run reproduces most but not all answers (Section 6 of the paper).

Table S1 of the supplementary material (the choice of embedder and reranker) comes from `src/compare_retrieval.py`, with each candidate component served on its own port.

The human check of the correctness judge (60 items, labelled blind) is summarised in `results/human_validation.csv`.

## Licence

The code is released under the MIT licence (`LICENSE`). The datasets and models keep their own licences.
