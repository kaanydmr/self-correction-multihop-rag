#!/usr/bin/env bash
# The full pipeline of the paper, in order.
#   bash scripts/run_all.sh
# The LLM server (port 8080) must hold the generator named in each step, and the embedder must run
# on port 8081 (scripts/start_servers.ps1 starts both; -OnlyLLM swaps the generator).
set -euo pipefail
cd "$(dirname "$0")/.."
DATASETS="hotpotqa 2wiki musique"
SYSTEMS="naive rerank crag selfrag_joint ecrag ecrag_hop ecrag_nohop rerank_search"
ABLATIONS="ecrag_dense ecrag_perpassage ecrag_nofloor rerank_k2 rerank_k3"

# 1. Questions, corpus and reranker candidates. Only the embedder may run here (the port 8081 line of
#    scripts/start_servers.ps1), because jina-reranker-v3.5 needs the GPU.
for ds in $DATASETS; do
    python src/prepare_data.py --dataset $ds
    python src/rerank_cache.py --dataset $ds --embedder harrier-0.6b --embed-url http://127.0.0.1:8081 \
        --reranker jina-v3.5
done
python src/retrieval_metrics.py

# 2. Answers with Qwen3.8-27B:  scripts/start_servers.ps1 -Model qwen38-27b
for ds in $DATASETS; do
    python src/run_experiment.py --dataset $ds --model qwen38-27b --limit 500 --workers 6 --systems $SYSTEMS $ABLATIONS
    python src/run_experiment.py --dataset $ds --model qwen38-27b --limit 100 --workers 6 --systems ecrag_hop_rerun
done
python src/determinism.py --model qwen38-27b --n 100

# 3. Answers with Gemma-4-12B:  scripts/start_servers.ps1 -Model gemma4-12b -OnlyLLM
for ds in $DATASETS; do
    python src/run_experiment.py --dataset $ds --model gemma4-12b --limit 500 --workers 6 --systems $SYSTEMS
done

# 4. Exact passages of every answer, then the cross-judge (Gemma still loaded)
python src/contexts.py
python src/judge.py --judge gemma4-12b
python src/faithfulness.py --judge gemma4-12b --fraction 0.15

# 5. Primary judge:  scripts/start_servers.ps1 -Model qwen38-27b -OnlyLLM
python src/judge.py
python src/faithfulness.py

# 6. Metrics, tests and figures
python src/analyze.py
