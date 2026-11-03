"""Shared settings: server endpoints, model names, experiment constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# llama-server instances (see scripts/start_servers.ps1)
LLM_URL = "http://127.0.0.1:8080"
EMBED_URL = "http://127.0.0.1:8081"
RERANK_URL = "http://127.0.0.1:8082"

# Generator models, keyed by the short name used on the command line
MODELS = {
    "qwen38-27b": "Qwen3.8-27B-UD-Q3_K_XL.gguf",
    "gemma4-12b": "gemma-4-12b-it-Q8_0.gguf",
}

# Datasets (dev splits, distractor setting); each lives in data/<name>/
DATASETS = ["hotpotqa", "2wiki", "musique"]


def data_dir(dataset):
    return DATA_ROOT / dataset


# Dataset construction
SEED = 42
N_QUESTIONS = 1000         # questions sampled per dataset, balanced over question types
N_UNANSWERABLE = 250       # questions whose gold paragraphs are removed from the corpus

# Retrieval
TOP_K = 5                  # passages handed to the generator
RERANK_CANDIDATES = 20     # first-stage candidates for the reranker system

# Embedding models served on EMBED_URL; queries get the model's recommended instruction prefix
EMBEDDERS = {
    "bge-base": {"file": "bge-base-en-v1.5-f16.gguf", "max_chars": 1500,
                 "query_prefix": "Represent this sentence for searching relevant passages: "},
    "qwen3-emb-0.6b": {"file": "Qwen3-Embedding-0.6B-Q8_0.gguf", "max_chars": 6000,
                       "query_prefix": "Instruct: Given a multi-hop question, retrieve Wikipedia passages "
                                       "that help answer it\nQuery:"},
    "qwen3-emb-4b": {"file": "Qwen3-Embedding-4B-Q8_0.gguf", "max_chars": 6000,
                     "query_prefix": "Instruct: Given a multi-hop question, retrieve Wikipedia passages "
                                     "that help answer it\nQuery:"},
    "harrier-0.6b": {"file": "harrier-oss-v1-0.6b.Q8_0.gguf", "max_chars": 6000,
                     "query_prefix": "Instruct: Given a multi-hop question, retrieve Wikipedia passages "
                                     "that help answer it\nQuery: "},
}
EMBEDDER = "harrier-0.6b"
RERANKER = "jina-v3.5"   # scores are precomputed by src/rerank_cache.py

# Generation
TEMPERATURE = 0.0
ABSTAIN = "I don't know"
