"""
Embedding-based evaluation harness.

Metrics:
  answer_similarity — cosine(embed(answer), embed(ground_truth))
  context_relevance — cosine(embed(query), embed(top retrieved chunk))
  avg_latency_s     — average retrieval + generation time per question

Reuses the SentenceTransformer already loaded by EmbeddingIndex — no extra
models, no LLM judge, no structured-output parsing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import yaml


def load_config(config_path: Path = Path("configs/config.yaml")) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_eval_set(path: Path = Path("data/eval_set.json")) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def run_eval(
    eval_set: list[dict],
    retriever,
    retrieval_strategy: str,
    config: dict,
) -> dict:
    """
    Runs retrieval + generation for every Q&A pair in eval_set, then scores
    with embedding cosine similarity against ground-truth answers.

    Returns a flat dict of metric scores plus average latency.
    """
    from generate import generate

    embed_model = retriever.index.model  # reuse already-loaded SentenceTransformer

    answer_sims: list[float] = []
    context_rels: list[float] = []
    latencies: list[float] = []

    for item in eval_set:
        question = item["question"]
        ground_truth = item["ground_truth"]
        where = {"ticker": item["ticker"]}

        t0 = time.time()
        chunks = retriever.retrieve(question, strategy=retrieval_strategy, where=where)
        result = generate(question, chunks, config)
        latencies.append(time.time() - t0)

        if chunks:
            q_emb = embed_model.encode(question, normalize_embeddings=True)
            c_emb = embed_model.encode(chunks[0]["text"], normalize_embeddings=True)
            context_rels.append(_cosine(q_emb, c_emb))
        else:
            context_rels.append(0.0)

        a_emb = embed_model.encode(result["answer"], normalize_embeddings=True)
        gt_emb = embed_model.encode(ground_truth, normalize_embeddings=True)
        answer_sims.append(_cosine(a_emb, gt_emb))

    def _mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else float("nan")

    return {
        "answer_similarity": _mean(answer_sims),
        "context_relevance": _mean(context_rels),
        "avg_latency_s": round(sum(latencies) / len(latencies), 2),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from embed import EmbeddingIndex
    from retrieve import HybridRetriever

    config = load_config()
    eval_set = load_eval_set()
    print(f"loaded {len(eval_set)} eval samples")

    index = EmbeddingIndex("fixed", config)
    retriever = HybridRetriever(index, config)

    scores = run_eval(eval_set, retriever, "hybrid_rerank", config)

    print("\nEval scores (fixed + hybrid_rerank):")
    for k, v in scores.items():
        print(f"  {k}: {v}")
