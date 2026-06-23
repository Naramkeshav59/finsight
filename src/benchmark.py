"""
Runs all chunking × retrieval strategy combinations against the eval set
and outputs a comparison table saved to data/benchmark_results.csv.

Run after embed.py has built all three indexes (fixed, recursive, section).
"""
from __future__ import annotations

import sys
from pathlib import Path

import chromadb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from embed import EmbeddingIndex, load_config
from eval import load_eval_set, run_eval
from retrieve import HybridRetriever

CHUNKING_STRATEGIES = ["fixed", "recursive", "section"]
RETRIEVAL_STRATEGIES = ["vector_only", "hybrid", "hybrid_rerank"]


def run_benchmark(config: dict, eval_set: list[dict]) -> pd.DataFrame:
    rows = []

    # share one ChromaDB client across all EmbeddingIndex instances
    chroma_client = chromadb.PersistentClient(path=config["paths"]["chroma_db"])

    for chunking in CHUNKING_STRATEGIES:
        print(f"\n=== chunking: {chunking} ===")
        index = EmbeddingIndex(chunking, config, client=chroma_client)

        for retrieval in RETRIEVAL_STRATEGIES:
            print(f"  retrieval: {retrieval} ...", end=" ", flush=True)
            retriever = HybridRetriever(index, config)

            scores = run_eval(eval_set, retriever, retrieval, config)
            scores["chunking"] = chunking
            scores["retrieval"] = retrieval
            rows.append(scores)
            print(f"answer_similarity={scores['answer_similarity']:.3f}  latency={scores['avg_latency_s']:.1f}s")

    cols = ["chunking", "retrieval", "answer_similarity", "context_relevance", "avg_latency_s"]
    return pd.DataFrame(rows)[cols]


if __name__ == "__main__":
    config = load_config()
    eval_set = load_eval_set()
    print(f"loaded {len(eval_set)} eval samples")
    print(f"running {len(CHUNKING_STRATEGIES) * len(RETRIEVAL_STRATEGIES)} combinations...\n")

    results = run_benchmark(config, eval_set)

    out_path = Path("data/benchmark_results.csv")
    results.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")
    print("\n" + results.sort_values("answer_similarity", ascending=False).to_string(index=False))
