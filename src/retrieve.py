from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from embed import EmbeddingIndex
from chunking import load_config


class HybridRetriever:
    """
    Supports three retrieval strategies configured in config.yaml:
      vector_only    — cosine similarity via ChromaDB
      hybrid         — vector + BM25 scores fused with weighted sum
      hybrid_rerank  — hybrid candidates re-scored by a cross-encoder
    """

    def __init__(self, index: EmbeddingIndex, config: dict):
        self.index = index
        self.config = config
        self._cross_encoder = None  # CrossEncoder, lazy-loaded on first rerank call
        self._build_bm25()

    # ------------------------------------------------------------------
    # BM25 setup
    # ------------------------------------------------------------------

    def _build_bm25(self) -> None:
        # pull every document from the chroma collection so BM25 covers the
        # same corpus as vector search — rebuilt in memory each time because
        # BM25 doesn't have a persistent format worth adding at this scale
        total = self.index.collection.count()
        result = self.index.collection.get(
            limit=total, include=["documents", "metadatas"]
        )
        self._bm25_docs: list[str] = result["documents"]
        self._bm25_metas: list[dict] = result["metadatas"]

        if not self._bm25_docs:
            raise ValueError(
                f"collection '{self.index.collection.name}' is empty — run embed.py first"
            )

        tokenized = [doc.lower().split() for doc in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized)
        print(f"  BM25 index built over {len(self._bm25_docs)} docs")

    # ------------------------------------------------------------------
    # Internal search helpers
    # ------------------------------------------------------------------

    def _normalize(self, scores: list[float]) -> list[float]:
        # min-max normalization so vector and BM25 scores are on the same [0,1] scale
        # before fusion; without this, unbounded BM25 scores dominate
        if not scores:
            return []
        arr = np.array(scores, dtype=float)
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            return [0.0] * len(scores)  # all scores identical — can't distinguish
        return ((arr - lo) / (hi - lo)).tolist()

    def _vector_search(self, query: str, top_k: int, where: dict | None) -> list[dict]:
        return self.index.query(query, n_results=top_k, where=where)

    def _bm25_search(self, query: str, top_k: int, where: dict | None) -> list[dict]:
        scores = self._bm25.get_scores(query.lower().split())

        if where:
            # BM25 has no native metadata filter — zero out non-matching docs
            # so they can't appear in the top-k after argsort
            mask = np.array([
                all(m.get(k) == v for k, v in where.items())
                for m in self._bm25_metas
            ])
            scores = np.where(mask, scores, 0.0)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0.0:
                break  # remaining docs have no BM25 signal for this query
            results.append({
                "text": self._bm25_docs[idx],
                "score": float(scores[idx]),
                **self._bm25_metas[idx],
            })
        return results

    def _fuse(
        self,
        query: str,
        top_k: int,
        where: dict | None,
        vector_weight: float,
        bm25_weight: float,
    ) -> list[dict]:
        # fetch 2× top_k from each source so the merged pool stays large enough
        # after deduplication — a strict top_k from each could lose good candidates
        vec_results = self._vector_search(query, top_k * 2, where)
        bm25_results = self._bm25_search(query, top_k * 2, where)

        # merge into a dict keyed by chunk text; chunks only in one list get
        # score 0.0 on the missing axis (they still benefit from the axis they appear in)
        combined: dict[str, dict] = {}

        vec_norm = self._normalize([r["score"] for r in vec_results])
        for r, s in zip(vec_results, vec_norm):
            combined[r["text"]] = {**r, "vec_score": s, "bm25_score": 0.0}

        bm25_norm = self._normalize([r["score"] for r in bm25_results])
        for r, s in zip(bm25_results, bm25_norm):
            key = r["text"]
            if key in combined:
                combined[key]["bm25_score"] = s
            else:
                combined[key] = {**r, "vec_score": 0.0, "bm25_score": s}

        for entry in combined.values():
            entry["score"] = (
                vector_weight * entry["vec_score"]
                + bm25_weight * entry["bm25_score"]
            )

        return sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    def _rerank(
        self, query: str, candidates: list[dict], rerank_top_k: int, model: str
    ) -> list[dict]:
        if self._cross_encoder is None:
            # cross-encoder is only loaded when hybrid_rerank is first called;
            # it scores (query, doc) pairs jointly — more accurate than cosine
            # similarity but O(k) inference calls, so we run it only on the
            # small hybrid candidate set, not the full corpus.
            # deferred import: a top-level import would load PyTorch at module load time,
            # before HybridRetriever.__init__ runs — that's what crashes ChromaDB on Windows
            from sentence_transformers import CrossEncoder
            print(f"  loading cross-encoder: {model}")
            self._cross_encoder = CrossEncoder(model)

        pairs = [[query, c["text"]] for c in candidates]
        scores = self._cross_encoder.predict(pairs).tolist()

        for c, s in zip(candidates, scores):
            c["rerank_score"] = s

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:rerank_top_k]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        strategy: str = "hybrid_rerank",
        where: dict | None = None,
    ) -> list[dict]:
        """
        Returns ranked chunks for `query`.

        Args:
            query:    natural-language question
            strategy: "vector_only" | "hybrid" | "hybrid_rerank"
            where:    optional ChromaDB metadata filter, e.g. {"ticker": "AAPL"}
        """
        cfg = self.config["retrieval"][strategy]
        top_k = cfg["top_k"]

        if strategy == "vector_only":
            return self._vector_search(query, top_k, where)

        if strategy == "hybrid":
            return self._fuse(
                query, top_k, where, cfg["vector_weight"], cfg["bm25_weight"]
            )

        if strategy == "hybrid_rerank":
            candidates = self._fuse(
                query, top_k, where, cfg["vector_weight"], cfg["bm25_weight"]
            )
            return self._rerank(
                query, candidates, cfg["rerank_top_k"], cfg["reranker_model"]
            )

        raise ValueError(f"unknown retrieval strategy: {strategy}")


if __name__ == "__main__":
    print("1. loading config...")
    config = load_config()
    print("2. loading embedding index (SentenceTransformer + ChromaDB)...")
    index = EmbeddingIndex("fixed", config)  # only finsight_fixed is populated right now
    print("3. building retriever (BM25)...")
    retriever = HybridRetriever(index, config)
    print("4. running queries...")

    query = "What are Apple's main risk factors?"
    for strategy in ["vector_only", "hybrid", "hybrid_rerank"]:
        print(f"\n=== {strategy} ===")
        results = retriever.retrieve(query, strategy=strategy, where={"ticker": "AAPL"})
        for i, r in enumerate(results[:3]):
            print(f"  [{i+1}] score={r['score']:.3f}  section={r.get('section', 'N/A')}")
            print(f"       {r['text'][:120]}...")
