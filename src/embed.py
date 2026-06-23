from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
import yaml

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer  # only for type checker, not runtime


def load_config(config_path: Path = Path("configs/config.yaml")) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class EmbeddingIndex:
    def __init__(self, strategy: str, config: dict, client=None):
        self._model_name = config["embedding"]["model"]
        self._model: SentenceTransformer | None = None  # lazy-loaded on first encode call

        chroma_path = config["paths"]["chroma_db"]
        # accept a shared client so __main__ can reuse one connection across strategies;
        # ChromaDB allows only one PersistentClient per path per process
        self.client = client if client is not None else chromadb.PersistentClient(path=chroma_path)

        # one collection per chunking strategy so benchmarks stay isolated
        collection_name = f"finsight_{strategy}"
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # use cosine similarity for search
        )

    @property
    def model(self):
        # deferred import + loading keeps ChromaDB init free of PyTorch/CUDA DLLs;
        # on Windows, importing sentence_transformers before ChromaDB native libs
        # causes a silent C-level crash — so we defer both import and instantiation
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def add_chunks(self, chunks: list[dict]) -> None:
        # embeds each chunk and upserts it into chromadb
        # upsert = insert if new, update if id already exists (safe to re-run)
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True).tolist()

        ids = [
            f"{c['ticker']}_{c['year']}_{c['chunk_index']}" for c in chunks
        ]

        # chromadb metadata values must be str, int, float, or bool — not lists
        metadatas = [
            {
                "ticker": c["ticker"],
                "year": c["year"],
                "section": c["section"],
                "chunk_index": c["chunk_index"],
                "strategy": c["strategy"],
            }
            for c in chunks
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"  upserted {len(chunks)} chunks into '{self.collection.name}'")

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        # embeds the query and returns the top-n most similar chunks with metadata
        query_embedding = self.model.encode([query_text]).tolist()[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,  # e.g. {"ticker": "AAPL"} to filter by company
            include=["documents", "metadatas", "distances"],
        )

        # flatten chroma's nested result format into a clean list of dicts
        output = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text": text,
                "score": 1 - dist,  # chroma returns distance; convert to similarity
                **meta,
            })

        return output


if __name__ == "__main__":
    from chunking import chunk_document

    config = load_config()
    raw_dir = Path(config["paths"]["raw_data"])

    # one shared client across all strategies — creating multiple PersistentClient
    # instances pointing to the same path in one process causes a silent crash on Windows
    chroma_client = chromadb.PersistentClient(path=config["paths"]["chroma_db"])

    for strategy in ["fixed", "recursive", "section"]:
        print(f"\n--- building index: {strategy} ---")
        index = EmbeddingIndex(strategy, config, client=chroma_client)

        for ticker_dir in sorted(raw_dir.iterdir()):
            ticker = ticker_dir.name
            txt_files = sorted(ticker_dir.glob("*.txt"))
            if not txt_files:
                print(f"  {ticker}: no files found — run ingest.py first")
                continue
            for txt_file in txt_files:
                year = int(txt_file.stem)
                print(f"  chunking {ticker} {year}...")
                text = txt_file.read_text(encoding="utf-8")
                chunks = chunk_document(text, ticker, year, strategy, config)
                index.add_chunks(chunks)

    print("\ndone — all indexes built")
