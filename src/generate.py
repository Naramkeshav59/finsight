from __future__ import annotations

import os
from pathlib import Path

import yaml


def load_config(config_path: Path = Path("configs/config.yaml")) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _chat(model: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
    """Dispatch to Groq (if GROQ_API_KEY set) or local Ollama."""
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        from openai import OpenAI
        client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        # Groq model names differ from Ollama; map or fall back to a fast default
        groq_model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
        resp = client.chat.completions.create(
            model=groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    else:
        import ollama as _ollama
        resp = _ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return resp["message"]["content"]


# system prompt is constant — no user input goes here, so no injection risk
SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer questions using ONLY the provided "
    "context excerpts from SEC 10-K filings. If the answer cannot be found in the "
    'provided excerpts, say "I don\'t have enough information in the provided excerpts '
    'to answer this." Do not use outside knowledge. When citing information, reference '
    'the excerpt number, e.g. "According to [2]...".'
)


def _build_context_block(chunks: list[dict]) -> str:
    # formats retrieved chunks as a numbered list with source metadata
    # numbering lets the LLM cite specific excerpts in its answer
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        source = f"{chunk['ticker']}, {chunk['year']}, {chunk.get('section', 'unknown')}"
        lines.append(f"[{i}] Source: {source}")
        lines.append(chunk["text"].strip())
        lines.append("")  # blank line between excerpts for readability
    return "\n".join(lines)


def generate(query: str, chunks: list[dict], config: dict) -> dict:
    """
    Builds a grounded RAG prompt from retrieved chunks, sends it to Ollama,
    and returns the answer with the source chunks used.

    Returns: {"answer": str, "sources": list[dict]}
    """
    cfg = config["generation"]

    user_message = f"Context:\n{_build_context_block(chunks)}\nQuestion: {query}"

    answer = _chat(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
    )

    return {"answer": answer, "sources": chunks}


def generate_diff(
    topic: str,
    chunks_year1: list[dict],
    chunks_year2: list[dict],
    year1: int,
    year2: int,
    config: dict,
) -> dict:
    """
    Compares the same section across two filing years and summarises what changed.
    Chunks from each year are presented as separate labelled blocks so the LLM
    can reason about the delta between them.

    Returns: {"answer": str, "sources": list[dict]}
    """
    cfg = config["generation"]

    block1 = _build_context_block(chunks_year1)
    block2 = _build_context_block(chunks_year2)

    user_message = (
        f"{year1} excerpts:\n{block1}\n"
        f"{year2} excerpts:\n{block2}\n"
        f"Topic: {topic}\n"
        "Summarise the key differences between the two years. "
        "Be specific about what was added, removed, or changed."
    )

    answer = _chat(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
    )

    return {"answer": answer, "sources": chunks_year1 + chunks_year2}


if __name__ == "__main__":
    from embed import EmbeddingIndex
    from retrieve import HybridRetriever

    config = load_config()
    index = EmbeddingIndex("fixed", config)
    retriever = HybridRetriever(index, config)

    query = "What are Apple's main risk factors?"
    chunks = retriever.retrieve(query, strategy="hybrid_rerank", where={"ticker": "AAPL"})

    result = generate(query, chunks, config)

    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources used ({len(result['sources'])}):")
    for i, s in enumerate(result["sources"], 1):
        print(f"  [{i}] {s['ticker']} {s['year']} — {s.get('section', 'unknown')}")
