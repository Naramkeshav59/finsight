"""
FinSight Streamlit dashboard.
Run with: streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import os

import streamlit as st

sys.path.insert(0, "src")

# expose Streamlit secrets as env vars so generate._chat() picks them up
for _key in ("GROQ_API_KEY", "GROQ_MODEL"):
    if _key in st.secrets:
        os.environ[_key] = st.secrets[_key]

from embed import EmbeddingIndex, load_config
from generate import generate, generate_diff
from retrieve import HybridRetriever


def _ensure_indexes(config: dict) -> None:
    """Build all three ChromaDB indexes if they don't exist yet (cloud cold-start)."""
    import chromadb
    from chunking import chunk_document

    chroma_path = config["paths"]["chroma_db"]
    raw_dir = Path(config["paths"]["raw_data"])
    client = chromadb.PersistentClient(path=chroma_path)

    for strategy in ("fixed", "recursive", "section"):
        col = client.get_or_create_collection(f"finsight_{strategy}")
        if col.count() > 0:
            continue  # already built

        st.info(f"Building '{strategy}' index for the first time — this takes ~10 min...")
        index = EmbeddingIndex(strategy, config, client=client)
        for ticker_dir in sorted(raw_dir.iterdir()):
            ticker = ticker_dir.name
            for txt_file in sorted(ticker_dir.glob("*.txt")):
                year = int(txt_file.stem)
                text = txt_file.read_text(encoding="utf-8")
                chunks = chunk_document(text, ticker, year, strategy, config)
                index.add_chunks(chunks)

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(page_title="FinSight", page_icon="📊", layout="wide")

# ------------------------------------------------------------------
# Cached resource loading — only initialises once per session
# ------------------------------------------------------------------

@st.cache_resource
def get_retriever(chunking_strategy: str):
    config = load_config()
    _ensure_indexes(config)
    index = EmbeddingIndex(chunking_strategy, config)
    return HybridRetriever(index, config), config


# ------------------------------------------------------------------
# Sidebar — global controls
# ------------------------------------------------------------------

with st.sidebar:
    st.title("FinSight")
    st.caption("RAG over SEC 10-K filings")
    st.divider()

    chunking = st.selectbox("Chunking strategy", ["section", "fixed", "recursive"])
    retrieval = st.selectbox(
        "Retrieval strategy",
        ["hybrid_rerank", "hybrid", "vector_only"],
    )

    st.divider()
    companies = st.multiselect(
        "Filter by company",
        ["AAPL", "JPM", "JNJ"],
        default=["AAPL"],
    )
    years = st.multiselect(
        "Filter by year",
        [2021, 2022, 2023],
        default=[2021, 2022, 2023],
    )

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------

tab_chat, tab_diff, tab_bench = st.tabs(["Chat", "Year-over-Year Diff", "Benchmark Results"])

# ---- Chat tab ----

with tab_chat:
    st.header("Ask a question")

    query = st.text_input(
        "Question",
        placeholder="What are Apple's main risk factors?",
    )

    if st.button("Ask", disabled=not query):
        retriever, config = get_retriever(chunking)

        # build where clause from sidebar selections
        where: dict | None = None
        if len(companies) == 1:
            where = {"ticker": companies[0]}
        elif companies:
            where = {"ticker": {"$in": companies}}

        with st.spinner("Retrieving and generating..."):
            chunks = retriever.retrieve(query, strategy=retrieval, where=where)
            result = generate(query, chunks, config)

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader(f"Sources ({len(result['sources'])} chunks)")
        for i, chunk in enumerate(result["sources"], 1):
            with st.expander(
                f"[{i}] {chunk['ticker']} {chunk['year']} — {chunk.get('section', 'unknown')}"
            ):
                st.write(chunk["text"])

# ---- Year-over-year diff tab ----

with tab_diff:
    st.header("Year-over-Year Comparison")
    st.caption("Retrieves the same section across two years and asks the LLM what changed.")

    col1, col2 = st.columns(2)
    with col1:
        diff_ticker = st.selectbox("Company", ["AAPL", "JPM", "JNJ"], key="diff_ticker")
        diff_year1 = st.selectbox("Earlier year", [2021, 2022], key="diff_year1")
    with col2:
        diff_year2 = st.selectbox("Later year", [2022, 2023], key="diff_year2")
        diff_section = st.text_input(
            "Section keyword (optional)",
            placeholder="risk factors",
            key="diff_section",
        )

    diff_query = diff_section or "risk factors and business overview"

    if st.button("Compare years", key="diff_btn"):
        retriever, config = get_retriever(chunking)

        with st.spinner("Retrieving both years..."):
            where1 = {"ticker": diff_ticker, "year": diff_year1}
            where2 = {"ticker": diff_ticker, "year": diff_year2}
            chunks1 = retriever.retrieve(diff_query, strategy="hybrid", where=where1)
            chunks2 = retriever.retrieve(diff_query, strategy="hybrid", where=where2)
            result = generate_diff(diff_query, chunks1, chunks2, diff_year1, diff_year2, config)

        st.subheader(f"What changed: {diff_ticker} {diff_year1} → {diff_year2}")
        st.write(result["answer"])

        col_a, col_b = st.columns(2)
        with col_a:
            st.caption(f"{diff_year1} sources")
            for i, c in enumerate(chunks1[:3], 1):
                with st.expander(f"[{i}] {c.get('section', 'unknown')}"):
                    st.write(c["text"])
        with col_b:
            st.caption(f"{diff_year2} sources")
            for i, c in enumerate(chunks2[:3], 1):
                with st.expander(f"[{i}] {c.get('section', 'unknown')}"):
                    st.write(c["text"])

# ---- Benchmark tab ----

with tab_bench:
    st.header("Benchmark Results")
    st.caption("Run `python src/benchmark.py` to generate results, then reload this page.")

    results_path = Path("data/benchmark_results.csv")
    if results_path.exists():
        import pandas as pd
        df = pd.read_csv(results_path)

        st.dataframe(
            df.sort_values("answer_similarity", ascending=False).style.highlight_max(
                subset=["answer_similarity", "context_relevance"],
                color="lightgreen",
            ).highlight_min(
                subset=["avg_latency_s"],
                color="lightgreen",
            ),
            use_container_width=True,
        )

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Answer similarity by strategy")
            pivot = df.pivot(index="chunking", columns="retrieval", values="answer_similarity")
            st.bar_chart(pivot)
        with col2:
            st.subheader("Latency (s) by strategy")
            pivot_lat = df.pivot(index="chunking", columns="retrieval", values="avg_latency_s")
            st.bar_chart(pivot_lat)
    else:
        st.info(
            "No benchmark results found. "
            "Run `python src/benchmark.py` first, then reload this page."
        )
