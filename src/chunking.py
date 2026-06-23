import re
from pathlib import Path

import tiktoken
import yaml


# section header pattern for SEC 10-K filings
# matches: "Item 1", "Item 1A", "Item 1B", "Item 7", etc.
SECTION_PATTERN = re.compile(
    r"(Item\s+\d+[A-Z]?\.?\s+[A-Z][^\n]{3,60})", re.IGNORECASE
)


def load_config(config_path: Path = Path("configs/config.yaml")) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def count_tokens(text: str, encoder: tiktoken.Encoding) -> int:
    # returns how many tokens a string contains
    return len(encoder.encode(text))


def split_into_fixed_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    encoder: tiktoken.Encoding,
) -> list[str]:
    # splits text into fixed-size token windows with overlap between chunks
    tokens = encoder.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(encoder.decode(chunk_tokens))
        start += chunk_size - chunk_overlap  # slide window forward, keeping overlap
    return chunks


def split_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    encoder: tiktoken.Encoding,
) -> list[str]:
    # splits on natural boundaries (paragraphs → newlines → sentences → words)
    # keeps chunks under chunk_size tokens
    if count_tokens(text, encoder) <= chunk_size:
        return [text]

    # try splitting points in order of preference
    for separator in ["\n\n", "\n", ". ", " "]:
        parts = text.split(separator)
        if len(parts) > 1:
            break

    chunks = []
    current = ""

    for part in parts:
        candidate = current + separator + part if current else part
        if count_tokens(candidate, encoder) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            # if a single part is still too big, recurse into it
            if count_tokens(part, encoder) > chunk_size:
                chunks.extend(split_recursive(part, chunk_size, chunk_overlap, encoder))
                current = ""
            else:
                current = part

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c.strip()]


def split_by_section(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    encoder: tiktoken.Encoding,
) -> list[tuple[str, str]]:
    # splits on 10-K Item headers first, then recursively chunks within each section
    # returns list of (section_name, chunk_text) tuples
    matches = list(SECTION_PATTERN.finditer(text))

    if not matches:
        # no section headers found — fall back to recursive chunking
        return [("unknown", chunk) for chunk in split_recursive(text, chunk_size, chunk_overlap, encoder)]

    sections = []
    for i, match in enumerate(matches):
        section_name = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        sections.append((section_name, section_text))

    result = []
    for section_name, section_text in sections:
        sub_chunks = split_recursive(section_text, chunk_size, chunk_overlap, encoder)
        for chunk in sub_chunks:
            result.append((section_name, chunk))

    return result


def chunk_document(
    text: str,
    ticker: str,
    year: int,
    strategy: str,
    config: dict,
) -> list[dict]:
    # main entry point: chunks a document and attaches metadata to each chunk
    encoder = tiktoken.get_encoding("cl100k_base")
    cfg = config["chunking"][strategy]
    chunk_size = cfg["chunk_size"]
    chunk_overlap = cfg["chunk_overlap"]

    if strategy == "fixed":
        texts = split_into_fixed_chunks(text, chunk_size, chunk_overlap, encoder)
        return [
            {
                "text": t,
                "ticker": ticker,
                "year": year,
                "section": "unknown",
                "chunk_index": i,
                "strategy": strategy,
            }
            for i, t in enumerate(texts)
        ]

    elif strategy == "recursive":
        texts = split_recursive(text, chunk_size, chunk_overlap, encoder)
        return [
            {
                "text": t,
                "ticker": ticker,
                "year": year,
                "section": "unknown",
                "chunk_index": i,
                "strategy": strategy,
            }
            for i, t in enumerate(texts)
        ]

    elif strategy == "section":
        section_chunks = split_by_section(text, chunk_size, chunk_overlap, encoder)
        return [
            {
                "text": chunk,
                "ticker": ticker,
                "year": year,
                "section": section,
                "chunk_index": i,
                "strategy": strategy,
            }
            for i, (section, chunk) in enumerate(section_chunks)
        ]

    else:
        raise ValueError(f"unknown chunking strategy: {strategy}")


if __name__ == "__main__":
    # quick smoke test against a saved raw file
    config = load_config()
    sample_path = Path("data/raw/AAPL/2023.txt")

    if not sample_path.exists():
        print("run ingest.py first to download raw filings")
    else:
        text = sample_path.read_text(encoding="utf-8")
        for strategy in ["fixed", "recursive", "section"]:
            chunks = chunk_document(text, "AAPL", 2023, strategy, config)
            print(f"{strategy}: {len(chunks)} chunks, first chunk {len(chunks[0]['text'])} chars")
