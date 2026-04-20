import re
from typing import Optional
from sentence_transformers import util, SentenceTransformer

from src.core import config as cfg


def fixed_sized_chunking(
    doc_content: str,
    size: int = cfg.DEFAULT_CHUNK_SIZE,
    overlap: int = cfg.DEFAULT_CHUNK_OVERLAP
) -> list[str]:
    """
    Split a document into fixed size chunks with optional overlap.

    Args:
        doc_content: The full text of the document to be chunked.
        size: The maximum size of each chunk.
        overlap: The number of characters to overlap between adjacent chunks.

    Returns:
        A list of text chunks.
    """
    chunks: list[str] = []
    start: int = 0
    content_length = len(doc_content)

    while start < content_length:
        end = start + size + overlap
        if end > content_length:
            chunk = doc_content[start:]
        else:
            chunk = doc_content[start:end]

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks


def _chunk_by_regex(doc_content: str, pattern: str) -> list[str]:
    """
    Helper function to split text by a given regex pattern.

    Args:
        doc_content: The text to split.
        pattern: The regex pattern to split by.

    Returns:
        A list of non-empty stripped chunks.
    """
    chunks = re.split(pattern, doc_content)
    return [chunk.strip(" \n\r\t-") for chunk in chunks if chunk.strip()]


def subsection_chunking(doc_content: str, max_size: int = cfg.DEFAULT_CHUNK_SIZE * 5) -> list[str]:
    """
    Split a document into chunks based on Markdown subsection headers (## or ###).
    If a subsection is too large, it is further split into fixed-size chunks.

    Args:
        doc_content: The Markdown content to be chunked.
        max_size: Maximum size of a chunk before further splitting.

    Returns:
        A list of subsection chunks.
    """
    # Use lookahead to split at ## or ### without removing the header itself
    pattern = r"(?<!#)(?=#{2,3}(?!#))"
    subsections = _chunk_by_regex(doc_content, pattern)
    
    final_chunks = []
    for sub in subsections:
        if len(sub) > max_size:
            # Further split large subsections using fixed-size chunking
            final_chunks.extend(fixed_sized_chunking(sub, size=max_size, overlap=cfg.DEFAULT_CHUNK_OVERLAP))
        else:
            final_chunks.append(sub)
    return final_chunks


def paragraph_chunking(doc_content: str, max_size: int = cfg.DEFAULT_CHUNK_SIZE * 10) -> list[str]:
    """
    Split a document into chunks based on Markdown section headers (##).
    If a section is too large, it is further split into fixed-size chunks.

    Args:
        doc_content: The Markdown content to be chunked.
        max_size: Maximum size of a chunk before further splitting.

    Returns:
        A list of section chunks.
    """
    pattern = r"(?<!#)(?=#{2}(?!#))"
    sections = _chunk_by_regex(doc_content, pattern)
    
    final_chunks = []
    for sec in sections:
        if len(sec) > max_size:
            final_chunks.extend(fixed_sized_chunking(sec, size=max_size, overlap=cfg.DEFAULT_CHUNK_OVERLAP))
        else:
            final_chunks.append(sec)
    return final_chunks


def _extract_sentences(doc_content: str) -> list[str]:
    """
    Split text into individual sentences based on punctuation followed by whitespace.

    Args:
        doc_content: The text content.

    Returns:
        A list of sentences.
    """
    raw_sentences = re.split(r"(?<=[.?!])\s+", doc_content)
    return [s.strip() for s in raw_sentences if s.strip()]


def semantic_chunking(
    doc_content: str,
    model: SentenceTransformer,
    threshold: float = cfg.SEMANTIC_THRESHOLD
) -> list[str]:
    """
    Split a document into chunks based on semantic similarity of sentences.

    This method groups adjacent sentences as long as their semantic similarity
    remains above a certain threshold.

    Args:
        doc_content: The text content to be chunked.
        model: A SentenceTransformer model used for embedding.
        threshold: The maximum semantic distance (1 - cosine similarity) allowed
            between adjacent sentences in the same chunk.

    Returns:
        A list of semantically coherent chunks.
    """
    sentences = _extract_sentences(doc_content)
    if not sentences:
        return []

    sentences_vectors = model.encode(sentences)
    chunks: list[str] = []

    current_chunk = sentences[0]

    for i in range(len(sentences_vectors) - 1):
        cosine_sim = util.cos_sim(sentences_vectors[i], sentences_vectors[i+1]).item()
        distance = 1.0 - cosine_sim

        if distance <= threshold:
            current_chunk += " " + sentences[i+1]
        else:
            chunks.append(current_chunk)
            current_chunk = sentences[i+1]

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def agentic_chunking(doc_content: str) -> list[str]:
    """
    Placeholder for LLM-based agentic chunking.

    This method will use an LLM to identify logical boundaries and provide context
    aware chunking.

    Args:
        doc_content: The text content to be chunked.

    Returns:
        Currently returns an empty list as it's not implemented yet.
    """
    # TODO: Implement agentic chunking using an LLM provider.
    return []
