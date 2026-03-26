import re
import sentence_transformers
from sentence_transformers import util, SentenceTransformer

from src.core import config as cfg


def fixed_sized_chunking(doc_content: str,
                         size: int = cfg.DEFAULT_CHUNK_SIZE, overlap: int = cfg.DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """Split a document into fixed size chunks"""
    chunks: list[str] = []
    start: int = 0
    while start < len(doc_content):
        end = start + size + overlap
        chunk: str = doc_content[start:end]

        if end > len(doc_content):
            chunk = doc_content[start:]

        chunks.append(chunk.strip())
        print(chunk)
        print('-' * 40)
        start = end - overlap

    return chunks


def subsection_chunking(doc_content: str) -> list[str]:
    """Split a document into subsection chunks"""
    chunks: list[str] = []
    chunks = re.split(r"(?=#{2,3})", doc_content)
    chunks = [chunk.strip(" \n\r\t-") for chunk in chunks if chunk.strip()]

    return chunks


def paragraph_chunking(doc_content: str) -> list[str]:
    """Split a document into paragraph/section chunks"""
    chunks: list[str] = []
    chunks = re.split(r"(?=#{2})", doc_content)
    chunks = [chunk.strip(" \n\r\t-") for chunk in chunks if chunk.strip()]
    return chunks

def _extract_sentences(doc_content: str) -> list[str]:
    raw_sentences = re.split(r"(?<=[.?!])\s+", doc_content)
    return [s.strip() for s in raw_sentences if s.strip()]

def semantic_chunking(doc_content: str, model: SentenceTransformer, threshold: float = cfg.SEMANTIC_THRESHOLD) -> list[str]:
    """
    Split a document into semantic chunks
    Not splitting based on document structure but based on meaning
    when document is not structured like film transcripts, walls of text.
    Action:
    The script reads sentence by sentence, converts them into vectors and calculates their ‘similarity’.
    As long as the meaning is the same, it groups them together.
    And when there is a decrease in similarity, it makes a ‘cut’ between them.
    """

    sentences: list[str] = _extract_sentences(doc_content)
    if not sentences:
        return []

    # model = sentence_transformers.SentenceTransformer(cfg.EMBEDDING_MODEL)
    sentences_vectors = model.encode(sentences)
    chunks: list[str] = []

    chunk = sentences[0]

    for i in range(len(sentences_vectors) - 1):

        cosine_sim = util.cos_sim(sentences_vectors[i], sentences_vectors[i+1]).item()
        distance = 1.0 - cosine_sim
        if distance <= threshold:
            chunk += " " + sentences[i+1]
        else:
            chunks.append(chunk)
            chunk = sentences[i+1]

    if chunk:
        chunks.append(chunk)

    return chunks

def agentic_chunking(doc_content):
    """
    Partition is made by LLM also based on semantic.
    Pros:
        - Standalone Context
        - Propositional Chunking
        - Elastic Boundaries
        - Augmenting with metadata
    Cons:
        - Expenses
        - Time
    """
    print("Method not implemented yet")
