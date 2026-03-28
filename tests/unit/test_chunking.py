from typing import Callable

import pytest
from src.core.chunking import fixed_sized_chunking, subsection_chunking, paragraph_chunking, semantic_chunking
from sentence_transformers import SentenceTransformer
from src.core import config as cfg

def test_fixed_sized_chunking():
    text = "0123456789" * 10 # 100 chars
    chunks = fixed_sized_chunking(text, size=20, overlap=5)
    
    # Text length 100, size 20, overlap 5
    # C1: 0-25 (0 to 20+5)
    # C2: 20-45 (20 to 40+5)
    # etc.
    assert len(chunks) > 1
    assert chunks[0] == text[0:25]

def test_subsection_chunking():
    text = "Intro\n## Section 1\nDetails\n### Subsection 1.1\nMore details"
    chunks = subsection_chunking(text)
    assert len(chunks) == 3
    assert chunks[0] == "Intro"
    assert "## Section 1" in chunks[1]
    assert "### Subsection 1.1" in chunks[2]

def test_paragraph_chunking():
    text = "Intro\n## Section 1\nDetails\n## Section 2\nMore details"
    chunks = paragraph_chunking(text)
    assert len(chunks) == 3
    assert "## Section 1" in chunks[1]
    assert "## Section 2" in chunks[2]

def test_semantic_chunking():
    # Use a very small model for testing if possible, or the default one if it's already downloaded
    model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    text = "This is a sentence about cats. Cats are fluffy. This is a sentence about dogs. Dogs bark at mailmen."
    chunks = semantic_chunking(text, model, threshold=0.5)
    
    assert len(chunks) > 0
    # The output depends on the model's embeddings, but we can verify it returns strings
    assert isinstance(chunks[0], str)
    assert len(" ".join(chunks)) >= len(text)


def _run_chunking_helper(docs_contents: list[str], docs_names: list[str],
                  chunking_func: Callable[..., list[str]], **details) -> list[str] | None:

    chunked_doc: list[str] = []

    for doc_content, doc_name in zip(docs_contents, docs_names):
        print(f"--- Chunking {doc_name}, size {len(doc_content)} ---")

        chunked_doc = chunking_func(doc_content, **details)

        chunked_doc_size: int = sum(len(chunk) for chunk in chunked_doc)
        """For fixed size chunking the cumulative size -> doc_size + (chunks_num-1) * overlap"""
        print(f"{len(chunked_doc)} chunks, cumulative size {chunked_doc_size}, average chunk size: {chunked_doc_size / len(chunked_doc)}")
        return chunked_doc
