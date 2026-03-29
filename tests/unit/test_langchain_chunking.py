import pytest
from src.core.chunking import langchain_recursive_chunking, langchain_semantic_chunking

def test_langchain_recursive_chunking():
    text = "This is a test document. It has multiple sentences to test the splitter. " * 20
    chunks = langchain_recursive_chunking(text, size=100, overlap=20)
    
    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks) # size + overlap margin

@pytest.mark.skip(reason="Requires HF Embeddings download, might be slow in CI")
def test_langchain_semantic_chunking():
    text = "Artificial Intelligence is great. Machine learning is a subset. I like pizza."
    chunks = langchain_semantic_chunking(text)
    assert len(chunks) > 0
