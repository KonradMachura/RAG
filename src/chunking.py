import re

def fixed_sized_chunking(doc_content: str, size: int = 200, overlap: int = 40) -> list[str]:
    """Split a document into fixed size chunks"""
    chunks: list[str] = []
    start: int = 0
    while start < len(doc_content):
        end = start + size + overlap
        chunk: str = doc_content[start:end]

        if end > len(doc_content):
            chunk = doc_content[start:]

        chunks.append(chunk.strip())
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

def semantic_chunking(doc_content):
    """
    Split a document into semantic chunks
    Not splitting based on document structure but based on meaning
    when document is not structured like film transcripts, walls of text.
    Action:
    The script reads sentence by sentence, converts them into vectors and calculates their ‘similarity’.
    As long as the meaning is the same, it groups them together.
    And when there is a decrease in similarity, it makes a ‘cut’ between them.
    """

    print("Method not implemented yet")

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
