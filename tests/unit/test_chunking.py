import pytest
from src.core.chunking import fixed_sized_chunking, subsection_chunking, paragraph_chunking

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
