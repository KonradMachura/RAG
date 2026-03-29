import pytest
import os
from src.services.langchain_service import get_langchain_rag

@pytest.fixture
def rag_service():
    return get_langchain_rag()

def test_add_and_query_langchain(rag_service):
    # Skip if no API key
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")
        
    texts = ["The capital of France is Paris.", "The capital of Germany is Berlin."]
    metadatas = [{"source": "france.pdf"}, {"source": "germany.pdf"}]
    
    rag_service.add_documents(texts, metadatas)
    
    response = rag_service.query("What is the capital of France?")
    
    assert "Paris" in response["answer"]
    assert len(response["source_documents"]) > 0
    assert response["source_documents"][0]["metadata"]["source"] == "france.pdf"
