from sentence_transformers import SentenceTransformer, util
from src.core import config as cfg
import pytest

def test_semantic_search_similarity():
    """Verify that semantic search identifies similar sentences."""
    model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    
    sentences = [
        "An employee is entitled to 26 days of holiday leave.",
        "Rules for taking days off work.",
        "How to set up a company email account?",
    ]
    query_text = "How long is holiday leave?"
    
    # Encode
    embeddings = model.encode(sentences)
    query_embedding = model.encode(query_text)
    
    # Search
    similarities = util.semantic_search(query_embedding, embeddings, top_k=1)
    
    # The first result should be the one about holiday leave (index 0)
    top_hit = similarities[0][0]
    assert top_hit['corpus_id'] == 0
    assert top_hit['score'] > 0.5 # Basic threshold for similarity
