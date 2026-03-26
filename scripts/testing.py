from dotenv import load_dotenv
from typing import Callable
from sentence_transformers import SentenceTransformer, util
from src.core import chunking as c, utils as u
from src.core import config as cfg


def test_chunking(docs_contents: list[str], docs_names: list[str],
                  chunking_func: Callable[..., list[str]], **details) -> list[str] | None:

    chunked_doc: list[str] = []

    for doc_content, doc_name in zip(docs_contents, docs_names):
        print(f"--- Chunking {doc_name}, size {len(doc_content)} ---")

        chunked_doc = chunking_func(doc_content, **details)

        chunked_doc_size: int = sum(len(chunk) for chunk in chunked_doc)
        """For fixed size chunking the cumulative size -> doc_size + (chunks_num-1) * overlap"""
        print(f"{len(chunked_doc)} chunks, cumulative size {chunked_doc_size}, average chunk size: {chunked_doc_size / len(chunked_doc)}")
        return chunked_doc


def test_embedding(model: SentenceTransformer):
    load_dotenv()

    sentences: list[str] = [
        "An employee is entitled to 26 days of holiday leave.",
        "Rules for taking days off work.",
        "How to set up a company email account?",
    ]
    query_text: str = "How long is holiday leave?"

    embedding = model.encode(sentences)
    query = model.encode(query_text)

    similarities = util.semantic_search(query, embedding, top_k= cfg.N_RESULTS)

    for item in similarities[0]:
        print(f"{item}")


if __name__ == "__main__":
    load_dotenv()
    docs_contents, docs_names, docs_paths = u.read_docs()
    test_model = SentenceTransformer(cfg.EMBEDDING_MODEL)

    chunks: list[str] = test_chunking(
        docs_contents=docs_contents,
        docs_names=docs_names,
        chunking_func=c.semantic_chunking,
        model=test_model
    )
    if chunks:
        print(chunks[5:10])

    print("--- Embedding test ---")
    test_embedding(model=test_model)