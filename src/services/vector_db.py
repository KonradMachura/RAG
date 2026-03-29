from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from src.core import utils as u
from src.core import chunking as c
from src.core import config as cfg

from src.services.langchain_service import get_langchain_rag

def save_chunks_to_vectordb(collection: Collection, chunks: list[str], doc_name: str):
    if not chunks:
        print(f"No chunks found for {doc_name}")
        return

    chunks_ids: list[str] = [f"chunk_{j}_{doc_name}" for j in range(len(chunks))]
    metadatas: list[dict[str, str]] = [{"source": doc_name} for _ in chunks]

    print(f"Adding {len(chunks_ids)} chunks from {doc_name}")
    collection.upsert(
        ids=chunks_ids,
        documents=chunks,
        metadatas=metadatas
    )

def configure_chroma_db() -> Collection:
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(cfg.EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=str(cfg.DB_PATH))
    collection = chroma_client.get_or_create_collection(
        name=cfg.DB_NAME,
        embedding_function=embedding_function)
    return collection

def process_with_langchain(doc_names: list[str], docs_contents: list[str], reset: bool = False):
    rag = get_langchain_rag()
    if reset:
        print(f"Resetting LangChain collection: {cfg.LC_COLLECTION_NAME}")
        rag.vector_store.delete_collection()
        # Re-initialize to create a fresh one
        rag = get_langchain_rag()

    for name, content in zip(doc_names, docs_contents):
        # Using recursive chunking for LangChain
        chunks = c.langchain_recursive_chunking(content)
        metadatas = [{"source": name} for _ in chunks]
        print(f"Adding {len(chunks)} chunks from {name} using LangChain")
        rag.add_documents(chunks, metadatas)

def main(use_langchain: bool = False, reset: bool = False):
    load_dotenv()
    """ sentence_transformer_ef is using Squared L2 for sentence embedding so lower results -> higher similarity
        First model is symmetric so it was trained to search sentences that mean more or less the same.
        Second model is asymmetric. It means it was designed for search tasks where the query 
        and the target documents are fundamentally different in structure, length, or intent.
        Instead of looking for a semantic "mirror image" (which is what symmetric models do)
        , they act like a "key and lock".
    """
    collection = configure_chroma_db()
    chunking_model = SentenceTransformer(cfg.EMBEDDING_MODEL)

    docs_contents, docs_names, docs_paths = u.read_docs()

    if use_langchain:
        process_with_langchain(docs_names, docs_contents, reset=reset)
    else:
        for i, (doc_name, doc_content) in enumerate(zip(docs_names, docs_contents)):
            chunks = c.semantic_chunking(doc_content, model=chunking_model)
            save_chunks_to_vectordb(collection, chunks, doc_name)

    print("\nDatabase has been updated and saved")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--langchain", action="store_true", help="Use LangChain processing")
    parser.add_argument("--reset", action="store_true", help="Reset LangChain collection before processing")
    args = parser.parse_args()
    main(use_langchain=args.langchain, reset=args.reset)


