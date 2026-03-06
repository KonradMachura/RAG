from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv
import utils as u
import chunking as c
import chromadb
from chromadb.utils import embedding_functions
import config as cfg

def chunk_and_save_document(collection: Collection, doc_content: str, doc_name: str):
    chunks: list[str] = c.semantic_chunking(doc_content)
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


def main():
    load_dotenv()
    """ sentence_transformer_ef is using Squared L2 for sentence embedding so lower results -> higher similarity
        First model is symmetric so it was trained to search sentences that mean more or less the same.
        Second model is asymmetric. It means it was designed for search tasks where the query 
        and the target documents are fundamentally different in structure, length, or intent.
        Instead of looking for a semantic "mirror image" (which is what symmetric models do)
        , they act like a "key and lock".
    """
    collection = configure_chroma_db()

    docs_contents, docs_names, docs_paths = u.read_docs()
    for i, (doc_name, doc_content) in enumerate(zip(docs_names, docs_contents)):
        chunk_and_save_document(collection, doc_content, doc_name)

    print("\nDatabase has been updated and saved")

if __name__ == '__main__':
    main()


