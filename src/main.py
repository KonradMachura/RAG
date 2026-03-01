from dotenv import load_dotenv
import utils as u
import chunking as c
import chromadb
from chromadb.utils import embedding_functions

def main():
    load_dotenv()
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction('all-MiniLM-L6-v2')

    chroma_client = chromadb.PersistentClient(path="./data")
    collection = chroma_client.get_or_create_collection(
        name='company_docs',
        embedding_function=sentence_transformer_ef)

    docs_contents, docs_names, docs_paths = u.read_docs()
    for i, (doc_name, doc_content) in enumerate(zip(docs_names, docs_contents)):
        # print(f"i={i}, doc_name={doc_name}, doc_content={doc_content[:10]}")
        chunks = c.paragraph_chunking(doc_content)
        chunks_ids = [f"chunk_{j}_{doc_name}" for j in range(len(chunks))]
        metadatas = [ {"source": doc_name} for _ in chunks]

        print(f"Adding {len(chunks_ids)} chunks from {doc_name}")
        collection.upsert(
            ids=chunks_ids,
            documents=chunks,
            metadatas=metadatas
        )

    print("\nDatabase has been updated and saved")

if __name__ == '__main__':
    main()


