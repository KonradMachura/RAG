from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

def main():
    load_dotenv()
    """sentence_transformer_ef is using Squared L2 for sentence embedding so lower results -> higher similarity"""
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction('all-MiniLM-L6-v2')
    chroma_client = chromadb.PersistentClient(path="./data")

    collection = chroma_client.get_collection(
        name='company_docs',
        embedding_function=sentence_transformer_ef
    )

    print(f"Db ready to use!")

    while(True):
        user_query_txt: str = input("What do you need to know? ")
        if user_query_txt.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        results = collection.query(
            query_texts=user_query_txt,
            n_results=3
        )

        documents: list[str] = results['documents'][0]
        metadatas: list[str] = results['metadatas'][0]
        distances: list[str] = results['distances'][0]

        print(f"\nResults for ({user_query_txt}):")

        for i in range(len(documents)):
            file_name: list[str] = metadatas[i].get("source", "Unknown file")
            print(f"\nFile {i+1}: {file_name} Distance: {distances[i]:.4f}")
            print("-" * 40)
            print(documents[i])
            print("-" * 40)

if __name__ == "__main__":
    main()