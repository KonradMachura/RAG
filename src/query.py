import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

def main():
    load_dotenv()
    """ sentence_transformer_ef is using Squared L2 for sentence embedding so lower results -> higher similarity
        First model is symmetric so it was trained to search sentences that mean more or less the same.
        Second model is asymmetric. It means it was designed for search tasks where the query 
        and the target documents are fundamentally different in structure, length, or intent.
        Instead of looking for a semantic "mirror image" (which is what symmetric models do)
        , they act like a "key and lock".
    """
    # sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction('all-MiniLM-L6-v2')
    # sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction('multi-qa-MiniLM-L6-cos-v1')
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction('paraphrase-multilingual-MiniLM-L12-v2')
    chroma_client = chromadb.PersistentClient(path="./data")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    collection = chroma_client.get_collection(
        name='company_docs',
        embedding_function=sentence_transformer_ef
    )

    print(f"Db ready to use!")

    while True:
        user_query_txt: str = input("What do you need to know? ")
        if user_query_txt.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        results = collection.query(
            query_texts=user_query_txt,
            n_results=3
        )

        documents: list[str] = results['documents'][0]
        metadatas: list[dict[str,str]] = results['metadatas'][0]
        distances: list[float] = results['distances'][0]
        file_names: list[str] = []

        print(f"\nResults for ({user_query_txt}):")


        for i in range(len(documents)):
            file_name: str = metadatas[i].get("source", "Unknown file")
            file_names.append(file_name)
            print(f"\nFile {i+1}: {file_name} Distance: {distances[i]:.4f}")
            print("-" * 40)
            print(documents[i])
            print("-" * 40)

        context: str = "\n\n---\n\n".join(documents)
        prompt = f"""
            You are a HR assistant in SuperTech company. Your task is to answer employees' questions,
            using only the following documents chunks. If there is no answer in documents,
            response "I am sorry, can't find any information related to your query."
            At the end of the answer emphasis source of the informations like
            document name and paragraph if is mentioned in context.
            Do it in format:"Document name: ,Paragraph(if mentioned in context): "
            Don't fabricate informations.
            
            Documents chunks (context):
            {context}
            
            Document name:
            {file_names}
            
            Employee query:
            {user_query_txt}
        """

        print("\nGenerating answer...\n")
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )


        print("-" * 40)
        print("HR assistant response:")
        print("-" * 40)
        print(chat_completion.choices[0].message.content)
        print("-" * 40)


if __name__ == "__main__":
    main()