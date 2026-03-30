import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from groq import Groq

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.core import utils as u
from src.core import chunking as c
from src.core import config as cfg

def main():
    load_dotenv()
    
    # 1. SETUP
    print("\n[STEP 1: SETUP]")
    print("-" * 20)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(cfg.EMBEDDING_MODEL)
    # Using a temporary or specific test DB path to avoid messing with main DB if needed
    test_db_path = cfg.BASE_DIR / "data" / "test_vector_db"
    test_db_path.mkdir(parents=True, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(path=str(test_db_path))
    collection_name = "full_pipeline_test"
    
    # Reset collection for a clean run if it exists
    try:
        chroma_client.delete_collection(name=collection_name)
    except:
        pass
        
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in .env")
        return
    groq_client = Groq(api_key=groq_api_key)
    
    chunking_model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    print("Setup complete.")

    # 2. INGESTION & CHUNKING VISUALIZATION
    print("\n[STEP 2: INGESTION & CHUNKING]")
    print("-" * 20)
    
    example_sources = cfg.BASE_DIR / "example_sources"
    files = list(example_sources.rglob("*.md")) + list(example_sources.rglob("*.pdf"))
    
    if not files:
        print("No files found in example_sources.")
        return

    for file_path in files:
        print(f"\nProcessing: {file_path.name}")
        
        # Load content
        parser_func = u.FILE_PARSER.get(file_path.suffix)
        if not parser_func:
            continue
        content = parser_func(file_path)
        
        # Chunk
        print(f"Chunking {file_path.name} using semantic_chunking...")
        chunks = c.semantic_chunking(content, model=chunking_model)
        print(f"Created {len(chunks)} chunks.")
        
        # Show some chunks as requested
        print(f"Preview of first 2 chunks:")
        for i, chunk in enumerate(chunks[:2]):
            print(f"  Chunk {i+1}: {chunk[:200]}...")
            
        # Save to Vector DB
        chunk_ids = [f"{file_path.stem}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_path.name} for _ in chunks]
        
        collection.add(
            ids=chunk_ids,
            documents=chunks,
            metadatas=metadatas
        )
        print(f"Added to Vector DB.")

    print("\nAll files processed and stored in Vector DB.")

    # 3. QUERY LOOP
    print("\n[STEP 3: RAG QUERY LOOP]")
    print("-" * 20)
    print("Type 'exit' or 'quit' to stop.")

    while True:
        user_query = input("\nQuery: ")
        if user_query.lower() in ["exit", "quit"]:
            break
            
        print("Retrieving relevant chunks...")
        results = collection.query(
            query_texts=[user_query],
            n_results=3
        )
        
        retrieved_docs = results['documents'][0]
        retrieved_metadatas = results['metadatas'][0]
        
        print(f"\nRetrieved {len(retrieved_docs)} chunks:")
        for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metadatas)):
            print(f"\n--- Retrieved Chunk {i+1} (Source: {meta['source']}) ---")
            print(doc)
            
        context = "\n\n---\n\n".join(retrieved_docs)
        prompt = f"""
            Answer the following query using ONLY the provided context.
            If the answer is not in the context, say you don't know.
            
            Context:
            {context}
            
            Query: {user_query}
        """
        
        print("\nGenerating AI answer...")
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=cfg.LLM_MODEL,
            temperature=cfg.TEMPERATURE
        )
        
        print("\nAI Response:")
        print("-" * 40)
        print(chat_completion.choices[0].message.content)
        print("-" * 40)

if __name__ == "__main__":
    main()
