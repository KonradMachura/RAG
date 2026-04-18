import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

from src.core import utils as u
from src.core import chunking as c
from src.core import config as cfg

# Define Test Data
TEST_DOCS = [
    {
        "path": "example_sources/company_documents/Polityka_bezpieczenstwa_i_IT.md",
        "type": "md",
        "questions": [
            "What laptops can employees choose from?",
            "What is the standard set for every employee?",
            "How much is the remote office allowance?",
            "How should IT problems be reported?",
            "What is the phone number for critical IT issues?",
            "What are the password requirements?",
            "How often must passwords be changed?",
            "Which password manager is used?",
            "What VPN must be used on public networks?",
            "Where must all company documents be stored?"
        ]
    },
    {
        "path": "example_sources/wybrane_filozoficzne_koncepcje_rozumu_ludzkiego.pdf",
        "type": "pdf",
        "questions": [
            "Is philosophy a static or dynamic field?",
            "What is the relationship between philosophy and 'detailed sciences' (nauki szczegółowe)?",
            "Why is the concept of mind fundamental for philosophy and science?",
            "What determines if a philosophy or science stays within the boundaries of rationality?",
            "What are the two distinguished concepts of mind mentioned?",
            "How does the concept of mind relate to human culture?",
            "What role does the human mind play in theoretical and practical cognition?",
            "Is the attitude of humans towards the mind indifferent to culture?",
            "According to the text, what decides the existence and shape of culture?",
            "What does the text say about the genetic and substantive connection between philosophy and detailed sciences?"
        ]
    },
    {
        "path": "example_sources/hobbit.pdf",
        "type": "pdf",
        "questions": [
            "Jakiego koloru były drzwi do nory Bilba Bagginsa?",
            "Kim byli rodzice Bilba Bagginsa?",
            "Jaki znak Gandalf wydrapał na drzwiach hobbita?",
            "Jaki udział w zyskach obiecano Bilbowi w liście od Thorina?",
            "Jak nazywały się trzy trolle, które schwytały krasnoludy?",
            "Co Elrond odczytał na mapie dzięki światłu księżyca?",
            "Jak Bilbo nazwał swój mieczyk po zabiciu pająka?",
            "W jaki sposób krasnoludy uciekły z więzienia króla leśnych elfów?",
            "Kto i jaką bronią zabił smoka Smauga?",
            "Co Bilbo zastał w swoim domu po powrocie z wyprawy?"
        ]
    },
    {
        "path": "example_sources/Do_Emotions_in_Prompts_Matter.pdf",
        "type": "pdf",
        "questions": [
            "What six emotion categories were used in the study?",
            "What are the three levels of emotional intensity defined in the validation study?",
            "Which three large language models were evaluated in the experiments?",
            "What is the name of the adaptive emotional prompting framework introduced in the paper?",
            "How does the paper describe the overall effect of static emotional prefixes on LLM performance?",
            "What benchmark was used to evaluate professional-level medical question answering?",
            "According to the study, in which type of tasks are emotional effects more variable?",
            "What does the paper conclude about human-written vs. LLM-generated emotional prefixes?",
            "What model was used to generate the short emotional expressions?",
            "At what temperature (T) was decoding set for the evaluations to ensure reproducibility?"
        ]
    }
]

CHUNK_METHODS = [
    {"name": "fixed", "func": lambda content, model: c.fixed_sized_chunking(content)},
    {"name": "semantic", "func": lambda content, model: c.semantic_chunking(content, model)},
    {"name": "subsection", "func": lambda content, model: c.subsection_chunking(content)},
    {"name": "paragraph", "func": lambda content, model: c.paragraph_chunking(content)}
]

def evaluate():
    load_dotenv()
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    embedding_model_name = cfg.EMBEDDING_MODEL
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(embedding_model_name)
    model = SentenceTransformer(embedding_model_name)
    
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)
    
    summary_file = results_dir / "effectiveness_summary.md"
    detailed_file = results_dir / "detailed_results.json"
    
    all_results = []
    
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("# Chunking Effectiveness Evaluation\n\n")

    for doc_info in TEST_DOCS:
        doc_path = Path(doc_info["path"])
        doc_type = doc_info["type"]
        questions = doc_info["questions"]
        
        print(f"\nProcessing document: {doc_path.name}")
        
        if doc_type == "pdf":
            # For structured evaluation, convert PDF to Markdown first
            processed_path = results_dir / f"{doc_path.stem}_processed.md"
            print(f"  Converting PDF to Markdown (Docling)...")
            content = u.convert_pdf_to_markdown_docling(doc_path, processed_path)
        else:
            content = u.FILE_PARSER[doc_path.suffix.lower()](doc_path)
        
        for method in CHUNK_METHODS:
            method_name = method["name"]
            print(f"  Using method: {method_name}")
            
            chunks = method["func"](content, model)
            
            # Setup temp ChromaDB collection
            client = chromadb.Client() # In-memory for testing
            collection_name = f"test_{int(time.time())}_{method_name}_{doc_path.stem[:10]}"
            collection = client.create_collection(name=collection_name, embedding_function=sentence_transformer_ef)
            
            if chunks:
                collection.add(
                    ids=[f"id_{i}" for i in range(len(chunks))],
                    documents=chunks,
                    metadatas=[{"source": doc_path.name} for _ in chunks]
                )
            else:
                print(f"    Warning: No chunks generated for {method_name}")
                continue

            method_results = {
                "document": doc_path.name,
                "method": method_name,
                "chunk_count": len(chunks),
                "avg_chunk_length": sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
                "evaluations": []
            }
            
            for q in questions:
                # Retrieve
                retrieval = collection.query(query_texts=[q], n_results=3)
                context_chunks = retrieval['documents'][0]
                context = "\n\n---\n\n".join(context_chunks)
                
                # Generate with retry mechanism
                prompt = f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer based ONLY on the context. If not found, say 'NOT_FOUND'."
                
                answer = "ERROR"
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        completion = groq_client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model="llama-3.1-8b-instant", # Use smaller model for faster/cheaper evaluation
                            temperature=0
                        )
                        answer = completion.choices[0].message.content.strip()
                        break
                    except Exception as e:
                        if "Rate limit" in str(e):
                            print(f"    Rate limit hit, sleeping for 60s (Attempt {attempt+1}/{max_retries})...")
                            time.sleep(60)
                        else:
                            print(f"    Error during generation: {e}")
                            break
                
                success = "NOT_FOUND" not in answer
                
                method_results["evaluations"].append({
                    "question": q,
                    "answer": answer,
                    "success": success,
                    "chunks": context_chunks
                })
            
            all_results.append(method_results)
            
            # Append to summary
            success_count = sum(1 for e in method_results["evaluations"] if e["success"])
            with open(summary_file, "a", encoding="utf-8") as f:
                f.write(f"## Doc: {doc_path.name} | Method: {method_name}\n")
                f.write(f"- Chunks: {len(chunks)}\n")
                f.write(f"- Success Rate: {success_count}/{len(questions)}\n\n")
                for e in method_results["evaluations"]:
                    status = "✅" if e["success"] else "❌"
                    f.write(f"### {status} Q: {e['question']}\n")
                    f.write(f"**A:** {e['answer']}\n\n")

    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"\nEvaluation complete. Results saved to {results_dir}")

if __name__ == "__main__":
    evaluate()
