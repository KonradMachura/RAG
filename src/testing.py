import chunking as c
import embedding as e

def test_chunking(docs_contents: list[str], docs_names: list[str],
                  chunking_type: str, **details) -> None:

    for doc_content, doc_name in zip(docs_contents, docs_names):
        print(f"--- Chunking {doc_name}, size {len(doc_content)} ---")

        match chunking_type:
            case "fixed_size":
                chunked_doc = c.fixed_sized_chunking(doc_content, details.get("size", 200),
                                                     details.get("overlap", 40))
            case "subsection":
                chunked_doc = c.subsection_chunking(doc_content)
            case "paragraph":
                chunked_doc = c.paragraph_chunking(doc_content)
            case _:
                raise ValueError(f"Nieznany typ chunkowania: {chunking_type}")

        chunked_doc_size: int = sum(len(chunk) for chunk in chunked_doc)
        """For fixed size chunking the cumulative size -> doc_size + (chunks_num-1) * overlap"""
        print(f"{len(chunked_doc)} chunks, cumulative size {chunked_doc_size}")


def test_embedding():
    e.load_dotenv()
    model = e.SentenceTransformer('all-MiniLM-L6-v2')

    sentences: list[str] = [
        "An employee is entitled to 26 days of holiday leave.",
        "Rules for taking days off work.",
        "How to set up a company email account?",
    ]
    query_text: str = "Hoe long is holiday leave?"

    embedding = model.encode(sentences)
    query = model.encode(query_text)

    similarities = e.util.semantic_search(query, embedding, top_k=3)

    for item in similarities[0]:
        print(f"{item}")