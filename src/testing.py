from typing import List
import chunking as c

def test_chunking(docs_contents: List[str], docs_names: List[str]):
    for doc_content, doc_name in zip(docs_contents, docs_names):
        print(f"--- Chunking {doc_name}, size {len(doc_content)} ---")

        chunked_doc = c.fixed_sized_chunking(doc_content, 200, 40)
        # chunked_doc = c.subsection_chunking(doc_content)
        chunked_doc_size: int = sum(len(chunk) for chunk in chunked_doc)
        """For fixed size chunking the cumulative size -> doc_size + (chunks_num-1) * overlap"""
        print(f"{len(chunked_doc)} chunks, cumulative size {chunked_doc_size}")
