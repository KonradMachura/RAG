"""Utilities for RAG search methods """
from pathlib import Path
from typing import List, Tuple

def read_docs() -> Tuple[List[str], List[str], List[Path]]:
    """Searching for specific files adn saving their paths and content if not empty"""
    docs = []
    doc_paths: List[Path] = []
    doc_names: List[str] = []

    project_path: Path = Path(__file__).parent
    file_paths = project_path.rglob(pattern='../company_documents/*.md')
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding="utf-8") as f:
                content = f.read()
                if content:
                    docs.append(content)
                    doc_paths.append(file_path)
                    doc_name = file_path.name
                    doc_names.append(doc_name)
        except Exception as e:
            print(f"Warning: Cannot read file {file_path}: {e} .")

    return docs, doc_names, doc_paths

def get_documents_info() -> Tuple[List[str], List[str], List[Path]]:
    """Reading files and returning info about num of documents, and their names"""
    docs: List[str]
    docs_path: List[Path]

    docs_names: List[str]

    docs, docs_names, docs_path = read_docs()

    print(f"Loaded {len(docs_path)} documents.")
    print("Documents info:")
    for i, (doc, doc_path) in enumerate(zip(docs, docs_path)):
        print(f"{i+1}. {docs_names[i]}")

    return docs, docs_names, docs_path