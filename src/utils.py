from pathlib import Path
from typing import Any
import fitz
from pymupdf import Page


def read_md_files(file_path: Path):
    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            return f.read().strip

    except Exception as e:
        print(f"Warning: Cannot read file {file_path}: {e} .")
        return ""

def read_pdf_files(file_path: Path):
    content: str = ""
    try:
        with fitz.open(file_path) as f:
            for page in f:
                content += page.get_text() + "\n\n"
        return content.strip()

    except Exception as e:
        print(f"Warning: Cannot read file {file_path}: {e} .")
        return ""

def read_docs() -> tuple[list[str], list[str], list[Path]]:
    """Searching for specific files adn saving their paths and content if not empty"""
    docs_content: list[str] = []
    doc_paths: list[Path] = []
    doc_names: list[str] = []

    project_path: Path = Path(__file__).parent.parent
    # file_paths = project_path.rglob(pattern='./company_documents/*.md')
    file_paths = project_path.rglob(pattern='./books/*.pdf')
    for file_path in file_paths:
        content: str = ""
        if file_path.suffix == '.md':
            content = read_md_files(file_path)  #should add another text files

        elif file_path.suffix == '.pdf':
            content = read_pdf_files(file_path)

        if content:
            docs_content.append(content)
            doc_paths.append(file_path)
            doc_names.append(file_path.name)

    return docs_content, doc_names, doc_paths


def get_documents_info() -> tuple[list[str], list[str], list[Path]]:
    """Reading files and returning info about num of documents, and their names"""
    docs_content: list[str]
    docs_path: list[Path]
    docs_names: list[str]

    docs_content, docs_names, docs_path = read_docs()

    print(f"Loaded {len(docs_path)} documents.")
    print("Documents info:")
    for i, (doc, doc_path) in enumerate(zip(docs_content, docs_path)):
        print(f"{i+1}. {docs_names[i]}")

    return docs_content, docs_names, docs_path


if __name__ == "__main__":
    docs_content, docs_names, docs_path = get_documents_info()