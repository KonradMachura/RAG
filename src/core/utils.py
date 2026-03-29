from pathlib import Path
from typing import Callable
import fitz  # old name for PyMuPDF
import re
from src.core import config as cfg

def clean_pdf_text(content: str) -> str:
    """Cleaning PDF text sticking cutted words"""
    """ /s - strings contains a white space char  /d - strings contains digits
        flag - MULTILINE returns matches at the beggining of each line"""
    content = re.sub(r'^\s*\d+\s*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'(\w+)[-\xad]\s*\n+\s*(\w+)', r'\1\2',content)
    content = re.sub(r'(?<![.!?])\n+', ' ', content)
    content = re.sub(r'(?<!\n)\n(?!\n)', ' ', content)
    content = re.sub(r' +', ' ', content)
    return content


def read_md_files(file_path: Path):
    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            return f.read().strip()

    except Exception as e:
        print(f"Warning: Cannot read file {file_path}: {e} .")
        return ""

def read_pdf_files(file_path: Path):
    content: str = ""
    try:
        with fitz.open(file_path) as f:
            for page in f:
                content += page.get_text()
        return clean_pdf_text(content)

    except Exception as e:
        print(f"Warning: Cannot read file {file_path}: {e} .")
        return ""


FILE_PARSER: dict[str, Callable[[Path], str]] = {
    ".md": read_md_files,
    ".pdf": read_pdf_files
}

def read_docs() -> tuple[list[str], list[str], list[Path]]:
    """Searching for specific files in data/sources/books and saving their paths and content"""
    docs_content: list[str] = []
    doc_paths: list[Path] = []
    doc_names: list[str] = []

    # Target ONLY the sources/books directory
    target_dir = cfg.SOURCES_DIR / cfg.DB_NAME
    if not target_dir.exists():
        print(f"Directory {target_dir} does not exist.")
        return [], [], []

    file_paths = target_dir.glob('*')
    for file_path in file_paths:
        # Skip system files or directories
        if file_path.name == "chroma.sqlite3" or file_path.is_dir():
            continue

        # If it's a hash (no extension), treat it as a PDF
        suffix = file_path.suffix.lower()
        if not suffix:
            suffix = ".pdf"
            
        parser_func = FILE_PARSER.get(suffix)
        if parser_func:
            content = parser_func(file_path)
            if content:
                docs_content.append(content)
                doc_paths.append(file_path)
                # Use the filename (hash) as the name for metadata
                doc_names.append(file_path.name)
        else:
            print(f"File {file_path} with unknown extension {file_path.suffix}.")

    return docs_content, doc_names, doc_paths


def print_documents_info(docs_names: list[Path], docs_path: list[str]) -> None:
    """Printing info about num of documents, and their names"""
    print(f"Loaded {len(docs_path)} documents.")
    print("Documents info:")
    for i, name in enumerate(docs_names):
        print(f"{i+1}. {name}")


if __name__ == "__main__":
    docs_content, docs_names, docs_path = read_docs()
    print_documents_info(docs_names, docs_path)
    if docs_content:
        print(docs_content[0][:5000])