import re
from pathlib import Path
from typing import Callable
import fitz  # PyMuPDF
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_EGRET_XLARGE
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling_core.types.doc import ContentLayer, DocItemLabel

from src.core import config as cfg

def convert_pdf_to_markdown_docling(pdf_path: Path, output_md_path: Path) -> str:
    """
    Convert a PDF file to Markdown using docling and save it.
    Uses the latest Egret XL model for high-fidelity layout reconstruction.

    Args:
        pdf_path: Path to the input PDF file.
        output_md_path: Path where the processed Markdown should be saved.

    Returns:
        The Markdown content as a string.
    """
    # 1. Configure the pipeline with the latest 2.84.0 API
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.do_ocr = True
    
    # Use standard OCR (not forced) to maintain text quality
    pipeline_options.ocr_options.force_full_page_ocr = False
    # Only perform OCR on pages with significant non-text areas to save memory
    pipeline_options.ocr_options.bitmap_area_threshold = 0.05
    
    # Select layout model based on config
    if cfg.DOCLING_MODEL == "egret_xl":
        pipeline_options.layout_options.model_spec = DOCLING_LAYOUT_EGRET_XLARGE
    # "default" (None) uses the standard lighter model, saving significant memory.
    
    # Enable hardware acceleration (auto-detects GPU if available)
    pipeline_options.accelerator_options.device = AcceleratorDevice.AUTO
    
    # 2. Initialize converter with configured options
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    result = converter.convert(pdf_path)
    doc = result.document

    # 1. Mark headers, footers and footnotes by modifying their text
    for item, _ in doc.iterate_items(
            included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
    ):
        if not hasattr(item, "text") or not item.text:
            continue

        text_content = item.text.strip()

        if not text_content:
            continue
        
        # Determine if it's furniture based on label or coordinates
        mark_prefix = ""

        # A. Trusted Labels
        if item.label == DocItemLabel.PAGE_HEADER:
            mark_prefix = "[HEADER]"
        elif item.label == DocItemLabel.PAGE_FOOTER:
            mark_prefix = "[FOOTER]"
        elif item.label == DocItemLabel.FOOTNOTE:
            mark_prefix = "[FOOTNOTE]"

        if mark_prefix:
            if not item.text.startswith(mark_prefix):
                item.text = f"{mark_prefix} {item.text}"

    # 2. Export the document including both layers
    markdown_content = doc.export_to_markdown(
        included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
    )

    # Replace NBSP (\xa0) with standard spaces for better RAG processing
    markdown_content = markdown_content.replace("\xa0", " ")

    # Ensure processed directory exists
    output_md_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return markdown_content


def clean_pdf_text(content: str) -> str:
    """
    Clean extracted PDF text by fixing common artifacts like cut-off words
    and unnecessary line breaks.

    Args:
        content: The raw text extracted from a PDF.

    Returns:
        The cleaned and normalized text.
    """
    # Remove standalone page numbers
    content = re.sub(r'^\s*\d+\s*$', '', content, flags=re.MULTILINE)
    # Join hyphenated words split across lines
    content = re.sub(r'(\w+)[-\xad]\s*\n+\s*(\w+)', r'\1\2', content)
    # Replace single newlines with spaces (preserving sentence boundaries)
    content = re.sub(r'(?<![.!?])\n+', ' ', content)
    # Replace any other newlines with spaces
    content = re.sub(r'(?<!\n)\n(?!\n)', ' ', content)
    # Normalize multiple spaces to a single space
    content = re.sub(r' +', ' ', content)
    return content.strip()


def read_md_file(file_path: Path) -> str:
    """
    Read the content of a Markdown file.

    Args:
        file_path: The path to the .md file.

    Returns:
        The file content as a string, or an empty string if reading fails.
    """
    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Warning: Cannot read Markdown file {file_path}: {e}")
        return ""


def read_pdf_file(file_path: Path) -> str:
    """
    Extract and clean text from a PDF file using PyMuPDF.

    Args:
        file_path: The path to the .pdf file.

    Returns:
        The cleaned text content as a string, or an empty string if extraction fails.
    """
    content: str = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                content += page.get_text()
        return clean_pdf_text(content)
    except Exception as e:
        print(f"Warning: Cannot read PDF file {file_path}: {e}")
        return ""


# Mapping of file extensions to their respective parser functions.
FILE_PARSER: dict[str, Callable[[Path], str]] = {
    ".md": read_md_file,
    ".pdf": read_pdf_file
}


def read_docs(directory_pattern: str = "documents/processed/*.md") -> tuple[list[str], list[str], list[Path]]:
    """
    Recursively search for and read documents within the project's sources directory.

    Args:
        directory_pattern: The glob pattern to search for files relative to SOURCES_DIR.

    Returns:
        A tuple containing three lists:
        - docs_content: The text content of each read document.
        - doc_names: The filenames (stems) of the documents.
        - doc_paths: The absolute paths to the documents.
    """
    docs_content: list[str] = []
    doc_paths: list[Path] = []
    doc_names: list[str] = []

    file_paths = cfg.SOURCES_DIR.rglob(pattern=directory_pattern)
    for file_path in file_paths:
        if file_path.is_dir():
            continue

        parser_func = FILE_PARSER.get(file_path.suffix.lower())
        if parser_func:
            content = parser_func(file_path)
            if content:
                docs_content.append(content)
                doc_paths.append(file_path)
                doc_names.append(file_path.stem)
        else:
            print(f"Skipping file {file_path}: Unsupported extension {file_path.suffix}")

    return docs_content, doc_names, doc_paths


def print_documents_info(docs_names: list[str], docs_paths: list[Path]) -> None:
    """
    Print a summary of the loaded documents.

    Args:
        docs_names: List of document stems.
        docs_paths: List of absolute paths to the documents.
    """
    print(f"Loaded {len(docs_paths)} documents.")
    print("Documents info:")
    for i, name in enumerate(docs_names):
        print(f"{i+1}. {name}")


if __name__ == "__main__":
    contents, names, paths = read_docs()
    print_documents_info(names, paths)
    if contents:
        # Print a snippet of the first document for verification
        print(f"\nSnippet from {names[0]}:\n{contents[0][:1000]}...")
