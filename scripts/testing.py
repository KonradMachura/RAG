import os
import sys
from pathlib import Path

# Dodaj katalog src do ścieżki wyszukiwania modułów
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.core.utils import convert_pdf_to_markdown_docling
def main():
    print(root_path)
    example_sources = Path("example_sources")
    test_output = Path("data/test_processed")
    test_output.mkdir(parents=True, exist_ok=True)
    
    # Znajdź wszystkie pliki PDF
    pdf_files = list(example_sources.rglob("*.pdf"))
    
    if not pdf_files:
        print("Nie znaleziono plików PDF w example_sources.")
        return

    print(f"Znaleziono {len(pdf_files)} plików PDF do przetestowania.")
    
    for pdf_path in pdf_files:
        print(f"\nPrzetwarzanie: {pdf_path.name}...")
        output_path = test_output / f"{pdf_path.stem}_test.md"
        
        try:
            content = convert_pdf_to_markdown_docling(pdf_path, output_path)
            print(f"Success! Result saved to: {output_path}")
        except Exception as e:
            print(f"BŁĄD podczas przetwarzania {pdf_path.name}: {e}")

if __name__ == "__main__":
    main()
