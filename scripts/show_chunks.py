import os
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.core import utils as u
from src.core import chunking as c
from src.core import config as cfg

def main():
    # Force UTF-8 for output
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    if len(sys.argv) < 2:
        print("Usage: python scripts/show_chunks.py <filename>")
        print("Example: python scripts/show_chunks.py hobbit.pdf")
        return

    target_name = sys.argv[1].lower()
    example_sources = cfg.BASE_DIR / "example_sources"
    
    # Search for the file
    found_file = None
    for file_path in example_sources.rglob("*"):
        if file_path.is_file() and target_name in file_path.name.lower():
            found_file = file_path
            break
            
    if not found_file:
        print(f"Error: File containing '{target_name}' not found in {example_sources}")
        return

    print(f"\nFound file: {found_file.relative_to(cfg.BASE_DIR)}")
    print("-" * 40)

    # Load content
    parser_func = u.FILE_PARSER.get(found_file.suffix)
    if not parser_func:
        print(f"Error: No parser available for extension {found_file.suffix}")
        return
        
    content = parser_func(found_file)
    if not content:
        print("Error: Could not read file content or file is empty.")
        return

    # Initialize model and chunk
    print("Loading model and generating semantic chunks...")
    model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    chunks = c.semantic_chunking(content, model=model)

    print(f"\nTotal chunks generated: {len(chunks)}")
    print("=" * 60)

    for i, chunk in enumerate(chunks):
        print(f"\n--- CHUNK {i+1} ---")
        print(chunk)
        print("-" * 40)

if __name__ == "__main__":
    main()
