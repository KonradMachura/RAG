import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists.
load_dotenv()

# Disable symlinks for Hugging Face downloads to avoid WinError 1314 on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# --- Project Paths ---
# The absolute path to the project root directory.
BASE_DIR = Path(__file__).parent.parent.parent
ROOT_PATH = BASE_DIR

# --- Database Configurations ---
# The absolute path to the vector database storage.
DB_PATH = BASE_DIR / "data" / "vector_db"
# The name of the collection/index in the vector database.
DB_NAME = "documents"
# The URL for the relational database (e.g., SQLite, PostgreSQL).
DATABASE_URL = os.getenv("DATABASE_URL")

# --- Directory Paths ---
# Directory where source documents are stored.
SOURCES_DIR = BASE_DIR / "data" / "sources"
DOCUMENTS_RAW_DIR = SOURCES_DIR / "documents" / "raw"
DOCUMENTS_PROCESSED_DIR = SOURCES_DIR / "documents" / "processed"

# --- Docling Model ---
# Options: "default" (light/fast), "egret_xl" (heavy/high-accuracy)
DOCLING_MODEL = "default"

# --- Retrieval Settings ---
# Number of search results to return from the vector database.
N_RESULTS = 10

# --- Embedding Model Configurations ---
# The identifier for the Sentence-Transformers model.
# Common choices: all-MiniLM-L6-v2, multi-qa-MiniLM-L6-cos-v1, paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# --- LLM Configurations ---
# The model ID for the Large Language Model used for generation.
LLM_MODEL = "llama-3.3-70b-versatile"
# Sampling temperature for LLM generation; lower is more deterministic.
TEMPERATURE = 0.1

# --- Chunking Configurations ---
# Threshold for semantic distance in semantic chunking (1 - cosine similarity).
SEMANTIC_THRESHOLD = 0.7
# Default character size for fixed-size chunks.
DEFAULT_CHUNK_SIZE = 200
# Character overlap between adjacent chunks in fixed-size chunking.
DEFAULT_CHUNK_OVERLAP = 50
