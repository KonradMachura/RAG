import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# Folder Paths
BASE_DIR = Path(__file__).parent.parent.parent
ROOT_PATH = BASE_DIR
VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db"
SOURCES_DIR = BASE_DIR / "data" / "sources"

# VECTOR_DB
DB_NAME = "books"
DB_PATH = VECTOR_DB_DIR / DB_NAME
N_RESULTS = 10

# EMBEDDING MODEL CONFIGS
""" all-MiniLM-L6-v2, multi-qa-MiniLM-L6-cos-v1, paraphrase-multilingual-MiniLM-L12-v2 """
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# LLM MODEL CONFIGS
LLM_MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.1

# CHUNKING CONFIGS
SEMANTIC_THRESHOLD = 0.7
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100

# LANGCHAIN CONFIGS
LC_COLLECTION_NAME = "langchain_books"
LC_SEARCH_K = 5
LC_SEARCH_TYPE = "similarity" # or "mmr"

#RELATIONAL_DB
DATABASE_URL = os.getenv("DATABASE_URL")