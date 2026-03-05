from pathlib import Path

# Folder Paths
BASE_DIR = Path(__file__).parent.parent
DATABASES_DIR = BASE_DIR / "databases"
SOURCES_DIR = BASE_DIR / "sources"

# DB
DB_NAME = "books"
DB_PATH = DATABASES_DIR / DB_NAME
N_RESULTS = 15

# EMBEDDING MODEL CONFIGS
""" all-MiniLM-L6-v2, multi-qa-MiniLM-L6-cos-v1, paraphrase-multilingual-MiniLM-L12-v2 """
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# LLM MODEL CONFIGS
LLM_MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.1

# CHUNKING CONFIGS
SEMANTIC_THRESHOLD = 0.8
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 50