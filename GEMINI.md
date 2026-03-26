# Bookipidia RAG System

## Project Overview
Bookipidia is a Retrieval-Augmented Generation (RAG) system designed to allow users to upload, manage, and chat with their library of books (PDFs). It uses a semantic search approach to find relevant context from uploaded documents and provides answers using an LLM.

### Main Technologies
- **Backend**: FastAPI (REST API), SQLAlchemy (Relational DB for users/metadata), ChromaDB (Vector DB for embeddings).
- **Frontend**: Streamlit.
- **LLM Provider**: Groq (utilizing `llama-3.3-70b-versatile`).
- **Embeddings**: Sentence-Transformers (`paraphrase-multilingual-MiniLM-L12-v2`).
- **Document Processing**: PyMuPDF (fitz) for PDF extraction and custom semantic chunking logic.

### Architecture
All application code is located in the `src/` directory for better modularity:
- `src/api/`: FastAPI routes and Pydantic schemas.
- `src/core/`: Shared core logic including configuration, security, chunking, and general utilities.
- `src/database/`: Relational database connection management and SQLAlchemy models.
- `src/services/`: Core business services such as Vector Database management and RAG logic.
- `src/frontend/`: Streamlit-based user interface.
- `data/`: Local storage for raw sources, relational database, and vector database.
- `tests/`: Comprehensive unit and integration tests.

## Building and Running

### Prerequisites
- Python 3.10+
- A `.env` file in the root directory with the following variables:
  ```env
  GROQ_API_KEY=your_groq_api_key
  DATABASE_URL=sqlite:///./sql_app.db
  ```

### Installation
```bash
pip install -r requirements.txt
```

### Running the Backend API
The backend handles user sessions and document metadata.
```bash
uvicorn src.api.main:app --reload
```

### Running the Frontend (Bookipidia)
This is the main application for library management and chatting.
```bash
streamlit run src/frontend/app.py
```

### Manual Vector DB Update
To process documents in `data/sources` manually:
```bash
python -m src.services.vector_db
```

## Development Conventions

### Coding Style
- **Type Hinting**: Used extensively throughout the project.
- **Async**: While FastAPI supports async, the current DB and core logic are largely synchronous.
- **Configuration**: Always use `src/core/config.py` for pathing and model settings.

### RAG Logic
- **Chunking**: Uses semantic chunking based on cosine similarity of sentence embeddings (defined in `src/core/chunking.py`).
- **Vector Search**: Performed via ChromaDB with a `where` filter to scope results to the user's selected documents.

### Testing
- **Unit & Integration Tests**: Run all tests using pytest:
  ```bash
  pytest
  ```
- **Manual Query Testing**: Can be done via `scripts/query.py`.
