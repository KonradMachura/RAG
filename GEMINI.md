
# Bookipidia RAG System

## Quick Start (Windows)
1. **Clone the repository**: `git clone <repo_url>`
2. **Run setup**: Open PowerShell and run `./setup.ps1`
3. **Configure API Key**: Open `.env` and add your `GROQ_API_KEY`.
4. **Start Application**:
   - Backend: `uvicorn src.api.main:app --reload`
   - Frontend: `streamlit run src/frontend/app.py`

## Project Overview
Bookipidia is a Retrieval-Augmented Generation (RAG) system designed to allow users to upload, manage, and chat with their library of books (PDFs). It uses a semantic search approach to find relevant context from uploaded documents and provides answers using an LLM.

### Main Technologies
- **Backend**: FastAPI (REST API), SQLAlchemy (Relational DB), ChromaDB (Vector DB).
- **Frontend**: Streamlit.
- **LLM Provider**: Groq (utilizing `llama-3.1-8b-instant`).
- **Embeddings**: Sentence-Transformers (`paraphrase-multilingual-MiniLM-L12-v2`).
- **Document Processing**: Docling for high-fidelity PDF-to-Markdown conversion and custom semantic chunking logic.

### Architecture
The project follows a modular structure in the `src/` directory:
- `src/api/`: FastAPI routes (`main.py`) and Pydantic schemas (`schemas.py`).
- `src/core/`: Shared logic including configuration (`config.py`), security (`security.py`), chunking (`chunking.py`), and document processing utilities (`utils.py`).
- `src/database/`: Relational database connection management (`connection.py`) and SQLAlchemy models (`models.py`) defining a many-to-many relationship between Users and Documents.
- `src/services/`: Core business services such as Vector Database management (`vector_db.py`) and RAG logic.
- `src/frontend/`: Streamlit-based user interface (`app.py`).
- `data/`: Local storage for raw sources, processed documents, relational database, and vector database.
- `scripts/`: Utility and evaluation scripts for chunking, querying, and full pipeline testing.
- `tests/`: Comprehensive unit and integration tests.

## API Endpoints
The backend provides a RESTful API for user and document management:
- **Authentication**: `POST /register`, `POST /token` (OAuth2).
- **Documents**:
    - `GET /documents`: List the current user's library.
    - `POST /document`: Register a new document with its file hash.
    - `PATCH /document/{doc_id}`: Update document processing status.
    - `DELETE /document/{doc_id}`: Unlink a document from a user (and delete if orphan).

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
```bash
uvicorn src.api.main:app --reload
```

### Running the Frontend (Bookipidia)
```bash
streamlit run src/frontend/app.py
```

### Manual Vector DB Update
To process documents manually:
```bash
python -m src.services.vector_db
```

## Development Conventions

### Coding Style
- **Type Hinting**: Mandatory for all new functions and classes.
- **Configuration**: Always use `src/core/config.py` for pathing, models, and thresholds.
- **Security**: Password hashing and JWT token handling are centralized in `src/core/security.py`.

### RAG Logic
- **Conversion**: Uses `docling` to convert PDFs to Markdown, preserving structural integrity.
- **Chunking**: Uses semantic chunking based on cosine similarity of sentence embeddings, with fallback to fixed-size chunking.
- **Vector Search**: Scoped to the user's selected documents using ChromaDB's `where` filter.

### Testing & Evaluation
- **Pytest**: Run all tests using `pytest`.
- **Evaluation**: Use `scripts/evaluate_chunking.py` to test different chunking strategies and `scripts/query.py` for direct RAG testing.
