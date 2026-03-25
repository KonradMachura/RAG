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
- `backend/api/`: FastAPI routes for user authentication (JWT) and document metadata management.
- `backend/core/`: Core RAG logic, including semantic chunking, embedding generation, and vector database operations.
- `backend/db/`: Database schema (User, Document, UserDocument) using SQLAlchemy.
- `frontend/`: Streamlit-based user interfaces.
- `config/`: Centralized configuration management.
- `data/`: Local storage for raw sources and the vector database.

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
uvicorn backend.api.main:app --reload
```

### Running the Frontend (Bookipidia)
This is the main application for library management and chatting.
```bash
streamlit run frontend/book_app.py
```

### Legacy/Internal Tools
- **HR Assistant**: A suspended version of the assistant focused on company documents.
  ```bash
  streamlit run frontend/app.py
  ```
- **Manual Vector DB Update**: To process documents in `data/sources` manually:
  ```bash
  python -m backend.core.build_db
  ```

## Development Conventions

### Coding Style
- **Type Hinting**: Used extensively throughout the backend.
- **Async**: While FastAPI supports async, the current DB and core logic are largely synchronous.
- **Configuration**: Always use `config/config.py` for pathing and model settings.

### RAG Logic
- **Chunking**: Uses semantic chunking based on cosine similarity of sentence embeddings (defined in `backend/core/chunking.py`).
- **Vector Search**: Performed via ChromaDB with a `where` filter to scope results to the user's selected documents.

### Testing
- Automated tests are located in `scripts/testing.py` (verify this for more specific testing instructions).
- Manual query testing can be done via `scripts/query.py`.
