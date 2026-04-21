# Bookipidia RAG System 📚

Bookipidia is a modern Retrieval-Augmented Generation (RAG) system that allows you to upload your personal library of PDFs and have intelligent, context-aware conversations with your documents.

## ✨ Features
- **Intelligent PDF Processing**: Uses `docling` for high-fidelity conversion of PDFs to Markdown.
- **Semantic Chunking**: Advanced text splitting based on sentence embeddings for better context.
- **Multi-User Support**: Secure registration and personal document libraries.
- **Two Operating Modes**: Choose between a full-featured Web App or a lightweight Terminal Mode.

---

## 🏗️ Project Architecture

```text
Bookipidia/
├── data/                 # Local storage for DBs and documents
│   ├── vector_db/        # ChromaDB persistent storage
│   └── sources/          # Raw and processed PDF/Markdown files
├── example_sources/      # Sample documents for testing and Terminal Mode
├── scripts/              # Standalone tools and Terminal Mode entry points
│   ├── full_pipeline.py  # [Terminal Mode] Complete RAG flow in one script
│   ├── query.py          # Direct CLI query against the main vector DB
│   ├── show_chunks.py    # Visualizes how a specific file is chunked
│   └── evaluate_chunking.py # Benchmarks different chunking strategies
├── src/                  # Main application source code
│   ├── api/              # FastAPI endpoints and Pydantic schemas
│   ├── core/             # RAG Engine: Chunking, Docling utilities, and Config
│   ├── database/         # Relational DB (SQLAlchemy) for users/metadata
│   ├── frontend/         # Streamlit-based UI
│   └── services/         # Business logic for Vector DB and RAG orchestration
├── .env.example          # Template for environment variables
├── setup.ps1             # Windows automation script
└── requirements.txt      # Python dependencies
```

---

## 🚀 Operating Modes

Bookipidia can be run in two ways depending on your needs:

### 1. Terminal Mode (Limited / No DB)
Ideal for testing or quick usage without setting up a user database. It uses files from `example_sources/` and a temporary vector database.
- **Required Key**: `GROQ_API_KEY`
- **Database**: Not required (Relational DB is bypassed).
- **Run**:
  ```bash
  python scripts/full_pipeline.py
  ```
  *This script will ingest sample files, chunk them, store them in a temporary vector DB, and start a chat loop immediately.*

### 2. Full Web Mode (Complete Features)
The full experience with user accounts, document management UI, and persistent history.
- **Required Keys**: `GROQ_API_KEY` and `DATABASE_URL` (defaults to local SQLite).
- **Run**:
  1. **Start Backend**: `uvicorn src.api.main:app --reload`
  2. **Start Frontend**: `streamlit run src/frontend/app.py`

---

## 🛠️ Configuration & Setup

### 1. Installation (Windows)
```powershell
./setup.ps1
```

### 2. Environment Variables (`.env`)
| Variable | Necessity | Description |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | **Mandatory** | Required for the Llama 3.1 LLM to generate answers. |
| `DATABASE_URL` | **Full Mode Only** | The connection string for the user database (e.g., `sqlite:///./sql_app.db`). |
| `DOCLING_MODEL` | Optional | Set to `default` or `egret_xl` for PDF processing quality. |

---

## 📜 Script Explanations

- **`full_pipeline.py`**: The "Zero-Setup" entry point. It automatically processes `example_sources/`, builds a vector index, and lets you query it in the terminal.
- **`query.py`**: A CLI tool to talk to your *actual* persistent library (the one used in Web Mode) without starting the UI.
- **`show_chunks.py`**: Debugging tool. Run `python scripts/show_chunks.py <filename>` to see exactly how the semantic chunker splits your text.
- **`evaluate_chunking.py`**: A research script used to compare the effectiveness of Paragraph, Fixed-Size, and Semantic chunking methods.

---

## 🧪 Testing
To run the automated test suite:
```bash
pytest
```
