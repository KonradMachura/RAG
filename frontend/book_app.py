import sys
import os
import time
from typing import Any

from dotenv import load_dotenv
from pathlib import Path

import requests
import hashlib
import streamlit as st
from groq import Groq
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.uploaded_file_manager import UploadedFile

root_path = str(Path(__file__).parent.parent.absolute())
if root_path not in sys.path:
    sys.path.append(root_path)

from config import config as cfg
from backend.core import build_db, chunking
from backend.core.utils import read_pdf_files

API_URL = "http://127.0.0.1:8000"

@st.cache_resource
def load_services() ->  tuple[Groq, Collection, SentenceTransformer]:
    load_dotenv()
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    collection = build_db.configure_chroma_db()
    chunking_model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    return groq_client, collection, chunking_model


def add_notification(message: str, notify_type: str) -> None:
    if "notifications" not in st.session_state:
        st.session_state["notifications"] = []
    else:
        st.session_state["notifications"].append({
            "msg": message,
            "type": notify_type
        })


def render_notifications():
    if "notifications" in st.session_state and st.session_state["notifications"]:
        for notif in st.session_state["notifications"]:
            match notif["type"]:
                case "success":
                    st.toast(notif["msg"], icon="✅", duration="short")
                case "error":
                    st.toast(notif["msg"], icon="❌", duration="long")
                case "info":
                    st.toast(notif["msg"], icon="ℹ️", duration="short")

        st.session_state["notifications"] = []


def get_stored_docs() -> list[dict]:
    try:
        response = requests.get(f"{API_URL}/documents")
        if response.status_code == 200:
            documents = response.json()
            if not documents:
                return []
            else:
                return documents
        else:
            err = response.json().get("detail", "Error") if response else "Connection error"
            add_notification(f"API error: {response.status_code}, {err}", notify_type="error")
            return []
    except requests.exceptions.ConnectionError:
        add_notification("Connection error. Try again later.", notify_type="error")
        return []


@st.fragment
def manage_sidebar_library(collection: Collection, chunking_model: SentenceTransformer) -> list[dict]:
    st.header("Your library")
    stored_documents = get_stored_docs()
    selected_documents = render_document_list(stored_documents)

    for _ in range(20 - len(stored_documents)):
        st.text("")

    action_space = st.empty()

    if "pending_processing" in st.session_state:
        handle_pending_processing(collection, chunking_model, action_space)
    else:
        render_file_uploader(stored_documents, action_space)

    return selected_documents


def api_add_document(payload: dict) -> requests.Response | None:
    try:
        return requests.post(f"{API_URL}/document", json=payload)
    except requests.exceptions.ConnectionError:
        return None


def api_delete_document(doc_id: str) -> requests.Response | None:
    try:
        return requests.delete(f"{API_URL}/document/{doc_id}")
    except requests.exceptions.ConnectionError:
        return None


def api_update_document(doc_id: str, chunks_num: int) -> requests.Response | None:
    try:
        payload = {"chunk_count": chunks_num}
        return requests.patch(f"{API_URL}/document/{doc_id}", json=payload)
    except requests.exceptions.ConnectionError:
        return None

def generate_file_hash(uploaded_file: UploadedFile) -> str:
    file_bytes = uploaded_file.getbuffer()
    uploaded_file_hash = hashlib.sha256(file_bytes).hexdigest()
    return uploaded_file_hash


def render_document_list(stored_documents: list[dict]) -> list[dict]:
    selected_documents: list[dict] = []

    if not stored_documents:
        return selected_documents

    for library_entry in stored_documents:
        col1, col2 = st.columns([0.85, 0.15])

        is_processing_this_doc = (
                "pending_processing" in st.session_state and
                st.session_state["pending_processing"]["file_name"] == library_entry["document"]["file_name"]
        )

        is_checked = col1.checkbox(
            label=library_entry["document"]["file_name"],
            value=False,
            key=f"chk_{library_entry['document']['id']}",
            disabled=is_processing_this_doc
        )

        with col2:
            if st.button(label="", icon=":material/delete:", key=f"del_{library_entry['document']['id']}",
                         type="tertiary", disabled=is_processing_this_doc
                         ):
                response = api_delete_document(library_entry["document"]['id'])

                if response and response.status_code == 200:
                    add_notification(f"{library_entry["document"]['file_name']} deleted successfully!", notify_type="success")
                    st.rerun()
                else:
                    err = response.json().get("detail", "Error") if response else "Connection error"
                    add_notification(f"Delete failed: {err}", notify_type="error")

        if is_checked:
            selected_documents.append(library_entry["document"])

    return selected_documents


def render_file_uploader(stored_documents: list[dict], placeholder: DeltaGenerator) -> None:
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 1

    if "pending_processing" in st.session_state:
        return

    uploaded_file = placeholder.file_uploader(
        "Upload your own book here",
        type="pdf",
        accept_multiple_files=False,
        key=f"uploader_{st.session_state['uploader_key']}",
    )

    if uploaded_file is not None:
        uploaded_file_hash = generate_file_hash(uploaded_file)
        is_already_uploaded = any(library_entry["document"]["file_hash"] == uploaded_file_hash for library_entry in stored_documents)
        if is_already_uploaded:
            add_notification(f"Book '{uploaded_file.name}' is already uploaded!", notify_type="info")
            st.session_state["uploader_key"] += 1
            st.rerun()

        file_path = build_target_file_path(uploaded_file_hash)
        payload = create_document_payload(uploaded_file, file_path, uploaded_file_hash)
        response = api_add_document(payload)

        if response and response.status_code == 200:
            new_library_entry = response.json()
            save_to_disk(file_path, uploaded_file)
            st.session_state["pending_processing"] = {
                "file_path": file_path,
                "file_name": uploaded_file.name,
                "id": new_library_entry["document"]["id"]
            }

            add_notification("Book added to library! Processing started...", notify_type="info")
            st.session_state["uploader_key"] += 1
            st.rerun()
        else:
            err = response.json().get("detail", "Error") if response else "Connection error"
            add_notification(f"Upload failed: {err}", notify_type="error")
            st.session_state["uploader_key"] += 1
            st.rerun()

def build_target_file_path(file_hash: str ) -> Path:
    return cfg.SOURCES_DIR / cfg.DB_NAME / file_hash

def create_document_payload(uploaded_file: UploadedFile, file_path: Path, file_hash: str) -> dict[str, str | Any]:
    payload = {"file_name": uploaded_file.name, "file_path": str(file_path), "file_hash": file_hash}
    return  payload


def save_to_disk(file_path: Path, uploaded_file: UploadedFile) -> None:
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

def vectorize_and_store_document(file_path: Path, collection: Collection, chunking_model: SentenceTransformer) -> int:
    doc_content = read_pdf_files(file_path)
    chunks = chunking.semantic_chunking(doc_content, model=chunking_model)
    build_db.save_chunks_to_vectordb(collection, chunks, file_path.name)
    return len(chunks)


def handle_pending_processing(collection: Collection, chunking_model: SentenceTransformer, placeholder) -> None:
    if "pending_processing" in st.session_state:
        task = st.session_state["pending_processing"]

        with placeholder.container():
            with st.spinner(f"Analyzing and vectorizing {task['file_name']}... This might take a minute ⏳"):
                time.sleep(0.2)
                chunks_num = vectorize_and_store_document(task["file_path"], collection, chunking_model)

                if "id" in task and chunks_num > 0:
                    api_update_document(task["id"], chunks_num)

        add_notification(f"Successfully processed {task['file_name']} into {chunks_num} chunks!", notify_type="success")

        del st.session_state["pending_processing"]
        st.rerun()

def check_empty_db_condition(collection: Collection) -> None:
    if collection.count() == 0:
        add_notification("Upload your first book and wait till we process it.", notify_type="info")
        st.stop()


def check_if_any_document_selected(selected_documents: list[dict]) -> None:
    if not selected_documents:
        add_notification("You have to mark at least one book to ask a question.", notify_type="info")
        st.stop()


def validate_search_conditions(collection: Collection, selected_documents: list[dict]) -> None:
    check_empty_db_condition(collection)
    check_if_any_document_selected(selected_documents)


def add_selected_documents_to_where_filter(selected_documents: list[dict]) -> (dict[str, dict] |
                                                                               dict[str, dict[str, list[dict]]]):
    sources = [library_entry["document"]["file_name"] for library_entry in selected_documents]
    if len(sources) == 1:
        return {"source": sources[0]}
    else:
        return {"source": {"$in": sources}}


def retrieve_context_and_sources(collection: Collection, user_query: str, where_filter: dict) -> tuple[str, set[str]]:
    results = collection.query(
        query_texts=[user_query],
        n_results=cfg.N_RESULTS,
        where=where_filter)

    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    context: str = "\n\n---\n\n".join(documents)
    sources = set([meta.get("source", "Not found") for meta in metadatas])
    return context, sources


def generate_llm_answer(groq_client: Groq, context: str, sources: set, user_query: str, chat_history: list) -> str:
    prompt = f"""
                You are a helpful and knowledgeable literary assistant. Your task is to answer the reader's questions using
                EXCLUSIVELY the provided book chunks. 
                Do not use your internal pre-trained knowledge about the book. 

                If the answer cannot be found in the provided text chunks, you must respond exactly: "I am sorry,
                I can't find any information related to your query in the provided text."
                Do not fabricate information, guess, or make up facts.
                Always answer in the exact same language as the Reader query
                (if the query is in Polish, you MUST answer in Polish).

                Book chunks (context):
                {context}

                Book name / Source:
                {sources}

                Reader query:
                {user_query}
            """

    api_messages = [{"role": "system", "content": prompt}]
    for msg in chat_history[-5:]:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
    api_messages.append({"role": "user", "content": user_query})

    chat_completion = groq_client.chat.completions.create(
        messages=api_messages,
        model=cfg.LLM_MODEL,
        temperature=cfg.TEMPERATURE
    )
    return chat_completion.choices[0].message.content


def main():
    st.set_page_config(page_title="Bookipidia", page_icon="📚")
    st.title("Bookipidia")
    st.markdown("Ask a question about your fav book!")

    groq_client, collection, chunking_model = load_services()

    render_notifications()

    with st.sidebar:
        selected_documents: list[dict] = manage_sidebar_library(collection, chunking_model)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container(height=380, border=False)

    if user_query := st.chat_input("Write down your question..."):

        validate_search_conditions(collection, selected_documents)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            with st.spinner(text="Searching procedure..."):
                where_filter = add_selected_documents_to_where_filter(selected_documents)
                print(where_filter)
                context, sources = retrieve_context_and_sources(collection, user_query, where_filter)
                print(context)
                print(sources)
                answer = generate_llm_answer(groq_client, context, sources, user_query, st.session_state.messages)
                final_response = f"{answer}\n\n"

            st.chat_message("assistant").markdown(final_response)

        st.session_state.messages.append({"role": "assistant", "content": final_response})

    else:
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

if __name__ == "__main__":
    main()