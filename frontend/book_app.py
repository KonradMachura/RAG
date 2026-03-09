import sys
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from groq import Groq
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer
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


def get_stored_books() -> list[str]:
    try:
        response = requests.get(f"{API_URL}/documents")
        if response.status_code == 200:
            documents = response.json()

            if not documents:
                st.info("Db is empty. Upload your first book.")
            else:
                return [doc["file_name"] for doc in documents]
        else:
            st.error(f"API error: {response.status_code}, {response.json()['detail']}")
    except requests.exceptions.ConnectionError:
        st.error("Connection error. Try again later.")


def process_uploaded_file(collection: Collection, chunking_model: SentenceTransformer,
                          file_path: Path | Any, uploaded_file: UploadedFile) -> None:
    with st.spinner(text="Saving and analyzing the book... This might take a minute ⏳"):
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        doc_content = read_pdf_files(Path(file_path))

        chunks = chunking.semantic_chunking(doc_content, model=chunking_model)
        print(chunks[0:20])
        build_db.save_chunks_to_vectordb(collection, chunks, Path(uploaded_file.name).stem)

        st.sidebar.success(f"Successfully processed and learned: {uploaded_file.name}!")


def upload_a_file(collection: Collection, chunking_model: SentenceTransformer) -> None:
    uploaded_file = st.sidebar.file_uploader("Upload your own book here",
                                     type="pdf",
                                     accept_multiple_files=False,
                                     key="saving_to_the_server")

    if uploaded_file is not None:
        file_path: Path = cfg.SOURCES_DIR / cfg.DB_NAME / uploaded_file.name
        payload = {
            "file_name": uploaded_file.name,
            "file_path": str(file_path)
        }
        try:
            response = requests.post(f"{API_URL}/document", json=payload)
            if response.status_code == 200:
                st.balloons()
                st.success("Successfully uploaded your book!")
                process_uploaded_file(collection, chunking_model, file_path, uploaded_file)
                st.info("Now we process the book.")
            else:
                st.error(f"API error: {response.status_code}, {response.json()["detail"]}")
        except requests.exceptions.ConnectionError:
            st.error("Connection error. Try again later.")


def render_sidebar_book_list_and_get_selection(stored_books: list[str]) -> list[str]:
    selected_books: list[str] = []
    if not stored_books:
        st.sidebar.info("You have to upload a book first!")
    else:
        for book in stored_books:
            is_checked = st.sidebar.checkbox(book, value=True)
            if is_checked:
                selected_books.append(book)

    selected_books_count = len(selected_books) if selected_books else 0
    for _ in range(25 - selected_books_count):
        st.sidebar.text("")

    return selected_books


def check_empty_db_condition(collection: Collection) -> None:
    if collection.count() == 0:
        st.error("DB is empty. Upload a book and wait till we process it.")
        st.stop()


def check_if_any_book_selected(selected_books: list[str]) -> None:
    if not selected_books:
        st.warning("You have to mark at least one book to ask a question.")
        st.stop()


def validate_search_conditions(collection: Collection, selected_books: list[str]) -> None:
    check_empty_db_condition(collection)
    check_if_any_book_selected(selected_books)


def add_selected_books_to_where_filter(selected_books: list[str]) -> dict[str, str] | dict[str, dict[str, list[str]]]:
    if len(selected_books) == 1:
        return {"source": selected_books[0]}
    else:
        return {"source": {"$in": selected_books}}


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
    st.sidebar.header("Your library")



    groq_client, collection, chunking_model = load_services()

    stored_books = get_stored_books()
    print(stored_books)
    selected_books: list[str] = render_sidebar_book_list_and_get_selection(stored_books)
    print(selected_books)
    upload_a_file(collection, chunking_model)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container(height=380, border=False)

    if user_query := st.chat_input("Write down your question..."):

        validate_search_conditions(collection, selected_books)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            with st.spinner(text="Searching procedure..."):
                where_filter = add_selected_books_to_where_filter(selected_books)
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