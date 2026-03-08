import sys
import streamlit as st
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
import os

root_path = str(Path(__file__).parent.parent.absolute())
if root_path not in sys.path:
    sys.path.append(root_path)

from config import config as cfg
from backend.core import build_db
from backend.core.utils import read_pdf_files

@st.cache_resource
def load_services():
    load_dotenv()
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    collection = build_db.configure_chroma_db()
    return groq_client, collection

def get_stored_books() -> list[str]:
    return [path.name.rstrip(".pdf") for path in cfg.SOURCES_DIR.glob('books/*.pdf')]

def upload_a_file(collection):
    uploaded_file = st.sidebar.file_uploader("Upload your own book here", type="pdf", accept_multiple_files=False)

    if uploaded_file is not None:
        file_path = cfg.SOURCES_DIR / cfg.DB_NAME / uploaded_file.name
        if not file_path.exists():
            with st.spinner(text="Saving and analyzing the book... This might take a minute ⏳"):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                doc_content = read_pdf_files(Path(file_path))
                build_db.chunk_and_save_document(collection, doc_content, uploaded_file.name)

                st.sidebar.success(f"Successfully processed and learned: {uploaded_file.name}!")

def render_sidebar_book_list_and_get_selection(stored_books: list[str]) -> list[str]:
    selected_books: list[str] = []
    if not stored_books:
        st.sidebar.info("You have to upload a book first!")
    else:
        for book in stored_books:
            is_checked = st.sidebar.checkbox(book, value=True)
            if is_checked:
                selected_books.append(book)

    for _ in range(25 - len(stored_books)):
        st.sidebar.text("")

    return selected_books

def check_empty_db_condition(collection):
    if collection.count() == 0:
        st.error("DB is empty. Upload a book and wait till we process it.")
        st.stop()

def check_if_any_book_selected(selected_books: list[str]):
    if not selected_books:
        st.warning("You have to mark at least one book to ask a question.")
        st.stop()

def validate_search_conditions(collection, selected_books: list[str]) -> None:
    check_empty_db_condition(collection)
    check_if_any_book_selected(selected_books)

def add_selected_books_to_where_filter(selected_books: list[str]) -> dict[str, str] | dict[str, dict[str, list[str]]]:
    if len(selected_books) == 1:
        return {"source": selected_books[0]}
    else:
        return {"source": {"$in": selected_books}}

def retrieve_context_and_sources(collection, user_query: str, where_filter: dict) -> tuple[str, set[str]]:
    results = collection.query(
        query_texts=[user_query],
        n_results=cfg.N_RESULTS,
        where=where_filter)

    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    context: str = "\n\n---\n\n".join(documents)
    sources = set([meta.get("source", "Not found") for meta in metadatas])
    return context, sources

def generate_llm_answer(groq_client: Groq, context: str, sources: set, user_query: str, chat_history: list):
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
    st.sidebar.header("Choose your book")

    groq_client, collection = load_services()
    stored_books = get_stored_books()
    selected_books: list[str] = render_sidebar_book_list_and_get_selection(stored_books)
    upload_a_file(collection)

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
                context, sources = retrieve_context_and_sources(collection, user_query, where_filter)
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