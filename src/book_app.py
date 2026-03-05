import streamlit as st
import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import config as cfg
import build_db

def save_uploaded_file(uploaded_file):
  with open(os.path.join(cfg.SOURCES_DIR / cfg.SOURCE_TYPE, uploaded_file.name),"wb") as f:
     f.write(uploaded_file.getbuffer())
  return st.success("Succesfully saved the book!")

@st.cache_resource
def load_services():
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(cfg.EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path= str(cfg.DB_PATH))
    collection = chroma_client.get_or_create_collection(name= cfg.DB_NAME, embedding_function=ef)
    return groq_client, collection

def get_stored_books() -> list[str]:
    return [path.name.rstrip(".pdf") for path in cfg.SOURCES_DIR.glob('books/*.pdf')]

st.set_page_config(page_title="Bookipidia", page_icon="📚")

load_dotenv()
groq_client, collection = load_services()

st.title("Bookipidia")
st.markdown("Ask a question about your fav book!")

if "messages" not in st.session_state:
    st.session_state.messages = []

chat_container = st.container(height=380, border=False)

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

st.sidebar.header("Choose your book")
stored_books = get_stored_books()
selected_books = []
if not stored_books:
    st.sidebar.info("You have to upload a book first!")
else:
    for book in stored_books:
        is_checked = st.sidebar.checkbox(book, value=True)
        if is_checked:
            selected_books.append(book)

for _ in range(25 - len(stored_books)):
    st.sidebar.text("")

uploaded_file = st.sidebar.file_uploader("Upload your own book here", type="pdf", accept_multiple_files=False)

if uploaded_file is not None:
    file_path = os.path.join(cfg.SOURCES_DIR / cfg.DB_NAME, uploaded_file.name)
    if not os.path.exists(file_path):
        with st.spinner("Saving and analyzing the book... This might take a minute ⏳"):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            build_db.main()

            st.sidebar.success(f"Successfully processed and learned: {uploaded_file.name}!")

if user_query := st.chat_input("Write down your question..."):

    if collection.count() == 0:
        st.error("DB is empty. Upload a book and wait till we process it.")
        st.stop()
    if not selected_books:
        st.warning("You have to mark at least one book to ask a question.")
        st.stop()

    with chat_container:
        st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.spinner("Searching procedure..."):

        if len(selected_books) == 1:
            where_filter = {"source": selected_books[0]}
        else:
            where_filter = {"source": {"$in": selected_books}}

        results = collection.query(query_texts=[user_query], n_results= cfg.N_RESULTS, where=where_filter)
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]

        context: str = "\n\n---\n\n".join(documents)

        sources = set([meta.get("source", "Not found") for meta in metadatas])

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
        for msg in st.session_state.messages[-5:]:
            api_messages.append({"role": msg["role"], "content": msg["content"]})
        api_messages.append({"role": "user", "content": user_query})

        chat_completion = groq_client.chat.completions.create(
            messages=api_messages,
            model=cfg.LLM_MODEL,
            temperature=cfg.TEMPERATURE
        )

        answer = chat_completion.choices[0].message.content
        final_response = f"{answer}\n\n"

    with chat_container:
        st.chat_message("assistant").markdown(final_response)
    st.session_state.messages.append({"role": "assistant", "content": final_response})