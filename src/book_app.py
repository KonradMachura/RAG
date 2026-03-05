import streamlit as st
import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import config as cfg

st.set_page_config(page_title="Bookipidia", page_icon="📚")

load_dotenv()


@st.cache_resource
def load_services():
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(cfg.EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path= str(cfg.DB_PATH))
    collection = chroma_client.get_collection(name= cfg.DB_NAME, embedding_function=ef)
    return groq_client, collection


groq_client, collection = load_services()

st.title("Bookipidia")
st.markdown("Ask a question about your fav book!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Write down your question..."):
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.spinner("Searching procedure..."):
        results = collection.query(query_texts=[user_query], n_results= cfg.N_RESULTS)
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

            At the end of your answer, emphasize the source of the information.
            Format it exactly like this:
            "Source: [Book name] (add chapter or page if it is mentioned in the context)"

            Book chunks (context):
            {context}

            Book name / Source:
            {sources}

            Reader query:
            {user_query}
        """

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_query}
            ],
            model= cfg.LLM_MODEL,
            temperature= cfg.TEMPERATURE
        )

        answer = chat_completion.choices[0].message.content
        final_response = f"{answer}\n\n"

    st.chat_message("assistant").markdown(final_response)
    st.session_state.messages.append({"role": "assistant", "content": final_response})