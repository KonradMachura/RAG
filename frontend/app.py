"""development suspended"""
import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from config import config as cfg
from backend.core import build_db

st.set_page_config(page_title="HR assistant", page_icon="😎")

load_dotenv()

@st.cache_resource
def load_services():
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    collection = build_db.configure_chroma_db()
    return groq_client, collection


groq_client, collection = load_services()

st.title("HR Assistant SuperTech")
st.markdown("Ask a question about regulations, holidays or equipment!")

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
            You are a HR assistant in SuperTech company. Your task is to answer employees' questions,
            using only the following documents chunks. If there is no answer in documents,
            response "I am sorry, can't find any information related to your query."
            At the end of the answer emphasis source of the informations like
            document name and paragraph if is mentioned in context.
            Do it in format:"Document name: ,Paragraph(if mentioned in context): "
            Don't fabricate informations.
            
            Documents chunks (context):
            {context}

            Document name:
            {sources}

            Employee query:
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