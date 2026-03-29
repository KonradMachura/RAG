import os
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document as LCDocument

from src.core import config as cfg

class LangChainRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDING_MODEL)
        self.vector_store = Chroma(
            collection_name=cfg.LC_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(cfg.VECTOR_DB_DIR)
        )
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=cfg.LLM_MODEL,
            temperature=cfg.TEMPERATURE
        )
        self.prompt_template = """
            SYSTEM INSTRUCTION:
            You are a strict factual assistant. You ONLY know what is provided in the Context below.

            RULES:
            1. If the answer is not contained within the Context, you MUST say: "I am sorry, I can't find any information related to your query in the provided text."
            2. Do NOT use your own internal knowledge.
            3. Answer in the same language as the Question.

            Context:
            {context}

            Question:
            {question}

            Answer:
        """
        self.QA_CHAIN_PROMPT = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        """Add documents to the LangChain vector store"""
        docs = [LCDocument(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        self.vector_store.add_documents(docs)

    def get_retrieval_qa_chain(self, search_kwargs: Dict[str, Any] = None):
        """Create a RetrievalQA chain with optional search filters"""
        kwargs = {"k": cfg.LC_SEARCH_K}
        if search_kwargs:
            kwargs.update(search_kwargs)

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type=cfg.LC_SEARCH_TYPE,
                search_kwargs=kwargs
            ),
            chain_type_kwargs={"prompt": self.QA_CHAIN_PROMPT},
            return_source_documents=True
        )

    def query(self, user_query: str, filenames: List[str] = None) -> Dict[str, Any]:
        """Execute a query with optional filename filtering"""
        search_kwargs = {}
        if filenames:
            if len(filenames) == 1:
                search_kwargs["filter"] = {"source": filenames[0]}
            else:
                search_kwargs["filter"] = {"source": {"$in": filenames}}

        qa_chain = self.get_retrieval_qa_chain(search_kwargs=search_kwargs)
        result = qa_chain.invoke({"query": user_query})
        
        # Debug: Print retrieved context to terminal
        print("\n--- DEBUG: LANGCHAIN CONTEXT ---")
        for i, doc in enumerate(result["source_documents"]):
            print(f"Chunk {i+1} (Source: {doc.metadata.get('source')}):")
            print(doc.page_content[:200] + "...")
        print("--- END DEBUG ---\n")

        return {
            "answer": result["result"],
            "source_documents": [
                {"content": doc.page_content, "metadata": doc.metadata} 
                for doc in result["source_documents"]
            ]
        }

def get_langchain_rag() -> LangChainRAG:
    return LangChainRAG()
