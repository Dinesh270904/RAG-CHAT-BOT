#!/usr/bin/env python3
"""
Agentic RAG Chatbot with MCP (Model Context Protocol)
Streamlit + LangChain + FAISS + Hugging Face (Flan-T5)
"""

import streamlit as st
import uuid
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# ---------------------
# MCP Message Utilities
# ---------------------
def create_message(sender, receiver, msg_type, payload):
    return {
        "sender": sender,
        "receiver": receiver,
        "type": msg_type,
        "trace_id": str(uuid.uuid4()),
        "payload": payload,
    }

# ---------------------
# Base Agent Class
# ---------------------
class Agent:
    def __init__(self, name):
        self.name = name

    def handle(self, message):
        raise NotImplementedError

# ---------------------
# Ingestion Agent
# ---------------------
class IngestionAgent(Agent):
    def __init__(self):
        super().__init__("IngestionAgent")

    def handle(self, message):
        file = message["payload"]["file"]
        text = ""
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        elif file.name.endswith(".txt"):
            text = file.read().decode("utf-8")
        else:
            text = ""
        return create_message(
            sender=self.name,
            receiver="RetrievalAgent",
            msg_type="DOC_INGESTED",
            payload={"text": text},
        )

# ---------------------
# Retrieval Agent
# ---------------------
class RetrievalAgent(Agent):
    def __init__(self):
        super().__init__("RetrievalAgent")
        self.vectorstore = None

    def build_vectorstore(self, text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(text)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_texts(chunks, embeddings)

    def handle(self, message):
        if message["type"] == "DOC_INGESTED":
            text = message["payload"]["text"]
            self.build_vectorstore(text)
            return create_message(
                sender=self.name,
                receiver="LLMResponseAgent",
                msg_type="VECTORSTORE_READY",
                payload={"status": "ready"},
            )
        elif message["type"] == "QUESTION":
            query = message["payload"]["query"]
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(query)
            top_chunks = [d.page_content for d in docs]
            return create_message(
                sender=self.name,
                receiver="LLMResponseAgent",
                msg_type="CONTEXT_RESPONSE",
                payload={"top_chunks": top_chunks, "query": query},
            )

# ---------------------
# LLM Response Agent
# ---------------------
class LLMResponseAgent(Agent):
    def __init__(self):
        super().__init__("LLMResponseAgent")
        pipe = pipeline("text2text-generation", model="google/flan-t5-base")
        self.local_llm = HuggingFacePipeline(pipeline=pipe)

    def handle(self, message):
        if message["type"] == "CONTEXT_RESPONSE":
            query = message["payload"]["query"]
            context = "\n\n".join(message["payload"]["top_chunks"])
            prompt = f"Answer the QUESTION based on the CONTEXT.\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer:"
            result = self.local_llm(prompt)
            answer = result if isinstance(result, str) else result[0]["generated_text"]
            return create_message(
                sender=self.name,
                receiver="UI",
                msg_type="ANSWER",
                payload={"answer": answer},
            )

# ---------------------
# Coordinator
# ---------------------
class Coordinator:
    def __init__(self):
        self.ingestion_agent = IngestionAgent()
        self.retrieval_agent = RetrievalAgent()
        self.llm_agent = LLMResponseAgent()

    def process_file(self, uploaded_file):
        msg = create_message("UI", "IngestionAgent", "DOC_UPLOAD", {"file": uploaded_file})
        msg = self.ingestion_agent.handle(msg)
        msg = self.retrieval_agent.handle(msg)
        return msg  # VECTORSTORE_READY

    def process_question(self, query):
        msg = create_message("UI", "RetrievalAgent", "QUESTION", {"query": query})
        msg = self.retrieval_agent.handle(msg)
        msg = self.llm_agent.handle(msg)
        return msg  # ANSWER

# ---------------------
# Streamlit App
# ---------------------
st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Agentic RAG Chatbot with MCP")
st.write("Upload documents and chat with them. Now powered by **MCP Agents**!")

# Session state init
if "coordinator" not in st.session_state:
    st.session_state.coordinator = Coordinator()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar file upload
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        msg = st.session_state.coordinator.process_file(uploaded_file)
        if msg["type"] == "VECTORSTORE_READY":
            st.success("✅ Document processed and ready for Q&A!")

# Chat UI
st.header("💬 Chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    # Show user question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Coordinator handles via agents
    msg = st.session_state.coordinator.process_question(prompt)

    # Show assistant answer
    answer = msg["payload"]["answer"]
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
