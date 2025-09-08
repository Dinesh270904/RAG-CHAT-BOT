#!/usr/bin/env python3
"""
Main entry point for the Agentic RAG Chatbot.
"""
import streamlit as st
import asyncio
from pathlib import Path
from uuid import uuid4
import tempfile
from typing import Dict, Any
import sys
import os

# Add the app directory and its subdirectories to Python path
app_dir = os.path.join(os.path.dirname(__file__), 'app')
sys.path.insert(0, app_dir)
sys.path.insert(0, os.path.join(app_dir, 'agents'))
sys.path.insert(0, os.path.join(app_dir, 'mcp'))
sys.path.insert(0, os.path.join(app_dir, 'parsers'))
sys.path.insert(0, os.path.join(app_dir, 'vector_store'))

from agents import (
    IngestionAgent,
    RetrievalAgent,
    LLMResponseAgent,
    CoordinatorAgent
)
from vector_store import VectorStore


def initialize_agents():
    """Initialize all agents."""
    # Create vector store
    vector_store = VectorStore()
    
    # Initialize agents
    ingestion_agent = IngestionAgent()
    retrieval_agent = RetrievalAgent(vector_store=vector_store)
    llm_agent = LLMResponseAgent()
    coordinator = CoordinatorAgent()
    
    return coordinator, ingestion_agent, retrieval_agent, llm_agent


def handle_document_upload(coordinator: CoordinatorAgent):
    """Handle document upload in the UI."""
    st.header("📄 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=['pdf', 'docx', 'pptx', 'csv', 'txt', 'md'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.type}") as tmp:
                tmp.write(file.getvalue())
                tmp_path = Path(tmp.name)
            
            # Process document
            asyncio.run(coordinator.process_document(
                file_path=str(tmp_path),
                file_type=file.type,
                content=file.getvalue()
            ))
            
            st.success(f"Processing {file.name}...")


def handle_chat(coordinator: CoordinatorAgent):
    """Handle chat interface in the UI."""
    st.header("💬 Chat")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.conversation_id = uuid4()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**\n{source}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🤔 Thinking...")
            
            # Process query through coordinator
            asyncio.run(coordinator.process_query(
                query=prompt,
                conversation_id=st.session_state.conversation_id
            ))
            
            # Note: The actual response will be handled by the callback


def handle_coordinator_callbacks(coordinator: CoordinatorAgent):
    """Set up coordinator callbacks."""
    
    def on_document_processed(data: Dict[str, Any]):
        st.toast(
            f"✅ Processed {Path(data['file_path']).name}: "
            f"{data['chunk_count']} chunks extracted"
        )
    
    def on_response_ready(data: Dict[str, Any]):
        # Update chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": data['response'],
            "sources": data['sources']
        })
        st.rerun()
    
    def on_error(data: Dict[str, Any]):
        st.error(
            f"Error: {data['error_type']}\n"
            f"Message: {data['message']}"
        )
    
    # Register callbacks
    coordinator.register_callback('document_processed', on_document_processed)
    coordinator.register_callback('response_ready', on_response_ready)
    coordinator.register_callback('error', on_error)


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG Chatbot")
    st.markdown(
        "Upload documents and ask questions about them! "
        "Supports PDF, DOCX, PPTX, CSV, and text files."
    )
    
    # Initialize agents
    coordinator, ingestion_agent, retrieval_agent, llm_agent = initialize_agents()
    
    # Set up coordinator callbacks
    handle_coordinator_callbacks(coordinator)
    
    # Start agent loops
    asyncio.run(asyncio.gather(
        coordinator.start(),
        ingestion_agent.start(),
        retrieval_agent.start(),
        llm_agent.start()
    ))
    
    # Create tabs for upload and chat
    tab1, tab2 = st.tabs(["Upload Documents", "Chat"])
    
    with tab1:
        handle_document_upload(coordinator)
    
    with tab2:
        handle_chat(coordinator)


if __name__ == "__main__":
    main()
