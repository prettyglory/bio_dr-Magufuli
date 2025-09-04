import sys
import sqlite3

# Check if we need to use pysqlite3
if sqlite3.sqlite_version_info < (3, 35, 0):
    try:
        import pysqlite3 as sqlite3
        sys.modules['sqlite3'] = sqlite3
    except ImportError:
        pass

import streamlit as st
import os
from dotenv import load_dotenv
from rag_pipeline import initialize_qa_system, build_index
from pathlib import Path

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
GENERATION_MODEL_NAME = os.getenv("GENERATION_MODEL_NAME")
PDF_FOLDER = Path("pdfs")
CHROMA_DIR = "chroma_db"

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_initialized" not in st.session_state:
    st.session_state.index_initialized = False

def main():
    st.title("📚 BioPro Chatbot")
    st.write("Ask questions about Dr. Magufuli's biography!")

    # Index PDF only if not already initialized
    if not st.session_state.index_initialized and not os.path.exists(CHROMA_DIR):
        pdf_files = list(PDF_FOLDER.glob("*.pdf"))
        if not pdf_files:
            st.error(f"No PDFs found in {PDF_FOLDER.resolve()}")
            return
        with st.spinner("Indexing PDF... This may take a moment."):
            for pdf_file in pdf_files:
                st.write(f"Indexing {pdf_file.name}...")
                build_index(str(pdf_file))
            st.session_state.index_initialized = True

    # Initialize QA system
    qa = initialize_qa_system()
    if not qa:
        st.error("Failed to initialize QA system. Please check configuration.")
        return

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask a question"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = qa.invoke({"query": prompt})["result"]
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")

    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # Footer
    st.markdown(
        """
        <hr style="border:1px solid #e0e0e0;">
        <p style="text-align: center; color: #666; font-size: 14px;">
        Developed by Gloria Mbilinyi
        </p>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
