import sys
import sqlite3

# Check if we need to use pysqlite3
if sqlite3.sqlite_version_info < (3, 35, 0):
    try:
        import pysqlite3 as sqlite3
        sys.modules['sqlite3'] = sqlite3
    except ImportError:
        pass


import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Load .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
GENERATION_MODEL_NAME = os.getenv("GENERATION_MODEL_NAME")
CHROMA_DIR = "chroma_db"
PDF_FOLDER = Path("pdfs")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env")
if not EMBEDDING_MODEL_NAME or not GENERATION_MODEL_NAME:
    raise ValueError("❌ EMBEDDING_MODEL_NAME or GENERATION_MODEL_NAME not set in .env")

def build_index(pdf_path, persist_dir=CHROMA_DIR):
    """Build and persist ChromaDB index from a PDF."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Convert to Path object
    pdf_path_obj = Path(pdf_path)
    
    # Load PDF
    loader = PyPDFLoader(str(pdf_path_obj))
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY
    )

    # Build ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectorstore.persist()
    print(f"✅ Indexed {pdf_path_obj.name} and persisted to {persist_dir}")
    return vectorstore

def initialize_qa_system():
    """Initialize the RAG QA system."""
    try:
        # Load embeddings and ChromaDB
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY
        )
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

        # Load chat LLM
        llm = ChatGoogleGenerativeAI(
            model=GENERATION_MODEL_NAME,
            temperature=0,
            google_api_key=GOOGLE_API_KEY
        )

        # Create RAG QA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
        )
        return qa
    except Exception as e:
        print(f"❌ Failed to initialize QA system: {str(e)}")
        return None

if __name__ == "__main__":
    # Index all PDFs in the pdfs folder
    pdf_files = list(PDF_FOLDER.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PDF_FOLDER.resolve()}")
    else:
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file.name} ...")
            build_index(str(pdf_file))

    # Test QA system
    qa = initialize_qa_system()
    if qa:
        print("✅ QA system ready. Type 'exit' to quit.")
        while True:
            query = input("\nAsk a question: ")
            if query.lower() == "exit":
                break
            try:
                answer = qa.invoke({"query": query})["result"]
                print("Answer:", answer)
            except Exception as e:
                print(f"Error processing query: {str(e)}")
