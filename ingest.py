#!/usr/bin/env python3
"""
RAG Chatbot Ingestion Script

This script ingests PDF documents from the docs/ folder, splits them into chunks,
generates embeddings, and stores them in a ChromaDB collection for retrieval.
"""

import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langsmith import traceable


def setup_logging():
    """Configure logging for the ingestion process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("ingest.log"), logging.StreamHandler()],
    )


def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()

    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")

    # Configure LangSmith if enabled
    if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if os.getenv("LANGCHAIN_API_KEY"):
            os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
        if os.getenv("LANGCHAIN_PROJECT"):
            os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
        logging.info("LangSmith tracing enabled")

    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "persist_dir": os.getenv("PERSIST_DIR", "db"),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
        "batch_size": int(os.getenv("BATCH_SIZE", "100")),
    }


def find_pdf_files(docs_dir: str = "docs") -> List[Path]:
    """Find all PDF files in the docs directory."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Documents directory '{docs_dir}' not found")

    pdf_files = list(docs_path.glob("*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDF files in {docs_dir}/")

    return pdf_files


@traceable(name="load_and_split_pdf")
def load_and_split_pdf(
    pdf_path: Path, text_splitter: RecursiveCharacterTextSplitter
) -> List[Document]:
    """Load a PDF file and split it into chunks."""
    try:
        start_time = time.time()
        logging.info(f"Processing {pdf_path.name}...")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load_and_split(text_splitter)

        # Add metadata to each document
        for i, doc in enumerate(pages):
            doc.metadata.update(
                {
                    "source_path": str(pdf_path),
                    "filename": pdf_path.name,
                    "page_number": i + 1,
                    "chunk_id": f"{pdf_path.stem}_page_{i + 1}",
                }
            )

        processing_time = time.time() - start_time
        logging.info(f"Split {pdf_path.name} into {len(pages)} chunks in {processing_time:.2f}s")
        return pages

    except Exception as e:
        logging.error(f"Error processing {pdf_path.name}: {str(e)}")
        return []


def create_embeddings(config: Dict[str, Any]) -> OpenAIEmbeddings:
    """Create OpenAI embeddings instance."""
    return OpenAIEmbeddings(
        openai_api_key=config["openai_api_key"], model=config["embedding_model"]
    )


def process_documents_in_batches(
    documents: List[Document], batch_size: int
) -> List[List[Document]]:
    """Split documents into batches for processing."""
    batches = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        batches.append(batch)
    return batches


@traceable(name="document_ingestion")
def ingest_documents():
    """Main ingestion function."""
    setup_logging()
    start_time = time.time()
    logging.info("Starting document ingestion process...")

    try:
        # Load configuration
        config = load_environment()
        logging.info("Environment configuration loaded successfully")

        # Find PDF files
        pdf_files = find_pdf_files()
        if not pdf_files:
            logging.warning("No PDF files found in docs/ directory")
            return

        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        # Process all PDFs
        all_documents = []
        for pdf_path in pdf_files:
            documents = load_and_split_pdf(pdf_path, text_splitter)
            all_documents.extend(documents)

        if not all_documents:
            logging.error("No documents were successfully processed")
            return

        logging.info(f"Total documents to process: {len(all_documents)}")

        # Initialize embeddings
        embeddings = create_embeddings(config)
        logging.info("Embeddings model initialized")

        # Initialize ChromaDB
        persist_directory = config["persist_dir"]
        os.makedirs(persist_directory, exist_ok=True)

        # Process documents in batches
        batches = process_documents_in_batches(all_documents, config["batch_size"])
        logging.info(f"Processing {len(batches)} batches...")

        # Create or load existing vector store
        vectorstore = None
        embedding_start_time = time.time()
        
        for i, batch in enumerate(batches):
            batch_start_time = time.time()
            logging.info(
                f"Processing batch {i + 1}/{len(batches)} ({len(batch)} documents)..."
            )

            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]

            if vectorstore is None:
                # Create new vector store with first batch
                vectorstore = Chroma.from_texts(
                    texts=texts,
                    metadatas=metadatas,
                    embedding=embeddings,
                    collection_name="pdf_knowledge",
                    persist_directory=persist_directory,
                )
            else:
                # Add to existing vector store
                vectorstore.add_texts(texts=texts, metadatas=metadatas)
            
            batch_time = time.time() - batch_start_time
            logging.info(f"Batch {i + 1} processed in {batch_time:.2f}s")

        embedding_time = time.time() - embedding_start_time
        
        # Persist the database
        vectorstore.persist()
        
        total_time = time.time() - start_time
        logging.info(
            f"Successfully ingested {len(all_documents)} documents into ChromaDB"
        )
        logging.info(f"Total embedding time: {embedding_time:.2f}s")
        logging.info(f"Total ingestion time: {total_time:.2f}s")
        logging.info(f"Database persisted to: {persist_directory}")

    except Exception as e:
        logging.error(f"Ingestion failed: {str(e)}")
        raise


if __name__ == "__main__":
    ingest_documents()
