#!/usr/bin/env python3
"""
RAG Chatbot CLI Interface

This script provides a command-line interface for querying documents stored in ChromaDB
using a conversational retrieval-augmented generation (RAG) approach.
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseRetriever
from langsmith import traceable


def setup_logging():
    """Configure logging for the chat interface."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("chat.log"), logging.StreamHandler()],
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
        logging.info("LangSmith tracing enabled for chat interface")

    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "persist_dir": os.getenv("PERSIST_DIR", "db"),
        "retrieval_k": int(os.getenv("RETRIEVAL_K", "4")),
    }


@traceable(name="initialize_vectorstore")
def initialize_vectorstore(config: Dict[str, Any]) -> Chroma:
    """Initialize and load the ChromaDB vector store."""
    start_time = time.time()
    persist_directory = config["persist_dir"]

    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"Database directory '{persist_directory}' not found. "
            "Please run ingest.py first to create the knowledge base."
        )

    embeddings = OpenAIEmbeddings(
        openai_api_key=config["openai_api_key"], model=config["embedding_model"]
    )

    vectorstore = Chroma(
        collection_name="pdf_knowledge",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    # Verify the collection has documents
    collection = vectorstore._collection
    count = collection.count()

    if count == 0:
        raise ValueError(
            "No documents found in the knowledge base. "
            "Please run ingest.py first to populate the database."
        )

    load_time = time.time() - start_time
    logging.info(f"Loaded vector store with {count} document chunks in {load_time:.2f}s")
    return vectorstore


def create_retrieval_chain(
    vectorstore: Chroma, config: Dict[str, Any]
) -> ConversationalRetrievalChain:
    """Create the conversational retrieval chain."""

    # Initialize the LLM
    llm = ChatOpenAI(
        openai_api_key=config["openai_api_key"],
        model_name=config["llm_model"],
        temperature=0.1,
        streaming=True,
    )

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": config["retrieval_k"]}
    )

    # Initialize conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    # Create the conversational retrieval chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )

    return qa_chain


def format_sources(source_documents) -> str:
    """Format source documents into a readable citation string."""
    if not source_documents:
        return "No sources found."

    citations = []
    seen_sources = set()

    for doc in source_documents:
        metadata = doc.metadata
        source_info = f"{metadata.get('filename', 'Unknown')} (page {metadata.get('page_number', '?')})"

        if source_info not in seen_sources:
            citations.append(source_info)
            seen_sources.add(source_info)

    return "Sources: " + ", ".join(citations)


def print_welcome():
    """Print welcome message and instructions."""
    print("=" * 60)
    print("🤖 Welcome to RAG Chatbot!")
    print("Ask questions about your documents and get AI-powered answers.")
    print("Type 'exit', 'quit', or 'q' to stop the conversation.")
    print("=" * 60)
    print()


def get_user_input() -> str:
    """Get user input with a friendly prompt."""
    try:
        return input("❓ You: ").strip()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except EOFError:
        return "exit"


@traceable(name="process_query")
def process_query(qa_chain: ConversationalRetrievalChain, user_input: str) -> Tuple[str, str, float]:
    """Process a user query through the RAG chain with tracking."""
    start_time = time.time()

    # Get response from the chain
    result = qa_chain(
        {
            "question": user_input,
            "chat_history": [],  # Memory is handled internally
        }
    )

    end_time = time.time()
    response_time = end_time - start_time

    # Extract answer and sources
    answer = result.get("answer", "I couldn't generate an answer.")
    source_docs = result.get("source_documents", [])
    sources = format_sources(source_docs)

    return answer, sources, response_time


def print_answer(answer: str, sources: str, response_time: float):
    """Print the chatbot's answer with formatting."""
    print(f"\n🤖 RAGbot: {answer}")
    print(f"\n📚 {sources}")
    print(f"⏱️  Response time: {response_time:.2f}s")
    print("-" * 60)


def chat_loop():
    """Main chat interaction loop."""
    setup_logging()
    logging.info("Starting RAG Chatbot...")

    try:
        # Load configuration
        config = load_environment()
        logging.info("Environment configuration loaded")

        # Initialize vector store
        vectorstore = initialize_vectorstore(config)

        # Create retrieval chain
        qa_chain = create_retrieval_chain(vectorstore, config)
        logging.info("Conversational retrieval chain initialized")

        # Print welcome message
        print_welcome()

        # Main conversation loop
        while True:
            user_input = get_user_input()

            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "q", ""]:
                print("\n👋 Thanks for using RAG Chatbot! Goodbye!")
                break

            # Process the query
            try:
                # Process query with LangSmith tracking
                answer, sources, response_time = process_query(qa_chain, user_input)

                # Print the response
                print_answer(answer, sources, response_time)

                # Log the interaction
                logging.info(f"Query processed in {response_time:.2f}s")

            except Exception as e:
                logging.error(f"Error processing query: {str(e)}")
                print(f"\n❌ Sorry, I encountered an error: {str(e)}")
                print("Please try rephrasing your question.")
                print("-" * 60)

    except Exception as e:
        logging.error(f"Chat initialization failed: {str(e)}")
        print(f"\n❌ Failed to initialize chatbot: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure you've run 'python ingest.py' first")
        print("2. Check that your .env file contains a valid OPENAI_API_KEY")
        print("3. Ensure the docs/ folder contains PDF files")
        sys.exit(1)


if __name__ == "__main__":
    chat_loop()
