#!/usr/bin/env python3
"""
RAG Chatbot FastAPI Backend

This FastAPI application provides a web interface for querying documents stored in ChromaDB
using a conversational retrieval-augmented generation (RAG) approach.
"""

import os
import time
import logging
from typing import Dict, Any, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langsmith import traceable


# Global variables to store the initialized components
vectorstore = None
qa_chain = None
config = None


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: str
    response_time: float


def setup_logging():
    """Configure logging for the FastAPI application."""
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
        logging.info("LangSmith tracing enabled for FastAPI interface")

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
    logging.info(
        f"Loaded vector store with {count} document chunks in {load_time:.2f}s"
    )
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
        streaming=False,  # Disable streaming for API responses
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

    return ", ".join(citations)


@traceable(name="process_query")
def process_query(
    qa_chain: ConversationalRetrievalChain, user_input: str
) -> Tuple[str, str, float]:
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG components on startup."""
    global vectorstore, qa_chain, config

    setup_logging()
    logging.info("Starting RAG Chatbot FastAPI application...")

    try:
        # Load configuration
        config = load_environment()
        logging.info("Environment configuration loaded")

        # Initialize vector store
        vectorstore = initialize_vectorstore(config)

        # Create retrieval chain
        qa_chain = create_retrieval_chain(vectorstore, config)
        logging.info("Conversational retrieval chain initialized")

        yield
    except Exception as e:
        logging.error(f"Failed to initialize RAG components: {str(e)}")
        raise
    finally:
        logging.info("Shutting down RAG Chatbot FastAPI application")


# Create FastAPI app with lifespan events
app = FastAPI(
    title="RAG Chatbot API",
    description="A simple RAG chatbot with FastAPI backend",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML interface."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Chatbot</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            
            .chat-container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                width: 100%;
                max-width: 800px;
                height: 600px;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            
            .chat-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                text-align: center;
            }
            
            .chat-header h1 {
                font-size: 24px;
                margin-bottom: 5px;
            }
            
            .chat-header p {
                opacity: 0.9;
                font-size: 14px;
            }
            
            .chat-messages {
                flex: 1;
                padding: 20px;
                overflow-y: auto;
                background: #f8f9fa;
            }
            
            .message {
                margin-bottom: 20px;
                display: flex;
                flex-direction: column;
            }
            
            .user-message {
                align-self: flex-end;
                background: #007bff;
                color: white;
                padding: 12px 16px;
                border-radius: 18px 18px 5px 18px;
                max-width: 70%;
                word-wrap: break-word;
            }
            
            .bot-message {
                align-self: flex-start;
                background: white;
                color: #333;
                padding: 12px 16px;
                border-radius: 18px 18px 18px 5px;
                max-width: 85%;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                word-wrap: break-word;
            }
            
            .sources {
                font-size: 12px;
                color: #666;
                margin-top: 8px;
                padding: 8px;
                background: #e9ecef;
                border-radius: 8px;
                border-left: 3px solid #007bff;
            }
            
            .response-time {
                font-size: 11px;
                color: #999;
                margin-top: 4px;
                text-align: right;
            }
            
            .chat-input {
                padding: 20px;
                background: white;
                border-top: 1px solid #e9ecef;
                display: flex;
                gap: 10px;
            }
            
            .chat-input input {
                flex: 1;
                padding: 12px 16px;
                border: 2px solid #e9ecef;
                border-radius: 25px;
                outline: none;
                font-size: 14px;
                transition: border-color 0.3s;
            }
            
            .chat-input input:focus {
                border-color: #007bff;
            }
            
            .chat-input button {
                padding: 12px 24px;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                transition: background-color 0.3s;
            }
            
            .chat-input button:hover:not(:disabled) {
                background: #0056b3;
            }
            
            .chat-input button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
                color: #666;
            }
            
            .loading.show {
                display: block;
            }
            
            .spinner {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #007bff;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .welcome-message {
                text-align: center;
                color: #666;
                padding: 40px 20px;
                line-height: 1.5;
            }
            
            .welcome-message h2 {
                color: #333;
                margin-bottom: 10px;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1>🤖 RAG Chatbot</h1>
                <p>Ask questions about your documents and get AI-powered answers</p>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="welcome-message">
                    <h2>Welcome!</h2>
                    <p>I'm your RAG chatbot assistant. I can help you find information from your uploaded documents.</p>
                    <p>Ask me anything about the content in your knowledge base!</p>
                </div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <span>Thinking...</span>
            </div>
            
            <div class="chat-input">
                <input type="text" id="messageInput" placeholder="Type your question here..." />
                <button onclick="sendMessage()" id="sendButton">Send</button>
            </div>
        </div>

        <script>
            const chatMessages = document.getElementById('chatMessages');
            const messageInput = document.getElementById('messageInput');
            const sendButton = document.getElementById('sendButton');
            const loading = document.getElementById('loading');

            messageInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });

            async function sendMessage() {
                const message = messageInput.value.trim();
                if (!message) return;

                // Add user message to chat
                addMessage(message, 'user');
                messageInput.value = '';
                
                // Show loading and disable input
                loading.classList.add('show');
                sendButton.disabled = true;
                messageInput.disabled = true;

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ message: message }),
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const data = await response.json();
                    
                    // Add bot response to chat
                    addMessage(data.answer, 'bot', data.sources, data.response_time);
                    
                } catch (error) {
                    console.error('Error:', error);
                    addMessage('Sorry, I encountered an error while processing your request. Please try again.', 'bot');
                } finally {
                    // Hide loading and enable input
                    loading.classList.remove('show');
                    sendButton.disabled = false;
                    messageInput.disabled = false;
                    messageInput.focus();
                }
            }

            function addMessage(text, type, sources = null, responseTime = null) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message';

                if (type === 'user') {
                    messageDiv.innerHTML = `<div class="user-message">${text}</div>`;
                } else {
                    let botMessageHTML = `<div class="bot-message">${text}`;
                    
                    if (sources) {
                        botMessageHTML += `<div class="sources">📚 Sources: ${sources}</div>`;
                    }
                    
                    if (responseTime) {
                        botMessageHTML += `<div class="response-time">⏱️ Response time: ${responseTime.toFixed(2)}s</div>`;
                    }
                    
                    botMessageHTML += `</div>`;
                    messageDiv.innerHTML = botMessageHTML;
                }

                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            // Focus on input when page loads
            window.addEventListener('load', function() {
                messageInput.focus();
            });
        </script>
    </body>
    </html>
    """
    return html_content


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return the response."""
    global qa_chain

    if qa_chain is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")

    try:
        # Process the query
        answer, sources, response_time = process_query(qa_chain, request.message)

        # Log the interaction
        logging.info(f"Query processed in {response_time:.2f}s")

        return ChatResponse(answer=answer, sources=sources, response_time=response_time)

    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global vectorstore, qa_chain

    if vectorstore is None or qa_chain is None:
        raise HTTPException(status_code=503, detail="RAG system not ready")

    return {"status": "healthy", "message": "RAG Chatbot is ready"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
