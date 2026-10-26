# Simple RAG Chatbot with FastAPI Documentation

## Project Overview

This project implements a lightweight Retrieval-Augmented Generation (RAG) chatbot that allows users to query PDF documents through both a modern web interface and a command-line interface. The system uses LangChain, ChromaDB, OpenAI's API, and FastAPI to provide accurate, contextual answers based on the content of uploaded documents with a beautiful, responsive web UI.

## Architecture

```
PDFs → PyPDFLoader → RecursiveCharacterTextSplitter → OpenAI Embeddings → ChromaDB → Retriever → ConversationalRetrievalChain → GPT-4o → FastAPI → Web UI/API
```

### Key Components

1. **Document Ingestion (`ingest.py`)**
   - Scans the `docs/` folder for PDF files
   - Splits documents into chunks using `RecursiveCharacterTextSplitter`
   - Generates embeddings using OpenAI's `text-embedding-3-small`
   - Stores chunks and metadata in ChromaDB

2. **FastAPI Web Application (`app.py`)**
   - Loads the ChromaDB collection on startup
   - Creates a conversational retrieval chain
   - Serves a beautiful HTML web interface at `/`
   - Provides RESTful API endpoints for chat interaction
   - Returns answers with source citations and response times

3. **Legacy Chat Interface (`chat.py`)**
   - Command-line interface for terminal users
   - Provides a CLI for user interaction
   - Returns answers with source citations

## File Structure

```
simple-rag-chatbot-with-interface/
├── docs/                           # PDF knowledge base
│   ├── A Comprehensive Plan to Control Type 2 Diabetes Naturally.pdf
│   ├── Yahya_Khan_CV.pdf
│   └── Yahya_Khan_CV_v2.pdf
├── db/                            # ChromaDB storage (created after ingestion)
├── app.py                         # FastAPI web application (primary interface)
├── ingest.py                      # Document ingestion script
├── chat.py                        # CLI chat interface (legacy)
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
├── .env                          # Environment variables (user created)
├── Makefile                      # Build and setup commands
├── CLAUDE.md                     # This documentation
├── FASTAPI_USAGE.md              # FastAPI specific usage guide
└── rag_chatbot_prd.md           # Product Requirements Document
```

## Installation & Setup

### Quick Start

```bash
# Clone or navigate to the repository
cd simple-rag-chatbot-with-interface

# Complete setup (recommended)
make setup

# Edit .env file with your OpenAI API key
nano .env

# Ingest documents
make ingest

# Start web interface (recommended)
python app.py
# Or start CLI interface
make chat
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your OpenAI API key

# Run ingestion
python ingest.py

# Start web interface
python app.py
# Or start CLI interface
python chat.py
```

## Configuration

The `.env` file supports the following variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-small` |
| `LLM_MODEL` | OpenAI chat model | `gpt-4o-mini` |
| `PERSIST_DIR` | ChromaDB storage directory | `db` |
| `CHUNK_SIZE` | Text chunk size for splitting | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `BATCH_SIZE` | Batch size for processing | `100` |
| `RETRIEVAL_K` | Number of chunks to retrieve | `4` |

### Langsmith Configuration (Optional)

For performance monitoring and model usage tracking:

| Variable | Description | Default |
|----------|-------------|---------|
| `LANGCHAIN_TRACING_V2` | Enable Langsmith tracing | `false` |
| `LANGCHAIN_API_KEY` | Your Langsmith API key | - |
| `LANGCHAIN_PROJECT` | Project name in Langsmith | `simple-rag-chatbot` |

## Usage

### Web Interface (Recommended)

Start the FastAPI web application:

```bash
# Start the web server
python app.py

# Or using uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000
```

The web interface will be available at `http://localhost:8000` and includes:

- **Modern UI**: Beautiful gradient design with responsive layout
- **Real-time Chat**: Interactive messaging with typing indicators
- **Source Citations**: Document sources displayed with page numbers
- **Response Timing**: Performance metrics for each query
- **Mobile Support**: Works on desktop, tablet, and mobile devices
- **API Access**: RESTful endpoints for programmatic access

#### API Endpoints

1. **Chat Endpoint**:
   ```bash
   POST /chat
   Content-Type: application/json
   
   {
     "message": "Your question here"
   }
   ```

2. **Health Check**:
   ```bash
   GET /health
   ```

3. **API Documentation**: Auto-generated docs at `/docs`

### Document Ingestion

```bash
# Basic ingestion
python ingest.py

# Or using make
make ingest
```

The ingestion process:
1. Scans `docs/` for PDF files
2. Loads and splits each PDF into chunks
3. Generates embeddings for each chunk
4. Stores everything in ChromaDB
5. Creates logs in `ingest.log`

### CLI Interface (Legacy)

```bash
# Start CLI chat
python chat.py

# Or using make
make chat
```

CLI features:
- Interactive CLI with colored prompts
- Streaming responses
- Source citations with page numbers
- Response time tracking
- Conversation logging in `chat.log`
- Exit with `exit`, `quit`, `q`, or Ctrl+C

### Example Web Interface Interaction

The web interface provides a modern chat experience where users can:

1. **Ask Questions**: Type questions in the chat input field
2. **Get Responses**: Receive AI-powered answers with source citations
3. **View Sources**: See which documents and page numbers were referenced
4. **Track Performance**: Monitor response times for each query
5. **Mobile Access**: Use the interface on any device with a web browser

### Example CLI Interaction

```
🤖 Welcome to RAG Chatbot!
Ask questions about your documents and get AI-powered answers.
Type 'exit', 'quit', or 'q' to stop the conversation.

❓ You: What are the main strategies for controlling diabetes?

🤖 RAGbot: Based on the document, the main strategies for controlling Type 2 diabetes naturally include:

1. **Dietary Management**: Following a low-carbohydrate diet, focusing on whole foods, and avoiding processed sugars and refined carbohydrates.

2. **Regular Exercise**: Incorporating both aerobic exercise and strength training to improve insulin sensitivity.

3. **Weight Management**: Maintaining a healthy weight through proper diet and exercise.

4. **Stress Management**: Using techniques like meditation, yoga, or other stress-reduction methods.

5. **Blood Sugar Monitoring**: Regular monitoring to track progress and adjust strategies as needed.

📚 Sources: A Comprehensive Plan to Control Type 2 Diabetes Naturally.pdf (page 1), A Comprehensive Plan to Control Type 2 Diabetes Naturally.pdf (page 3)
⏱️  Response time: 2.34s
```

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make help` | Show available commands |
| `make setup` | Complete setup (venv, deps, .env) |
| `make install` | Install dependencies only |
| `make ingest` | Run document ingestion |
| `make chat` | Start the CLI chatbot |
| `make web` | Start the FastAPI web interface |
| `make test` | Run basic functionality tests |
| `make clean` | Clean up generated files |

## Testing

### Basic Functionality Test

```bash
make test
```

This checks:
- Python package imports
- docs/ directory and PDF files
- .env file existence
- Database directory

### Manual Testing

1. **Ingestion Test (AC-1)**:
   ```bash
   python ingest.py
   # Should complete without errors
   ```

2. **Chat Test (AC-2)**:
   ```bash
   python chat.py
   # Ask: "What is this document about?"
   # Should return answer with citations
   ```

3. **Performance Test (AC-3)**:
   - Monitor response times in chat interface
   - Should be ≤ 3 seconds for typical queries

## Troubleshooting

### Common Issues

1. **"Missing required environment variables"**
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key

2. **"Database directory not found"**
   - Run `python ingest.py` first

3. **"No documents found in the knowledge base"**
   - Ensure PDFs exist in `docs/` folder
   - Re-run ingestion

4. **Import errors**
   - Activate virtual environment
   - Run `pip install -r requirements.txt`

5. **OpenAI API errors**
   - Check API key validity
   - Verify account has sufficient credits

### Logs

- `ingest.log`: Document ingestion logs
- `chat.log`: Chat interaction logs
- Both scripts log to console and files

## Performance Monitoring with Langsmith

The system includes integrated Langsmith tracking for comprehensive monitoring of model usage, performance metrics, and debugging capabilities.

### Tracked Operations

1. **Document Ingestion (`ingest.py`)**:
   - `@traceable(name="document_ingestion")`: Main ingestion process
   - `@traceable(name="load_and_split_pdf")`: Individual PDF processing
   - Metrics: Total ingestion time, embedding generation time, batch processing times

2. **Chat Interface (`chat.py`)**:
   - `@traceable(name="initialize_vectorstore")`: Vector store loading
   - `@traceable(name="process_query")`: Query processing and response generation
   - Metrics: Vector store load time, query response time, retrieval performance

### Performance Metrics Logged

- **Ingestion Process**:
  - PDF loading and splitting time per document
  - Embedding generation time per batch
  - Total ingestion time and document count
  - Individual batch processing times

- **Chat Queries**:
  - Vector store initialization time
  - Query processing and response generation time
  - Retrieved document count and relevance scores
  - OpenAI API call latencies

### Setup Instructions

1. **Create Langsmith Account**:
   ```bash
   # Visit https://smith.langchain.com/ and sign up
   ```

2. **Configure Environment**:
   ```bash
   # Add to your .env file
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langsmith_api_key_here
   LANGCHAIN_PROJECT=simple-rag-chatbot
   ```

3. **View Traces**:
   - Access your Langsmith dashboard
   - Monitor real-time performance metrics
   - Debug failed operations with detailed traces
   - Analyze cost and usage patterns

### Benefits

- **Cost Tracking**: Monitor OpenAI API usage and costs
- **Performance Optimization**: Identify bottlenecks in ingestion and queries
- **Error Debugging**: Detailed traces for failed operations
- **Usage Analytics**: Understand user interaction patterns
- **Model Comparison**: A/B test different models and configurations

## Performance Considerations

- **Chunk Size**: Larger chunks provide more context but slower retrieval
- **Retrieval K**: More chunks provide better context but increase latency
- **Batch Size**: Larger batches use more memory but faster ingestion
- **Model Choice**: `gpt-4o-mini` is faster and cheaper than `gpt-4o`
- **Langsmith Overhead**: Minimal performance impact (~1-2% latency increase)

## Security Notes

- Store OpenAI API keys securely in `.env` file
- `.env` file is gitignored to prevent accidental commits
- No external network calls except to OpenAI API
- Local ChromaDB storage for privacy

## Extending the System

### Adding New Document Types

Modify `ingest.py` to support additional loaders:
```python
from langchain.document_loaders import TextLoader, CSVLoader

# Add new file type handling
if file_path.suffix == '.txt':
    loader = TextLoader(str(file_path))
elif file_path.suffix == '.csv':
    loader = CSVLoader(str(file_path))
```

### Custom Embedding Models

Switch to local embeddings for privacy:
```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### Additional Web Features

The FastAPI web interface can be extended with:

- **User Authentication**: Add login/logout functionality
- **Document Upload**: Allow users to upload PDFs via web interface
- **Conversation History**: Persist chat sessions in database
- **Advanced Filters**: Filter by document type, date, relevance
- **Export Options**: Export conversations or search results

## Dependencies

Core dependencies in `requirements.txt`:
- `langchain==0.1.0`: LLM application framework
- `langchain-openai==0.0.5`: OpenAI integration
- `chromadb==0.4.22`: Vector database
- `openai>=1.10.0,<2.0.0`: OpenAI API client
- `pypdf==3.17.3`: PDF processing
- `python-dotenv==1.0.0`: Environment variables
- `langsmith==0.0.87`: Performance monitoring and tracing
- `fastapi==0.104.1`: Modern web framework for APIs
- `uvicorn[standard]==0.24.0`: ASGI server for FastAPI

## Performance Targets

- **Ingestion**: Process ~5 PDFs in under 2 minutes
- **Query Response**: ≤ 3 seconds per query
- **Memory Usage**: < 1GB RAM for typical documents
- **Storage**: ~10MB per 100 pages of PDFs

## Future Enhancements

- [x] **Langsmith Integration**: Performance monitoring and model usage tracking
- [x] **FastAPI Web Interface**: Modern web UI with responsive design
- [ ] **User Authentication**: Login/logout and user management
- [ ] **Document Upload**: Web-based PDF upload functionality
- [ ] Support for more document types (Word, HTML, Markdown)
- [ ] Advanced chunking strategies
- [ ] Multi-user support with session management
- [ ] Cloud deployment options (Docker, AWS, GCP)
- [ ] Integration with managed vector databases
- [ ] Conversation history persistence
- [ ] Advanced search filters and export options