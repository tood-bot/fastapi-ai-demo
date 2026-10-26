
# PRD — Simple RAG Chatbot (CLI) with LangChain & ChromaDB

---

## 1. Purpose
Build a lightweight Retrieval‑Augmented Generation (RAG) chatbot that:
1. **Indexes** all PDF documents located in a `docs/` sub‑folder.
2. **Stores** embeddings and metadata in a local ChromaDB instance.
3. **Answers** user questions from the terminal by retrieving relevant chunks and generating concise responses via an LLM (e.g., OpenAI GPT‑4o).

---

## 2. Goals & Success Metrics
| Goal | Metric | Target |
|------|--------|--------|
| Accurate retrieval | Top‑k retrieved chunks are relevant | ≥ 90% manual relevance on test set |
| Fast response | End‑to‑end latency per query | ≤ 3 s on laptop CPU |
| Easy setup | One‑command install & run | `make setup` then `python chat.py` |

---

## 3. Background
Many teams want an offline, scriptable RAG reference. This project demonstrates a **minimal two‑file** pattern—one ingestion script, one chat script—using the LangChain and ChromaDB ecosystem.

---

## 4. Scope
### 4.1 In Scope
* Local ingestion of PDFs in `docs/`
* Embedding with `langchain.embeddings.OpenAIEmbeddings` (pluggable)
* Storage in **ChromaDB** (persistent, local)
* CLI chat with streaming answers
* Basic error handling & logging

### 4.2 Out of Scope
* Web UI, API server, or multi‑user auth
* Non‑PDF formats (can be future work)
* Cloud deployment / vector DB as a service

---

## 5. Functional Requirements
| ID | Requirement |
|----|-------------|
| **FR‑1** | The system **ingests** every PDF in `docs/` at startup or on demand. |
| **FR‑2** | Embeddings are generated in batches (configurable batch size). |
| **FR‑3** | Metadata includes `source_path`, `page_number`, and `chunk_id`. |
| **FR‑4** | The **chat** script loads the same ChromaDB collection. |
| **FR‑5** | For each user query, top‑k (default = 4) chunks are retrieved. |
| **FR‑6** | Retrieved chunks are passed to an LLM via LangChain `ConversationalRetrievalChain`. |
| **FR‑7** | The terminal prints the answer and, optionally, citations. |

---

## 6. Non‑Functional Requirements
* **Performance:** ≤ 3 s latency on an M1 MacBook @ 4 chunks, GPT‑4o.
* **Extensibility:** Swap embedding/LLM classes via `.env` or CLI flags.
* **Reliability:** Graceful failure if PDFs are missing or ChromaDB is corrupt.
* **Security:** No external network calls beyond the LLM provider.

---

## 7. User Stories
1. **As a user**, I want to run `python ingest.py` so that my PDFs are indexed.
2. **As a user**, I want to ask “_What is the refund policy?_” and get an answer sourced from my PDFs.
3. **As a maintainer**, I want clear logging so I can debug ingestion failures.

---

## 8. Technical Design

### 8.1 Architecture (Textual)
`PDFs → LangChain loaders (PyPDFLoader) → Splitter (RecursiveCharacterTextSplitter) → Embeddings → ChromaDB (persistent) ← Retriever ← ConversationalRetrievalChain ← LLM (GPT‑4o)`

### 8.2 File & Folder Structure
```
rag-chatbot/
├── docs/               # PDF knowledge base
├── ingest.py           # Ingest & index PDFs
├── chat.py             # CLI chat interface
├── .env.example        # API keys & config
└── requirements.txt
```

### 8.3 Key Components
| File | Responsibility |
|------|----------------|
| `ingest.py` | Scan `docs/`, split pages into chunks, embed, and upsert into ChromaDB collection `pdf_knowledge`. |
| `chat.py` | Load collection, build retrieval‑augmented chain, prompt user for queries in REPL loop, stream answers. |

#### `ingest.py` High‑Level Flow
1. Load `.env` (API keys, chunk size, batch size, persist_directory).
2. Initialize `Chroma` with `persist_directory="db"` and `collection_name="pdf_knowledge"`.
3. For each PDF:
   * Use `PyPDFLoader` → `load_and_split`.
   * Add `metadata={"source": path, "page": page_no}`.
4. Embed with OpenAI or HuggingFace embeddings.
5. `collection.add_texts(texts, metadatas)`.
6. Persist DB (`client.persist()`).

#### `chat.py` High‑Level Flow
1. Load `.env`.
2. Instantiate retriever: `client.get_collection("pdf_knowledge").as_retriever(search_type="similarity", k=4)`.
3. Wrap in `ConversationalRetrievalChain` (memory optional).
4. Start REPL:

```python
while True:
    query = input("❓  You: ")
    if query.lower() in {"exit", "quit"}:
        break
    result = qa({"question": query, "chat_history": chat_history})
    print("🤖 RAGbot:", result["answer"])
    chat_history.append((query, result["answer"]))
```

### 8.4 Environment Variables
| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Key for GPT‑4o or chosen model. |
| `EMBEDDING_MODEL` | e.g., `text-embedding-3-small`. |
| `PERSIST_DIR` | Folder for Chroma files (default: `db`). |

### 8.5 Installation & Setup
```bash
git clone https://github.com/your‑org/rag-chatbot.git
cd rag-chatbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your keys
python ingest.py     # index PDFs
python chat.py       # start chatting
```

---

## 9. Acceptance Criteria
* AC‑1: Running `python ingest.py` with 5 sample PDFs completes without errors.
* AC‑2: `chat.py` returns an answer containing at least one citation token like “(source.pdf, p. 2)”.
* AC‑3: Latency ≤ 3 s for queries under 100 tokens.

---

## 10. Timeline (Tentative)
| Milestone | Deliverable | Owner | ETA |
|-----------|-------------|-------|-----|
| M1 | Repo scaffolding & env setup | Dev | Day 1 |
| M2 | `ingest.py` functional | Dev | Day 2 |
| M3 | `chat.py` functional | Dev | Day 3 |
| M4 | README & sample PDFs | Dev | Day 3 |
| M5 | Testing & polish | Dev | Day 4 |

---

## 11. Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|-----------|
| OpenAI API quota limits | High | Allow HF embeddings / LLM fallback; caching. |
| Large PDFs exceed token limits | Medium | Recursive splitter, chunk size tuning. |
| Non‑PDF files present | Low | Filter by `.pdf` extension. |

---

## 12. Future Enhancements
* Add web UI (Streamlit, Next.js).
* Support additional file types (Markdown, HTML).
* Switch to managed vector DB (Pinecone, Weaviate).
* Multi‑user conversation context with auth.

---
