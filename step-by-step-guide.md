**Step-by-Step Annotated Guide to Building Your Own RAG LLM Server**
1.  **Step 1: Basic Server Structure (FastAPI)**
    *   **What:** Set up a minimal FastAPI application (`app/main.py`) run by Uvicorn. Created a root endpoint (`/`).
    *   **Why:** To establish the core web server framework using a modern, high-performance, async-capable library. Provides basic API structure and automatic documentation (`/docs`).

2.  **Step 2: Configuration & Database Initialization**
    *   **What:** Defined settings using Pydantic (`core/config.py`), set up SQLite table schemas using SQLAlchemy Core (`db/models.py`), initialized ChromaDB client, and managed database connections (`db/database.py`). Created data directories.
    *   **Why:** Centralize configuration; define persistent storage for metadata (SQLite) and vectors (ChromaDB); ensure database connections are handled correctly during server startup/shutdown.

3.  **Step 3: Document Upload and Background Processing**
    *   **What:** Created API endpoints for file upload (`app/api/endpoints/documents.py`). Implemented file handling, text extraction (`services/document_processor.py` using `python-docx`, `pypdf`, `beautifulsoup4`), chunking (`langchain-text-splitters`), and initial SQLite record creation. Used FastAPI `BackgroundTasks` for processing.
    *   **Why:** To allow users to ingest documents; handle different file types; break large documents into manageable chunks; offload time-consuming processing from the initial API request for better responsiveness.

4.  **Step 4: Vector Embedding Integration**
    *   **What:** Integrated `sentence-transformers` (`services/embedding_service.py`) to load an embedding model (pre-loaded on startup). Modified the background task to generate embeddings for chunks and store them, along with text and metadata, in ChromaDB. Updated document status in SQLite.
    *   **Why:** To convert text chunks into numerical representations (vectors) suitable for semantic similarity search, enabling the core RAG functionality.

5.  **Step 5: Search Endpoints**
    *   **What:** Added API endpoints for keyword search (using SQL `LIKE` against SQLite chunks) and semantic search (embedding the query and querying ChromaDB) within `app/api/endpoints/documents.py`.
    *   **Why:** To provide tools for retrieving relevant document chunks based on user queries, forming the retrieval part of RAG.

6.  **Step 6: Chat Session and Message Management API**
    *   **What:** Created API models (`app/api/models/chat.py`) and endpoints (`app/api/endpoints/sessions.py`) for managing chat sessions (CRUD) and messages (add/list). Stored session metadata and message history in SQLite tables (`sessions`, `chat_messages`).
    *   **Why:** To maintain conversation state, link RAG documents to sessions, and store the prompt-response history needed for context and future interaction.

7.  **Step 7: Integrate RAG and Placeholder LLM Call**
    *   **What:** Modified the "add message" endpoint to: fetch associated RAG documents, perform semantic search based on the user query, retrieve chat history, construct a detailed prompt (including context/history), simulate an LLM call, and store both user/assistant messages. Added context truncation logic. Added summarization step.
    *   **Why:** To implement the full RAG loop within a chat context, assembling the necessary information before calling the LLM and handling context window limits. The placeholder simulates the final generation step.

8.  **Step 8: LLM Loading and Management API**
    *   **What:** Introduced an abstraction layer (`services/llm_backends.py`) for different LLM backends (local, Ollama, vLLM). Refactored `services/llm_service.py` to manage the active backend. Created API endpoints (`app/api/endpoints/models.py`) to list available models, dynamically load/configure the active backend and model (via API request or config defaults), check status, and update generation parameters. Handled local model loading (including quantization via `bitsandbytes`) in background threads.
    *   **Why:** To provide flexibility in choosing the LLM inference source without changing the core chat logic; manage the complex process of loading/unloading local models safely; allow runtime configuration.

9.  **Step 9: Real LLM Integration & Final Auth/Fixes**
    *   **What:** Replaced the placeholder LLM call with a call to the active backend via `llm_service.generate_text`, ensuring blocking calls run in an executor. Added a startup check (`app/main.py`) for Hugging Face authentication (`huggingface_hub` token via env/`.env` or CLI cache). Debugged various errors related to argument passing, state access, and library interactions.
    *   **Why:** To connect the RAG pipeline to an actual LLM for response generation; ensure prerequisites for gated models are met; resolve runtime errors for a functional system.

10. **Step 10: System Status API**
    *   **What:** Created API endpoints (`app/api/endpoints/system.py`) and a service (`services/system_service.py`) using `psutil` and `pynvml` to report CPU, RAM, Disk, GPU (optional), and Temperature (optional) metrics. Added NVML shutdown to the FastAPI shutdown event.
    *   **Why:** To provide visibility into the server's resource usage and operational health, aiding monitoring and debugging.
