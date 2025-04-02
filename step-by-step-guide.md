**Step-by-Step Annotated Guide to Building Your Own RAG LLM Server**

Here's a breakdown of the thought process behind generating the initial project structure and `main.py` for the RAG LLM server:

1.  **Deconstruct the Request:**

    *   **Core Goal:** Build a RAG LLM server, step-by-step.
    *   **Functionality:**
        *   Document Management: Upload, chunk, embed, manage (CRUD?). Store in SQLite/ChromaDB. Accessible by ID.
        *   Document Processing: Use BeautifulSoup (HTML/XML), python-docx (DOCX), `docling` (needs clarification, assumed generic text processing/chunking for now).
        *   Search: Keyword and Semantic search over documents.
        *   Chat Sessions: Threaded, store prompt/response pairs, RAG/embed these exchanges, session management (ID, timestamp, model, RAG docs, name, auto meta-tags).
        *   LLM Management: Load from HuggingFace CLI/local, status, list local models, modify settings, confirmation.
        *   System Monitoring: CPU/GPU status, temp, disk, RAM.
        *   API: Expose all functionality via an API.
    *   **Technology Stack:**
        *   Python
        *   API Framework: FastAPI (good choice for async, modern Python APIs)
        *   Databases: SQLite (metadata, sessions), ChromaDB (vector embeddings)
        *   Document Processing: BeautifulSoup, python-docx, `docling` (placeholder)
        *   LLM Interaction: Hugging Face libraries (`transformers`, maybe `accelerate`)
        *   System Monitoring: `psutil`, potentially GPU-specific libraries (like `pynvml` for NVIDIA).
    *   **Development Process:** Step-by-step, explain the code. Start with the most logical component.

2.  **Identify the "Most Logical Starting Component":**

    *   What's the foundation? The server itself. We need a way to run the code and expose endpoints. An API framework is the natural starting point.
    *   Within the API, what's the *most* fundamental piece? Just getting the server *running* and maybe a basic health check endpoint. This ensures the core infrastructure is in place before adding complex logic.
    *   Alternative starting points considered:
        *   *Document Processing:* Could start here, but without an API or database, it's just standalone scripts. Less integrated.
        *   *LLM Loading:* Complex, depends on external libraries and potentially hardware. Better to establish the server structure first.
        *   *Database Setup:* Important, but needs the API context to be truly useful.
    *   Conclusion: Start with the basic FastAPI application structure and a simple root or health endpoint.

3.  **Design the Project Structure:**

    *   A well-organized project is crucial for maintainability, especially as complexity grows.
    *   **`main.py`:** The entry point for the FastAPI application. Keeps the top level clean.
    *   **`app/` directory:** Contains the core application logic, separated by concern.
        *   **`api/`:** Holds the API endpoint definitions (routers). This promotes modularity. We'll have routers for documents, sessions, models, system status, etc.
        *   **`core/`:** Configuration, settings, maybe startup/shutdown events.
        *   **`services/`:** Business logic layer. Document processing, embedding, chat handling, LLM interaction, system monitoring logic will live here. Separates logic from API definitions.
        *   **`models/`:** Pydantic models for API request/response validation and data structures (e.g., Document, Session, ChatMessage). Also potentially database models if using an ORM later (though SQLite might be simple enough without one initially).
        *   **`db/`:** Database interaction logic. Functions to interact with SQLite and ChromaDB.
        *   **`utils/`:** Helper functions, common utilities (e.g., ID generation, logging setup).
    *   **`data/`:** Placeholder for persistent data (SQLite DB, ChromaDB files, potentially uploaded docs if not stored elsewhere). Should be in `.gitignore`.
    *   **`models_local/`:** Placeholder for locally downloaded LLM files. Should be in `.gitignore`.
    *   **`requirements.txt`:** Lists project dependencies. Essential for reproducibility.
    *   **`.env` (optional but recommended):** For environment variables (API keys, database paths, etc.). Should be in `.gitignore`.
    *   **`.gitignore`:** Standard practice to exclude unnecessary files/directories from version control.

4.  **Implement the First Step (Basic Server Setup):**

    *   **`requirements.txt`:** Start with the absolute minimum: `fastapi` and `uvicorn` (the ASGI server to run FastAPI).
    *   **`main.py`:**
        *   Import `FastAPI`.
        *   Create a `FastAPI` app instance (`app = FastAPI(...)`). Add a title and description for good measure (helps with docs).
        *   Define a simple root endpoint (`@app.get("/")`) that returns a welcome message. This serves as a basic health check and confirms the server is running.
    *   **Running the Server:** Explain the `uvicorn main:app --reload` command. `--reload` is crucial during development.

5.  **Explain the Code and Next Steps:**

    *   Clearly explain *why* this structure was chosen (modularity, maintainability).
    *   Explain the purpose of `main.py`, `FastAPI`, `@app.get("/")`, and `uvicorn`.
    *   Explicitly state the next logical step: Setting up configuration management and potentially the database connections, as these are foundational elements needed by many other components. This sets expectations for the next interaction.

6.  **Refinement during Thought Process:**

    *   Initial thought: Maybe put *all* API endpoints in `main.py`? No, that gets messy quickly. Routers are better (`app/api/`).
    *   Should database logic be in `services/`? No, better to have a dedicated `db/` layer for separation of concerns. Services *use* the DB layer.
    *   What about `docling`? It's not a standard library. Acknowledge this and treat it as a placeholder for now, focusing on the *concept* of document processing. We can integrate specific libraries later.
    *   How to handle threading for chat sessions? FastAPI is async, which handles I/O-bound concurrency well. CPU-bound tasks (like LLM inference) might need `run_in_threadpool` or separate worker processes later, but don't overcomplicate the initial setup. Acknowledge threading is needed *later*.
    *   Error handling? Add basic exception handlers later. Focus on the core structure first.

This structured approach ensures that the foundation is solid before adding more complex features.

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
