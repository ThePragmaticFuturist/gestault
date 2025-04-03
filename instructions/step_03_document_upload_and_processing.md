# Step 3: Document Upload and Background Processing #

1.  **Install New Dependencies:**
    We need libraries for handling different file formats and detecting file types. We'll also use LangChain's text splitter for convenience.
    ```bash
    pip install python-docx pypdf beautifulsoup4 lxml langchain-text-splitters python-magic aiofiles
    ```
    *   `python-docx`: For reading `.docx` files.
    *   `pypdf`: For reading `.pdf` files.
    *   `beautifulsoup4`: For potentially parsing HTML/XML.
    *   `lxml`: A fast parser often used with BeautifulSoup.
    *   `langchain-text-splitters`: Provides various text splitting algorithms.
    *   `python-magic`: For reliable file type detection (might require installing `libmagic1` on Debian/Ubuntu or `libmagic` on macOS/other Linux distros via system package manager, e.g., `sudo apt-get update && sudo apt-get install -y libmagic1` or `brew install libmagic`).
    *   `aiofiles`: For asynchronous file operations (reading/writing temporary files).

2.  **Update Configuration (`core/config.py`):**
    Add settings for the upload directory and chunking parameters.

    ```python
    # core/config.py
    import os
    from pydantic_settings import BaseSettings
    from pydantic import Field # Import Field
    from pathlib import Path
    import tempfile # Import tempfile

    BASE_DIR = Path(__file__).resolve().parent.parent

    class Settings(BaseSettings):
        # --- General Settings ---
        APP_NAME: str = "RAG LLM Server"
        BASE_DIR: Path = BASE_DIR

        # --- Database Settings ---
        DATA_DIR: Path = BASE_DIR / "data"
        SQLITE_DB_NAME: str = "rag_server.db"
        CHROMA_DB_DIR_NAME: str = "chroma_db"

        # --- Calculated Paths ---
        @property
        def SQLITE_DATABASE_URL(self) -> str:
            db_dir = self.DATA_DIR
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / self.SQLITE_DB_NAME
            return f"sqlite+aiosqlite:///{db_path.resolve()}"

        @property
        def CHROMA_PERSIST_PATH(self) -> str:
            chroma_path = self.DATA_DIR / self.CHROMA_DB_DIR_NAME
            chroma_path.mkdir(parents=True, exist_ok=True)
            return str(chroma_path.resolve())

        # --- LLM Settings (Placeholders) ---
        DEFAULT_MODEL_NAME: str = "placeholder-model"
        HUGGINGFACE_HUB_CACHE: Path = BASE_DIR / "models_cache"

        # --- Server Settings ---
        SERVER_HOST: str = "0.0.0.0"
        SERVER_PORT: int = 8000

        # --- ChromaDB Settings ---
        DOCUMENT_COLLECTION_NAME: str = "rag_documents"
        SESSION_COLLECTION_NAME: str = "chat_sessions"

        # --- NEW: Document Processing Settings ---
        # Use system's temp dir for uploads before processing
        UPLOAD_DIR: Path = Path(tempfile.gettempdir()) / "rag_server_uploads"
        CHUNK_SIZE: int = Field(default=1000, description="Target size for text chunks (in characters)")
        CHUNK_OVERLAP: int = Field(default=150, description="Overlap between consecutive chunks (in characters)")
        SUPPORTED_FILE_TYPES: list[str] = Field(default=[
            "text/plain",
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # .docx
            "text/markdown",
            "text/html",
        ])

        # --- Pydantic settings configuration ---
        class Config:
            env_file = '.env'
            env_file_encoding = 'utf-8'
            extra = 'ignore'

    settings = Settings()

    # --- Ensure necessary directories exist on import ---
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.HUGGINGFACE_HUB_CACHE.mkdir(parents=True, exist_ok=True)
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True) # Create upload dir
    _ = settings.SQLITE_DATABASE_URL
    _ = settings.CHROMA_PERSIST_PATH

    print(f"--- Configuration Loaded ---")
    print(f"SQLite DB URL: {settings.SQLITE_DATABASE_URL}")
    print(f"ChromaDB Path: {settings.CHROMA_PERSIST_PATH}")
    print(f"Upload Dir: {settings.UPLOAD_DIR}") # Print upload dir
    print(f"--------------------------")
    ```
    *   Added `UPLOAD_DIR`: Uses the system's temporary directory to store uploads during processing.
    *   Added `CHUNK_SIZE`, `CHUNK_OVERLAP`: Parameters for the text splitter.
    *   Added `SUPPORTED_FILE_TYPES`: A list of MIME types we intend to support.

3.  **Create Service for Document Processing (`services/document_processor.py`):**
    This module will contain the core logic for handling files, extracting text, chunking, and interacting with the database.

    ```python
    # services/document_processor.py
    import os
    import uuid
    import magic # For file type detection
    import traceback
    import logging
    from pathlib import Path
    from typing import List, Dict, Optional, Tuple

    import docx # python-docx
    import pypdf # pypdf
    from bs4 import BeautifulSoup
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import aiofiles # For async file operations

    from core.config import settings
    from db.database import get_sqlite_db
    from db.models import documents_table, document_chunks_table

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # --- Text Extraction Functions ---

    async def extract_text_from_txt(file_path: Path) -> str:
        async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
            return await f.read()

    async def extract_text_from_docx(file_path: Path) -> str:
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs if para.text])
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {e}")
            raise # Re-raise to be caught by the main processor

    async def extract_text_from_pdf(file_path: Path) -> str:
        try:
            reader = pypdf.PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            raise

    async def extract_text_from_html(file_path: Path) -> str:
        async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
            content = await f.read()
        soup = BeautifulSoup(content, 'lxml')
        # Basic extraction: get text from body, remove scripts/styles
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        # Get text, strip leading/trailing whitespace, join lines
        text = ' '.join(t.strip() for t in soup.stripped_strings)
        return text


    # --- Text Chunking Function ---

    def chunk_text(text: str) -> List[str]:
        """Splits text into chunks using RecursiveCharacterTextSplitter."""
        if not text:
            return []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_text(text)

    # --- Main Background Task Function ---

    async def process_document_upload(temp_file_path: Path, doc_id: str, filename: str):
        """
        Background task to process an uploaded document:
        1. Detect file type.
        2. Extract text.
        3. Chunk text.
        4. Save chunks to database.
        5. Update document status.
        6. Clean up temp file.
        """
        db = get_sqlite_db() # Get the database connection instance
        final_status = "failed" # Assume failure initially
        error_message = None
        extracted_text = ""
        chunks = []
        detected_mime_type = "unknown"

        try:
            logger.info(f"[{doc_id}] Starting processing for: {filename}")

            # 1. Detect file type using python-magic
            try:
                detected_mime_type = magic.from_file(str(temp_file_path), mime=True)
                logger.info(f"[{doc_id}] Detected MIME type: {detected_mime_type}")
                if detected_mime_type not in settings.SUPPORTED_FILE_TYPES:
                    raise ValueError(f"Unsupported file type: {detected_mime_type}")
            except Exception as e:
                logger.error(f"[{doc_id}] Error detecting file type or unsupported: {e}")
                raise ValueError(f"Could not process file type: {e}") # Raise specific error

            # Update status to 'processing'
            await db.execute(
                query=documents_table.update().where(documents_table.c.id == doc_id),
                values={"processing_status": "processing", "metadata": {"mime_type": detected_mime_type}}
            )

            # 2. Extract text based on MIME type
            logger.info(f"[{doc_id}] Extracting text...")
            if detected_mime_type == "application/pdf":
                extracted_text = await extract_text_from_pdf(temp_file_path)
            elif detected_mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                extracted_text = await extract_text_from_docx(temp_file_path)
            elif detected_mime_type == "text/plain" or detected_mime_type == "text/markdown":
                extracted_text = await extract_text_from_txt(temp_file_path)
            elif detected_mime_type == "text/html":
                 extracted_text = await extract_text_from_html(temp_file_path)
            else:
                # Should have been caught earlier, but as a safeguard:
                raise ValueError(f"Extraction logic not implemented for: {detected_mime_type}")

            if not extracted_text or extracted_text.isspace():
                logger.warning(f"[{doc_id}] No text extracted from {filename}.")
                # Decide if this is an error or just an empty doc
                # For now, let's treat it as 'completed' but with 0 chunks.
                final_status = "completed_empty" # Custom status
                chunks = []
            else:
                logger.info(f"[{doc_id}] Extracted {len(extracted_text)} characters.")
                await db.execute(
                    query=documents_table.update().where(documents_table.c.id == doc_id),
                    values={"processing_status": "chunking"}
                )

                # 3. Chunk text
                logger.info(f"[{doc_id}] Chunking text...")
                chunks = chunk_text(extracted_text)
                logger.info(f"[{doc_id}] Created {len(chunks)} chunks.")

            # 4. Save chunks to database
            if chunks:
                logger.info(f"[{doc_id}] Saving {len(chunks)} chunks to database...")
                chunk_records = []
                for i, chunk_content in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i}" # Simple, predictable chunk ID
                    chunk_records.append({
                        "id": chunk_id,
                        "document_id": doc_id,
                        "chunk_index": i,
                        "text_content": chunk_content,
                        "metadata": None, # Add metadata later if needed
                        "vector_id": None # Will be populated after embedding
                    })

                # Use execute_many for potential performance benefit
                if chunk_records:
                     # Ensure connection is active (important if task runs long)
                    async with db.transaction(): # Ensures atomicity for chunk insertion
                        await db.execute_many(query=document_chunks_table.insert(), values=chunk_records)
                    logger.info(f"[{doc_id}] Saved {len(chunks)} chunks.")
                # Status after successful chunking & saving
                # We'll use 'chunked' for now, later can be 'pending_embedding'
                final_status = "chunked"
            elif final_status != "completed_empty":
                # No chunks generated, and wasn't explicitly marked as empty
                 logger.warning(f"[{doc_id}] No chunks were generated for non-empty extracted text. Marking as 'chunked' with 0 chunks.")
                 final_status = "chunked" # Still mark as chunked, count will be 0

            # Update final status in 'documents' table
            await db.execute(
                query=documents_table.update().where(documents_table.c.id == doc_id),
                values={
                    "processing_status": final_status,
                    "chunk_count": len(chunks),
                    "error_message": None # Clear any previous error
                }
            )
            logger.info(f"[{doc_id}] Processing finished. Final status: {final_status}")

        except Exception as e:
            # 5. Handle errors during processing
            final_status = "failed"
            error_message = f"Error during processing: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error(f"[{doc_id}] Processing failed for {filename}: {error_message}")
            # Update status and error message in DB
            try:
                 # Update status and error message in DB
                await db.execute(
                    query=documents_table.update().where(documents_table.c.id == doc_id),
                    values={
                        "processing_status": final_status,
                        "error_message": error_message,
                        "chunk_count": 0 # Ensure chunk count is 0 on failure
                    }
                )
            except Exception as db_err:
                 logger.error(f"[{doc_id}] CRITICAL: Failed to update document status to 'failed' in DB after processing error: {db_err}")

        finally:
            # 6. Clean up the temporary file
            try:
                if temp_file_path.exists():
                    os.remove(temp_file_path)
                    logger.info(f"[{doc_id}] Cleaned up temporary file: {temp_file_path}")
            except OSError as e:
                logger.error(f"[{doc_id}] Error deleting temporary file {temp_file_path}: {e}")

    ```
    *   **Extraction Functions:** Separate async functions for each supported file type.
    *   **`chunk_text`:** Uses `RecursiveCharacterTextSplitter` from `langchain-text-splitters`.
    *   **`process_document_upload`:**
        *   Takes the temp file path, document ID, and original filename.
        *   Gets the async DB connection.
        *   Uses `python-magic` to detect MIME type (more reliable than extension).
        *   Updates the document status in SQLite at various stages (`processing`, `chunking`, `chunked`, `failed`).
        *   Calls the appropriate extraction function.
        *   Calls the chunking function.
        *   Generates unique IDs for chunks (simple sequential for now).
        *   Inserts chunk records into `document_chunks` table (using `execute_many`).
        *   Includes extensive logging and error handling (`try...except...finally`).
        *   Crucially, cleans up the temporary file in the `finally` block.

4.  **Create API Endpoint (`app/api/endpoints/documents.py`):**
    This file will contain the FastAPI router and the `/upload` endpoint.

    ```python
    # app/api/endpoints/documents.py
    import uuid
    import shutil
    import logging
    from pathlib import Path
    from typing import List, Optional

    from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends, status
    from pydantic import BaseModel, Field

    from core.config import settings
    from services.document_processor import process_document_upload # Import the background task
    from db.database import get_sqlite_db, database # Need async database access here too
    from db.models import documents_table

    logger = logging.getLogger(__name__)

    router = APIRouter()

    class UploadResponse(BaseModel):
        message: str
        document_id: str
        filename: str

    class DocumentStatus(BaseModel):
        id: str
        filename: str
        status: str = Field(alias="processing_status") # Map DB column to field name
        upload_time: str
        error_message: Optional[str] = None
        chunk_count: Optional[int] = None

        class Config:
            orm_mode = True # Enable mapping from DB RowProxy
            allow_population_by_field_name = True # Allow using alias


    @router.post(
        "/upload",
        response_model=UploadResponse,
        status_code=status.HTTP_202_ACCEPTED, # Use 202 Accepted for background tasks
        summary="Upload a document for processing",
        description="Uploads a file, saves it temporarily, creates a document record, "
                    "and schedules background processing (extraction, chunking).",
    )
    async def upload_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        # db: databases.Database = Depends(get_sqlite_db) # Inject DB connection
    ):
        """
        Handles file upload, creates initial DB record, and starts background processing.
        """
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Generate a unique document ID
        doc_id = str(uuid.uuid4())
        temp_file_path = settings.UPLOAD_DIR / f"{doc_id}_{file.filename}"

        logger.info(f"Received upload: {file.filename}, size: {file.size}, assigned ID: {doc_id}")

        # Save the uploaded file temporarily using a synchronous approach for simplicity here,
        # but ideally use aiofiles if dealing with very large files concurrently often.
        try:
            # Ensure the upload directory exists (should be handled by config loading, but check again)
            settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

            with temp_file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"Saved temporary file to: {temp_file_path}")
        except Exception as e:
            logger.error(f"Failed to save temporary file for {doc_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {e}")
        finally:
            # Close the file stream provided by FastAPI
             await file.close()


        # Insert initial record into the documents table
        insert_query = documents_table.insert().values(
            id=doc_id,
            filename=file.filename,
            source_path=str(temp_file_path), # Store temp path initially
            processing_status="pending",
            # metadata can be added here if needed immediately
        )
        try:
            # Use the globally available `database` instance directly
            await database.execute(insert_query)
            logger.info(f"[{doc_id}] Initial document record created in DB.")
        except Exception as e:
            logger.error(f"[{doc_id}] Failed to insert initial document record: {e}")
            # Clean up the saved file if DB insert fails
            if temp_file_path.exists():
                os.remove(temp_file_path)
            raise HTTPException(status_code=500, detail=f"Database error creating document record: {e}")


        # Add the processing task to the background
        background_tasks.add_task(
            process_document_upload, # The function to run
            temp_file_path=temp_file_path,
            doc_id=doc_id,
            filename=file.filename
        )
        logger.info(f"[{doc_id}] Background processing task scheduled for {file.filename}.")

        return UploadResponse(
            message="File upload accepted, processing started in background.",
            document_id=doc_id,
            filename=file.filename,
        )

    @router.get(
        "/status/{document_id}",
        response_model=DocumentStatus,
        summary="Get document processing status",
        description="Retrieves the current status and metadata for a given document ID.",
    )
    async def get_document_status(
        document_id: str,
        # db: databases.Database = Depends(get_sqlite_db) # Inject DB
    ):
        """
        Retrieves the processing status of a document by its ID.
        """
        select_query = documents_table.select().where(documents_table.c.id == document_id)
        result = await database.fetch_one(select_query) # Use global 'database'

        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

        # Map the RowProxy object to our Pydantic model
        # Note: Timestamps are often returned as datetime objects by the driver
        # Convert datetime to string if necessary for JSON serialization (FastAPI usually handles this)
        status_data = dict(result) # Convert RowProxy to dict
        # Ensure timestamp is string (isoformat)
        status_data["upload_time"] = result["upload_time"].isoformat() if result["upload_time"] else None

        # Map processing_status from DB to 'status' field in Pydantic model
        # Pydantic alias handles this if allow_population_by_field_name=True
        # return DocumentStatus(**status_data)
        # Use parse_obj for better validation with aliases
        return DocumentStatus.parse_obj(status_data)

    @router.get(
        "/",
        response_model=List[DocumentStatus],
        summary="List processed documents",
        description="Retrieves a list of all documents and their current status.",
    )
    async def list_documents(
        skip: int = 0,
        limit: int = 100,
        # db: databases.Database = Depends(get_sqlite_db) # Inject DB
    ):
        """
        Lists documents with pagination.
        """
        query = documents_table.select().offset(skip).limit(limit).order_by(documents_table.c.upload_time.desc())
        results = await database.fetch_all(query) # Use global 'database'

        # Convert results to DocumentStatus models
        doc_statuses = []
        for row in results:
            status_data = dict(row)
            status_data["upload_time"] = row["upload_time"].isoformat() if row["upload_time"] else None
            doc_statuses.append(DocumentStatus.parse_obj(status_data))

        return doc_statuses

    # TODO: Add endpoint to get specific document chunks?
    # TODO: Add endpoint to delete a document and its chunks/vectors?

    ```
    *   **Router:** Creates an `APIRouter`.
    *   **Models:** `UploadResponse` defines the success response for the upload. `DocumentStatus` defines the structure for retrieving document info. Note the use of `alias` and `Config` for mapping DB columns.
    *   **`/upload` Endpoint:**
        *   Accepts `UploadFile` and `BackgroundTasks`.
        *   Generates a unique `doc_id`.
        *   Constructs a temporary file path in the `UPLOAD_DIR`.
        *   Saves the uploaded file to the temporary path (using standard `shutil`, could be enhanced with `aiofiles` for true async I/O if needed).
        *   Inserts a record into the `documents` table with status `pending`. Uses the global `database` object directly (simpler than Depends for now, as it's available everywhere).
        *   Uses `background_tasks.add_task` to schedule `process_document_upload` to run after the response is sent.
        *   Returns a `202 Accepted` status code.
    *   **`/status/{document_id}` Endpoint:** Retrieves a document's record from the DB by ID and returns its status. Handles `404 Not Found`.
    *   **`/` Endpoint:** Lists documents with basic pagination.

5.  **Update Main Application (`app/main.py`):**
    Include the new documents router.

    ```python
    # app/main.py
    # ... (keep existing imports)

    # Import the new router
    from app.api.endpoints import documents

    # ... (keep database initialization logic)

    # --- FastAPI Application Setup ---
    app = FastAPI(
        title=settings.APP_NAME,
        description="API server for managing RAG documents, interacting with LLMs, and managing chat sessions.",
        version="0.1.0",
    )

    # --- Event Handlers for Database Connections ---
    # ... (keep startup/shutdown events)

    # --- Basic Root Endpoint ---
    # ... (keep root endpoint)


    # --- Include API Routers ---  ### MODIFIED SECTION ###
    # Add the documents router
    app.include_router(
        documents.router,
        prefix="/api/v1/documents", # Add a prefix for versioning/namespacing
        tags=["Documents"]          # Tag for OpenAPI documentation grouping
    )

    # Placeholder for Future Routers
    # Example: from app.api.endpoints import sessions, models, system
    # Example: app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["Sessions"])
    # Example: app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
    # Example: app.include_router(system.router, prefix="/api/v1/system", tags=["System Status"])
    ### END MODIFIED SECTION ###


    # --- Running the Server (for direct execution) ---
    # ... (keep if __name__ == "__main__": block)

    ```
    *   We import the `documents` router.
    *   We use `app.include_router` to add the document endpoints under the `/api/v1/documents` path prefix. The `tags` argument helps organize the automatic `/docs` page.

6.  **Run and Test:**
    *   Make sure `libmagic` is installed on your system if you haven't already (`sudo apt-get update && sudo apt-get install -y libmagic1` or `brew install libmagic`).
    *   Run the server: `uvicorn app.main:app --reload --reload-dirs app --reload-dirs core --reload-dirs db --reload-dirs services` (add `services` to reload dirs).
    *   Open the interactive API docs at `http://localhost:8000/docs`.
    *   Expand the "Documents" section.
    *   Try the `POST /api/v1/documents/upload` endpoint:
        *   Click "Try it out".
        *   Choose a file (`.txt`, `.pdf`, `.docx`).
        *   Click "Execute".
    *   You should get a `202 Accepted` response with the `document_id`.
    *   Check the server logs. You should see messages about receiving the file, saving it, creating the DB record, scheduling the task, and then messages from the background task itself (detecting type, extracting, chunking, saving chunks, updating status, cleaning up).
    *   Use the `GET /api/v1/documents/status/{document_id}` endpoint (paste the ID from the upload response) to check the status. It should change from `pending` to `processing` to `chunking` (if applicable) to `chunked` or `failed`.
    *   Use the `GET /api/v1/documents/` endpoint to see the list of uploaded documents.
    *   Check your SQLite database (`data/rag_server.db`) using a tool like DB Browser for SQLite. Verify records in `documents` (status, chunk_count, error_message) and `document_chunks` (text content, etc.). Check the `data/rag_server_uploads` directory in your system's temp folder is cleaned up after processing.

**Summary:**

We now have a functional document ingestion pipeline:
*   API endpoint to upload files.
*   Temporary storage of uploaded files.
*   Background processing using FastAPI's `BackgroundTasks`.
*   File type detection using `python-magic`.
*   Text extraction for `.txt`, `.pdf`, `.docx`, `.html`.
*   Text chunking using `langchain-text-splitters`.
*   Storage of document metadata and text chunks in the SQLite database.
*   Status tracking and error handling during processing.
*   API endpoints to check document status and list documents.

**Next Steps:**

The next logical step is to **integrate vector embedding**. This involves:

1.  Choosing an embedding model (e.g., from Hugging Face Sentence Transformers).
2.  Loading the embedding model (possibly on server startup or on demand).
3.  Modifying the document processing pipeline (or adding a new step/task) to:
    *   Take the text chunks from the `document_chunks` table (or directly after chunking).
    *   Generate vector embeddings for each chunk using the loaded model.
    *   Store these embeddings, along with chunk text and metadata (like document ID, chunk ID), into the ChromaDB vector store.
    *   Update the document status to `completed` or `embedding_failed`.
