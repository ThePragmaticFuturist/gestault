# Step 4: Vector Embedding Integration #

Let's add vector embedding to the pipeline. We'll use the popular `sentence-transformers` library to load a model from Hugging Face and generate embeddings, then store them in ChromaDB.

1.  **Install Dependencies:**
    ```bash
    pip install sentence-transformers torch torchvision torchaudio
    ```
    *   `sentence-transformers`: The core library for using embedding models easily.
    *   `torch`, `torchvision`, `torchaudio`: PyTorch is required by `sentence-transformers`. Install the version appropriate for your system (CPU or CUDA-enabled GPU). The command above usually fetches a suitable version. If you have a specific CUDA setup, you might consult the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

2.  **Update Configuration (`core/config.py`):**
    Add settings for the embedding model.

    ```python
    # core/config.py
    # ... (previous imports)

    class Settings(BaseSettings):
        # ... (previous settings)

        # --- NEW: Embedding Settings ---
        EMBEDDING_MODEL_NAME: str = Field(
            default="all-MiniLM-L6-v2", # Good default, balance of speed/performance
            description="The Hugging Face sentence-transformers model name to use for embeddings."
        )
        EMBEDDING_DEVICE: str = Field(
            default="cpu", # Default to CPU, change to "cuda" if GPU is available and configured
            description="Device to run the embedding model on ('cpu', 'cuda')."
        )
        EMBEDDING_BATCH_SIZE: int = Field(
            default=32, # Process chunks in batches for efficiency
            description="Batch size for generating embeddings."
        )

        # --- ChromaDB Settings ---
        DOCUMENT_COLLECTION_NAME: str = "rag_documents"
        SESSION_COLLECTION_NAME: str = "chat_sessions"

        # --- Document Processing Settings ---
        UPLOAD_DIR: Path = Path(tempfile.gettempdir()) / "rag_server_uploads"
        CHUNK_SIZE: int = Field(default=1000, description="Target size for text chunks (in characters)")
        CHUNK_OVERLAP: int = Field(default=150, description="Overlap between consecutive chunks (in characters)")
        SUPPORTED_FILE_TYPES: list[str] = Field(default=[...]) # Keep existing list

        # --- Pydantic settings configuration ---
        # ... (rest of the file)

    settings = Settings()

    # ... (directory creation and print statements)
    print(f"Embedding Model: {settings.EMBEDDING_MODEL_NAME} on {settings.EMBEDDING_DEVICE}")
    print(f"--------------------------")
    ```
    *   `EMBEDDING_MODEL_NAME`: Specifies the model. `all-MiniLM-L6-v2` is a good starting point.
    *   `EMBEDDING_DEVICE`: Controls whether to use CPU or GPU. Set to `"cuda"` if you have a compatible NVIDIA GPU and CUDA toolkit installed.
    *   `EMBEDDING_BATCH_SIZE`: How many chunks to process at once by the model.

3.  **Create Embedding Service (`services/embedding_service.py`):**
    This service will handle loading the model and generating embeddings.

    ```python
    # services/embedding_service.py
    import time
    import logging
    from typing import List, Optional
    import torch
    from sentence_transformers import SentenceTransformer
    from core.config import settings

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # --- Global variable to hold the loaded model ---
    # Avoids reloading the model on every request/task
    _embedding_model: Optional[SentenceTransformer] = None

    def _load_embedding_model() -> SentenceTransformer:
        """Loads the Sentence Transformer model."""
        global _embedding_model
        if _embedding_model is None:
            logger.info(f"Loading embedding model '{settings.EMBEDDING_MODEL_NAME}' onto device '{settings.EMBEDDING_DEVICE}'...")
            start_time = time.time()
            try:
                # Check for CUDA availability if requested
                if settings.EMBEDDING_DEVICE == "cuda" and not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available. Falling back to CPU.")
                    settings.EMBEDDING_DEVICE = "cpu"

                _embedding_model = SentenceTransformer(
                    settings.EMBEDDING_MODEL_NAME,
                    device=settings.EMBEDDING_DEVICE,
                    cache_folder=str(settings.HUGGINGFACE_HUB_CACHE.resolve()) # Use configured cache path
                )
                # You might want to warm up the model here if needed
                # _embedding_model.encode(["Warm-up sentence."])
                load_time = time.time() - start_time
                logger.info(f"Embedding model loaded successfully in {load_time:.2f} seconds.")
            except Exception as e:
                logger.error(f"Failed to load embedding model '{settings.EMBEDDING_MODEL_NAME}': {e}")
                # Depending on requirements, you might want to raise the exception
                # or handle it gracefully (e.g., disable embedding features)
                raise RuntimeError(f"Could not load embedding model: {e}") from e
        return _embedding_model

    def get_embedding_model() -> SentenceTransformer:
        """Returns the loaded embedding model, loading it if necessary."""
        # This function ensures the model is loaded only once.
        return _load_embedding_model()

    def generate_embeddings(texts: List[str]) -> Optional[List[List[float]]]:
        """Generates embeddings for a list of texts using the loaded model."""
        if not texts:
            return []
        try:
            model = get_embedding_model()
            logger.info(f"Generating embeddings for {len(texts)} text chunks...")
            start_time = time.time()
            # Use model.encode for generating embeddings.
            # `show_progress_bar=False` as logs provide feedback.
            # Set batch size from config.
            embeddings = model.encode(
                texts,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=False # Keep as tensors for potential GPU efficiency, convert later if needed
            ) # final result is a list of lists for JSON/DB compatibility
            gen_time = time.time() - start_time
            logger.info(f"Generated {len(embeddings)} embeddings in {gen_time:.2f} seconds.")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return None or an empty list to indicate failure? Let's return None.
            return None

    # --- Optional: Pre-load model during startup ---
    # You could call get_embedding_model() once when the app starts
    # in main.py's startup event if you want the loading delay to happen then.

    ```
    *   **Global Model Variable:** `_embedding_model` stores the loaded model instance to prevent reloading.
    *   **`_load_embedding_model`:** Handles the actual loading logic, including device selection, cache folder setting, timing, and error handling. It updates the global variable.
    *   **`get_embedding_model`:** The public function to access the model. It calls `_load_embedding_model` only if the model isn't loaded yet.
    *   **`generate_embeddings`:** Takes a list of texts, gets the loaded model, uses `model.encode()` (with batching) to create embeddings, logs timing, handles errors, and returns embeddings as a list of lists of floats.

4.  **Update ChromaDB Initialization (Optional but Recommended):**
    While ChromaDB *can* use its own embedding function, we're generating embeddings manually. Let's ensure our `get_or_create_collections` doesn't unintentionally set one up that conflicts. We can also explicitly set the distance metric if desired.

    ```python
    # db/database.py
    # ... (imports and other code)

    # --- ChromaDB Collection Management ---
    def get_or_create_collections():
        """Ensures the necessary ChromaDB collections exist."""
        try:
            # Document collection
            doc_collection_name = settings.DOCUMENT_COLLECTION_NAME
            chroma_client.get_or_create_collection(
                name=doc_collection_name,
                # No embedding_function specified, as we generate manually
                metadata={"hnsw:space": "cosine"} # Specify cosine distance, common for sentence embeddings
            )
            print(f"ChromaDB document collection '{doc_collection_name}' ensured (metric: cosine).")

            # Session collection (for chat history embedding)
            session_collection_name = settings.SESSION_COLLECTION_NAME
            chroma_client.get_or_create_collection(
                name=session_collection_name,
                # No embedding_function specified
                metadata={"hnsw:space": "cosine"} # Use cosine here too
            )
            print(f"ChromaDB session collection '{session_collection_name}' ensured (metric: cosine).")

        except Exception as e:
            print(f"Error setting up ChromaDB collections: {e}")
            raise

    # ... (rest of the file)
    ```
    *   Added `metadata={"hnsw:space": "cosine"}`. This tells Chroma to use the cosine similarity metric when searching, which is generally suitable for sentence transformer embeddings.

5.  **Modify Document Processing Service (`services/document_processor.py`):**
    Integrate the embedding generation and ChromaDB storage into the background task.

    ```python
    # services/document_processor.py
    import os
    import uuid
    import magic
    import traceback
    import logging
    from pathlib import Path
    from typing import List, Dict, Optional, Tuple

    import docx
    import pypdf
    from bs4 import BeautifulSoup
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import aiofiles

    from core.config import settings
    # Import database and chroma client getters
    from db.database import get_sqlite_db, get_chroma_client
    from db.models import documents_table, document_chunks_table
    # Import the embedding service function
    from services.embedding_service import generate_embeddings

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # --- Text Extraction Functions (Keep as they are) ---
    # ... extract_text_from_txt, extract_text_from_docx, etc. ...

    # --- Text Chunking Function (Keep as it is) ---
    # ... chunk_text ...

    # --- Main Background Task Function (Modified) ---
    async def process_document_upload(temp_file_path: Path, doc_id: str, filename: str):
        """
        Background task: Extract, Chunk, Embed, Store.
        """
        db = get_sqlite_db()
        chroma_client = get_chroma_client()
        doc_collection = chroma_client.get_collection(settings.DOCUMENT_COLLECTION_NAME)

        final_status = "failed"
        error_message = None
        extracted_text = ""
        chunk_records_to_save: List[Dict] = [] # Store chunk data before inserting
        chunks_for_embedding: List[str] = []
        chunk_ids: List[str] = []
        detected_mime_type = "unknown"
        chunk_count = 0

        try:
            logger.info(f"[{doc_id}] Starting processing for: {filename}")

            # 1. Detect file type
            try:
                detected_mime_type = magic.from_file(str(temp_file_path), mime=True)
                logger.info(f"[{doc_id}] Detected MIME type: {detected_mime_type}")
                if detected_mime_type not in settings.SUPPORTED_FILE_TYPES:
                    raise ValueError(f"Unsupported file type: {detected_mime_type}")
            except Exception as e:
                logger.error(f"[{doc_id}] Error detecting file type or unsupported: {e}")
                raise ValueError(f"Could not process file type: {e}")

            await db.execute(
                query=documents_table.update().where(documents_table.c.id == doc_id),
                values={"processing_status": "processing", "metadata": {"mime_type": detected_mime_type}}
            )

            # 2. Extract text
            # ... (keep extraction logic as before) ...
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
                raise ValueError(f"Extraction logic not implemented for: {detected_mime_type}")


            if not extracted_text or extracted_text.isspace():
                logger.warning(f"[{doc_id}] No text extracted from {filename}. Marking as completed_empty.")
                final_status = "completed_empty"
                chunk_count = 0
            else:
                logger.info(f"[{doc_id}] Extracted {len(extracted_text)} characters.")
                await db.execute(
                    query=documents_table.update().where(documents_table.c.id == doc_id),
                    values={"processing_status": "chunking"}
                )

                # 3. Chunk text
                logger.info(f"[{doc_id}] Chunking text...")
                raw_chunks = chunk_text(extracted_text)
                chunk_count = len(raw_chunks)
                logger.info(f"[{doc_id}] Created {chunk_count} chunks.")

                if chunk_count > 0:
                    # Prepare chunk data for SQLite and for embedding
                    for i, chunk_content in enumerate(raw_chunks):
                        chunk_id = f"{doc_id}_chunk_{i}"
                        chunk_ids.append(chunk_id)
                        chunks_for_embedding.append(chunk_content)
                        chunk_records_to_save.append({
                            "id": chunk_id,
                            "document_id": doc_id,
                            "chunk_index": i,
                            "text_content": chunk_content,
                            "metadata": None,
                            "vector_id": chunk_id # Use chunk_id as the vector_id for simplicity
                        })

                    # 4. Save chunk metadata to SQLite
                    logger.info(f"[{doc_id}] Saving {chunk_count} chunk records to SQLite...")
                    async with db.transaction():
                        await db.execute_many(query=document_chunks_table.insert(), values=chunk_records_to_save)
                    logger.info(f"[{doc_id}] Saved chunk records to SQLite.")

                    # 5. Generate Embeddings
                    await db.execute(
                        query=documents_table.update().where(documents_table.c.id == doc_id),
                        values={"processing_status": "embedding"}
                    )
                    logger.info(f"[{doc_id}] Generating embeddings for {chunk_count} chunks...")
                    # NOTE: generate_embeddings is synchronous CPU/GPU bound code.
                    # Running it directly in the async task *will block* the event loop.
                    # For true async, need `run_in_executor` or a separate process/queue.
                    # For simplicity now, we accept the blocking nature here.
                    # Consider libraries like `httpx` or dedicated task queues (Celery, RQ, Arq)
                    # if the embedding time becomes problematic for server responsiveness.
                    embeddings = generate_embeddings(chunks_for_embedding)

                    if embeddings is None or len(embeddings) != chunk_count:
                        raise RuntimeError(f"Embedding generation failed or returned incorrect number of vectors.")

                    logger.info(f"[{doc_id}] Generated {len(embeddings)} embeddings.")

                    # 6. Store Embeddings in ChromaDB
                    logger.info(f"[{doc_id}] Adding {len(embeddings)} embeddings to ChromaDB collection '{settings.DOCUMENT_COLLECTION_NAME}'...")

                    # Prepare metadata for ChromaDB vectors
                    # Include document_id and original filename for filtering later
                    chroma_metadatas = [
                        {"document_id": doc_id, "filename": filename, "chunk_index": i}
                        for i in range(chunk_count)
                    ]

                    # Add to ChromaDB collection
                    # Use chunk_ids as the primary IDs in ChromaDB
                    try:
                        doc_collection.add(
                            ids=chunk_ids,
                            embeddings=embeddings,
                            documents=chunks_for_embedding, # Store the chunk text itself
                            metadatas=chroma_metadatas
                        )
                        logger.info(f"[{doc_id}] Successfully added embeddings to ChromaDB.")
                        final_status = "completed" # Final success state
                        await db.execute(
                            query=documents_table.update().where(documents_table.c.id == doc_id),
                            values={"vector_count": chunk_count} # Update vector count
                        )

                    except Exception as chroma_err:
                        logger.error(f"[{doc_id}] Failed to add embeddings to ChromaDB: {chroma_err}")
                        raise RuntimeError(f"ChromaDB add operation failed: {chroma_err}") from chroma_err

                else: # No chunks generated from non-empty text
                    logger.warning(f"[{doc_id}] No chunks generated, marking as 'completed_no_chunks'.")
                    final_status = "completed_no_chunks" # More descriptive status

            # Update final status in 'documents' table (outside the chunking 'if')
            await db.execute(
                query=documents_table.update().where(documents_table.c.id == doc_id),
                values={
                    "processing_status": final_status,
                    "chunk_count": chunk_count, # Already set
                    "error_message": None # Clear error on success
                }
            )
            logger.info(f"[{doc_id}] Processing finished. Final status: {final_status}")

        except Exception as e:
            final_status = "failed"
            # Distinguish between embedding failure and other failures if possible
            if "embedding" in str(e).lower() or "chroma" in str(e).lower():
                 final_status = "embedding_failed"

            error_message = f"Error during processing ({final_status}): {type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error(f"[{doc_id}] Processing failed for {filename}: {error_message}")
            try:
                await db.execute(
                    query=documents_table.update().where(documents_table.c.id == doc_id),
                    values={
                        "processing_status": final_status,
                        "error_message": error_message,
                        "chunk_count": chunk_count if chunk_count else 0, # Record chunk count even on failure
                        "vector_count": 0 # Ensure vector count is 0 on failure
                    }
                )
            except Exception as db_err:
                 logger.error(f"[{doc_id}] CRITICAL: Failed to update document status to '{final_status}' in DB after processing error: {db_err}")

        finally:
            # Clean up the temporary file
            try:
                if temp_file_path.exists():
                    os.remove(temp_file_path)
                    logger.info(f"[{doc_id}] Cleaned up temporary file: {temp_file_path}")
            except OSError as e:
                logger.error(f"[{doc_id}] Error deleting temporary file {temp_file_path}: {e}")

    ```
    *   **Imports:** Added imports for `get_chroma_client`, `generate_embeddings`.
    *   **Chroma Client:** Get the ChromaDB client and the specific document collection at the start.
    *   **Workflow:**
        1.  Detect type, update status (`processing`).
        2.  Extract text.
        3.  Update status (`chunking`).
        4.  Chunk text.
        5.  **If chunks exist:**
            *   Prepare chunk data for SQLite (`chunk_records_to_save`) and embedding (`chunks_for_embedding`, `chunk_ids`). Use the `chunk_id` as the `vector_id` in SQLite.
            *   Save chunk records to `document_chunks` table in SQLite.
            *   Update status (`embedding`).
            *   Call `generate_embeddings`. **Important Note:** This call is blocking! For high-throughput servers, you'd use `loop.run_in_executor` or a task queue (Celery, RQ, Arq) to run the CPU/GPU-bound embedding task without blocking the main async event loop. We'll keep it simple for now.
            *   Handle embedding failures.
            *   Prepare metadata for ChromaDB (including `document_id`, `filename`, `chunk_index`).
            *   Add embeddings, texts (documents), and metadata to the ChromaDB collection using `doc_collection.add()`.
            *   Handle ChromaDB errors.
            *   Update status to `completed` and set `vector_count` in SQLite.
        6.  **If no chunks:** Update status (`completed_empty` or `completed_no_chunks`).
    *   **Error Handling:** Updated error handling to potentially set `embedding_failed` status and log ChromaDB errors. Records `chunk_count` even if embedding fails. Ensures `vector_count` is 0 on failure.
    *   **Cleanup:** Still ensures temp file cleanup.

6.  **Modify Document API Endpoint (Optional):**
    Update the `DocumentStatus` model in `app/api/endpoints/documents.py` to include the `vector_count`.

    ```python
    # app/api/endpoints/documents.py
    # ... (imports)

    class DocumentStatus(BaseModel):
        id: str
        filename: str
        status: str = Field(alias="processing_status")
        upload_time: str
        error_message: Optional[str] = None
        chunk_count: Optional[int] = None
        vector_count: Optional[int] = None # <--- ADDED: Vector count

        class Config:
            orm_mode = True
            allow_population_by_field_name = True

    # ... (rest of the file, GET /status/{document_id} and GET / will now include vector_count if present)
    ```

7.  **Pre-load Model on Startup (Recommended):**
    To avoid the model loading delay on the first upload, trigger the loading during server startup.

    ```python
    # app/main.py
    # ... (imports)
    from services.embedding_service import get_embedding_model # Import the getter
    # ...

    # --- FastAPI Application Setup ---
    # ... (app definition)

    # --- Event Handlers for Database Connections ---
    @app.on_event("startup")
    async def startup_event():
        """Connect to database and pre-load models on server startup."""
        print("Server starting up...")
        await connect_db()
        # --- ADDED: Pre-load embedding model ---
        print("Pre-loading embedding model...")
        try:
            # This call will trigger the loading if not already loaded
            get_embedding_model()
            print("Embedding model loading initiated (check logs for completion).")
        except Exception as e:
            print(f"ERROR: Failed to pre-load embedding model during startup: {e}")
            # Decide if this is fatal or if the server can run without embeddings initially
        # --- END ADDED ---
        print("Server startup complete.")

    # ... (rest of the file)
    ```

8.  **Run and Test:**
    *   Restart the server: `uvicorn app.main:app --reload --reload-dirs app --reload-dirs core --reload-dirs db --reload-dirs services`
    *   Observe the startup logs. You should see messages about the embedding model loading (this might take some time on the first run as the model is downloaded).
    *   Upload a document via `POST /api/v1/documents/upload`.
    *   Monitor the server logs. You should see:
        *   File processing steps (extraction, chunking).
        *   Saving chunks to SQLite.
        *   Status update to `embedding`.
        *   Messages from `embedding_service` about generating embeddings.
        *   Messages about adding embeddings to ChromaDB.
        *   Final status update (hopefully `completed`).
    *   Check the document status using `GET /api/v1/documents/status/{document_id}`. Verify the status is `completed` and `vector_count` matches `chunk_count`.
    *   Inspect the SQLite `document_chunks` table: the `vector_id` column should be populated (with the chunk ID in our case).
    *   **Verify in ChromaDB (Manual Check):** You might need a separate small script or an interactive Python session to query ChromaDB directly for now:
        ```python
        import chromadb
        from core.config import settings # Your project's settings

        client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_PATH)
        collection = client.get_collection(settings.DOCUMENT_COLLECTION_NAME)

        print(f"Collection: {collection.name}")
        print(f"Number of items: {collection.count()}")

        # Optional: Get a few items to inspect
        results = collection.get(limit=5, include=['metadatas', 'documents'])
        print("\nSample items in ChromaDB:")
        print(results)

        # Optional: Query (we'll build API endpoints for this later)
        # query_results = collection.query(
        #     query_texts=["some query text"],
        #     n_results=2,
        #     include=['metadatas', 'documents', 'distances']
        # )
        # print("\nSample Query Results:")
        # print(query_results)
        ```

**Summary:**

We have now successfully integrated vector embedding:
*   Configured an embedding model (`sentence-transformers`).
*   Created a service to load the model efficiently and generate embeddings.
*   Modified the document processing task to generate embeddings for text chunks after saving them to SQLite.
*   Stored the embeddings, chunk text, and relevant metadata (document ID, filename, chunk index) into ChromaDB.
*   Updated the document status to `completed` upon successful embedding.
*   Added status (`embedding`, `embedding_failed`, `completed`) and counts (`vector_count`) tracking.

**Next Steps:**

With documents uploaded, chunked, and embedded, the next logical step is to implement the **Search Functionality**:

1.  Create API endpoints for searching:
    *   Keyword search (potentially using SQLite's FTS capabilities or simple `LIKE` queries on `document_chunks`).
    *   Semantic search (vector search) using ChromaDB.
2.  Implement the search logic:
    *   For semantic search: Take a query text, generate its embedding using the *same* embedding model, and query the ChromaDB collection.
    *   Return relevant chunks (text, metadata) based on the search results.
