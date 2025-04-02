# app/api/endpoints/documents.py
import uuid
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends, status
from pydantic import BaseModel, Field

from core.config import settings
from services.document_processor import process_document_upload # Import the background task
from db.database import get_sqlite_db, database # Need async database access here too
from db.models import documents_table

import sqlalchemy # Import sqlalchemy for query building

logger = logging.getLogger(__name__)

router = APIRouter()

class UploadResponse(BaseModel):
    message: str
    document_id: str
    filename: str

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

class SearchQuery(BaseModel):
    query: str = Field(..., description="The search query text.")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of top results to return.")
    # Optional filter by document ID(s)
    document_ids: Optional[List[str]] = Field(default=None, description="Optional list of document IDs to search within.")

class ChunkResult(BaseModel):
    document_id: str
    chunk_id: str
    chunk_index: int
    text_content: str
    # For semantic search, include similarity score (distance)
    score: Optional[float] = None
    # Include other relevant metadata from the chunk or document
    metadata: Optional[Dict[str, Any]] = None # Store Chroma metadata or other info

class SearchResponse(BaseModel):
    results: List[ChunkResult]
    query: str
    search_type: str # 'keyword' or 'semantic'

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

# --- Dependency for ChromaDB client (optional but good practice) ---
# If we start using dependencies more, this pattern is cleaner
# from db.database import get_chroma_client
# def get_chroma(): return get_chroma_client()
# Then use: chroma_client: chromadb.ClientAPI = Depends(get_chroma)

# --- NEW: Keyword Search Endpoint ---
@router.post(
    "/search/keyword",
    response_model=SearchResponse,
    summary="Perform keyword search on document chunks",
    description="Searches the text content of document chunks using SQL LIKE queries.",
)
async def keyword_search(
    search_request: SearchQuery,
    # db: databases.Database = Depends(get_sqlite_db) # Inject if needed, or use global 'database'
):
    """
    Performs a case-insensitive keyword search on the 'text_content' field
    of the document chunks stored in SQLite.
    """
    logger.info(f"Keyword search request received: query='{search_request.query}', top_k={search_request.top_k}")

    # Build the base query
    query = document_chunks_table.select().where(
        # Case-insensitive LIKE search
        document_chunks_table.c.text_content.ilike(f"%{search_request.query}%")
    )

    # Apply document ID filter if provided
    if search_request.document_ids:
        query = query.where(document_chunks_table.c.document_id.in_(search_request.document_ids))

    # Add limit
    query = query.limit(search_request.top_k)
    # Optional: Add ordering, e.g., by chunk index or document ID
    query = query.order_by(document_chunks_table.c.document_id, document_chunks_table.c.chunk_index)


    try:
        results = await database.fetch_all(query) # Use global 'database' instance
        logger.info(f"Keyword search found {len(results)} results for query '{search_request.query}'.")

        # Format results
        chunk_results = [
            ChunkResult(
                document_id=row["document_id"],
                chunk_id=row["id"],
                chunk_index=row["chunk_index"],
                text_content=row["text_content"],
                score=None, # No score for keyword search
                metadata=row["metadata"] # Pass along any metadata stored in the chunk row
            )
            for row in results
        ]

        return SearchResponse(
            results=chunk_results,
            query=search_request.query,
            search_type="keyword"
        )

    except Exception as e:
        logger.error(f"Error during keyword search for query '{search_request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Keyword search failed: {e}")


# --- NEW: Semantic Search Endpoint ---
@router.post(
    "/search/semantic",
    response_model=SearchResponse,
    summary="Perform semantic search using vector embeddings",
    description="Generates embedding for the query and searches ChromaDB for similar document chunks.",
)
async def semantic_search(
    search_request: SearchQuery,
    # db: databases.Database = Depends(get_sqlite_db) # Not needed here directly
    # chroma_client: chromadb.ClientAPI = Depends(get_chroma) # Use dependency injection if preferred
):
    """
    Performs semantic search using embeddings stored in ChromaDB.
    """
    logger.info(f"Semantic search request received: query='{search_request.query}', top_k={search_request.top_k}")

    chroma_client = get_chroma_client() # Get client directly
    try:
        doc_collection = chroma_client.get_collection(settings.DOCUMENT_COLLECTION_NAME)
    except Exception as e:
         logger.error(f"Could not get ChromaDB collection '{settings.DOCUMENT_COLLECTION_NAME}': {e}")
         raise HTTPException(status_code=500, detail="Could not connect to vector database collection.")

    # 1. Generate embedding for the query
    # Again, note this is blocking. Use run_in_executor for production.
    try:
        logger.info(f"Generating embedding for query: '{search_request.query}'")
        query_embedding = generate_embeddings([search_request.query]) # Pass query as a list
        if not query_embedding or not query_embedding[0]:
            raise ValueError("Failed to generate query embedding.")
        logger.info("Query embedding generated successfully.")
    except Exception as e:
        logger.error(f"Failed to generate embedding for query '{search_request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate query embedding: {e}")

    # 2. Build ChromaDB query filter (if document_ids are provided)
    chroma_where_filter = None
    if search_request.document_ids:
        if len(search_request.document_ids) == 1:
             chroma_where_filter = {"document_id": search_request.document_ids[0]}
        else:
             chroma_where_filter = {
                 "$or": [{"document_id": doc_id} for doc_id in search_request.document_ids]
             }
        logger.info(f"Applying ChromaDB where filter: {chroma_where_filter}")


    # 3. Query ChromaDB
    try:
        logger.info(f"Querying ChromaDB collection '{doc_collection.name}'...")
        results = doc_collection.query(
            query_embeddings=query_embedding, # Pass the generated embedding (inside a list)
            n_results=search_request.top_k,
            # Specify what data to include in the results
            include=['metadatas', 'documents', 'distances'],
            where=chroma_where_filter # Apply the filter if created
        )
        logger.info(f"ChromaDB query returned {len(results.get('ids', [[]])[0])} results.") # Chroma returns results nested in lists

    except Exception as e:
        logger.error(f"Error querying ChromaDB for query '{search_request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Vector database query failed: {e}")

    # 4. Process and format ChromaDB results
    chunk_results = []
    # Results are returned as lists of lists (one inner list per query, we only have one query)
    ids = results.get('ids', [[]])[0]
    distances = results.get('distances', [[]])[0]
    metadatas = results.get('metadatas', [[]])[0]
    documents = results.get('documents', [[]])[0]

    for i, chunk_id in enumerate(ids):
        metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
        # Construct the ChunkResult
        chunk_results.append(
            ChunkResult(
                # Extract info primarily from metadata if stored there during ingestion
                document_id=metadata.get("document_id", "Unknown"),
                chunk_id=chunk_id,
                chunk_index=metadata.get("chunk_index", -1), # Get index from metadata
                text_content=documents[i] if documents and i < len(documents) else "",
                score=distances[i] if distances and i < len(distances) else None, # Lower distance = more similar
                metadata=metadata # Pass along the full metadata dict
            )
        )

    return SearchResponse(
        results=chunk_results,
        query=search_request.query,
        search_type="semantic"
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