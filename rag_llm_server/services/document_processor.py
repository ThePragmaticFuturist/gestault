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
# Import database and chroma client getters
from db.database import get_sqlite_db, get_chroma_client
from db.models import documents_table, document_chunks_table
# Import the embedding service function
from services.embedding_service import generate_embeddings

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