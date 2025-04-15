# app/api/endpoints/sessions.py
import uuid
import logging
import datetime
import asyncio
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks

# Import models and db access
from app.api.models.chat import (
    SessionCreateRequest,
    SessionMetadataResponse,
    MessageCreateRequest,
    ChatMessageResponse
)

# --- MODIFIED IMPORTS ---
from db.database import database, get_chroma_client # Import Chroma client getter
from db.models import sessions_table, chat_messages_table, documents_table
from core.config import settings
# Import embedding and search functions/models if needed directly
from services.embedding_service import generate_embeddings
# Import LLM service functions and status
from services.llm_service import generate_text, get_llm_status, LLMStatus, llm_state, summarize_text_with_query ##, summarize_text_with_query for the optional idea of summarizing chunks to reduce tokens
from services.llm_backends import LocalTransformersBackend
# --- END MODIFIED IMPORTS ---
from app.api.endpoints.documents import SearchQuery

logger = logging.getLogger(__name__)
router = APIRouter()

class SessionMemoryUpdateRequest(BaseModel):
    enabled: bool

# --- Helper Function (Optional but recommended for validation) ---
async def get_session_or_404(session_id: str) -> dict:
    """Fetches session metadata or raises HTTPException 404."""
    query = sessions_table.select().where(sessions_table.c.id == session_id)
    session = await database.fetch_one(query)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session with ID '{session_id}' not found.")

    # Ensure JSON fields are parsed (databases library might return strings)
    # Example for rag_document_ids (assuming it's stored as JSON string):
    # if isinstance(session_dict.get("rag_document_ids"), str):
    #     try:
    #         session_dict["rag_document_ids"] = json.loads(session_dict["rag_document_ids"])
    #     except json.JSONDecodeError:
    #         logger.warning(f"Could not decode rag_document_ids JSON for session {session_id}")
    #         session_dict["rag_document_ids"] = None # Or handle as error

    # Note: SQLAlchemy's JSON type with 'databases' might handle this automatically depending on dialect/setup.
    # Let's assume it returns a list/dict directly for now.

    return dict(session) # Convert RowProxy to dict

# --- Session Endpoints ---

@router.post(
    "/",
    response_model=SessionMetadataResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new chat session",
)
async def create_session(
    session_data: SessionCreateRequest,
):
    """
    Starts a new chat session, optionally associating RAG documents.
    """
    session_id = str(uuid.uuid4())
    now = datetime.datetime.now(datetime.timezone.utc)

    # TODO: Validate if rag_document_ids actually exist in documents table?
    # For now, assume they are valid if provided.

    insert_query = sessions_table.insert().values(
        id=session_id,
        name=session_data.name,
        created_at=now,
        last_updated_at=now,
        llm_model_name=settings.DEFAULT_MODEL_NAME, # Use default model from settings for now
        rag_document_ids=session_data.rag_document_ids,
        metadata_tags=None, # Add logic for automatic tags later
    )
    try:
        await database.execute(insert_query)
        logger.info(f"Created new session with ID: {session_id}, Name: {session_data.name}")
        # Fetch the created session to return its data
        created_session = await get_session_or_404(session_id)
        return SessionMetadataResponse.parse_obj(created_session)
    except Exception as e:
        logger.error(f"Failed to create session: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create session.")

@router.get(
    "/",
    response_model=List[SessionMetadataResponse],
    summary="List existing chat sessions",
)
async def list_sessions(
    skip: int = 0,
    limit: int = 50,
):
    """
    Retrieves a list of chat sessions, ordered by last update time (most recent first).
    """
    query = sessions_table.select().order_by(sessions_table.c.last_updated_at.desc()).offset(skip).limit(limit)
    try:
        results = await database.fetch_all(query)
        return [SessionMetadataResponse.parse_obj(dict(row)) for row in results]
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve sessions.")

@router.get(
    "/{session_id}",
    response_model=SessionMetadataResponse,
    summary="Get details of a specific session",
)
async def get_session(session_id: str):
    """
    Retrieves metadata for a single chat session by its ID.
    """
    session = await get_session_or_404(session_id)
    return SessionMetadataResponse.parse_obj(session)


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a chat session",
)
async def delete_session(session_id: str):
    """
    Deletes a chat session and all its associated messages.
    """
    await get_session_or_404(session_id) # Ensure session exists first

    # Use a transaction to ensure atomicity
    async with database.transaction():
        try:
            # Delete messages associated with the session
            delete_messages_query = chat_messages_table.delete().where(chat_messages_table.c.session_id == session_id)
            await database.execute(delete_messages_query)

            # Delete the session itself
            delete_session_query = sessions_table.delete().where(sessions_table.c.id == session_id)
            await database.execute(delete_session_query)

            logger.info(f"Deleted session {session_id} and its messages.")
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}", exc_info=True)
            # Transaction will be rolled back automatically on exception
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete session {session_id}.")
    return None # Return None for 204 status code


# --- Message Endpoints (within a session) ---

@router.post(
    "/{session_id}/messages",
    response_model=ChatMessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Send a message and get AI response (with RAG)",
    description="Adds user message, performs RAG search, constructs prompt, calls LLM, "
                "stores response, and returns the assistant message.",
)

async def add_message_to_session(
    session_id: str,
    message_data: MessageCreateRequest,
    background_tasks: BackgroundTasks,
):
    if message_data.role != "user":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only 'user' role messages can be posted by the client.")

    session = await get_session_or_404(session_id)

    # --- Get the memory setting ---
    long_term_memory_enabled = session.get("long_term_memory_enabled", True) # Default to True if missing? Or rely on DB default.

    user_message_content = message_data.content
    now = datetime.datetime.now(datetime.timezone.utc)
    chroma_client = get_chroma_client()

    # --- Store User Message ---
    user_message_id = None
    try:
        insert_user_message_query = chat_messages_table.insert().values(
            session_id=session_id, timestamp=now, role="user", content=user_message_content, metadata=None
        ).returning(chat_messages_table.c.id)
        user_message_id = await database.execute(insert_user_message_query)
        logger.info(f"[Session:{session_id}] Stored user message (ID: {user_message_id}).")
        update_session_query = sessions_table.update().where(sessions_table.c.id == session_id).values(last_updated_at=now)
        await database.execute(update_session_query)
    except Exception as e:
        logger.error(f"[Session:{session_id}] Failed to store user message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to store user message.")


    # --- Perform Document RAG Search & Summarization ---
    rag_context = "" # Stores summarized document context
    rag_chunk_ids = []
    retrieved_docs = [] # Keep original docs for potential summarization
    retrieved_metadatas = []
    rag_document_ids_in_session = session.get("rag_document_ids")
    if rag_document_ids_in_session:
        logger.info(f"[Session:{session_id}] Performing RAG search within documents: {rag_document_ids_in_session}")
        try:
            # ... (code to perform ChromaDB query for documents, populate retrieved_docs, retrieved_metadatas, rag_chunk_ids) ...
            doc_collection = chroma_client.get_collection(settings.DOCUMENT_COLLECTION_NAME)
            query_embedding = generate_embeddings([user_message_content])
            if query_embedding and query_embedding[0]:
                chroma_where_filter = {"$and": [{"type": "document"}, {"document_id": {"$in": rag_document_ids_in_session}}]} # Filter for type:document
                results = doc_collection.query(query_embeddings=query_embedding, n_results=settings.RAG_TOP_K, where=chroma_where_filter, include=['documents', 'metadatas', 'distances'])
                retrieved_docs = results.get('documents', [[]])[0]
                retrieved_metadatas = results.get('metadatas', [[]])[0]
                rag_chunk_ids = results.get('ids', [[]])[0] # Original chunk IDs

                if retrieved_docs:
                    logger.info(f"[Session:{session_id}] Retrieved {len(retrieved_docs)} document chunks for summarization.")
                    # --- Summarize Retrieved Document Chunks ---
                    summarized_doc_parts = []
                    loop = asyncio.get_running_loop()
                    for i, chunk_text in enumerate(retrieved_docs):
                         try:
                             summary = await loop.run_in_executor(None, summarize_text_with_query, chunk_text, user_message_content, 150)
                             if summary: summarized_doc_parts.append(summary)
                         except Exception as summary_err: logger.error(f"Error summarizing doc chunk {i}: {summary_err}", exc_info=True)
                    rag_context = "\n---\n".join(summarized_doc_parts) # rag_context now holds summarized doc info
                    if not rag_context: rag_context = "[Document Summarization Failed]"
                    logger.info(f"[Session:{session_id}] Finished summarizing doc chunks.")
                else:
                     logger.info(f"[Session:{session_id}] No relevant document chunks found.")
            else:
                 logger.error("[Session:{session_id}] Failed to generate embedding for document search.")
                 rag_context = "[Document RAG search embedding failed]"
        except Exception as e:
            logger.error(f"[Session:{session_id}] Document RAG search failed: {e}", exc_info=True)
            rag_context = "[Document RAG search failed]"
    else:
         logger.info(f"[Session:{session_id}] No RAG document IDs associated. Skipping document RAG search.")


    # --- Perform Chat History RAG Search & Summarization ---
    retrieved_history_context = "" # Stores summarized history context
    retrieved_history_ids = []

    if long_term_memory_enabled:
        try:
            logger.info(f"[Session:{session_id}] Performing RAG search on chat history...")
            # ... (code to perform ChromaDB query for chat history, populate retrieved_history_docs, retrieved_history_ids) ...
            # ... (code to summarize retrieved_history_docs into retrieved_history_context) ...
            history_query_embedding = generate_embeddings([user_message_content])
            if history_query_embedding and history_query_embedding[0]:
                 chroma_client = get_chroma_client()
                 collection = chroma_client.get_collection(settings.DOCUMENT_COLLECTION_NAME)
                 history_where_filter = {"$and": [{"type": "chat"}, {"session_id": session_id}]}
                 history_rag_k = settings.RAG_TOP_K
                 history_results = collection.query(query_embeddings=history_query_embedding, n_results=history_rag_k, where=history_where_filter, include=['documents', 'metadatas', 'distances'])
                 retrieved_history_docs = history_results.get('documents', [[]])[0]
                 retrieved_history_metadatas = history_results.get('metadatas', [[]])[0]
                 retrieved_history_ids = history_results.get('ids', [[]])[0]

                 if retrieved_history_docs:
                      logger.info(f"[Session:{session_id}] Retrieved {len(retrieved_history_docs)} relevant past messages for summarization.")
                      summarized_history_parts = []
                      loop = asyncio.get_running_loop()
                      for i, history_text in enumerate(retrieved_history_docs):
                           hist_meta = retrieved_history_metadatas[i] if i < len(retrieved_history_metadatas) else {}
                           hist_role = hist_meta.get('role', 'Unknown')
                           try:
                                summary = await loop.run_in_executor(None, summarize_text_with_query, f"{hist_role.capitalize()}: {history_text}", user_message_content, 100)
                                if summary: summarized_history_parts.append(summary)
                           except Exception as summary_err: logger.error(f"Error summarizing history snippet {i}: {summary_err}", exc_info=True)
                      retrieved_history_context = "\n---\n".join(summarized_history_parts)
                      if not retrieved_history_context: retrieved_history_context = "[History Summarization Failed]"
                      logger.info(f"[Session:{session_id}] Finished summarizing history snippets.")
                 else:
                      logger.info(f"[Session:{session_id}] No relevant chat history found for this query.")
            else:
                 logger.error("[Session:{session_id}] Failed to generate embedding for history search.")
                 retrieved_history_context = "[History RAG search embedding failed]"
        except Exception as e:
            logger.error(f"[Session:{session_id}] Chat history RAG search failed: {e}", exc_info=True)
            retrieved_history_context = "[History RAG search failed]"
    else:
        logger.info(f"[Session:{session_id}] Long term memory disabled. Skipping chat history RAG search.")
        # Ensure context is empty if disabled
        retrieved_history_context = ""
        retrieved_history_ids = []

    # --- Retrieve RECENT Chat History (Verbatim) ---
    chat_history_str = "" # Stores recent verbatim history
    try:
        # ... (code to fetch last N messages into chat_history_str - unchanged) ...
        history_limit = settings.CHAT_HISTORY_LENGTH * 2
        history_query = chat_messages_table.select().where(chat_messages_table.c.session_id == session_id).order_by(chat_messages_table.c.timestamp.desc()).limit(history_limit)
        recent_messages = await database.fetch_all(history_query)
        recent_messages.reverse()
        if recent_messages:
             history_parts = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_messages]
             chat_history_str = "\n".join(history_parts)
             logger.info(f"[Session:{session_id}] Retrieved last {len(recent_messages)} verbatim messages for history.")
    except Exception as e:
        logger.error(f"[Session:{session_id}] Failed to retrieve recent chat history: {e}", exc_info=True)
        chat_history_str = "[Failed to retrieve recent history]"


    logger.info("Constructing final prompt with RAG context and history...")

    prompt_for_llm = "[ERROR: Prompt construction did not complete]"
    llm_ready = False
    final_token_count = 0

    assistant_message_metadata = {
        "prompt_preview": None,
        "rag_chunks_retrieved": rag_chunk_ids,
        "relevant_history_retrieved": retrieved_history_ids, # Now correctly populated or []
        "llm_call_error": None
    }

    # 1. Check LLM Status and get essentials
    llm_status_data = get_llm_status()
    # tokenizer = llm_state.get("tokenizer")
    # max_length = llm_state.get("max_model_length")
    backend_instance = llm_state.get("backend_instance") # Get the backend instance
    model_name = llm_status_data.get("active_model", "N/A")

    tokenizer = None
    max_length = None
    if isinstance(backend_instance, LocalTransformersBackend):
        tokenizer = backend_instance.tokenizer
        max_length = backend_instance.max_model_length

    if llm_status_data["status"] != LLMStatus.READY.value:
        error_detail = f"LLM not ready (Status: {llm_status_data['status']}). Please load/configure a model first."
        logger.warning(f"[Session:{session_id}] {error_detail}")
        prompt_for_llm = f"[ERROR: {error_detail}]"
        assistant_message_metadata["llm_call_error"] = error_detail
        # llm_ready remains False

    # 2. Check if Tokenizer/Max Length are needed and available FOR LOCAL backend
    # Only perform token-based truncation if using local backend AND have tools
    elif isinstance(backend_instance, LocalTransformersBackend) and (not tokenizer or not max_length):
        # We are using local backend but missing tools for truncation
        logger.warning("Local backend active but Tokenizer or max_length missing from state. Cannot perform accurate truncation. Sending raw context.")
        llm_ready = True # Allow attempt
        # Assemble simple prompt without truncation
        # ... (Assemble simple prompt logic - unchanged, uses rag_context, retrieved_history_context, chat_history_str) ...
        prompt_parts = [instruction_prompt]
        if rag_context and "[ERROR" not in rag_context: prompt_parts.append(f"\n### RETRIEVED DOCUMENT CONTEXT (Summarized):\n{rag_context}")
        if retrieved_history_context and "[ERROR" not in retrieved_history_context: prompt_parts.append(f"\n### RELEVANT PAST CONVERSATION (Summarized):\n{retrieved_history_context}")
        if chat_history_str and "[ERROR" not in chat_history_str: prompt_parts.append(f"\n### RECENT CHAT HISTORY:\n{chat_history_str}")
        prompt_parts.append(user_query_prompt)
        prompt_parts.append(response_marker)
        prompt_for_llm = "\n".join(prompt_parts)

    # 3. Perform Full Prompt Construction with Truncation
    else:
        llm_ready = True # LLM is ready

        # Define fixed parts
        instruction_text = "You are a helpful AI assistant. Respond to the user's query based on the provided context and chat history."
        user_query_text = user_message_content
        instruction_prompt = "### INSTRUCTION:\n" + instruction_text
        user_query_prompt = "\n### USER QUERY:\n" + user_query_text
        response_marker = "\n### RESPONSE:"

        # --- Only do token math if local backend ---
        if isinstance(backend_instance, LocalTransformersBackend) and tokenizer and max_length:
            logger.info("Proceeding with token-based prompt truncation logic for local backend.")
            # ... (Calculate reserved_tokens, max_context_tokens using tokenizer/max_length) ...
            # ... (Tokenize context parts using tokenizer) ...
            # ... (Perform truncation logic on token lists) ...
            # ... (Assemble final_token_ids) ...
            # ... (Decode final_token_ids into prompt_for_llm) ...
            # ... (Log final_token_count) ...
            generation_config = llm_status_data.get("generation_config", {})
            max_new_tokens = generation_config.get("max_new_tokens", settings.DEFAULT_LLM_MAX_NEW_TOKENS)
            fixed_parts_text = instruction_prompt + user_query_prompt + response_marker
            fixed_tokens_count = len(tokenizer.encode(fixed_parts_text, add_special_tokens=False))
            reserved_tokens = fixed_tokens_count + max_new_tokens + 10
            max_context_tokens = max_length - reserved_tokens
            if max_context_tokens < 0: max_context_tokens = 0
            logger.info(f"Max Length: {max_length}, Reserved: {reserved_tokens}, Available for ALL Context: {max_context_tokens}")

            # Tokenize and truncate... (Copy the full logic from previous step here)
 
            doc_context_tokens = tokenizer.encode(f"\n### RETRIEVED DOCUMENT CONTEXT (Summarized):\n{rag_context}", add_special_tokens=False) if rag_context and "[ERROR" not in rag_context else []
            history_summary_tokens = tokenizer.encode(f"\n### RELEVANT PAST CONVERSATION (Summarized):\n{retrieved_history_context}", add_special_tokens=False) if retrieved_history_context and "[ERROR" not in retrieved_history_context else []
            recent_history_tokens = tokenizer.encode(f"\n### RECENT CHAT HISTORY:\n{chat_history_str}", add_special_tokens=False) if chat_history_str and "[ERROR" not in chat_history_str else []

            total_context_tokens = len(doc_context_tokens) + len(history_summary_tokens) + len(recent_history_tokens)
            available_space = max_context_tokens

            if total_context_tokens > available_space:
                logger.info(f"Combined context ({total_context_tokens} tokens) exceeds available space ({available_space}). Truncating...")
                overflow = total_context_tokens - available_space
                # Truncate Recent History first (keep end)
                if overflow > 0 and len(recent_history_tokens) > 0:
                    # ... (truncation logic for recent_history_tokens) ...
                     if len(recent_history_tokens) >= overflow: recent_history_tokens = recent_history_tokens[overflow:]; overflow = 0
                     else: overflow -= len(recent_history_tokens); recent_history_tokens = []
                # Truncate History Summary next (keep beginning)
                if overflow > 0 and len(history_summary_tokens) > 0:
                    # ... (truncation logic for history_summary_tokens) ...
                    if len(history_summary_tokens) >= overflow: history_summary_tokens = history_summary_tokens[:len(history_summary_tokens)-overflow]; overflow = 0
                    else: overflow -= len(history_summary_tokens); history_summary_tokens = []
                # Truncate Document Context last (keep beginning)
                if overflow > 0 and len(doc_context_tokens) > 0:
                    # ... (truncation logic for doc_context_tokens) ...
                    if len(doc_context_tokens) >= overflow: doc_context_tokens = doc_context_tokens[:len(doc_context_tokens)-overflow]; overflow = 0
                    else: doc_context_tokens = []

            # Assemble final prompt from potentially truncated token lists
            instruction_tokens = tokenizer.encode(instruction_prompt, add_special_tokens=False)
            query_tokens = tokenizer.encode(user_query_prompt, add_special_tokens=False)
            marker_tokens = tokenizer.encode(response_marker, add_special_tokens=False)
            final_token_ids = instruction_tokens + doc_context_tokens + history_summary_tokens + recent_history_tokens + query_tokens + marker_tokens

            prompt_for_llm = tokenizer.decode(final_token_ids)
            final_token_count = len(final_token_ids)
            logger.info(f"Constructed final prompt with {final_token_count} tokens using truncation.")

        else: # API Backend - Assemble without local truncation
            logger.info("Using API backend. Assembling prompt without local token truncation.")
            prompt_parts = [instruction_prompt]
            if rag_context and "[ERROR" not in rag_context: prompt_parts.append(f"\n### RETRIEVED DOCUMENT CONTEXT (Summarized):\n{rag_context}")
            if retrieved_history_context and "[ERROR" not in retrieved_history_context: prompt_parts.append(f"\n### RELEVANT PAST CONVERSATION (Summarized):\n{retrieved_history_context}")
            if chat_history_str and "[ERROR" not in chat_history_str: prompt_parts.append(f"\n### RECENT CHAT HISTORY:\n{chat_history_str}")
            prompt_parts.append(user_query_prompt)
            prompt_parts.append(response_marker)
            prompt_for_llm = "\n".join(prompt_parts)
            # We don't have an accurate token count here
            final_token_count = -1 # Indicate unknown count
            logger.debug("Prompt assembled for API backend.")

        # Set prompt preview metadata regardless of truncation method
        assistant_message_metadata["prompt_preview"] = prompt_for_llm[:200] + "..."


    # Log final prompt
    log_token_count = f"Token Count: {final_token_count}" if final_token_count != -1 else "Token Count: N/A (API Backend)"
    if llm_ready and "[ERROR" not in prompt_for_llm:
         logger.debug(f"[Session:{session_id}] Constructed prompt for LLM ({log_token_count}):\n{prompt_for_llm[:200]}...\n...{prompt_for_llm[-200:]}")
    else:
         logger.debug(f"[Session:{session_id}] Prompt construction failed or LLM not ready. Final prompt value: {prompt_for_llm}")


    # --- Call LLM Service (Checks llm_ready flag) ---
    if llm_ready:
        try:
            logger.info(f"[Session:{session_id}] Sending prompt to LLM '{model_name}' via backend '{llm_status_data['backend_type']}'...")
            llm_response = await generate_text(prompt_for_llm) # Await the service call

            if llm_response is None:
                raise Exception("LLM generation returned None or failed internally.")

            assistant_response_content = llm_response # Set content on success
            logger.info(f"[Session:{session_id}] Received response from LLM.")

        except Exception as e:
            error_detail = f"LLM generation failed: {type(e).__name__}"
            logger.error(f"[Session:{session_id}] {error_detail}: {e}", exc_info=True)
            assistant_response_content = f"[ERROR: {error_detail}]" # Set error content
            assistant_message_metadata["llm_call_error"] = f"{error_detail}: {e}" # Record error in metadata
    # If llm_ready was False, assistant_response_content and metadata already set

    # --- Store Assistant Message ---
    try:
        assistant_timestamp = datetime.datetime.now(datetime.timezone.utc)
        insert_assistant_message_query = chat_messages_table.insert().values(
            session_id=session_id,
            timestamp=assistant_timestamp,
            role="assistant",
            content=assistant_response_content, # Use the actual or error content
            metadata=assistant_message_metadata, # Include metadata
        ).returning(chat_messages_table.c.id, *[c for c in chat_messages_table.c])

        new_assistant_message_row = await database.fetch_one(insert_assistant_message_query)
        if not new_assistant_message_row:
            raise Exception("Failed to retrieve assistant message after insert.")
        assistant_message_id = new_assistant_message_row['id']
        logger.info(f"[Session:{session_id}] Stored assistant message (ID: {assistant_message_id}).")

        update_session_query_after_assist = sessions_table.update().where(sessions_table.c.id == session_id).values(last_updated_at=assistant_timestamp)
        await database.execute(update_session_query_after_assist)

        # --- Schedule Background Indexing Task ---
        if user_message_id is not None and assistant_message_id is not None:
             logger.info(f"[Session:{session_id}] Scheduling background task to index chat turn ({user_message_id}, {assistant_message_id}).")
             # Ensure chat indexer is imported if used
             # from services.chat_indexer import index_chat_turn_task
             # background_tasks.add_task(index_chat_turn_task, user_message_id, assistant_message_id, session_id)
             pass # Commented out as chat_indexer.py wasn't provided/requested in this thread
        else:
              logger.warning(f"[Session:{session_id}] Missing user or assistant message ID, cannot schedule indexing.")

        return ChatMessageResponse.parse_obj(dict(new_assistant_message_row))

    except Exception as e:
        logger.error(f"[Session:{session_id}] Failed to store assistant message: {e}", exc_info=True)
        # Note: If storing fails, the client won't get the response even if generated
        raise HTTPException(status_code=500, detail="Failed to store assistant response.")

@router.put(
    "/{session_id}/memory",
    response_model=SessionMetadataResponse, # Return updated session metadata
    summary="Enable or disable long-term memory (chat history RAG) for a session",
)
async def update_session_memory(
    session_id: str,
    update_request: SessionMemoryUpdateRequest
):
    """
    Updates the long_term_memory_enabled setting for the specified session.
    """
    # Ensure session exists
    session = await get_session_or_404(session_id)

    new_value = update_request.enabled
    current_value = session.get("long_term_memory_enabled")

    if current_value == new_value:
         logger.info(f"Long term memory for session {session_id} already set to {new_value}. No change.")
         # Return current data without DB update
         return SessionMetadataResponse.parse_obj(session)

    update_query = sessions_table.update().where(
        sessions_table.c.id == session_id
    ).values(
        long_term_memory_enabled=new_value,
        # Optionally update last_updated_at timestamp? Debatable for settings change.
        # last_updated_at=datetime.datetime.now(datetime.timezone.utc)
    )

    try:
        await database.execute(update_query)
        logger.info(f"Set long_term_memory_enabled to {new_value} for session {session_id}.")
        # Fetch updated session data to return
        updated_session = await get_session_or_404(session_id)
        return SessionMetadataResponse.parse_obj(updated_session)
    except Exception as e:
        logger.error(f"Failed to update memory setting for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update session memory setting.")

# --- List Messages Endpoint (Keep as is) ---
@router.get(
    "/{session_id}/messages",
    response_model=List[ChatMessageResponse],
    summary="List messages in a session",
)
async def list_messages_in_session(
    session_id: str,
    skip: int = 0,
    limit: int = 1000,
):


    await get_session_or_404(session_id) # Ensure session exists
    query = chat_messages_table.select().where(
        chat_messages_table.c.session_id == session_id
    ).order_by(chat_messages_table.c.timestamp.asc()).offset(skip).limit(limit) # Order by ascending time
    try:
        results = await database.fetch_all(query)
        return [ChatMessageResponse.parse_obj(dict(row)) for row in results]
    except Exception as e:
        logger.error(f"Failed to list messages for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve messages for session {session_id}.")
