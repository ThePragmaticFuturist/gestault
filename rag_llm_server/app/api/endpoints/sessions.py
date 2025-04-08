# app/api/endpoints/sessions.py
import uuid
import logging
import datetime
import asyncio
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, status

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
from services.llm_service import generate_text, get_llm_status, LLMStatus, llm_state ##, summarize_text_with_query for the optional idea of summarizing chunks to reduce tokens
# --- END MODIFIED IMPORTS ---
from app.api.endpoints.documents import SearchQuery

logger = logging.getLogger(__name__)
router = APIRouter()

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

async def add_message_to_session( # Renamed from post_message for clarity internally
    session_id: str,
    message_data: MessageCreateRequest,
    # chroma_client: chromadb.ClientAPI = Depends(get_chroma_client) # Inject Chroma
):
    """
    Handles user message, RAG, placeholder LLM call, and stores conversation turn.
    """
    if message_data.role != "user":
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only 'user' role messages can be posted by the client.")

    session = await get_session_or_404(session_id) # Ensure session exists and get its data
    user_message_content = message_data.content
    now = datetime.datetime.now(datetime.timezone.utc)
    chroma_client = get_chroma_client() # Get client directly for now

    # --- Store User Message ---
    # Use a transaction? Maybe better to store user msg, then do RAG/LLM, then store assistant msg.
    # Let's store user message immediately first.
    try:
        insert_user_message_query = chat_messages_table.insert().values(
            session_id=session_id,
            timestamp=now,
            role="user",
            content=user_message_content,
            metadata=None,
        ).returning(chat_messages_table.c.id)
        user_message_id = await database.execute(insert_user_message_query)
        logger.info(f"[Session:{session_id}] Stored user message (ID: {user_message_id}).")

        # Update session timestamp for user message (or wait until assistant message?)
        # Let's update now to reflect activity immediately.
        update_session_query = sessions_table.update().where(sessions_table.c.id == session_id).values(
             last_updated_at=now
        )
        await database.execute(update_session_query)

    except Exception as e:
        logger.error(f"[Session:{session_id}] Failed to store user message: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to store user message.")

    # --- Perform RAG Search ---
    rag_context = ""
    rag_chunk_ids = []
    rag_document_ids_in_session = session.get("rag_document_ids") # Get associated doc IDs

    if rag_document_ids_in_session:
        logger.info(f"[Session:{session_id}] Performing RAG search within documents: {rag_document_ids_in_session}")
        try:
            doc_collection = chroma_client.get_collection(settings.DOCUMENT_COLLECTION_NAME)

            # 1. Embed the user query
            query_embedding = generate_embeddings([user_message_content])
            if not query_embedding or not query_embedding[0]:
                raise ValueError("Failed to generate query embedding for RAG.")

            # 2. Build ChromaDB filter
            chroma_where_filter = {
                # Filter results to only include chunks from the documents linked to this session
                "document_id": {"$in": rag_document_ids_in_session}
            }
            logger.debug(f"[Session:{session_id}] RAG ChromaDB filter: {chroma_where_filter}")

            # 3. Query ChromaDB
            results = doc_collection.query(
                query_embeddings=query_embedding,
                n_results=settings.RAG_TOP_K,
                where=chroma_where_filter,
                include=['documents', 'metadatas', 'distances'] # Get text, metadata, distance
            )

            # 4. Format RAG context
            retrieved_docs = results.get('documents', [[]])[0]
            retrieved_metadatas = results.get('metadatas', [[]])[0]
            retrieved_ids = results.get('ids', [[]])[0] # Get chunk IDs

            if retrieved_docs:
                rag_chunk_ids = retrieved_ids # Store the IDs of retrieved chunks
                context_parts = []
                for i, doc_text in enumerate(retrieved_docs):
                    metadata = retrieved_metadatas[i] if i < len(retrieved_metadatas) else {}
                    source_info = f"Source(Doc: {metadata.get('document_id', 'N/A')}, Chunk: {metadata.get('chunk_index', 'N/A')})"
                    context_parts.append(f"{source_info}:\n{doc_text}")
                rag_context = "\n\n---\n\n".join(context_parts)
                logger.info(f"[Session:{session_id}] Retrieved {len(retrieved_docs)} chunks for RAG context.")
                logger.debug(f"[Session:{session_id}] RAG Context:\n{rag_context[:500]}...") # Log beginning of context

        except Exception as e:
            logger.error(f"[Session:{session_id}] RAG search failed: {e}", exc_info=True)
            # Proceed without RAG context, maybe log a warning or add to assistant metadata
            rag_context = "[RAG search failed]" # Indicate failure in context
    else:
         logger.info(f"[Session:{session_id}] No RAG document IDs associated. Skipping RAG search.")


    # --- Retrieve Chat History ---
    chat_history_str = ""
    try:
        # Fetch last N messages (user and assistant), ordered oldest first for prompt
        # N pairs = 2 * N messages. Add 1 to potentially fetch the just-added user message if needed.
        history_limit = settings.CHAT_HISTORY_LENGTH * 2
        history_query = chat_messages_table.select().where(
            chat_messages_table.c.session_id == session_id
        ).order_by(chat_messages_table.c.timestamp.desc()).limit(history_limit) # Fetch recent, then reverse

        recent_messages = await database.fetch_all(history_query)
        recent_messages.reverse() # Reverse to get chronological order

        if recent_messages:
             history_parts = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_messages]
             # Exclude the very last message if it's the user message we just added?
             # Let's assume the LLM prompt format handles the current user query separately.
             chat_history_str = "\n".join(history_parts)
             logger.info(f"[Session:{session_id}] Retrieved last {len(recent_messages)} messages for history.")
             logger.debug(f"[Session:{session_id}] Chat History:\n{chat_history_str[-500:]}...") # Log end of history

    except Exception as e:
        logger.error(f"[Session:{session_id}] Failed to retrieve chat history: {e}", exc_info=True)
        chat_history_str = "[Failed to retrieve history]"

    # --- Construct Prompt ---
    # --- Construct & Truncate Prompt ---
    

    prompt_parts = []

    prompt_parts.append("### INSTRUCTION:")
    prompt_parts.append("You are a helpful AI assistant. Respond to the user's query based on the provided context and chat history.")

    if rag_context and rag_context != "[RAG search failed]":
        prompt_parts.append("\n### CONTEXT:")
        prompt_parts.append(rag_context)

    if chat_history_str:
        prompt_parts.append("\n### CHAT HISTORY:")
        prompt_parts.append(chat_history_str) # History already includes "User:" / "Assistant:"

    prompt_parts.append("\n### USER QUERY:")
    prompt_parts.append(user_message_content)

    prompt_parts.append("\n### RESPONSE:") # Model should generate text after this marker

    prompt_for_llm = "\n".join(prompt_parts)

    logger.debug(f"[Session:{session_id}] Constructed prompt (first/last 200 chars):\n{prompt_for_llm[:200]}...\n...{prompt_for_llm[-200:]}") 

    assistant_response_content = None
    assistant_message_metadata = {
        "prompt_preview": prompt_for_llm[:200] + "...",
        "rag_chunks_retrieved": rag_chunk_ids,
        "llm_call_error": None # Add field for potential errors
    }

    # Check LLM status before attempting generation
    # Get necessary info from LLM state
    # llm_status_data = get_llm_status()
    # tokenizer = llm_state.get("tokenizer") # Get tokenizer object from state
    # max_length = llm_state.get("max_model_length")
    # model_name = llm_status_data.get("model_name_or_path", "N/A")

    # logger.info("Constructing final prompt using summarized context...")
    llm_status_data = get_llm_status() # Gets the status dict
    tokenizer = llm_state.get("tokenizer")
    max_length = llm_state.get("max_model_length")
    # --- Use 'active_model' key from status dict ---
    model_name = llm_status_data.get("active_model", "N/A") # Use 'active_model' key

    if llm_status_data["status"] != LLMStatus.READY.value: # Check for READY status
        error_detail = f"LLM not ready (Status: {llm_status_data['status']}). Please load/configure a model first."
        logger.warning(f"[Session:{session_id}] {error_detail}")
        assistant_response_content = f"[ERROR: {error_detail}]"
        # Need to initialize metadata here if erroring out early
        assistant_message_metadata = {
            "prompt_preview": "[ERROR: LLM not ready]",
            "rag_chunks_retrieved": rag_chunk_ids, # Keep RAG info if available
            "llm_call_error": error_detail
        }
        llm_ready = False # Flag to skip generation
        # No prompt needed if we can't generate
        prompt_for_llm = "[ERROR: LLM/Tokenizer not ready for prompt construction]"
    # --- Handle potential missing tokenizer/max_length needed for truncation ---
    # These are primarily needed if we need to truncate context/history for the prompt
    # API backends might handle truncation server-side, but good practice to check here

    elif not tokenizer or not max_length:
         # Only critical if context/history is long and needs truncation
         # For now, let's allow proceeding but log a warning if they are missing
         # (relevant if switching from local to API backend without restarting server maybe?)
         logger.warning("Tokenizer or max_length not found in llm_state, prompt truncation might be inaccurate if needed.")
         llm_ready = True # Allow attempt, API backend might handle length

    else:
        # --- Define Prompt Components ---
        # Estimate tokens needed for instruction, markers, query, and response buffer
        # This is an approximation; actual token count depends on specific words.
        instruction_text = "You are a helpful AI assistant. Respond to the user's query based on the provided context and chat history."
        user_query_text = user_message_content
        # Reserve tokens for instruction, markers, query, and some buffer for the response itself
        # Let's reserve ~150 tokens for non-context/history parts + response buffer
        # And also account for the model's requested max_new_tokens
        generation_config = llm_status_data.get("generation_config", {})
        max_new_tokens = generation_config.get("max_new_tokens", settings.DEFAULT_LLM_MAX_NEW_TOKENS)
        reserved_tokens = 150 + max_new_tokens
        max_context_history_tokens = max_length - reserved_tokens
        logger.info(f"Max Length: {max_length}, Reserved: {reserved_tokens}, Available for Context/History: {max_context_history_tokens}")

        # --- Tokenize and Truncate Components ---
        # 1. RAG Context: Truncate if needed
        rag_context_tokens = []
        if rag_context and rag_context != "[RAG search failed]":
            rag_context_tokens = tokenizer.encode(f"\n### CONTEXT:\n{rag_context}", add_special_tokens=False)
            if len(rag_context_tokens) > max_context_history_tokens:
                logger.warning(f"Truncating RAG context ({len(rag_context_tokens)} tokens) to fit limit.")
                rag_context_tokens = rag_context_tokens[:max_context_history_tokens]
                # Optional: Decode back to string to see truncated context? Might be slow.
                # rag_context = tokenizer.decode(rag_context_tokens) # Use truncated context text

        available_tokens_for_history = max_context_history_tokens - len(rag_context_tokens)

        # 2. Chat History: Truncate from the OLDEST messages if needed
        chat_history_tokens = []
        if chat_history_str:
            # Start with full history prompt part
            full_history_prompt = f"\n### CHAT HISTORY:\n{chat_history_str}"
            chat_history_tokens = tokenizer.encode(full_history_prompt, add_special_tokens=False)
            if len(chat_history_tokens) > available_tokens_for_history:
                 logger.warning(f"Truncating chat history ({len(chat_history_tokens)} tokens) to fit limit.")
                 # Simple truncation: keep only the last N tokens
                 chat_history_tokens = chat_history_tokens[-available_tokens_for_history:]
                 # More robust: Re-tokenize messages one by one from recent to old until limit hit? Too complex for now.

        # --- Assemble Final Prompt (using tokens is safer, but harder to format) ---
        # For simplicity, let's assemble with potentially truncated *strings* derived from tokens
        # NOTE: Decoding tokens back to strings might introduce slight changes / tokenization artifacts.
        # A more advanced method would keep everything as token IDs until the final input.

        final_prompt_parts = []
        final_prompt_parts.append("### INSTRUCTION:")
        final_prompt_parts.append(instruction_text)

        if rag_context_tokens:
             # Decode the potentially truncated RAG context tokens
             decoded_rag_context_header = tokenizer.decode(tokenizer.encode("\n### CONTEXT:", add_special_tokens=False))
             decoded_rag_context_body = tokenizer.decode(rag_context_tokens[len(tokenizer.encode(decoded_rag_context_header, add_special_tokens=False)):])
             final_prompt_parts.append(decoded_rag_context_header + decoded_rag_context_body)


        if chat_history_tokens:
            # Decode the potentially truncated chat history tokens
            decoded_hist_header = tokenizer.decode(tokenizer.encode("\n### CHAT HISTORY:", add_special_tokens=False))
            decoded_hist_body = tokenizer.decode(chat_history_tokens[len(tokenizer.encode(decoded_hist_header, add_special_tokens=False)):])
            final_prompt_parts.append(decoded_hist_header + decoded_hist_body)


        final_prompt_parts.append("\n### USER QUERY:")
        final_prompt_parts.append(user_query_text)
        final_prompt_parts.append("\n### RESPONSE:")

        prompt_for_llm = "\n".join(final_prompt_parts)

        # Final Token Check (Optional but recommended)
        final_tokens = tokenizer.encode(prompt_for_llm)
        final_token_count = len(final_tokens)
        logger.info(f"Constructed final prompt with {final_token_count} tokens (Max allowed for input: {max_length - max_new_tokens}).")
        if final_token_count >= max_length:
            # This shouldn't happen often with the truncation, but as a safeguard:
            logger.error(f"FATAL: Final prompt ({final_token_count} tokens) still exceeds model max length ({max_length}). Truncating forcefully.")
            # Forcefully truncate the token list (might cut mid-word)
            truncated_tokens = final_tokens[:max_length - 5] # Leave a tiny buffer
            prompt_for_llm = tokenizer.decode(truncated_tokens)
            logger.debug(f"Forcefully truncated prompt:\n{prompt_for_llm}")

        llm_ready = True 

    logger.debug(f"[Session:{session_id}] Constructed prompt for LLM (Token Count: {final_token_count if 'final_token_count' in locals() else 'N/A'}):\n{prompt_for_llm[:200]}...\n...{prompt_for_llm[-200:]}")

    if llm_ready:
        try:
            logger.info(f"[Session:{session_id}] Sending prompt to LLM '{model_name}' via backend '{llm_status_data['backend_type']}'...")
            # Run the potentially blocking LLM generation in an executor thread
            # loop = asyncio.get_running_loop()

            # llm_response = await loop.run_in_executor(
            #     None,  # Use default ThreadPoolExecutor
            #     generate_text, # The function to run
            #     prompt_for_llm # The argument(s) for the function
            # )

            llm_response = await generate_text(prompt_for_llm)

            if llm_response is None:
                # Generation failed within the service (error already logged there)
                raise Exception("LLM generation failed or returned None.")

            assistant_response_content = llm_response
            logger.info(f"[Session:{session_id}] Received response from LLM.")
            logger.debug(f"[Session:{session_id}] LLM Response:\n{assistant_response_content[:500]}...")

        except Exception as e:
            error_detail = f"LLM generation failed: {type(e).__name__}"
            logger.error(f"[Session:{session_id}] {error_detail}: {e}", exc_info=True)
            assistant_response_content = f"[ERROR: {error_detail}]"
            assistant_message_metadata["llm_call_error"] = f"{error_detail}: {e}"

    if not llm_ready:
        # We already set assistant_response_content and metadata earlier
        pass
    elif 'assistant_message_metadata' not in locals():
         # Initialize metadata if generation path didn't run due to other issues
          assistant_message_metadata = {
             "prompt_preview": prompt_for_llm[:200] + "...",
             "rag_chunks_retrieved": rag_chunk_ids,
             "llm_call_error": "Unknown error before storing message"
         }
         
    # --- Store Assistant Message (Using actual or error response) ---
    try:
        assistant_timestamp = datetime.datetime.now(datetime.timezone.utc)
        insert_assistant_message_query = chat_messages_table.insert().values(
            session_id=session_id,
            timestamp=assistant_timestamp,
            role="assistant",
            content=assistant_response_content, # Use the actual or error content
            metadata=assistant_message_metadata, # Include metadata with potential error
        ).returning(chat_messages_table.c.id, *[c for c in chat_messages_table.c])

        new_assistant_message_row = await database.fetch_one(insert_assistant_message_query)
        if not new_assistant_message_row:
            raise Exception("Failed to retrieve assistant message after insert.")

        logger.info(f"[Session:{session_id}] Stored assistant message (ID: {new_assistant_message_row['id']}).")

        # Update session timestamp
        update_session_query_after_assist = sessions_table.update().where(sessions_table.c.id == session_id).values(
             last_updated_at=assistant_timestamp
        )
        await database.execute(update_session_query_after_assist)

        # Return the structured assistant message response
        return ChatMessageResponse.parse_obj(dict(new_assistant_message_row))

    except Exception as e:
        logger.error(f"[Session:{session_id}] Failed to store assistant message: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to store assistant response.")


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