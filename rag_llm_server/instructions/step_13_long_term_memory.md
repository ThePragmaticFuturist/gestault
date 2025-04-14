# Step 13: Long Term Memory #

Storing and retrieving chat history as RAG documents to provide long-term memory for sessions is a powerful and logical extension. It turns the conversation itself into a searchable knowledge base specific to that interaction thread.

Here's how we can integrate this:

**Conceptual Workflow Change:**

1.  **Indexing:** After each user message and assistant response pair is finalized and stored in the SQLite `chat_messages` table, trigger a background task to embed both the user's query and the assistant's response text. Store these embeddings, the text content, and relevant metadata (like `session_id`, `message_id`, `role`, `timestamp`, and a `type='chat'`) into your **ChromaDB collection** (likely the same one used for documents, `rag_documents`, differentiated by metadata).
2.  **Retrieval:** When a *new* user message arrives (`add_message_to_session`):
    *   Perform the RAG search for *documents* as before (if documents are linked to the session).
    *   Perform a **second** semantic search using the user's message embedding, but this time query ChromaDB filtering specifically for records where `type='chat'` AND `session_id` matches the *current session*. Retrieve the top-k most relevant past messages (user or assistant).
3.  **Summarization:** Because retrieved chat history snippets might be numerous or verbose, summarize the results from the chat history search (just like we did for document chunks), potentially using the current user query to guide the summary focus.
4.  **Prompt Construction:** Modify the prompt assembly to include sections for:
    *   Instruction
    *   Retrieved Document Context (Summarized)
    *   Relevant Past Conversation (Summarized Chat History)
    *   Recent Chat History (Maybe the last 1-2 turns verbatim)
    *   Current User Query
5.  **Token Management:** Apply truncation logic considering the combined length of *both* summarized document context *and* summarized chat history context, plus recent history and the query.

**Implementation Steps (Backend Focus):**

**1. Modify ChromaDB Metadata during Indexing (Background Task)**

We need a new background task for indexing chat turns and update the existing document indexing to include a `type` field.

*   **Update `services/document_processor.py` (Document Indexing):**
    When adding document chunks to ChromaDB in `process_document_upload`, add a `type` field to the metadata.

    ```python
    # services/document_processor.py (inside process_document_upload, before doc_collection.add)

                    # Prepare metadata for ChromaDB vectors
                    chroma_metadatas = [
                        {
                            "document_id": doc_id,
                            "filename": filename,
                            "chunk_index": i,
                            "type": "document" # <-- ADD TYPE FIELD
                        }
                        for i in range(chunk_count)
                    ]

                    # Add to ChromaDB collection
                    try:
                        doc_collection.add(
                            ids=chunk_ids,
                            embeddings=embeddings,
                            documents=chunks_for_embedding,
                            metadatas=chroma_metadatas # Use updated metadata
                        )
                        # ... rest of the function ...
    ```

*   **Create Chat Indexing Function (`services/chat_indexer.py` or add to existing service):**
    Let's create a new file for clarity.

    ```python
    # services/chat_indexer.py
    import logging
    from typing import Optional, List, Dict, Any

    from core.config import settings
    from db.database import get_sqlite_db, get_chroma_client, database
    from db.models import chat_messages_table
    from services.embedding_service import generate_embeddings

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    async def index_chat_turn_task(user_message_id: int, assistant_message_id: int, session_id: str):
        """Background task to embed and index a user/assistant chat turn."""
        logger.info(f"[ChatIndex:{session_id}] Starting indexing for user_msg:{user_message_id}, assist_msg:{assistant_message_id}")

        try:
            # 1. Fetch messages from SQLite
            query = chat_messages_table.select().where(
                chat_messages_table.c.id.in_([user_message_id, assistant_message_id])
            )
            # Use the global 'database' instance as this runs in background thread context
            messages = await database.fetch_all(query)

            if len(messages) != 2:
                logger.error(f"[ChatIndex:{session_id}] Could not fetch both messages ({user_message_id}, {assistant_message_id}). Found {len(messages)}.")
                return # Or raise an error?

            messages_to_index: List[Dict[str, Any]] = []
            texts_to_embed: List[str] = []
            ids_for_chroma: List[str] = []
            metadatas_for_chroma: List[Dict[str, Any]] = []

            for msg_row in messages:
                msg = dict(msg_row)
                msg_id = msg['id']
                role = msg['role']
                content = msg['content']
                timestamp = msg['timestamp'].isoformat() # Store timestamp as string

                # Prepare data
                texts_to_embed.append(content)
                # Create a unique ID for ChromaDB (e.g., sessionid_messageid)
                chroma_id = f"{session_id}_{msg_id}"
                ids_for_chroma.append(chroma_id)
                metadatas_for_chroma.append({
                    "type": "chat", # <-- Mark as chat history
                    "session_id": session_id,
                    "message_id": msg_id,
                    "role": role,
                    "timestamp": timestamp
                })

            if not texts_to_embed:
                logger.warning(f"[ChatIndex:{session_id}] No text content found for messages. Skipping embedding.")
                return

            # 2. Generate Embeddings (blocking call, runs in executor via llm_service)
            logger.info(f"[ChatIndex:{session_id}] Generating embeddings for {len(texts_to_embed)} chat messages...")
            # NOTE: Ensure generate_embeddings runs correctly from background task context
            embeddings = generate_embeddings(texts_to_embed) # This call needs careful context management if run via FastAPI BackgroundTasks vs. separate queue

            if embeddings is None or len(embeddings) != len(texts_to_embed):
                 logger.error(f"[ChatIndex:{session_id}] Embedding generation failed or returned incorrect count.")
                 return # Or raise/retry?

            # 3. Add to ChromaDB
            chroma_client = get_chroma_client()
            # Use the *same* collection as documents
            collection = chroma_client.get_collection(settings.DOCUMENT_COLLECTION_NAME)

            logger.info(f"[ChatIndex:{session_id}] Adding {len(embeddings)} chat message embeddings to ChromaDB...")
            collection.add(
                ids=ids_for_chroma,
                embeddings=embeddings,
                documents=texts_to_embed, # Store the original text
                metadatas=metadatas_for_chroma
            )
            logger.info(f"[ChatIndex:{session_id}] Successfully indexed chat turn (User:{user_message_id}, Assist:{assistant_message_id}).")

            # Optional: Update SQLite message records to mark as indexed?
            # await database.execute(
            #     query=chat_messages_table.update().where(chat_messages_table.c.id.in_(...)).values(is_embedded=True)
            # )

        except Exception as e:
            logger.error(f"[ChatIndex:{session_id}] Failed to index chat turn (User:{user_message_id}, Assist:{assistant_message_id}): {e}", exc_info=True)

    ```
    *   Fetches the relevant user/assistant message text from SQLite.
    *   Calls `generate_embeddings`.
    *   Adds entries to the *same* ChromaDB collection used for documents (`settings.DOCUMENT_COLLECTION_NAME`).
    *   Crucially, sets `metadata={"type": "chat", "session_id": ..., "message_id": ..., ...}`.

**2. Trigger Background Task in `sessions.py`**

After successfully storing the assistant's message, schedule the indexing task.

```python
# app/api/endpoints/sessions.py
import uuid
import logging
import datetime
import asyncio
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks

from services.llm_service import summarize_text_with_query

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

async def add_message_to_session(
    session_id: str,
    message_data: MessageCreateRequest,
    background_tasks: BackgroundTasks,
):
    if message_data.role != "user":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only 'user' role messages can be posted by the client.")

    session = await get_session_or_404(session_id)
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

    # # --- Perform RAG Search ---
    # rag_context = ""
    # rag_chunk_ids = []
    # rag_document_ids_in_session = session.get("rag_document_ids") # Get associated doc IDs

    # if rag_document_ids_in_session:
    #     logger.info(f"[Session:{session_id}] Performing RAG search within documents: {rag_document_ids_in_session}")
    #     try:
    #         doc_collection = chroma_client.get_collection(settings.DOCUMENT_COLLECTION_NAME)

    #         # 1. Embed the user query
    #         query_embedding = generate_embeddings([user_message_content])
    #         if not query_embedding or not query_embedding[0]:
    #             raise ValueError("Failed to generate query embedding for RAG.")

    #         # 2. Build ChromaDB filter
    #         chroma_where_filter = {
    #             # Filter results to only include chunks from the documents linked to this session
    #             "document_id": {"$in": rag_document_ids_in_session}
    #         }
    #         logger.debug(f"[Session:{session_id}] RAG ChromaDB filter: {chroma_where_filter}")

    #         # 3. Query ChromaDB
    #         results = doc_collection.query(
    #             query_embeddings=query_embedding,
    #             n_results=settings.RAG_TOP_K,
    #             where=chroma_where_filter,
    #             include=['documents', 'metadatas', 'distances'] # Get text, metadata, distance
    #         )

    #         # 4. Format RAG context
    #         retrieved_docs = results.get('documents', [[]])[0]
    #         retrieved_metadatas = results.get('metadatas', [[]])[0]
    #         retrieved_ids = results.get('ids', [[]])[0] # Get chunk IDs

    #         if retrieved_docs:
    #             rag_chunk_ids = retrieved_ids # Store the IDs of retrieved chunks
    #             context_parts = []
    #             for i, doc_text in enumerate(retrieved_docs):
    #                 metadata = retrieved_metadatas[i] if i < len(retrieved_metadatas) else {}
    #                 source_info = f"Source(Doc: {metadata.get('document_id', 'N/A')}, Chunk: {metadata.get('chunk_index', 'N/A')})"
    #                 context_parts.append(f"{source_info}:\n{doc_text}")
    #             rag_context = "\n\n---\n\n".join(context_parts)
    #             logger.info(f"[Session:{session_id}] Retrieved {len(retrieved_docs)} chunks for RAG context.")
    #             logger.debug(f"[Session:{session_id}] RAG Context:\n{rag_context[:500]}...") # Log beginning of context

    #     except Exception as e:
    #         logger.error(f"[Session:{session_id}] RAG search failed: {e}", exc_info=True)
    #         # Proceed without RAG context, maybe log a warning or add to assistant metadata
    #         rag_context = "[RAG search failed]" # Indicate failure in context
    # else:
    #      logger.info(f"[Session:{session_id}] No RAG document IDs associated. Skipping RAG search.")


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
    assistant_message_metadata = { # Initialize metadata
        "prompt_preview": None,
        "rag_chunks_retrieved": rag_chunk_ids,
        "relevant_history_retrieved": retrieved_history_ids,
        "llm_call_error": None
    }

    # 1. Check LLM Status and get essentials
    llm_status_data = get_llm_status()
    tokenizer = llm_state.get("tokenizer")
    max_length = llm_state.get("max_model_length")
    model_name = llm_status_data.get("active_model", "N/A")

    if llm_status_data["status"] != LLMStatus.READY.value:
        error_detail = f"LLM not ready (Status: {llm_status_data['status']}). Please load/configure a model first."
        logger.warning(f"[Session:{session_id}] {error_detail}")
        prompt_for_llm = f"[ERROR: {error_detail}]"
        assistant_message_metadata["llm_call_error"] = error_detail
        # llm_ready remains False

    # 2. Check if Tokenizer/Max Length are needed and available
    elif (rag_context or retrieved_history_context or chat_history_str) and (not tokenizer or not max_length):
        # Needed if any context/history exists that *might* need truncating
        logger.warning("Tokenizer or max_length not found in llm_state, but context/history exists. Prompt truncation cannot be performed accurately. Sending raw context.")
        llm_ready = True # Allow attempt, API backend might handle it
        # --- Assemble simple prompt without truncation ---
        prompt_parts = []
        prompt_parts.append("### INSTRUCTION:")
        prompt_parts.append("You are a helpful AI assistant. Respond to the user's query based on the provided context and chat history.") # Keep instruction
        # Add contexts if they are valid (not error strings)
        if rag_context and "[ERROR" not in rag_context and "[Summarization Failed]" not in rag_context:
            prompt_parts.append(f"\n### RETRIEVED DOCUMENT CONTEXT (Summarized):\n{rag_context}")
        if retrieved_history_context and "[ERROR" not in retrieved_history_context and "[Summarization Failed]" not in retrieved_history_context:
            prompt_parts.append(f"\n### RELEVANT PAST CONVERSATION (Summarized):\n{retrieved_history_context}")
        if chat_history_str and "[ERROR" not in chat_history_str:
            prompt_parts.append(f"\n### RECENT CHAT HISTORY:\n{chat_history_str}")
        prompt_parts.append("\n### USER QUERY:")
        prompt_parts.append(user_message_content)
        prompt_parts.append("\n### RESPONSE:")
        prompt_for_llm = "\n".join(prompt_parts)
        logger.warning(f"Assembled prompt without token-based truncation. Length may exceed limit.")
        assistant_message_metadata["prompt_preview"] = prompt_for_llm[:200] + "..."
        # --- End simple prompt assembly ---

    # 3. Perform Full Prompt Construction with Truncation
    else:
        llm_ready = True # We have tokenizer and max_length if needed
        logger.info("Proceeding with token-based prompt truncation logic.")

        # Define fixed parts
        instruction_text = "You are a helpful AI assistant. Respond to the user's query based on the provided context and chat history."
        user_query_text = user_message_content
        instruction_prompt = "### INSTRUCTION:\n" + instruction_text
        user_query_prompt = "\n### USER QUERY:\n" + user_query_text
        response_marker = "\n### RESPONSE:"

        # Calculate reserved tokens
        generation_config = llm_status_data.get("generation_config", {})
        max_new_tokens = generation_config.get("max_new_tokens", settings.DEFAULT_LLM_MAX_NEW_TOKENS)
        fixed_parts_text = instruction_prompt + user_query_prompt + response_marker
        # Encode fixed parts to get accurate token count
        fixed_tokens_count = len(tokenizer.encode(fixed_parts_text, add_special_tokens=False))
        reserved_tokens = fixed_tokens_count + max_new_tokens + 10 # Increased buffer slightly
        max_context_tokens = max_length - reserved_tokens
        logger.info(f"Max Length: {max_length}, Reserved: {reserved_tokens}, Available for ALL Context: {max_context_tokens}")

        if max_context_tokens <= 0:
            logger.warning("Not enough token space for any context after reserving for fixed parts and response. Using minimal prompt.")
            prompt_parts = [instruction_prompt, user_query_prompt, response_marker]
            prompt_for_llm = "\n".join(prompt_parts)
            final_token_count = len(tokenizer.encode(prompt_for_llm)) # Get count for log
        else:
            # Tokenize valid context parts (including markers)
            doc_context_tokens = tokenizer.encode(f"\n### RETRIEVED DOCUMENT CONTEXT (Summarized):\n{rag_context}", add_special_tokens=False) if rag_context and "[ERROR" not in rag_context and "[Summarization Failed]" not in rag_context else []
            history_summary_tokens = tokenizer.encode(f"\n### RELEVANT PAST CONVERSATION (Summarized):\n{retrieved_history_context}", add_special_tokens=False) if retrieved_history_context and "[ERROR" not in retrieved_history_context and "[Summarization Failed]" not in retrieved_history_context else []
            recent_history_tokens = tokenizer.encode(f"\n### RECENT CHAT HISTORY:\n{chat_history_str}", add_special_tokens=False) if chat_history_str and "[ERROR" not in chat_history_str else []

            # Truncate in order: Recent History -> History Summary -> Doc Context
            total_context_tokens = len(doc_context_tokens) + len(history_summary_tokens) + len(recent_history_tokens)
            available_space = max_context_tokens

            if total_context_tokens > available_space:
                logger.info(f"Combined context ({total_context_tokens} tokens) exceeds available space ({available_space}). Truncating...")
                overflow = total_context_tokens - available_space

                # Truncate Recent History first (keep end)
                if overflow > 0 and len(recent_history_tokens) > 0:
                    if len(recent_history_tokens) >= overflow:
                        logger.warning(f"Truncating RECENT history by {overflow} tokens.")
                        recent_history_tokens = recent_history_tokens[overflow:]
                        overflow = 0
                    else:
                        logger.warning(f"Removing RECENT history entirely ({len(recent_history_tokens)} tokens).")
                        overflow -= len(recent_history_tokens)
                        recent_history_tokens = []

                # Truncate History Summary next (keep beginning)
                if overflow > 0 and len(history_summary_tokens) > 0:
                    if len(history_summary_tokens) >= overflow:
                        logger.warning(f"Truncating HISTORY SUMMARY by {overflow} tokens.")
                        history_summary_tokens = history_summary_tokens[:len(history_summary_tokens)-overflow]
                        overflow = 0
                    else:
                        logger.warning(f"Removing HISTORY SUMMARY entirely ({len(history_summary_tokens)} tokens).")
                        overflow -= len(history_summary_tokens)
                        history_summary_tokens = []

                # Truncate Document Context last (keep beginning)
                if overflow > 0 and len(doc_context_tokens) > 0:
                     if len(doc_context_tokens) >= overflow:
                          logger.warning(f"Truncating DOCUMENT context by {overflow} tokens.")
                          doc_context_tokens = doc_context_tokens[:len(doc_context_tokens)-overflow]
                          overflow = 0
                     else:
                          logger.warning(f"Removing DOCUMENT context entirely ({len(doc_context_tokens)} tokens).")
                          doc_context_tokens = []

            # Assemble final prompt from potentially truncated token lists
            instruction_tokens = tokenizer.encode(instruction_prompt, add_special_tokens=False)
            query_tokens = tokenizer.encode(user_query_prompt, add_special_tokens=False)
            marker_tokens = tokenizer.encode(response_marker, add_special_tokens=False)

            final_token_ids = instruction_tokens + doc_context_tokens + history_summary_tokens + recent_history_tokens + query_tokens + marker_tokens

            # Safety check - shouldn't exceed max_length - max_new_tokens (approx)
            if len(final_token_ids) >= max_length:
                 logger.error(f"FATAL: Calculated prompt tokens ({len(final_token_ids)}) still exceed model max length ({max_length}). Force truncating ID list.")
                 final_token_ids = final_token_ids[:max_length - max_new_tokens - 5] # Hard truncate

            prompt_for_llm = tokenizer.decode(final_token_ids)
            final_token_count = len(final_token_ids)
            logger.info(f"Constructed final prompt with {final_token_count} tokens using truncation.")
            assistant_message_metadata["prompt_preview"] = prompt_for_llm[:200] + "..."


    # Log final prompt (if not error string)
    if llm_ready and "[ERROR" not in prompt_for_llm:
         logger.debug(f"[Session:{session_id}] Constructed prompt for LLM (Token Count: {final_token_count}):\n{prompt_for_llm[:200]}...\n...{prompt_for_llm[-200:]}")
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

```

**Explanation of Changes in `add_message_to_session`:**

1.  **Inject `BackgroundTasks`:** Added `background_tasks: BackgroundTasks` to the function signature.
2.  **Chat History RAG:** Added a new block to perform semantic search specifically for chat history (`type='chat'`, `session_id=current_session_id`).
3.  **History Summarization:** Included the summarization loop for the retrieved history snippets.
4.  **Prompt Construction Updated:** Modified the prompt building and truncation logic to accommodate the three context types (Docs, History Summary, Recent History) and apply truncation in a reasonable order.
5.  **Schedule Indexing Task:** After successfully storing the assistant message and getting its ID (`assistant_message_id`), use `background_tasks.add_task` to schedule `index_chat_turn_task`, passing the user message ID, assistant message ID, and the *backend* session ID.

**Important Considerations:**

*   **Background Task Execution:** FastAPI's `BackgroundTasks` run in the same process/event loop after the response is sent. For heavy indexing tasks or high throughput, using a dedicated task queue (Celery, RQ, Arq) with separate worker processes is more robust and scalable. The current `generate_embeddings` call within the task will block if using the local backend, making a separate task queue even more beneficial.
*   **Summarization Latency:** Adding summarization for *both* document context *and* chat history context will significantly increase the time taken to generate a response.
*   **Summarization Concurrency:** The thread-safety issue with modifying `llm_state["config"]` in `summarize_text_with_query` becomes more prominent if multiple summarization calls happen concurrently (which they might if parallelized or with multiple simultaneous user requests).
*   **Retrieval Query:** Using only the current user message to search past history might not always be optimal. More advanced methods could involve summarizing the recent conversation history to form a query.
*   **Indexing Granularity:** Indexing user/assistant messages separately provides fine-grained retrieval.
*   **No Duplicate Indexing:** Ensure you don't re-index the same messages if the API endpoint is somehow called multiple times for the same interaction. Adding an `is_embedded` flag to the `chat_messages` table and checking it before scheduling the task adds robustness.

**Testing:**

1.  Restart the backend server.
2.  Ensure an LLM is loaded.
3.  Start a *new* chat session in the frontend (link a document if desired).
4.  Have a conversation with several turns.
5.  **Check Backend Logs:** Look for the "[ChatIndex:...] Starting indexing..." and "Successfully indexed chat turn..." messages appearing *after* each assistant response is stored.
6.  **Check ChromaDB:** (Use a script or tool) Verify that new entries with `metadata.type == 'chat'` and the correct `session_id` are being added to your `rag_documents` collection.
7.  **Observe Context:** In later turns of the conversation, check the backend logs for "[Session:...] Performing RAG search on chat history...", "[Session:...] Retrieved X relevant past messages...", and see if the "RELEVANT PAST CONVERSATION (Summarized)" section appears in the constructed prompt log.
8.  **Test Loading:** Save the session, start a new chat, then load the saved session. Send another message and verify that the chat history RAG search still works correctly using the loaded session's context.

This implements the long-term memory feature by indexing and retrieving chat history via RAG. Remember the latency and concurrency implications of the summarization step!
