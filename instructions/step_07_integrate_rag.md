# Step 7: Integrate RAG and Placeholder LLM Call #

Okay, let's modify the `add_message_to_session` endpoint to incorporate the RAG search using ChromaDB and simulate an LLM response.

1.  **Update Configuration (Optional):** Add settings for RAG/Prompting if needed.
    ```python
    # core/config.py
    # ... (inside Settings class) ...

    class Settings(BaseSettings):
        # ... other settings ...

        # --- NEW: RAG/Chat Settings ---
        RAG_TOP_K: int = Field(default=3, description="Number of relevant document chunks to retrieve for RAG.")
        CHAT_HISTORY_LENGTH: int = Field(default=5, description="Number of recent message pairs (user+assistant) to include in the prompt.")

        # ... rest of settings ...

    settings = Settings()
    # ... rest of config.py ...
    ```
    *   `RAG_TOP_K`: How many chunks to fetch from ChromaDB.
    *   `CHAT_HISTORY_LENGTH`: How many previous messages to include in the prompt context.

2.  **Modify `add_message_to_session` Endpoint (`app/api/endpoints/sessions.py`):**
    This is where the main changes happen. We need to fetch session details, perform the search, construct context, and add the placeholder response.

    ```python
    # app/api/endpoints/sessions.py
    import uuid
    import logging
    import datetime
    from typing import List, Optional, Dict, Any # Added Dict, Any

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
    from app.api.endpoints.documents import SearchQuery # We might reuse SearchQuery or parts of it
    import sqlalchemy
    # --- END MODIFIED IMPORTS ---


    logger = logging.getLogger(__name__)
    router = APIRouter()

    # --- Helper Function (Keep as is) ---
    async def get_session_or_404(session_id: str) -> dict:
        # ... (implementation unchanged) ...
        """Fetches session metadata or raises HTTPException 404."""
        query = sessions_table.select().where(sessions_table.c.id == session_id)
        session = await database.fetch_one(query)
        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session with ID '{session_id}' not found.")
        # Convert RowProxy to dict, handle potential JSON fields if needed
        session_dict = dict(session)
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
        return session_dict


    # --- Session Endpoints (Keep / , GET /, GET /{id}, DELETE /{id} as they are) ---
    # ... create_session, list_sessions, get_session, delete_session ...


    # --- Message Endpoints (within a session) ---

    @router.post(
        "/{session_id}/messages",
        response_model=ChatMessageResponse, # Now returns the Assistant's message
        status_code=status.HTTP_201_CREATED,
        summary="Send a message and get AI response (with RAG)",
        description="Adds user message, performs RAG search on associated documents, "
                    "constructs prompt, simulates LLM response, stores it, and returns the assistant message.",
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

        # --- Construct Prompt (Placeholder) ---
        # This is where you'd format the final prompt for the LLM
        prompt_for_llm = f"""CONTEXT:
{rag_context if rag_context else "No RAG context available."}

CHAT HISTORY:
{chat_history_str if chat_history_str else "No history available."}

USER QUERY:
{user_message_content}

ASSISTANT RESPONSE:""" # LLM is expected to continue from here

        logger.debug(f"[Session:{session_id}] Constructed prompt (first/last 200 chars):\n{prompt_for_llm[:200]}...\n...{prompt_for_llm[-200:]}")

        # --- Placeholder LLM Call ---
        # Replace this with your actual LLM interaction logic later
        logger.info(f"[Session:{session_id}] Simulating LLM call...")
        # Simulate a delay if needed: await asyncio.sleep(1)
        assistant_response_content = f"Placeholder response based on your query: '{user_message_content[:50]}...'"
        if rag_context and rag_context != "[RAG search failed]":
            assistant_response_content += f"\nRetrieved context from {len(rag_chunk_ids)} chunk(s)."
        # You could even include parts of the context/prompt for debugging:
        # assistant_response_content += f"\nDEBUG_CONTEXT_START: {rag_context[:100]}..."
        logger.info(f"[Session:{session_id}] Simulated LLM response generated.")


        # --- Store Assistant Message ---
        assistant_message_metadata = {
            "prompt_preview": prompt_for_llm[:200] + "...", # Store a preview
            "rag_chunks_retrieved": rag_chunk_ids, # Store which chunks were used
        }
        try:
            insert_assistant_message_query = chat_messages_table.insert().values(
                session_id=session_id,
                timestamp=datetime.datetime.now(datetime.timezone.utc), # Use current time for assistant msg
                role="assistant",
                content=assistant_response_content,
                metadata=assistant_message_metadata,
            ).returning(chat_messages_table.c.id, *[c for c in chat_messages_table.c]) # Return all columns

            # Use fetch_one to get the newly inserted row directly
            new_assistant_message_row = await database.fetch_one(insert_assistant_message_query)

            if not new_assistant_message_row:
                raise Exception("Failed to retrieve assistant message after insert.")

            logger.info(f"[Session:{session_id}] Stored assistant message (ID: {new_assistant_message_row['id']}).")

            # Update session timestamp again to reflect the assistant's response time
            update_session_query_after_assist = sessions_table.update().where(sessions_table.c.id == session_id).values(
                 last_updated_at=new_assistant_message_row['timestamp'] # Use the assistant msg timestamp
            )
            await database.execute(update_session_query_after_assist)

            # Return the structured assistant message response
            return ChatMessageResponse.parse_obj(dict(new_assistant_message_row))

        except Exception as e:
            logger.error(f"[Session:{session_id}] Failed to store assistant message: {e}", exc_info=True)
            # If storing the assistant message fails, should we indicate this to the user?
            # For now, raise 500. A more robust system might return the user message ID
            # and an error state for the assistant response.
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
        # ... (implementation unchanged) ...
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

3.  **Explanation of Changes in `add_message_to_session`:**
    *   **Imports:** Added necessary imports like `get_chroma_client`, `generate_embeddings`, `settings`.
    *   **Get Session Data:** Fetches the full session dictionary using `get_session_or_404`.
    *   **Store User Message:** Stores the incoming user message immediately and updates `last_updated_at`. This ensures the user's input is saved even if subsequent steps fail.
    *   **RAG Search:**
        *   Checks if `rag_document_ids` are present in the session data.
        *   If yes:
            *   Gets the ChromaDB collection.
            *   Generates embedding for the `user_message_content`.
            *   Builds a ChromaDB `where` filter using `$in` to match the session's document IDs.
            *   Queries ChromaDB using `query_embeddings`, `n_results` (from settings), and the `where` filter. Includes `documents`, `metadatas`, and `distances`.
            *   Formats the retrieved document chunks into a `rag_context` string, including source info. Stores retrieved chunk IDs in `rag_chunk_ids`.
            *   Includes error handling for the RAG process.
    *   **Retrieve Chat History:**
        *   Queries the `chat_messages` table for the current session.
        *   Orders by timestamp (`desc` then `reverse` to get recent N in chronological order). Limits based on `CHAT_HISTORY_LENGTH` setting.
        *   Formats the history into a simple string.
    *   **Construct Prompt:** Creates a basic text prompt string combining the RAG context, chat history, and the user query, clearly demarcated.
    *   **Placeholder LLM Call:** Generates a simple hardcoded string as the assistant's response, potentially including some info about whether context was retrieved.
    *   **Store Assistant Message:**
        *   Creates metadata for the assistant message (prompt preview, RAG chunk IDs).
        *   Inserts the placeholder response with `role='assistant'` into `chat_messages`. Uses `.returning()` and `fetch_one` to get the full row data back efficiently.
        *   Updates the session's `last_updated_at` again to reflect the time of the assistant's response.
    *   **Return Value:** Returns the `ChatMessageResponse` for the *assistant's* message.

4.  **Run and Test:**
    *   If you added the new settings, update your `.env` file or environment variables if needed.
    *   Restart the server: `uvicorn app.main:app --reload ...`
    *   Go to `/docs`.
    *   **Create a session and link documents:**
        *   First, make sure you have uploaded and processed documents (e.g., one with ID `doc-abc-123`).
        *   Use `POST /api/v1/sessions/` with a body like: `{"name": "RAG Test Session", "rag_document_ids": ["doc-abc-123"]}`. Note the `session_id`.
    *   **Send a message:**
        *   Use `POST /api/v1/sessions/{session_id}/messages` (using the ID from the previous step).
        *   Send a user message relevant to the content of `doc-abc-123`: `{"content": "What does document abc say about topic X?"}`.
    *   **Check Response:** The response should be a `ChatMessageResponse` with `role: "assistant"` and the placeholder content (e.g., "...Retrieved context from 3 chunk(s).").
    *   **Check Logs:** Examine the server logs for messages about:
        *   Storing the user message.
        *   Performing RAG search (and the filter used).
        *   Retrieving chunks.
        *   Retrieving chat history.
        *   The constructed prompt preview.
        *   Storing the assistant message.
    *   **Check Database:** Look at the `chat_messages` table. You should see both the user message and the assistant message stored correctly, including the metadata field for the assistant message.
    *   **Test without RAG docs:** Create a session without `rag_document_ids` and send a message. Verify the logs show "Skipping RAG search" and the response doesn't mention retrieved context.

**Summary:**

The chat endpoint now performs the core RAG loop:
*   Stores the user message.
*   Retrieves relevant document chunks based on the user query and session-linked documents.
*   Retrieves recent chat history.
*   Constructs a prompt (ready for a real LLM).
*   Simulates and stores an assistant response.
*   Returns the assistant's response.

**Next Steps:**

We are very close to a fully functional RAG system. The main remaining pieces are:

1.  **LLM Loading and Management:** Implement endpoints and logic to load actual LLMs (local or Hugging Face) and manage their settings.
2.  **Replace Placeholder LLM Call:** Substitute the placeholder logic in `add_message_to_session` with a real call to the loaded LLM using the constructed prompt.
3.  **System Status API:** Add endpoints to monitor server resources.
4.  **Advanced Features (Optional Later):** Session search/tagging, embedding chat history, hybrid search, UI, etc.
