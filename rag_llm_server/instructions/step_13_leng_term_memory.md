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
import logging
import asyncio
from typing import Dict, Any
# --- ADD BackgroundTasks import ---
from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
# --- ADD Import for chat indexer ---
from services.chat_indexer import index_chat_turn_task
# ... (other imports: models, db, services) ...

@router.post(
    "/{session_id}/messages",
    # ... (decorator) ...
)
async def add_message_to_session(
    session_id: str,
    message_data: MessageCreateRequest,
    background_tasks: BackgroundTasks, # <-- Inject BackgroundTasks
    # ... (other parameters if using Depends)
):
    # ... (message validation, get session, store user message - get user_message_id) ...
    user_message_id = None # Initialize
    try:
        # ... (code to insert user message) ...
        user_message_id = await database.execute(insert_user_message_query) # Get ID
        # ... (update session timestamp) ...
    except Exception as e:
        # ... (error handling) ...

    # --- Perform RAG Search (unchanged) ---
    # ... (gets rag_context, rag_chunk_ids) ...

    # --- ### NEW: Perform Chat History RAG Search ### ---
    retrieved_history_context = ""
    retrieved_history_ids = []
    try:
        logger.info(f"[Session:{session_id}] Performing RAG search on chat history...")
        history_query_embedding = generate_embeddings([user_message_content]) # Embed user query
        if history_query_embedding and history_query_embedding[0]:
            chroma_client = get_chroma_client()
            collection = chroma_client.get_collection(settings.DOCUMENT_COLLECTION_NAME)
            # Filter for type='chat' AND current session_id
            history_where_filter = {
                "$and": [
                    {"type": "chat"},
                    {"session_id": session_id}
                ]
            }
            # How many history pieces to retrieve? Use RAG_TOP_K or a new setting?
            history_rag_k = settings.RAG_TOP_K # Reuse setting for now
            history_results = collection.query(
                query_embeddings=history_query_embedding,
                n_results=history_rag_k,
                where=history_where_filter,
                include=['documents', 'metadatas', 'distances']
            )
            retrieved_history_docs = history_results.get('documents', [[]])[0]
            retrieved_history_metadatas = history_results.get('metadatas', [[]])[0]
            retrieved_history_ids = history_results.get('ids', [[]])[0] # Chroma IDs like session_msgId

            if retrieved_history_docs:
                logger.info(f"[Session:{session_id}] Retrieved {len(retrieved_history_docs)} relevant past messages.")
                # --- ### Summarize Retrieved History ### ---
                summarized_history_parts = []
                loop = asyncio.get_running_loop()
                for i, history_text in enumerate(retrieved_history_docs):
                    hist_meta = retrieved_history_metadatas[i] if i < len(retrieved_history_metadatas) else {}
                    hist_role = hist_meta.get('role', 'Unknown')
                    # Maybe add role/timestamp to summarization prompt?
                    try:
                        summary = await loop.run_in_executor(
                            None, summarize_text_with_query,
                            f"{hist_role.capitalize()}: {history_text}", # Add role prefix for context
                            user_message_content, 100 # Max tokens per summary
                        )
                        if summary:
                            summarized_history_parts.append(summary)
                        else:
                            logger.warning(f"Failed to summarize history snippet {i}. Skipping.")
                    except Exception as summary_err:
                        logger.error(f"Error summarizing history snippet {i}: {summary_err}", exc_info=True)
                retrieved_history_context = "\n---\n".join(summarized_history_parts)
                if not retrieved_history_context:
                     retrieved_history_context = "[History Summarization Failed]"
                logger.debug(f"[Session:{session_id}] Summarized History Context:\n{retrieved_history_context[:500]}...")
            else:
                 logger.info(f"[Session:{session_id}] No relevant chat history found for this query.")

        else:
             logger.error(f"[Session:{session_id}] Failed to generate embedding for history search.")

    except Exception as e:
        logger.error(f"[Session:{session_id}] Chat history RAG search failed: {e}", exc_info=True)
        retrieved_history_context = "[History RAG search failed]"
    # --- ### END Chat History RAG Search ### ---


    # --- Retrieve RECENT Chat History (unchanged) ---
    # ... (gets chat_history_str) ...

    # --- Construct & Truncate Prompt (MODIFIED) ---
    # ... (get llm status, tokenizer, max_length) ...
    if llm_status_data["status"] == LLMStatus.READY.value and tokenizer and max_length:
        llm_ready = True
        # ... (calculate reserved_tokens, max_context_history_tokens) ...

        # --- Tokenize and Truncate Components (Order matters!) ---
        # Prioritize: Docs, then History Summary, then Recent History?
        
        # 1. Document Context Summaries (Tokenize/Truncate)
        doc_context_tokens = []
        if rag_context and "[ERROR" not in rag_context: # Check for valid doc context
            doc_context_prompt_part = f"\n### RETRIEVED DOCUMENT CONTEXT (Summarized):\n{rag_context}"
            doc_context_tokens = tokenizer.encode(doc_context_prompt_part, add_special_tokens=False)
            if len(doc_context_tokens) > max_context_history_tokens:
                logger.warning(f"Truncating DOC context ({len(doc_context_tokens)} tokens) to fit limit {max_context_history_tokens}.")
                doc_context_tokens = doc_context_tokens[:max_context_history_tokens]

        available_tokens_after_docs = max_context_history_tokens - len(doc_context_tokens)

        # 2. History Summaries (Tokenize/Truncate)
        history_summary_tokens = []
        if retrieved_history_context and "[ERROR" not in retrieved_history_context and available_tokens_after_docs > 0:
             hist_summary_prompt_part = f"\n### RELEVANT PAST CONVERSATION (Summarized):\n{retrieved_history_context}"
             history_summary_tokens = tokenizer.encode(hist_summary_prompt_part, add_special_tokens=False)
             if len(history_summary_tokens) > available_tokens_after_docs:
                  logger.warning(f"Truncating HISTORY SUMMARY context ({len(history_summary_tokens)} tokens) to fit limit {available_tokens_after_docs}.")
                  history_summary_tokens = history_summary_tokens[:available_tokens_after_docs]

        available_tokens_for_recent_hist = available_tokens_after_docs - len(history_summary_tokens)

        # 3. Recent Chat History (Tokenize/Truncate from OLD)
        recent_history_tokens = []
        if chat_history_str and available_tokens_for_recent_hist > 0:
            recent_hist_prompt_part = f"\n### RECENT CHAT HISTORY:\n{chat_history_str}"
            recent_history_tokens = tokenizer.encode(recent_hist_prompt_part, add_special_tokens=False)
            if len(recent_history_tokens) > available_tokens_for_recent_hist:
                 logger.warning(f"Truncating RECENT history ({len(recent_history_tokens)} tokens) to fit limit {available_tokens_for_recent_hist}.")
                 recent_history_tokens = recent_history_tokens[-available_tokens_for_recent_hist:]

        # --- Assemble Final Prompt ---
        final_prompt_parts = []
        final_prompt_parts.append("### INSTRUCTION:")
        # ... (instruction text) ...

        if doc_context_tokens: # Add if exists
            # ... (decode doc context tokens) ...
            decoded_doc_context_header = tokenizer.decode(tokenizer.encode("\n### RETRIEVED DOCUMENT CONTEXT (Summarized):", add_special_tokens=False))
            header_token_len = len(tokenizer.encode(decoded_doc_context_header, add_special_tokens=False))
            decoded_doc_context_body = tokenizer.decode(doc_context_tokens[header_token_len:])
            final_prompt_parts.append(decoded_doc_context_header + decoded_doc_context_body)


        if history_summary_tokens: # Add if exists
             # ... (decode history summary tokens) ...
             decoded_hist_summary_header = tokenizer.decode(tokenizer.encode("\n### RELEVANT PAST CONVERSATION (Summarized):", add_special_tokens=False))
             header_token_len = len(tokenizer.encode(decoded_hist_summary_header, add_special_tokens=False))
             decoded_hist_summary_body = tokenizer.decode(history_summary_tokens[header_token_len:])
             final_prompt_parts.append(decoded_hist_summary_header + decoded_hist_summary_body)


        if recent_history_tokens: # Add if exists
            # ... (decode recent history tokens) ...
            decoded_recent_hist_header = tokenizer.decode(tokenizer.encode("\n### RECENT CHAT HISTORY:", add_special_tokens=False))
            header_token_len = len(tokenizer.encode(decoded_recent_hist_header, add_special_tokens=False))
            decoded_recent_hist_body = tokenizer.decode(recent_history_tokens[header_token_len:])
            final_prompt_parts.append(decoded_recent_hist_header + decoded_recent_hist_body)


        # ... (user query, response marker) ...
        final_prompt_parts.append("\n### USER QUERY:")
        # ... (user_query_text) ...
        final_prompt_parts.append("\n### RESPONSE:")

        prompt_for_llm = "\n".join(final_prompt_parts)
        # ... (Final Token Check) ...

    # ... (Check llm_ready flag) ...

    # --- Call LLM Service ---
    if llm_ready:
        try:
            # ... (await generate_text(prompt_for_llm)) ...
            llm_response = await generate_text(prompt_for_llm)
            # ... (handle response/error) ...
        except Exception as e:
            # ... (handle generation error) ...

    # ... (Store Assistant Message - Get assistant_message_id) ...
    assistant_message_id = None # Initialize
    try:
        # ... (code to insert assistant message) ...
        new_assistant_message_row = await database.fetch_one(insert_assistant_message_query)
        if not new_assistant_message_row: # ... error handling ...
        assistant_message_id = new_assistant_message_row['id'] # Get ID
        # ... (update session timestamp) ...

        # --- ### Schedule Background Indexing Task ### ---
        if user_message_id is not None and assistant_message_id is not None:
             logger.info(f"[Session:{session_id}] Scheduling background task to index chat turn ({user_message_id}, {assistant_message_id}).")
             background_tasks.add_task(
                 index_chat_turn_task,
                 user_message_id=user_message_id,
                 assistant_message_id=assistant_message_id,
                 session_id=session_id # Pass backend session ID
             )
        else:
              logger.warning(f"[Session:{session_id}] Missing user or assistant message ID, cannot schedule indexing.")
        # --- ### End Scheduling ### ---

        return ChatMessageResponse.parse_obj(dict(new_assistant_message_row))

    except Exception as e:
        # ... (handle storing assistant message error) ...

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
