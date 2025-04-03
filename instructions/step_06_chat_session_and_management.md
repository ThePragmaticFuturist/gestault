# Step 6: Chat Session and Message Management API #

We’re building a chat session API using FastAPI, where:

*    Users can start chat sessions (like a conversation thread).

*    Users can add messages to those sessions.

*    The system saves those messages in a SQLite database.

*    We can later retrieve or delete those sessions and messages.

Imagine it like how chat works in ChatGPT, but simplified.

Okay, let's create the API endpoints and database logic for managing chat sessions and their message history. We'll focus on storing the data correctly in our SQLite tables (`sessions` and `chat_messages`).
   
[Reasoning behind the code](reasoning/r6.md)
   
1.  **Define API Models (`app/api/models/chat.py`):**
    It's good practice to keep Pydantic models separate. Let's create a new directory `app/api/models` and a file `chat.py`.

    ```python
    # app/api/models/__init__.py
    # (Leave this file empty or add imports if needed later)
    ```

    ```python
    # app/api/models/chat.py
    import datetime
    from pydantic import BaseModel, Field
    from typing import List, Optional, Dict, Any

    # --- Request Models ---

    class SessionCreateRequest(BaseModel):
        name: Optional[str] = Field(None, description="Optional user-friendly name for the session.")
        rag_document_ids: Optional[List[str]] = Field(None, description="List of document IDs to associate with this session for RAG.")
        # llm_model_name: Optional[str] = Field(None, description="Specify LLM model for the session (overrides default).") # Add later if needed

    class MessageCreateRequest(BaseModel):
        content: str = Field(..., description="The text content of the message.")
        role: str = Field(default="user", description="Role of the sender ('user' or 'assistant'). For now, API typically receives 'user'.")


    # --- Response Models ---

    class SessionMetadataResponse(BaseModel):
        id: str
        name: Optional[str] = None
        created_at: datetime.datetime
        last_updated_at: datetime.datetime
        llm_model_name: Optional[str] = None
        rag_document_ids: Optional[List[str]] = None
        metadata_tags: Optional[Dict[str, Any]] = None # Or List[str] depending on tag format

        class Config:
            orm_mode = True # Allow mapping from SQLAlchemy RowProxy objects

    class ChatMessageResponse(BaseModel):
        id: int # Auto-incrementing ID from chat_messages table
        session_id: str
        timestamp: datetime.datetime
        role: str
        content: str
        metadata: Optional[Dict[str, Any]] = None

        class Config:
            orm_mode = True

    # Could combine SessionMetadata and Messages for a full session detail response later
    # class SessionDetailResponse(SessionMetadataResponse):
    #     messages: List[ChatMessageResponse] = []

    ```
    *   We define request models (`SessionCreateRequest`, `MessageCreateRequest`) and response models (`SessionMetadataResponse`, `ChatMessageResponse`).
    *   We use `orm_mode = True` in the response models' `Config` to easily create them from database query results.

2.  **Create Session API Endpoints (`app/api/endpoints/sessions.py`):**
    Create a new file for the session-related endpoints.

    ```python
    # app/api/endpoints/sessions.py
    import uuid
    import logging
    import datetime
    from typing import List, Optional

    from fastapi import APIRouter, HTTPException, Depends, status

    # Import models and db access
    from app.api.models.chat import (
        SessionCreateRequest,
        SessionMetadataResponse,
        MessageCreateRequest,
        ChatMessageResponse
    )
    from db.database import database # Use global 'database' instance
    from db.models import sessions_table, chat_messages_table, documents_table
    from core.config import settings
    import sqlalchemy

    logger = logging.getLogger(__name__)
    router = APIRouter()

    # --- Helper Function (Optional but recommended for validation) ---
    async def get_session_or_404(session_id: str) -> dict:
        """Fetches session metadata or raises HTTPException 404."""
        query = sessions_table.select().where(sessions_table.c.id == session_id)
        session = await database.fetch_one(query)
        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session with ID '{session_id}' not found.")
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
        response_model=ChatMessageResponse, # Return the created message
        status_code=status.HTTP_201_CREATED,
        summary="Add a message to a session",
    )
    async def add_message_to_session(
        session_id: str,
        message_data: MessageCreateRequest,
    ):
        """
        Adds a new message (typically from the user) to a specific chat session.
        Updates the session's last_updated_at timestamp.
        **Note:** This endpoint currently only stores the user message.
                 It does NOT generate or store an assistant response yet.
        """
        session = await get_session_or_404(session_id) # Ensure session exists

        # Validate role if necessary (e.g., allow only 'user' for now via API?)
        if message_data.role not in ["user", "assistant", "system"]: # Adjust allowed roles as needed
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid role: {message_data.role}")

        now = datetime.datetime.now(datetime.timezone.utc)

        # Use a transaction to insert message and update session timestamp
        async with database.transaction():
            try:
                # Insert the new message
                insert_message_query = chat_messages_table.insert().values(
                    session_id=session_id,
                    timestamp=now,
                    role=message_data.role,
                    content=message_data.content,
                    metadata=None, # Add message metadata later if needed
                ).returning(chat_messages_table.c.id) # Get the auto-incremented ID back

                created_message_id = await database.execute(insert_message_query)

                # Update the session's last_updated_at timestamp
                update_session_query = sessions_table.update().where(sessions_table.c.id == session_id).values(
                    last_updated_at=now
                )
                await database.execute(update_session_query)

                logger.info(f"Added message (ID: {created_message_id}) to session {session_id}.")

                # Fetch the newly created message to return it
                select_message_query = chat_messages_table.select().where(chat_messages_table.c.id == created_message_id)
                new_message = await database.fetch_one(select_message_query)

                if not new_message: # Should not happen if insert succeeded, but good check
                     raise HTTPException(status_code=500, detail="Failed to retrieve newly created message.")

                return ChatMessageResponse.parse_obj(dict(new_message))

            except Exception as e:
                logger.error(f"Failed to add message to session {session_id}: {e}", exc_info=True)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to add message to session {session_id}.")


    @router.get(
        "/{session_id}/messages",
        response_model=List[ChatMessageResponse],
        summary="List messages in a session",
    )
    async def list_messages_in_session(
        session_id: str,
        skip: int = 0,
        limit: int = 1000, # Default to a large limit for messages? Or make configurable
    ):
        """
        Retrieves the message history for a specific chat session, ordered by timestamp (oldest first).
        """
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
    *   **Structure:** Creates a new router for session endpoints.
    *   **Helper:** `get_session_or_404` reduces code duplication for checking if a session exists.
    *   **Create:** Generates UUID, uses current time, inserts into `sessions_table`, returns the created session data.
    *   **List:** Selects from `sessions_table` with ordering and pagination.
    *   **Get:** Uses the helper to fetch and return single session metadata.
    *   **Delete:** Uses a transaction to delete messages first, then the session, ensuring atomicity. Returns `204 No Content`.
    *   **Add Message:** Checks session exists, validates role, uses a transaction to insert the message and **importantly** update `last_updated_at` on the `sessions` table. Returns the created message data. *This is the core interaction point for the user during a chat.*
    *   **List Messages:** Selects messages for a session, ordered by timestamp (ascending for chat history).

3.  **Integrate Session Router in `app/main.py`:**
    Include the new router in your main application file.

    ```python
    # app/main.py
    # ... (keep existing imports)
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import routers
    from app.api.endpoints import documents
    from app.api.endpoints import sessions

    from core.config import settings
    from db.database import connect_db, disconnect_db, create_metadata_tables, get_or_create_collections
    from services.embedding_service import get_embedding_model

    # --- Create database tables and Chroma collections ---
    # ... (keep initialization logic)

    # --- FastAPI Application Setup ---
    app = FastAPI(
        title=settings.APP_NAME,
        description="API server for managing RAG documents, interacting with LLMs, and managing chat sessions.",
        version="0.1.0",
         # Add lifespan context manager for cleaner startup/shutdown with model loading
        # lifespan=lifespan # We can implement this later if needed
    )

    # --- Event Handlers for Database Connections ---
    @app.on_event("startup")
    async def startup_event():
        """Connect to database and pre-load models on server startup."""
        print("Server starting up...")
        await connect_db()
        print("Pre-loading embedding model...")
        try:
            get_embedding_model()
            print("Embedding model loading initiated (check logs for completion).")
        except Exception as e:
            print(f"ERROR: Failed to pre-load embedding model during startup: {e}")
        print("Server startup complete.")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Disconnect from database on server shutdown."""
        print("Server shutting down...")
        await disconnect_db()
        print("Server shutdown complete.")


    # --- Basic Root Endpoint ---
    @app.get("/")
    async def read_root():
        """ Root endpoint providing basic server information. """
        return {
            "message": f"Welcome to the {settings.APP_NAME}!",
            "status": "running",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }


    # --- Include API Routers ---
    app.include_router(
        documents.router,
        prefix="/api/v1/documents",
        tags=["Documents"]
    )
    # --- ADDED: Include sessions router ---
    app.include_router(
        sessions.router,
        prefix="/api/v1/sessions",
        tags=["Chat Sessions"] # New tag for organization
    )
    # --- END ADDED ---

    # Placeholder for Future Routers
    # Example: from app.api.endpoints import models, system
    # Example: app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
    # Example: app.include_router(system.router, prefix="/api/v1/system", tags=["System Status"])

    # --- Running the Server (for direct execution) ---
    if __name__ == "__main__":
        print(f"Starting server on http://{settings.SERVER_HOST}:{settings.SERVER_PORT}")
        uvicorn.run(
            "app.main:app",
            host=settings.SERVER_HOST,
            port=settings.SERVER_PORT,
            reload=True,
            reload_dirs=[
                str(settings.BASE_DIR / 'app'),
                str(settings.BASE_DIR / 'core'),
                str(settings.BASE_DIR / 'db'),
                str(settings.BASE_DIR / 'services')
            ]
        )

    ```

4.  **Run and Test:**
    *   Restart the server: `uvicorn app.main:app --reload --reload-dirs app --reload-dirs core --reload-dirs db --reload-dirs services`
    *   Go to the API docs (`http://localhost:8000/docs`). You should see a new section "Chat Sessions".
    *   **Test Session Creation:**
        *   Use `POST /api/v1/sessions/`. Click "Try it out".
        *   Provide an optional name and/or a list of (dummy, for now) `rag_document_ids`. Example: `{"name": "My Test Chat", "rag_document_ids": ["doc-abc-123"]}` or just `{}`.
        *   Execute. You should get a `201 Created` response with the new session's details (ID, timestamps, etc.). Note the `session_id`.
    *   **Test List Sessions:**
        *   Use `GET /api/v1/sessions/`. Execute. You should see the session(s) you created.
    *   **Test Get Session:**
        *   Use `GET /api/v1/sessions/{session_id}`. Enter the `session_id` you noted. Execute. You should get the details for that specific session.
    *   **Test Adding Messages:**
        *   Use `POST /api/v1/sessions/{session_id}/messages`. Enter the `session_id`.
        *   In the request body, provide the user's message: `{"content": "Hello, this is my first message!"}` (role defaults to 'user').
        *   Execute. You should get a `201 Created` response showing the message details (including its own ID).
        *   Add another message: `{"content": "This is a second message."}`. Execute.
    *   **Test Listing Messages:**
        *   Use `GET /api/v1/sessions/{session_id}/messages`. Enter the `session_id`. Execute. You should see the two messages you added, ordered correctly by timestamp.
    *   **Test Session Update Time:**
        *   Use `GET /api/v1/sessions/{session_id}` again. Note the `last_updated_at` timestamp. It should reflect the time you added the last message.
    *   **Test Delete Session:**
        *   Use `DELETE /api/v1/sessions/{session_id}`. Enter the `session_id`. Execute. You should get a `204 No Content` response.
        *   Try `GET /api/v1/sessions/{session_id}` again. You should now get a `404 Not Found`.
        *   Try `GET /api/v1/sessions/{session_id}/messages`. You should also get a `404 Not Found` (because the parent session is gone).
        *   Check your SQLite database directly: the session and its corresponding messages should be removed from `sessions` and `chat_messages` tables.

**Summary:**

We have now established the basic framework for managing chat sessions:
*   API endpoints to create, list, get, and delete sessions.
*   API endpoints to add user messages to a session and list a session's message history.
*   Database logic to store session metadata in the `sessions` table and message history in the `chat_messages` table.
*   Correct handling of timestamps (`created_at`, `last_updated_at`) and relationships between sessions and messages.
*   Use of transactions for data integrity (deleting sessions, adding messages).

**Next Steps:**

The server can now store conversations. The next logical steps involve making the chat interactive and leveraging our RAG setup:

1.  **LLM Integration:**
    *   Loading an actual LLM (local or via API).
    *   Modifying the `POST /api/v1/sessions/{session_id}/messages` endpoint:
        *   After storing the user message.
        *   **(RAG Step)** Perform semantic search on associated documents (using `rag_document_ids` from the session) and potentially past messages in the current session.
        *   Construct a prompt including the retrieved context and the latest user message (and possibly some prior conversation turns).
        *   Send the prompt to the loaded LLM.
        *   Receive the LLM response.
        *   Store the LLM response as an 'assistant' message in the `chat_messages` table.
        *   Return *both* the user message *and* the assistant response (or just the assistant response) to the client.
2.  **LLM Management API:**
    *   Endpoints to list available local models.
    *   Endpoint to load a specific model (from Hugging Face Hub or local path).
    *   Endpoint to get the status of the currently loaded model.
    *   Endpoints to get/set LLM parameters (temperature, top_k, etc.).
3.  **System Status API:** Endpoints to report CPU/GPU usage, RAM, disk space, etc.
