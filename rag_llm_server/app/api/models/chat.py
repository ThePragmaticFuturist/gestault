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