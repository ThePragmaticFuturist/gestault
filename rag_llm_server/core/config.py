# core/config.py
import os
import torch
from pydantic_settings import BaseSettings
from pydantic import Field # Import Field
from pathlib import Path
import tempfile # Import tempfile
from typing import Optional, List, Any, Literal

BASE_DIR = Path(__file__).resolve().parent.parent

def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    # Add checks for MPS (Apple Silicon) or other accelerators if needed
    # elif torch.backends.mps.is_available():
    #     return "mps"
    else:
        return "cpu"

class Settings(BaseSettings):
    # --- General Settings ---
    APP_NAME: str = "RAG LLM Server"
    BASE_DIR: Path = BASE_DIR

    # --- Database Settings ---
    DATA_DIR: Path = BASE_DIR / "data"
    SQLITE_DB_NAME: str = "rag_server.db"
    CHROMA_DB_DIR_NAME: str = "chroma_db"

    # --- Calculated Paths ---
    @property
    def SQLITE_DATABASE_URL(self) -> str:
        db_dir = self.DATA_DIR
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / self.SQLITE_DB_NAME
        return f"sqlite+aiosqlite:///{db_path.resolve()}"

    @property
    def CHROMA_PERSIST_PATH(self) -> str:
        chroma_path = self.DATA_DIR / self.CHROMA_DB_DIR_NAME
        chroma_path.mkdir(parents=True, exist_ok=True)
        return str(chroma_path.resolve())

    # --- LLM Settings (Placeholders) ---
    DEFAULT_MODEL_NAME: str = "placeholder-model"
    HUGGINGFACE_HUB_CACHE: Path = BASE_DIR / "models_cache"

    # --- Server Settings ---
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000

    # --- NEW: Embedding Settings ---
    EMBEDDING_MODEL_NAME: str = Field(
        default="all-MiniLM-L6-v2", # Good default, balance of speed/performance
        description="The Hugging Face sentence-transformers model name to use for embeddings."
    )
    EMBEDDING_DEVICE: str = Field(
        default="cuda", # Default to CPU, change to "cuda" if GPU is available and configured
        description="Device to run the embedding model on ('cpu', 'cuda')."
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default=32, # Process chunks in batches for efficiency
        description="Batch size for generating embeddings."
    )

    # --- LLM Settings ---
    LLM_BACKEND_TYPE: Literal["local", "ollama", "vllm", "instructlab"] = Field(
        default="local",
        description="The default LLM backend to use if not specified in the /load request ('local', 'ollama', 'vllm', 'instructlab')."
    )
    # Base URLs for API backends (only needed if corresponding type is selected)
    OLLAMA_BASE_URL: Optional[str] = Field(
        default="http://localhost:11434", # Common default
        description="Base URL for Ollama API (e.g., http://localhost:11434)."
    )
    VLLM_BASE_URL: Optional[str] = Field(
        default=None, # No standard default, often runs on 8000
        description="Base URL for vLLM OpenAI-compatible API (e.g., http://localhost:8000)."
    )
    INSTRUCTLAB_BASE_URL: Optional[str] = Field(
        default=None, # No standard default
        description="Base URL for InstructLab server API (if applicable)."
    )
    # Model name to use by default *on the configured backend*
    # Can be overridden by the /load request
    DEFAULT_MODEL_NAME_OR_PATH: str = Field(
        default="gpt2", # A reasonable default for 'local'
        description="Default model identifier (HF ID, local path, or name on API backend)."
    )

    HUGGING_FACE_HUB_TOKEN: Optional[str] = Field(
        default=None,
        description="Optional Hugging Face Hub token. If set, overrides login via CLI. Loaded from .env or environment variable."
    )

    # Directory to scan for locally stored models (e.g., downloaded via snapshot_download)
    LOCAL_MODELS_DIR: Path = BASE_DIR / "local_models"
    # Default device for LLM inference ('cuda', 'cpu', 'mps', 'auto')
    LLM_DEVICE: str = Field(default_factory=get_default_device, # Automatically detect best device
                            description="Device for LLM inference ('cuda', 'cpu', 'auto').")

    # Default quantization setting (can be overridden in load request)
    # Options: None, "8bit", "4bit"
    DEFAULT_LLM_QUANTIZATION: Optional[str] = Field(default=None, description="Default quantization (None, '8bit', '4bit'). Requires bitsandbytes.")
    # Model parameters (can be overridden in config request)
    DEFAULT_LLM_MAX_NEW_TOKENS: int = Field(default=512, description="Default max new tokens for LLM generation.")
    DEFAULT_LLM_TEMPERATURE: float = Field(default=0.7, description="Default temperature for LLM generation.")
    DEFAULT_LLM_TOP_P: float = Field(default=0.9, description="Default top_p for LLM generation.")
    DEFAULT_LLM_TOP_K: int = Field(default=50, description="Default top_k for LLM generation.") # Note: Not used by standard OpenAI Chat API

    # --- ChromaDB Settings ---
    DOCUMENT_COLLECTION_NAME: str = "rag_documents"
    SESSION_COLLECTION_NAME: str = "chat_sessions"

    # --- NEW: RAG/Chat Settings ---
    RAG_TOP_K: int = Field(default=3, description="Number of relevant document chunks to retrieve for RAG.")
    CHAT_HISTORY_LENGTH: int = Field(default=5, description="Number of recent message pairs (user+assistant) to include in the prompt.")

    # --- NEW: Document Processing Settings ---
    # Use system's temp dir for uploads before processing
    UPLOAD_DIR: Path = Path(tempfile.gettempdir()) / "rag_server_uploads"
    CHUNK_SIZE: int = Field(default=1000, description="Target size for text chunks (in characters)")
    CHUNK_OVERLAP: int = Field(default=150, description="Overlap between consecutive chunks (in characters)")
    SUPPORTED_FILE_TYPES: list[str] = Field(default=[
        "text/plain",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # .docx
        "text/markdown",
        "text/html",
    ])

    # --- Pydantic settings configuration ---
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore'
        # Allow reading Literal values case-insensitively if needed
        # case_sensitive = False

settings = Settings()

# --- Ensure necessary directories exist on import ---
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.HUGGINGFACE_HUB_CACHE.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True) # Create local models dir
_ = settings.SQLITE_DATABASE_URL
_ = settings.CHROMA_PERSIST_PATH

print(f"--- Configuration Loaded ---")
print(f"SQLite DB URL: {settings.SQLITE_DATABASE_URL}")
print(f"ChromaDB Path: {settings.CHROMA_PERSIST_PATH}")
print(f"Upload Dir: {settings.UPLOAD_DIR}")
print(f"Embedding Model: {settings.EMBEDDING_MODEL_NAME} on {settings.EMBEDDING_DEVICE}")
print(f"Local Models Dir: {settings.LOCAL_MODELS_DIR}") # Print new dir
print(f"LLM Device: {settings.LLM_DEVICE}") # Print LLM device
print(f"LLM Backend Type: {settings.LLM_BACKEND_TYPE}") # Print backend type
print(f"Default LLM Identifier: {settings.DEFAULT_MODEL_NAME_OR_PATH}")
if settings.LLM_BACKEND_TYPE == 'ollama':
    print(f"Ollama URL: {settings.OLLAMA_BASE_URL}")
if settings.LLM_BACKEND_TYPE == 'vllm':
    print(f"vLLM URL: {settings.VLLM_BASE_URL}")
print(f"--------------------------")