#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM Server optimized for Python 3.10 and CUDA 11.8/12.4
Fixed for SQLAlchemy reserved keyword issue
"""
import sys
import os
import time
import json
import uuid
import logging
import uvicorn
import re
import asyncio
import tempfile
import uuid

import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query, Body, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from threading import Lock
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Text, Boolean, ForeignKey, Integer
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from urllib.parse import unquote
from document_processor import DocumentProcessor
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("llm_server")

try:
    import chromadb
    CHROMA_AVAILABLE = True
    logger.info("ChromaDB available for vector storage")
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not installed - vector storage features will be disabled")



# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Add this near the beginning of your file, after the imports
def setup_huggingface_credentials():
    """Setup Hugging Face credentials at startup"""
    try:
        # Get token from environment variable for security
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.warning("HF_TOKEN environment variable not set. Some models might not be accessible.")
            return False
            
        try:
            from huggingface_hub import login
            login(token=hf_token)
            logger.info("Successfully logged in to Hugging Face Hub")
            return True
        except ImportError:
            logger.warning("huggingface_hub not installed - cannot login to Hugging Face")
            return False
    except Exception as e:
        logger.error(f"Error setting up Hugging Face credentials: {e}")
        return False

huggingface_login_success = setup_huggingface_credentials()
logger.info(f"Hugging Face login {'successful' if huggingface_login_success else 'failed or skipped'}")

# First try to import torch to set CUDA configuration before other imports
try:
    import torch
    # CUDA configuration for PyTorch
    if torch.cuda.is_available():
        # Set memory strategies appropriate for CUDA 11.8/12.4
        torch.cuda.empty_cache()
        # Use TF32 precision (greatly improves speed on Ampere/newer GPUs)
        if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        logger.info(f"CUDA available, found {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            logger.info(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("CUDA not available, using CPU")
except ImportError:
    logger.warning("PyTorch not installed, some functionality will be limited")

# Import other dependencies
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers library available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available - LLM functionality will be limited")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("Sentence Transformers library available")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence Transformers not available - document similarity search will be disabled")

try:
    import fitz  # PyMuPDF
    PDF_PROCESSING_AVAILABLE = True
    logger.info("PyMuPDF library available for PDF processing")
except ImportError:
    PDF_PROCESSING_AVAILABLE = False
    logger.warning("PyMuPDF not available - PDF processing will be disabled")

# Define fallback embedding class if sentence-transformers is not available
if not SENTENCE_TRANSFORMERS_AVAILABLE:
    class SimpleSentenceEmbedder:
        def __init__(self, model_name):
            logger.warning(f"Using simple fallback embedder instead of {model_name}")
            import hashlib
            import numpy as np
            self.hashlib = hashlib
            self.np = np
            
        def encode(self, text):
            """Create a simple hash-based embedding for demo purposes"""
            # Handle both single strings and lists of strings
            if isinstance(text, list):
                # If given a list, process each item and return array of embeddings
                return self.np.array([self._encode_single(t) for t in text])
            else:
                # Process a single string
                return self._encode_single(text)
                
        def _encode_single(self, text):
            """Process a single text string to create an embedding"""
            # Create a simple embedding based on text hash (NOT FOR PRODUCTION)
            hash_obj = self.hashlib.sha256(text.encode('utf-8'))
            hash_bytes = hash_obj.digest()
            # Convert to a simple numpy array (this is NOT semantic, just for demo)
            return self.np.array([float(b) for b in hash_bytes[:64]]) / 255.0

# Database setup
Base = declarative_base()
DATABASE_URL = "sqlite:///./llm_server.db"
engine = create_engine(DATABASE_URL)

CHROMA_DIR = "./chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True)

# Create new table for vector store documents in the SQLAlchemy model definitions
class VectorDocument(Base):
    __tablename__ = "vector_documents"
    
    id = Column(String, primary_key=True, index=True)
    original_document_id = Column(String, ForeignKey("documents.id"))
    chunk_count = Column(Integer, default=0)
    is_indexed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    meta_info = Column(JSON)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

processing_websockets = {}

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, index=True)
    filename = Column(String, index=True)
    content = Column(Text)
    embedding_path = Column(String)
    meta_info = Column(JSON) #, server_default=text("'{}'"))  # Changed from metadata to meta_info
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"))
    role = Column(String)  # user or assistant
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("ChatSession", back_populates="messages")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(String, primary_key=True, index=True)
    model_name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    meta_info = Column(JSON) #, server_default=text("'{}'"))
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ProcessDocumentRequest(BaseModel):
    # Make document_id optional since it's already in the path
    document_id: Optional[str] = None
    prompt_template: str
    combine_prompt: Optional[str] = None
    chunk_size: int = 2000
    overlap: int = 200
    max_concurrent: int = 3
    llm_model_id: Optional[str] = None  # Use this for model ID
    processing_type: Optional[str] = "summarization"  # Add this field
    use_embeddings: bool = True  # Add this field
    
    # Add model config to disable protected namespaces if needed
    class Config:
        protected_namespaces = []

class ProcessDocumentResponse(BaseModel):
    status: str
    result: str
    processing_info: Dict[str, Any]

class ProcessedDocument(Base):
    __tablename__ = "processed_documents"
    
    id = Column(String, primary_key=True, index=True)
    original_document_id = Column(String, ForeignKey("documents.id"))
    content = Column(Text)
    processing_type = Column(String)
    meta_info = Column(JSON) #, server_default=text("'{}'"))  # Keep consistent with other models
    created_at = Column(DateTime, default=datetime.utcnow)

# Create all tables
Base.metadata.create_all(bind=engine)

# Pydantic Models for API
class ModelInfo(BaseModel):
    id: str
    name: str
    is_loaded: bool
    meta_info: Dict = {}  

    class Config:
        protected_namespaces = []

class DocumentInfo(BaseModel):
    id: str
    filename: str
    meta_info: Dict = {}  
    created_at: datetime


class DocumentContent(BaseModel):
    id: str
    filename: str
    content: str
    meta_info: Dict = {}  
    created_at: datetime


class ChatMessageModel(BaseModel):
    role: str
    content: str


class ChatSessionInfo(BaseModel):
    id: str
    model_name: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class ModelParameters(BaseModel):
    temperature: float = 0.7
    max_length: int = 1024
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    use_fp16: bool = True  # For CUDA optimization


class StatusResponse(BaseModel):
    status: str
    message: str

# Add this new Pydantic model for RAG queries
class RAGQuery(BaseModel):
    query: str
    document_ids: List[str] = []
    limit: int = 5
    
# Add this new Pydantic model for document processing with advanced options
class AdvancedProcessDocumentRequest(BaseModel):
    document_id: Optional[str] = None
    prompt_template: str
    combine_prompt: Optional[str] = None
    chunk_size: int = 2000
    overlap: int = 200
    max_concurrent: int = 3
    llm_model_id: Optional[str] = None
    processing_type: Optional[str] = "summarization"
    use_embeddings: bool = True
    use_hierarchical_chunking: bool = True
    store_in_vectordb: bool = True


# LLM Server Class
class LLMServer:
    def __init__(self, models_dir: str = "models", embeddings_dir: str = "embeddings"):
        self.models_dir = Path(models_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.embeddings_dir.mkdir(exist_ok=True)
        
        # Model management
        self.available_models: Dict[str, Dict] = {}
        self.active_model: Optional[Dict] = None
        self.model_lock = Lock()
        
        # Default embedding model
        self.embedding_model = None
        
        # Initialize default parameters
        self.parameters = ModelParameters()
        
        # CUDA batch size optimization
        self.cuda_available = torch.cuda.is_available() if 'torch' in sys.modules else False
        if self.cuda_available:
            self.parameters.use_fp16 = True
            
        # Scan for available models
        self._scan_models()
        
        # Initialize embedding model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self._initialize_embedding_model()
        else:
            # Use fallback embedder
            self.embedding_model = SimpleSentenceEmbedder("all-MiniLM-L6-v2")
    
    def _scan_models(self):
        """Scan for locally available models including those in Hugging Face cache"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers library not available - cannot scan for models")
            # Add a dummy model for testing
            self.available_models["dummy-model"] = {
                "id": "dummy-model",
                "name": "Dummy Model (No Transformers)",
                "is_loaded": False,
                "is_local": True,
                "path": "dummy_path"
            }
            return
            
        # First check models in our local models directory
        local_models = [d.name for d in self.models_dir.iterdir() if d.is_dir()]
        
        for model_id in local_models:
            self.available_models[model_id] = {
                "id": model_id,
                "name": model_id,
                "is_loaded": False,
                "is_local": True,
                "path": str(self.models_dir / model_id)
            }
        
        # Now scan the Hugging Face cache
        try:
            from huggingface_hub import scan_cache_dir
            cache_info = scan_cache_dir()
            
            for repo in cache_info.repos:
                # Extract the model_id which is the repo_id
                model_id = repo.repo_id
                
                # Skip if we already have this model
                if model_id in self.available_models:
                    continue
                    
                # Get the latest revision for this model
                if repo.revisions:
                    latest_revision = list(repo.revisions)[0]  # Assuming the first one is the latest
                    
                    self.available_models[model_id] = {
                        "id": model_id,
                        "name": model_id.split("/")[-1] if "/" in model_id else model_id,
                        "is_loaded": False,
                        "is_local": True,  # It's local since it's in the cache
                        "path": model_id,  # Use the model_id as path - from_pretrained will find it in cache
                        "cache_info": {
                            "size_mb": round(latest_revision.size_on_disk / (1024**2), 2),
                            "commit_hash": latest_revision.commit_hash,
                        }
                    }
                    
            # Add additional known models that work well
            hf_models = [
                "meta-llama/Llama-2-7b-chat-hf",
                "mistralai/Mistral-7B-Instruct-v0.2",
                "TheBloke/Llama-2-7B-Chat-GPTQ",
                "microsoft/phi-2",
                "stabilityai/stablelm-tuned-alpha-7b",
                "gpt2",
                "facebook/opt-1.3b",
                "EleutherAI/pythia-1.4b",
                "ibm-granite/granite-3.2-2b-instruct",
                "ibm-granite/granite-3.2-8b-instruct",
                "ibm-granite/granite-code:3b",
                "ibm-granite/granite-code:8b",
                "microsoft/phi-4"
            ]
            
            for model_id in hf_models:
                if model_id not in self.available_models:
                    self.available_models[model_id] = {
                        "id": model_id,
                        "name": model_id.split("/")[-1] if "/" in model_id else model_id,
                        "is_loaded": False,
                        "is_local": False,
                        "path": model_id
                    }
                    
        except ImportError:
            logger.warning("huggingface_hub not installed or up-to-date - cannot scan cache")
            # Fall back to just adding known models
            hf_models = [
                "meta-llama/Llama-2-7b-chat-hf",
                "mistralai/Mistral-7B-Instruct-v0.2",
                "TheBloke/Llama-2-7B-Chat-GPTQ",
                "microsoft/phi-2",
                "stabilityai/stablelm-tuned-alpha-7b",
                "gpt2",
                "facebook/opt-1.3b",
                "EleutherAI/pythia-1.4b",
                "ibm-granite/granite-3.2-2b-instruct",
                "ibm-granite/granite-3.2-8b-instruct",
                "ibm-granite/granite-code:3b",
                "ibm-granite/granite-code:8b",
                "microsoft/phi-4"
            ]
            
            for model_id in hf_models:
                if model_id not in self.available_models:
                    self.available_models[model_id] = {
                        "id": model_id,
                        "name": model_id.split("/")[-1] if "/" in model_id else model_id,
                        "is_loaded": False,
                        "is_local": False,
                        "path": model_id
                    }
        
        logger.info(f"Found {len(self.available_models)} available models")
        # Log actual model IDs at debug level
        logger.debug(f"Available models: {list(self.available_models.keys())}")     
           
    def _initialize_embedding_model(self):
        """Initialize the sentence embedding model"""
        try:
            # Use a model known to work well with CUDA 11.8/12.4
            self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            # Move to GPU if available
            if self.cuda_available:
                self.embedding_model.to(torch.device('cuda'))
            logger.info(f"Initialized embedding model: sentence-transformers/all-mpnet-base-v2 on {'GPU' if self.cuda_available else 'CPU'}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            try:
                # Try a simpler model as fallback
                self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
                if self.cuda_available:
                    self.embedding_model.to(torch.device('cuda'))
                logger.info(f"Initialized fallback embedding model: paraphrase-MiniLM-L3-v2 on {'GPU' if self.cuda_available else 'CPU'}")
            except Exception as e2:
                logger.error(f"Failed to initialize fallback embedding model: {e2}")
                logger.critical("Could not initialize any embedding model - using simple hash-based embedder")
                self.embedding_model = SimpleSentenceEmbedder("all-MiniLM-L6-v2")

    async def check_and_add_model(self, model_id: str) -> bool:
        """Check if a model exists on Hugging Face and add it to available models if it does"""
        if model_id in self.available_models:
            return True
            
        try:
            from huggingface_hub import model_info
            from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
            
            try:
                info = model_info(model_id)
                
                self.available_models[model_id] = {
                    "id": model_id,
                    "name": model_id.split("/")[-1] if "/" in model_id else model_id,
                    "is_loaded": False,
                    "is_local": False,
                    "path": model_id
                }
                
                logger.info(f"Found model {model_id} on Hugging Face - added to available models")
                return True
                
            except (RepositoryNotFoundError, RevisionNotFoundError):
                logger.warning(f"Model {model_id} not found on Hugging Face")
                return False
                
        except ImportError:
            logger.warning("huggingface_hub not installed - cannot check models on Hugging Face")
            return False

    async def load_model(self, model_id: str) -> Dict:
        """Load a model from local cache or Hugging Face"""
        if not TRANSFORMERS_AVAILABLE:
            return {"status": "error", "message": "Transformers library not available - cannot load models"}

        # Strip the "models/" prefix if it exists in the model_id
        original_model_id = model_id
        if model_id.startswith("models/"):
            model_id = model_id[7:]  # Remove "models/" prefix
            logger.info(f"Stripped 'models/' prefix from model ID: {original_model_id} -> {model_id}")

        # Check if model exists in our list or can be found on HuggingFace
        if model_id not in self.available_models:
            model_exists = await self.check_and_add_model(model_id)
            if not model_exists:
                return {"status": "error", "message": f"Model {model_id} not found locally or on Hugging Face"}
        
        try:
            # Now proceed with loading the model
            with self.model_lock:
                if self.available_models[model_id]["is_loaded"]:
                    logger.info(f"Model {model_id} already loaded")
                    self.active_model = self.available_models[model_id]
                    return {"status": "success", "message": f"Model {model_id} already loaded"}
                
                # Actual model loading logic
                model_info = self.available_models[model_id]
                logger.info(f"Loading model: {model_id}")
                
                # Clear CUDA cache first
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Get the correct path
                model_path = model_info["path"]
                
                # Special handling for IBM Granite models
                is_granite_model = "granite" in model_id.lower()
                is_phi_model = "phi" in model_id.lower()
                
                # Load tokenizer with specific handling for IBM Granite models
                try:
                    if is_granite_model:
                        # For IBM Granite models, use trust_remote_code=True
                        tokenizer = AutoTokenizer.from_pretrained(
                            model_path, 
                            trust_remote_code=True
                        )
                    else:
                        # For other models, try with legacy=False first
                        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
                except Exception as tokenizer_error:
                    logger.warning(f"Error loading tokenizer with first method, trying alternative: {tokenizer_error}")
                    try:
                        # Fall back to default parameters
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                    except Exception as second_error:
                        # Last resort with trust_remote_code=True for all models
                        logger.warning(f"Second tokenizer error, trying with trust_remote_code=True: {second_error}")
                        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

                # Load model with appropriate settings for CUDA
                if torch.cuda.is_available() and self.parameters.use_fp16:
                    # Use half precision (FP16) for faster inference on GPU
                    logger.info(f"Loading model with FP16 precision on GPU")
                    
                    if is_phi_model:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            torch_dtype=torch.float16,
                            low_cpu_mem_usage=True
                        )
                        # Manually move to CUDA after loading
                        model = model.to("cuda")
                    elif is_granite_model:
                        # For IBM Granite models
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            torch_dtype=torch.float16,
                            low_cpu_mem_usage=True,
                            device_map="auto",
                            trust_remote_code=True
                        )
                    else:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            torch_dtype=torch.float16,
                            low_cpu_mem_usage=True,
                            device_map="auto"
                        )

                else:
                    # Use CPU or full precision on GPU
                    logger.info(f"Loading model with full precision")
                    if is_phi_model and torch.cuda.is_available():
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            low_cpu_mem_usage=True
                        )
                        # Manually move to CUDA after loading
                        model = model.to("cuda")
                    elif is_granite_model:
                        # For IBM Granite models
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            device_map="auto" if torch.cuda.is_available() else None,
                            trust_remote_code=True
                        )
                    else:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            device_map="auto" if torch.cuda.is_available() and not is_phi_model else None
                        )
                        if torch.cuda.is_available() and not is_phi_model:
                            model = model.to("cuda")
                
                # Create pipeline with optimized batch size for GPU
                self.active_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=self.parameters.max_length,
                    temperature=self.parameters.temperature,
                    top_p=self.parameters.top_p,
                    top_k=self.parameters.top_k,
                    repetition_penalty=self.parameters.repetition_penalty,
                    no_repeat_ngram_size=self.parameters.no_repeat_ngram_size,
                    # Add trust_remote_code for IBM Granite models
                    trust_remote_code=True if is_granite_model else False
                )
                
                # Update model info
                model_info["is_loaded"] = True
                model_info["tokenizer"] = tokenizer
                model_info["model"] = model
                model_info["pipeline"] = self.active_pipeline
                model_info["loaded_at"] = datetime.utcnow()
                model_info["cuda_enabled"] = torch.cuda.is_available()
                
                self.active_model = model_info
                
                logger.info(f"Successfully loaded model: {model_id} on {'GPU' if torch.cuda.is_available() else 'CPU'}")
                return {"status": "success", "message": f"Model {model_id} loaded successfully on {'GPU' if torch.cuda.is_available() else 'CPU'}"}
                
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return {"status": "error", "message": f"Failed to load model: {str(e)}"}

                
    def list_models(self) -> List[ModelInfo]:
        """List all available models"""
        return [
            ModelInfo(
                id=model_id,
                name=model_info["name"],
                is_loaded=model_info["is_loaded"],
                meta_info={  # Changed from metadata to meta_info
                    "is_local": model_info["is_local"],
                    "loaded_at": model_info.get("loaded_at", None),
                    "cuda_enabled": model_info.get("cuda_enabled", False) if model_info.get("is_loaded", False) else None
                }
            ) 
            for model_id, model_info in self.available_models.items()
        ]
        
    def get_active_model(self) -> Optional[ModelInfo]:
        """Get the currently active model"""
        if not self.active_model:
            return None
            
        return ModelInfo(
            id=self.active_model["id"],
            name=self.active_model["name"],
            is_loaded=True,
            meta_info={  # Changed from metadata to meta_info
                "is_local": self.active_model["is_local"],
                "loaded_at": self.active_model.get("loaded_at", None),
                "cuda_enabled": self.active_model.get("cuda_enabled", False)
            }
        )
        
    def set_parameters(self, parameters: ModelParameters) -> Dict:
        """Set parameters for the LLM"""
        old_parameters = self.parameters
        self.parameters = parameters
        
        # Update pipeline if a model is loaded and transformers is available
        if TRANSFORMERS_AVAILABLE and self.active_model and "pipeline" in self.active_model:
            # If FP16 setting changed and we're on CUDA, reload the model
            if torch.cuda.is_available() and old_parameters.use_fp16 != parameters.use_fp16:
                logger.info(f"FP16 setting changed, reloading model with new precision")
                # Save current model ID
                model_id = self.active_model["id"]
                # Reset is_loaded flag
                self.available_models[model_id]["is_loaded"] = False
                # Schedule reload in background
                asyncio.create_task(self.load_model(model_id))
                return {"status": "success", "message": "Parameters updated, model reloading with new precision"}
            
            # Otherwise just update pipeline parameters
            try:
                self.active_pipeline = pipeline(
                    "text-generation",
                    model=self.active_model["model"],
                    tokenizer=self.active_model["tokenizer"],
                    max_length=parameters.max_length,
                    temperature=parameters.temperature,
                    top_p=parameters.top_p,
                    top_k=parameters.top_k,
                    repetition_penalty=parameters.repetition_penalty,
                    no_repeat_ngram_size=parameters.no_repeat_ngram_size
                )
                self.active_model["pipeline"] = self.active_pipeline
            except Exception as e:
                logger.error(f"Error updating pipeline parameters: {e}")
                return {"status": "error", "message": f"Error updating parameters: {str(e)}"}
            
        return {"status": "success", "message": "Parameters updated successfully"}
        
    def get_parameters(self) -> ModelParameters:
        """Get current parameters"""
        return self.parameters
        
    async def process_document(self, file_path: str, filename: str, meta_info: Dict = {}) -> str:  # Changed from metadata to meta_info
        """Process a document and create embeddings"""
        try:
            # Extract text based on file type
            file_extension = Path(filename).suffix.lower()
            
            if file_extension == '.pdf' and PDF_PROCESSING_AVAILABLE:
                content = self._extract_text_from_pdf(file_path)
            elif file_extension == '.pdf' and not PDF_PROCESSING_AVAILABLE:
                logger.warning("PDF processing not available - treating as text file")
                with open(file_path, 'rb') as f:
                    content = f"[PDF content not extracted - PyMuPDF not installed. Raw bytes: {len(f.read())} bytes]"
            elif file_extension in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json']:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
            else:
                # Try to read as text as fallback for unknown types
                logger.warning(f"Attempting to read unsupported file type as text: {file_extension}")
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                except Exception as text_read_error:
                    logger.error(f"Failed to read {file_extension} as text: {text_read_error}")
                    raise ValueError(f"Unsupported file type: {file_extension}")
                
            # Create document ID
            doc_id = str(uuid.uuid4())
            
            # Create embeddings if available
            embedding_path = str(self.embeddings_dir / f"{doc_id}.npy")
            
            if hasattr(self.embedding_model, 'encode'):
                # Create batched embedding for long documents to optimize CUDA memory
                if len(content) > 10000 and hasattr(self, 'cuda_available') and self.cuda_available:
                    # Split long text into chunks
                    chunks = [content[i:i+2000] for i in range(0, len(content), 2000)]
                    # Process in batches to avoid CUDA OOM errors
                    embeddings = self.embedding_model.encode(chunks)
                    # Average embeddings
                    embedding = np.mean(embeddings, axis=0)
                else:
                    embedding = self.embedding_model.encode(content)
                
                np.save(embedding_path, embedding)
            else:
                logger.warning("Embedding model not available - skipping document embedding")
                embedding_path = None
            
            # Save to database
            with SessionLocal() as db:
                db_document = Document(
                    id=doc_id,
                    filename=filename,
                    content=content,
                    embedding_path=embedding_path,
                    meta_info=meta_info if meta_info else {} # Changed from metadata to meta_info
                )
                db.add(db_document)
                db.commit()
                
            logger.info(f"Processed document: {filename}, ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            raise
            
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        if not PDF_PROCESSING_AVAILABLE:
            return "[PDF content not extracted - PyMuPDF not installed]"
            
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
        return text
        
    def get_document_info(self, doc_id: Optional[str] = None, filename: Optional[str] = None) -> List[DocumentInfo]:
        """Get document information by ID or filename"""
        with SessionLocal() as db:
            query = db.query(Document)
            
            if doc_id:
                query = query.filter(Document.id == doc_id)
            elif filename:
                query = query.filter(Document.filename == filename)
                
            documents = query.all()
            
            return [
                DocumentInfo(
                    id=doc.id,
                    filename=doc.filename,
                    meta_info=doc.meta_info,  # Changed from metadata to meta_info
                    created_at=doc.created_at
                )
                for doc in documents
            ]
            
    def get_document_content(self, doc_id: str) -> DocumentContent:
        """Get document content by ID"""
        with SessionLocal() as db:
            doc = db.query(Document).filter(Document.id == doc_id).first()
            
            if not doc:
                raise ValueError(f"Document with ID {doc_id} not found")
                
            return DocumentContent(
                id=doc.id,
                filename=doc.filename,
                content=doc.content,
                meta_info=doc.meta_info,  # Changed from metadata to meta_info
                created_at=doc.created_at
            )
            
    def delete_document(self, doc_id: str) -> Dict:
        """Delete a document by ID"""
        with SessionLocal() as db:
            doc = db.query(Document).filter(Document.id == doc_id).first()
            
            if not doc:
                raise ValueError(f"Document with ID {doc_id} not found")
                
            # Delete embedding file if it exists
            if doc.embedding_path and os.path.exists(doc.embedding_path):
                os.remove(doc.embedding_path)
                
            db.delete(doc)
            db.commit()
            
        logger.info(f"Deleted document: {doc_id}")
        return {"status": "success", "message": f"Document {doc_id} deleted successfully"}
        
    def delete_all_documents(self) -> Dict:
        """Delete all documents"""
        with SessionLocal() as db:
            docs = db.query(Document).all()
            
            # Delete embedding files
            for doc in docs:
                if doc.embedding_path and os.path.exists(doc.embedding_path):
                    os.remove(doc.embedding_path)
                    
            # Delete from database
            db.query(Document).delete()
            db.commit()
            
        logger.info("Deleted all documents")
        return {"status": "success", "message": "All documents deleted successfully"}
        
    def clean_response(self, response: str) -> str:
        """Clean LLM response by removing repetitions and fixing formatting"""
        # Remove repetitive sentences
        sentences = re.split(r'(?<=[.!?])\s+', response)
        unique_sentences = []
        
        for sentence in sentences:
            # Skip if it's empty or just whitespace
            if not sentence.strip():
                continue
                
            # Skip if very similar to the last sentence
            if unique_sentences and self._is_similar(sentence, unique_sentences[-1]):
                continue
                
            unique_sentences.append(sentence)
            
        cleaned_response = ' '.join(unique_sentences)
        
        # Fix formatting issues
        cleaned_response = re.sub(r'\s+', ' ', cleaned_response).strip()
        cleaned_response = re.sub(r'\.{2,}', '...', cleaned_response)
        
        return cleaned_response
        
    def _is_similar(self, str1: str, str2: str, threshold: float = 0.8) -> bool:
        """Check if two strings are similar using character overlap"""
        # Simple similarity check
        if not str1 or not str2:
            return False
            
        str1, str2 = str1.lower(), str2.lower()
        
        # If one is a subset of the other
        if str1 in str2 or str2 in str1:
            return True
            
        # Count common characters
        common_chars = set(str1) & set(str2)
        
        if not common_chars:
            return False
            
        # Jaccard similarity
        similarity = len(common_chars) / len(set(str1) | set(str2))
        
        return similarity > threshold
            
    # Add this method to the LLMServer class in app.py to ensure device consistency

    def _ensure_tensor_device_consistency(self, model, input_ids, attention_mask=None):
        """
        Ensure all tensors are on the same device as the model
        """
        device = next(model.parameters()).device
        
        # Move input_ids to the model's device
        input_ids = input_ids.to(device)
        
        # Move attention_mask if it exists
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            return input_ids, attention_mask
        
        return input_ids

    # Then modify the generate_response method to use this helper

    async def generate_response(self, prompt: str) -> str:
        """Generate a response from the active model"""
        if not TRANSFORMERS_AVAILABLE:
            return "Transformers library not available - cannot generate responses"
            
        if not self.active_model or not self.active_model.get("is_loaded", False):
            raise ValueError("No model loaded")
            
        try:
            # Clear CUDA cache before generation if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # If using the pipeline directly
            # Ensure the input is on the same device as the model
            # Get model from the pipeline
            model = self.active_model["model"]
            tokenizer = self.active_model["tokenizer"] 
            
            # Tokenize the input
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Ensure device consistency
            input_ids = self._ensure_tensor_device_consistency(
                model, 
                inputs["input_ids"]
            )
            
            if "attention_mask" in inputs:
                input_ids, attention_mask = self._ensure_tensor_device_consistency(
                    model,
                    inputs["input_ids"],
                    inputs["attention_mask"]
                )
                inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            else:
                inputs = {"input_ids": input_ids}
            
            # Generate with the model directly if pipeline is causing issues
            if "phi" in self.active_model["id"].lower():
                # For Phi models, use direct generation
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=self.parameters.max_length,
                        temperature=self.parameters.temperature,
                        top_p=self.parameters.top_p,
                        top_k=self.parameters.top_k,
                        repetition_penalty=self.parameters.repetition_penalty,
                        no_repeat_ngram_size=self.parameters.no_repeat_ngram_size,
                    )
                
                # Decode the output
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                # Use the pipeline for other models
                result = self.active_pipeline(prompt)[0]["generated_text"]
            
            # Remove the prompt from the response
            response = result[len(prompt):].strip()
            
            # Clean response
            cleaned_response = self.clean_response(response)
            
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    async def create_chat_session(self, model_name: Optional[str] = None) -> str:
        """Create a new chat session"""
        if not model_name and self.active_model:
            model_name = self.active_model["id"]
        elif not model_name:
            if len(self.available_models) > 0:
                model_name = next(iter(self.available_models.keys()))
            else:
                model_name = "dummy-model"
            logger.warning(f"No model specified and no active model loaded, using {model_name}")
            
        session_id = str(uuid.uuid4())
        
        with SessionLocal() as db:
            session = ChatSession(
                id=session_id,
                model_name=model_name
            )
            db.add(session)
            db.commit()
            
        logger.info(f"Created chat session: {session_id}")
        return session_id
        
    async def add_message_to_session(self, session_id: str, role: str, content: str) -> Dict:
        """Add a message to a chat session"""
        with SessionLocal() as db:
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            
            if not session:
                raise ValueError(f"Chat session with ID {session_id} not found")
                
            message = ChatMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                role=role,
                content=content
            )
            db.add(message)
            
            # Update session timestamp
            session.updated_at = datetime.utcnow()
            
            db.commit()
            
        return {"status": "success", "message": "Message added to session"}
        
    async def get_session_messages(self, session_id: str) -> List[ChatMessageModel]:
        """Get all messages in a chat session"""
        with SessionLocal() as db:
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            
            if not session:
                raise ValueError(f"Chat session with ID {session_id} not found")
                
            # Fixed: Use ChatMessage (SQLAlchemy model) instead of ChatMessageModel (Pydantic model)
            messages = db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).order_by(ChatMessage.created_at).all()
            
            return [
                ChatMessageModel(
                    role=msg.role,
                    content=msg.content
                )
                for msg in messages
            ]
            
    async def chat_completion(self, session_id: str, message: str, document_ids: list = None):
        """Process a chat message and generate a response"""
        print(message)

        # Get session
        with SessionLocal() as db:
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            
            if not session:
                raise ValueError(f"Chat session with ID {session_id} not found")
                
            # Load the model if it's not the active one
            if TRANSFORMERS_AVAILABLE and (not self.active_model or self.active_model["id"] != session.model_name):
                await self.load_model(session.model_name)
                
            # Get existing messages
            db_messages = db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).order_by(ChatMessage.created_at).all()
            
            history = [{"role": msg.role, "content": msg.content} for msg in db_messages]
            
            # Add document context if document_ids are provided
            document_context = ""
            if document_ids:
                docs = db.query(Document).filter(Document.id.in_(document_ids)).all()
                if docs:
                    document_context = "\n\nREFERENCE DOCUMENTS:\n"
                    for doc in docs:
                        document_context += f"\n{doc.filename}:\n{doc.content[:1000]}\n"
                    
                    # Append document context to the user message
                    message = f"{message}\n\n{document_context}"
            
        # Add user message to session
        await self.add_message_to_session(session_id, "user", message)
        
        if not TRANSFORMERS_AVAILABLE:
            # Add a dummy response if transformers not available
            dummy_response = f"[Dummy response - Transformers library not available] Received: {message}"
            await self.add_message_to_session(session_id, "assistant", dummy_response)
            return dummy_response
            
        # Build prompt with context
        prompt = self._build_chat_prompt(history, message)
        
        # Generate response
        response = await self.generate_response(prompt)
        
        # Add assistant response to session
        await self.add_message_to_session(session_id, "assistant", response)
        
        return response
        
    
    def _build_chat_prompt(self, history: List[Dict], new_message: str) -> str:
        """
        Build a prompt for chat completion based on history, with improved RAG handling.
        """
        # Extract RAG context if present
        rag_context = ""
        if "REFERENCE DOCUMENTS:" in new_message or "PROCESSED KNOWLEDGE:" in new_message:
            # Split user message from RAG context
            message_parts = new_message.split("\n\n", 1)
            user_message = message_parts[0]
            if len(message_parts) > 1:
                rag_context = message_parts[1]
        else:
            user_message = new_message
        
        # For Llama-2 and Mistral models, use their specific instruction format with RAG
        if "llama" in self.active_model["id"].lower() or "mistral" in self.active_model["id"].lower():
            system_prompt = (
                "You are a helpful, precise, and accurate AI assistant. Answer based on the provided context. "
                "If the context doesn't contain the information needed, acknowledge what you don't know. "
                "Keep responses informative and concise."
            )
            
            # Start with system prompt
            prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
            
            # Add RAG context to system prompt if available
            if rag_context:
                prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n\nContext information:\n{rag_context}\n<</SYS>>\n\n"
            
            # Add history in Llama format
            for i, msg in enumerate(history):
                if msg["role"] == "user":
                    if i > 0:  # Not the first message
                        prompt += f"<s>[INST] {msg['content']}"
                    else:  # First message
                        prompt += f"{msg['content']}"
                else:  # assistant message
                    prompt += f" [/INST] {msg['content']} </s>"
            
            # Add the new message
            if history:  # If there's history
                prompt += f"<s>[INST] {user_message} [/INST]"
            else:  # If this is the first message
                prompt += f"{user_message} [/INST]"
        else:
            # More generic format for other models with RAG improvements
            prompt = ""
            
            # Add RAG context as system information if available
            if rag_context:
                prompt += f"System: I'll provide you with context to help answer the user's questions. Use this information when relevant.\n\n{rag_context}\n\n"
            
            # Add history
            for msg in history:
                if msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                else:
                    prompt += f"Assistant: {msg['content']}\n"
                    
            # Add the new message
            prompt += f"User: {user_message}\nAssistant:"
        
        return prompt

    async def delete_chat_session(self, session_id: str) -> Dict:
        """Delete a chat session"""
        with SessionLocal() as db:
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            
            if not session:
                raise ValueError(f"Chat session with ID {session_id} not found")
                
            db.delete(session)
            db.commit()
            
        logger.info(f"Deleted chat session: {session_id}")
        return {"status": "success", "message": f"Chat session {session_id} deleted successfully"}
        
    async def delete_all_chat_sessions(self) -> Dict:
        """Delete all chat sessions"""
        with SessionLocal() as db:
            db.query(ChatSession).delete()
            db.commit()
            
        logger.info("Deleted all chat sessions")
        return {"status": "success", "message": "All chat sessions deleted successfully"}
        
    async def list_chat_sessions(self) -> List[ChatSessionInfo]:
        """List all chat sessions"""
        with SessionLocal() as db:
            sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
            
            result = []
            for session in sessions:
                # Count messages
                message_count = db.query(ChatMessage).filter(
                    ChatMessage.session_id == session.id
                ).count()
                
                result.append(ChatSessionInfo(
                    id=session.id,
                    model_name=session.model_name,
                    created_at=session.created_at,
                    updated_at=session.updated_at,
                    message_count=message_count
                ))
                
            return result
            
    async def search_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for documents similar to the query"""
        if not hasattr(self.embedding_model, 'encode'):
            logger.warning("Embedding model not available - returning random documents")
            # Return some documents without similarity ranking
            with SessionLocal() as db:
                documents = db.query(Document).limit(top_k).all()
                return [
                    {
                        "id": doc.id,
                        "filename": doc.filename,
                        "similarity": 0.0,  # No similarity score
                        "meta_info": doc.meta_info  # Changed from metadata to meta_info
                    }
                    for doc in documents
                ]
        
        # Convert query to embedding
        query_embedding = self.embedding_model.encode(query)
        
        results = []
        
        with SessionLocal() as db:
            documents = db.query(Document).all()
            
            for doc in documents:
                if doc.embedding_path and os.path.exists(doc.embedding_path):
                    doc_embedding = np.load(doc.embedding_path)
                    
                    # Calculate similarity
                    similarity = self._cosine_similarity(query_embedding, doc_embedding)
                    
                    results.append({
                        "id": doc.id,
                        "filename": doc.filename,
                        "similarity": float(similarity),
                        "meta_info": doc.meta_info  # Changed from metadata to meta_info
                    })
                    
        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top_k results
        return results[:top_k]
        
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def create_search_index_for_document(self, document_id):
        """Create a search index for a document for better RAG"""
        with SessionLocal() as db:
            doc = db.query(Document).filter(Document.id == document_id).first()
            
            if not doc or not doc.content:
                return False
                
            # Create embedding if not exists
            if not doc.embedding_path or not os.path.exists(doc.embedding_path):
                # Split content into paragraphs
                paragraphs = [p.strip() for p in doc.content.split('\n\n') if p.strip()]
                
                if not paragraphs:
                    return False
                    
                # Create embeddings for paragraphs
                if self.embedding_model and hasattr(self.embedding_model, 'encode'):
                    try:
                        paragraph_embeddings = self.embedding_model.encode(paragraphs)
                        
                        # Save paragraph embeddings
                        embeddings_dir = Path(self.embeddings_dir)
                        embedding_path = str(embeddings_dir / f"{doc.id}_paragraphs.npz")
                        
                        np.savez(
                            embedding_path, 
                            embeddings=paragraph_embeddings, 
                            paragraphs=paragraphs
                        )
                        
                        # Also update document embedding
                        doc_embedding = np.mean(paragraph_embeddings, axis=0)
                        doc_embedding_path = str(embeddings_dir / f"{doc.id}.npy")
                        np.save(doc_embedding_path, doc_embedding)
                        
                        # Update document record
                        doc.embedding_path = doc_embedding_path
                        db.commit()
                        
                        return True
                    except Exception as e:
                        logger.error(f"Error creating search index for document {document_id}: {e}")
                        return False
                
            return True

# FastAPI App
app = FastAPI(title="LLM Server API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - in production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize LLM Server
llm_server = LLMServer()


# API Routes
@app.get("/")
async def root():
    return {"message": "LLM Server API", "version": "1.0.0"}


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models"""
    return llm_server.list_models()

@app.post("/models/{model_id:path}/load")
async def load_model(model_id: str):
    """Trigger model loading process"""
    # Strip the "models/" prefix if it exists
    if model_id.startswith("models/"):
        model_id = model_id[7:]  # Remove "models/" prefix
        logger.info(f"Stripped 'models/' prefix from model ID for loading")
    
    # Check if model exists or can be found on HuggingFace
    if model_id not in llm_server.available_models:
        model_exists = await llm_server.check_and_add_model(model_id)
        if not model_exists:
            return {"status": "error", "message": f"Model {model_id} not found locally or on Hugging Face"}
    
    # Return success - actual loading happens via WebSocket
    return {"status": "loading", "message": f"Loading model {model_id} - connect to WebSocket for progress"}

@app.get("/models/active", response_model=Optional[ModelInfo])
async def get_active_model():
    """Get the active model"""
    return llm_server.get_active_model()

        
@app.websocket("/ws/models/{model_id:path}/loading")
async def websocket_model_loading(websocket: WebSocket, model_id: str):
    await websocket.accept()
        
    try:
        logger.info(f"WebSocket connection accepted for model {model_id}")
        
        # The model_id may be URL-encoded, so decode it
        from urllib.parse import unquote
        model_id = unquote(model_id)
        
        # Strip the "models/" prefix if it exists
        if model_id.startswith("models/"):
            model_id = model_id[7:]  # Remove "models/" prefix
            logger.info(f"Stripped 'models/' prefix from model ID for WebSocket")
        
        # Check if model exists or can be found on HuggingFace
        if model_id not in llm_server.available_models:
            model_exists = await llm_server.check_and_add_model(model_id)
            if not model_exists:
                await websocket.send_json({"status": "error", "message": f"Model {model_id} not found locally or on Hugging Face"})
                await asyncio.sleep(1.0)  # Give client time to process
                return
        
        # Check if model is already loaded - send a message if it is
        if llm_server.available_models[model_id]["is_loaded"]:
            logger.info(f"Model {model_id} already loaded")
            await websocket.send_json({
                "status": "success", 
                "message": f"Model {model_id} already loaded"
            })
            # Wait a moment before closing to ensure the client receives the message
            await asyncio.sleep(1.0)
            return
        
        # Send initial loading status
        await websocket.send_json({"status": "loading", "message": f"Starting to load model {model_id}..."})

        # await websocket.send_json({"status": "loading", "progress": 10, "message": "Checking model..."})
        # ... later
        # await websocket.send_json({"status": "loading", "progress": 50, "message": "Downloading model files..."})
        
        # Perform the actual model loading
        await clear_cuda_cache()

        result = await llm_server.load_model(model_id)
        
        # Send the final result from the load_model function
        await websocket.send_json(result)
        
        # Wait a moment before closing to ensure the client receives the message
        await asyncio.sleep(1.0)
        
    except Exception as e:
        logger.error(f"Error in model loading websocket: {e}")
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
            # Wait a moment before closing to ensure the client receives the message
            await asyncio.sleep(1.0)
        except:
            pass
    finally:
        logger.info(f"Closing WebSocket connection for model {model_id}")
        await websocket.close()

@app.get("/parameters", response_model=ModelParameters)
async def get_parameters():
    """Get the current model parameters"""
    return llm_server.get_parameters()


@app.post("/parameters", response_model=StatusResponse)
async def set_parameters(parameters: ModelParameters):
    """Set model parameters"""
    result = llm_server.set_parameters(parameters)
    return StatusResponse(status=result["status"], message=result["message"])


@app.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    meta_info: Optional[str] = None  # Changed from metadata to meta_info
):
    """Upload and process a document"""
    # Save uploaded file
    file_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    # Parse meta_info if provided
    meta_dict = {}
    if meta_info:
        try:
            meta_dict = json.loads(meta_info)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON")
            
    # Process document in background
    doc_id = await llm_server.process_document(file_path, file.filename, meta_dict)
    
    return {"status": "success", "document_id": doc_id, "message": "Document processed successfully"}


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents(filename: Optional[str] = None):
    """List all documents or filter by filename"""
    return llm_server.get_document_info(filename=filename)


@app.get("/documents/{doc_id}", response_model=DocumentContent)
async def get_document(doc_id: str):
    """Get document content by ID"""
    try:
        return llm_server.get_document_content(doc_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/documents/{doc_id}", response_model=StatusResponse)
async def delete_document(doc_id: str):
    """Delete a document by ID"""
    try:
        result = llm_server.delete_document(doc_id)
        return StatusResponse(status=result["status"], message=result["message"])
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/documents", response_model=StatusResponse)
async def delete_all_documents():
    """Delete all documents"""
    result = llm_server.delete_all_documents()
    return StatusResponse(status=result["status"], message=result["message"])

@app.get("/processed-documents", response_model=List[Dict])
async def get_processed_documents():
    """Get all processed documents"""
    with SessionLocal() as db:
        docs = db.query(ProcessedDocument).all()
        return [
            {
                "id": doc.id,
                "original_document_id": doc.original_document_id,
                "content": doc.content,
                "processing_type": doc.processing_type,
                "created_at": doc.created_at,
                "metadata": doc.meta_info
            }
            for doc in docs
        ]

@app.get("/processed-documents/{doc_id}", response_model=Dict)
async def get_processed_document(doc_id: str):
    """Get a processed document by ID"""
    with SessionLocal() as db:
        doc = db.query(ProcessedDocument).filter(ProcessedDocument.id == doc_id).first()
        if not doc:
            # Try to find it in local storage (client-side) - for compatible clients
            raise HTTPException(status_code=404, detail="Processed document not found")
            
        return {
            "id": doc.id,
            "original_document_id": doc.original_document_id,
            "content": doc.content,
            "processing_type": doc.processing_type,
            "created_at": doc.created_at,
            "metadata": doc.meta_info,
            "result": doc.content  # Add this for compatibility with client expectations
        }

@app.post("/documents/{document_id}/process")
async def process_document(document_id: str, request: ProcessDocumentRequest):
    """Process a document by chunks with the LLM"""
    logger.info(f"Processing document: {document_id} with request: {request}")
    
    try:
        # Get document content
        with SessionLocal() as db:
            doc = db.query(Document).filter(Document.id == document_id).first()
            
            if not doc:
                raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")
            
            if not doc.content:
                raise HTTPException(status_code=400, detail="Document has no content to process")
                
            # Check if embeddings exist and are loadable
            embedding_available = False
            if doc.embedding_path and os.path.exists(doc.embedding_path):
                try:
                    embedding = np.load(doc.embedding_path)
                    embedding_available = True
                except Exception as e:
                    logger.warning(f"Could not load embedding for document {document_id}: {e}")
        
        # Initialize document processor
        processor = DocumentProcessor(api_url="http://localhost:8000")
        await processor.initialize()
        
        try:
            
            # Add a progress callback function that sends WebSocket updates
            async def progress_update(data):
                if document_id in processing_websockets:
                    try:
                        await processing_websockets[document_id].send_json(data)
                    except Exception as ws_error:
                        logger.error(f"Error sending WebSocket update: {ws_error}")

            # Extract parameters from request
            # Handle both camelCase (frontend) and snake_case (backend) naming conventions
            # prompt_template = request.get('prompt_template') or request.get('promptTemplate', '')
            # combine_prompt = request.get('combine_prompt') or request.get('combinePrompt', '')
            # chunk_size = request.get('chunk_size') or request.get('chunkSize', 2000)
            # overlap = request.get('overlap', 200)
            # model_id = request.get('llm_model_id') or request.get('modelId', None)
            # max_concurrent = request.get('max_concurrent') or request.get('maxConcurrent', 3)
            # processing_type = request.get('processing_type') or request.get('processingType', 'summarization')
            # use_embeddings = request.get('use_embeddings', True)
            
            # Add embedding information to the processor if available
            if embedding_available and hasattr(processor, 'set_document_embedding'):
                processor.set_document_embedding(embedding)
            
            # Process the document
            start_time = time.time()

            logger.warning(f"chunk_size: {request.chunk_size}")

            result = await processor.process_document(
                document_text=doc.content,
                prompt_template=request.prompt_template,
                chunk_size=request.chunk_size or request.chunkSize or 2000,
                overlap=request.overlap,
                model_id=request.llm_model_id,
                max_concurrent=request.max_concurrent,
                combine_prompt=request.combine_prompt,
                use_embeddings=request.use_embeddings if hasattr(request, 'use_embeddings') else True,
                progress_callback=progress_update  # Pass the callback
            )
            processing_time = time.time() - start_time
            
            # Determine number of chunks processed
            chunks = processor.split_document(doc.content, request.chunk_size, request.overlap)
            
            # Store processed result
            processed_doc_id = str(uuid.uuid4())

            print(f"result1: {result[0]}")
            
            with SessionLocal() as db:
                processed_doc = ProcessedDocument(
                    id=processed_doc_id,
                    original_document_id=document_id,
                    content=result[0],
                    processing_type=request.processing_type,
                    meta_info={
                        "original_document_name": doc.filename,
                        "processing_type": request.processing_type,
                        "processing_date": datetime.utcnow().isoformat(),
                        "chunks_processed": len(chunks),
                        "processing_time_seconds": processing_time
                    }
                )
                db.add(processed_doc)
                db.commit()
            
            # Return response with consistent field names
            return {
                "status": "success",
                "result": result,  # Keep result field name consistent
                "processing_info": {  # Keep processing_info field name consistent
                    "document_id": doc.id,
                    "document_name": doc.filename,
                    "chunks_processed": len(chunks),
                    "processing_time_seconds": processing_time,
                    "processing_type": request.processing_type,
                    "model_id": request.llm_model_id or (llm_server.active_model["id"] if llm_server.active_model else None),
                    "processed_at": datetime.utcnow().isoformat()
                },
                "processed_document_id": processed_doc_id
            }
        except Exception as proc_error:
            logger.exception(f"Error in document processor: {proc_error}")
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(proc_error)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error processing document: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error processing document: {str(e)}")
    finally:
        if 'processor' in locals():
            await processor.close()

@app.post("/chat/sessions", response_model=Dict[str, str])
async def create_chat_session(model_id: Optional[str] = None):
    """Create a new chat session"""
    try:
        session_id = await llm_server.create_chat_session(model_id)
        return {"session_id": session_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/chat/sessions", response_model=List[ChatSessionInfo])
async def list_chat_sessions():
    """List all chat sessions"""
    return await llm_server.list_chat_sessions()


@app.get("/chat/sessions/{session_id}/messages", response_model=List[ChatMessageModel])
async def get_session_messages(session_id: str):
    """Get all messages in a chat session"""
    try:
        return await llm_server.get_session_messages(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/chat/sessions/{session_id}/messages", response_model=Dict[str, str])
async def chat_completion(session_id: str, message_data: Dict):
    role = message_data.get("role")
    content = message_data.get("content")
    document_ids = message_data.get("document_ids", [])
    processed_documents = message_data.get("processed_documents", [])
    
    if role != "user":
        raise HTTPException(status_code=400, detail="Only user messages can be sent")
    
    try:
        # Get session
        with SessionLocal() as db:
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            
            if not session:
                raise ValueError(f"Chat session with ID {session_id} not found")
                
            # Load the model if it's not the active one
            if TRANSFORMERS_AVAILABLE and (not llm_server.active_model or llm_server.active_model["id"] != session.model_name):
                await llm_server.load_model(session.model_name)
            
            # Get existing messages for context
            db_messages = db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).order_by(ChatMessage.created_at).all()
            
            history = [{"role": msg.role, "content": msg.content} for msg in db_messages]
        
        # Build document context
        document_context = ""
        
        # Add raw document context if document_ids are provided
        if document_ids:
            with SessionLocal() as db:
                docs = db.query(Document).filter(Document.id.in_(document_ids)).all()
                if docs:
                    document_context += "\n\nREFERENCE DOCUMENTS:\n"
                    for doc in docs:
                        document_context += f"\n{doc.filename}:\n{doc.content[:1000]}\n"
        
        # Add processed document context if provided
        if processed_documents:
            document_context += "\n\nPROCESSED KNOWLEDGE:\n"
            for doc in processed_documents:
                doc_title = doc.get("metadata", {}).get("original_document_name", f"Document {doc['id'][:8]}")
                document_context += f"\n{doc_title}:\n{doc['content']}\n"
        
        # Append document context to the user message if any context exists
        if document_context:
            message = f"{content}\n\n{document_context}"
        else:
            message = content
            
        # Add user message to session
        await llm_server.add_message_to_session(session_id, "user", message)
        
        # Generate and add assistant response
        if TRANSFORMERS_AVAILABLE:
            # Build prompt with context
            prompt = llm_server._build_chat_prompt(history, message)
            response = await llm_server.generate_response(prompt)
        else:
            # Dummy response for when transformers not available
            response = f"[Dummy response - Transformers library not available] Received: {message}"
        
        # Add assistant response to session
        await llm_server.add_message_to_session(session_id, "assistant", response)
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/sessions/{session_id}", response_model=StatusResponse)
async def delete_chat_session(session_id: str):
    """Delete a chat session"""
    try:
        result = await llm_server.delete_chat_session(session_id)
        return StatusResponse(status=result["status"], message=result["message"])
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/chat/sessions", response_model=StatusResponse)
async def delete_all_chat_sessions():
    """Delete all chat sessions"""
    result = await llm_server.delete_all_chat_sessions()
    return StatusResponse(status=result["status"], message=result["message"])


@app.post("/chat/generate", response_model=Dict[str, str])
async def generate_response(prompt: str = Body(..., embed=True)):
    
    logger.info(f"generate: {prompt}")

    """Generate a response without creating a session"""
    if not TRANSFORMERS_AVAILABLE:
        return {"response": "Transformers library not available - cannot generate responses"}
        
    if not llm_server.active_model:
        raise HTTPException(status_code=400, detail="No active model loaded")
        
    try:
        response = await llm_server.generate_response(prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/search", response_model=List[Dict])
async def search_documents(query: str = Body(..., embed=True), top_k: int = 3):
    """Search for documents similar to the query"""
    try:
        results = await llm_server.search_documents(query, top_k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vector-db/index-document/{document_id}")
async def index_document_in_vectordb(document_id: str, background_tasks: BackgroundTasks):
    """Index a document in the vector database"""
    if not CHROMA_AVAILABLE:
        raise HTTPException(status_code=400, detail="ChromaDB is not available. Please install it with 'pip install chromadb'")
    
    try:
        # Check if document exists
        with SessionLocal() as db:
            doc = db.query(Document).filter(Document.id == document_id).first()
            
            if not doc:
                raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")
            
            # Check if already indexed
            existing_vector_doc = db.query(VectorDocument).filter(
                VectorDocument.original_document_id == document_id
            ).first()
            
            if existing_vector_doc and existing_vector_doc.is_indexed:
                return {"status": "success", "message": f"Document {document_id} already indexed", "vector_document_id": existing_vector_doc.id}
                
            # Create vector document record if not exists
            if not existing_vector_doc:
                vector_doc_id = str(uuid.uuid4())
                vector_doc = VectorDocument(
                    id=vector_doc_id,
                    original_document_id=document_id,
                    is_indexed=False,
                    meta_info={
                        "original_filename": doc.filename,
                        "indexing_date": datetime.utcnow().isoformat()
                    }
                )
                db.add(vector_doc)
                db.commit()
            else:
                vector_doc_id = existing_vector_doc.id
        
        # Process document for vector indexing in background
        background_tasks.add_task(
            index_document_in_background,
            document_id,
            vector_doc_id
        )
        
        return {
            "status": "processing", 
            "message": f"Document {document_id} is being indexed in the background",
            "vector_document_id": vector_doc_id
        }
        
    except Exception as e:
        logger.error(f"Error indexing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def index_document_in_background(document_id: str, vector_doc_id: str):
    """Background task to index document in ChromaDB"""
    try:
        from document_processor import DocumentProcessor, ChromaVectorStore
        
        # Initialize processor and vector store
        processor = DocumentProcessor(api_url="http://localhost:8000", chroma_persist_dir=CHROMA_DIR)
        vector_store = ChromaVectorStore(persist_directory=CHROMA_DIR)
        
        # Get document content
        with SessionLocal() as db:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if not doc or not doc.content:
                logger.error(f"Document {document_id} not found or empty")
                return
                
            # Create semantic chunks
            if hasattr(processor, 'split_document_semantic'):
                chunks = processor.split_document_semantic(doc.content, chunk_size=2000)
            else:
                # Fallback to basic chunking
                basic_chunks = processor.split_document(doc.content, chunk_size=2000)
                chunks = [{"title": f"Chunk {i+1}", "text": chunk} for i, chunk in enumerate(basic_chunks)]
            
            # Store document and chunks in vector store
            vector_store.add_document(
                doc_id=vector_doc_id,
                content=doc.content,
                metadata={
                    "original_document_id": document_id,
                    "filename": doc.filename,
                    "created_at": doc.created_at.isoformat() if hasattr(doc.created_at, 'isoformat') else str(doc.created_at)
                }
            )
            
            # Store chunks
            vector_store.add_chunks(
                doc_id=vector_doc_id,
                chunks=chunks
            )
            
            # Update database record
            vector_doc = db.query(VectorDocument).filter(VectorDocument.id == vector_doc_id).first()
            if vector_doc:
                vector_doc.is_indexed = True
                vector_doc.chunk_count = len(chunks)
                vector_doc.meta_info = {
                    **vector_doc.meta_info,
                    "indexed_at": datetime.utcnow().isoformat(),
                    "chunk_count": len(chunks)
                }
                db.commit()
                
            logger.info(f"Successfully indexed document {document_id} in vector store with {len(chunks)} chunks")
            
    except Exception as e:
        logger.error(f"Error in background indexing of document {document_id}: {e}")
        # Update failure status in database
        try:
            with SessionLocal() as db:
                vector_doc = db.query(VectorDocument).filter(VectorDocument.id == vector_doc_id).first()
                if vector_doc:
                    vector_doc.meta_info = {
                        **vector_doc.meta_info,
                        "indexing_error": str(e),
                        "indexing_error_time": datetime.utcnow().isoformat()
                    }
                    db.commit()
        except Exception as db_error:
            logger.error(f"Error updating vector document status: {db_error}")


########## NEW ROUTES

@app.get("/vector-db/documents")
async def list_vector_documents():
    """List all documents in the vector database"""
    if not CHROMA_AVAILABLE:
        raise HTTPException(status_code=400, detail="ChromaDB is not available")
    
    try:
        with SessionLocal() as db:
            vector_docs = db.query(VectorDocument).all()
            result = []
            
            for vector_doc in vector_docs:
                # Get original document info
                doc = db.query(Document).filter(Document.id == vector_doc.original_document_id).first()
                
                result.append({
                    "id": vector_doc.id,
                    "original_document_id": vector_doc.original_document_id,
                    "filename": doc.filename if doc else "Unknown",
                    "chunk_count": vector_doc.chunk_count,
                    "is_indexed": vector_doc.is_indexed,
                    "created_at": vector_doc.created_at.isoformat(),
                    "meta_info": vector_doc.meta_info
                })
                
            return result
    except Exception as e:
        logger.error(f"Error listing vector documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector-db/query")
async def query_vector_db(rag_query: RAGQuery):
    """Query the vector database for similar chunks"""
    if not CHROMA_AVAILABLE:
        raise HTTPException(status_code=400, detail="ChromaDB is not available")
    
    try:
        # Initialize vector store
        from document_processor import ChromaVectorStore
        vector_store = ChromaVectorStore(persist_directory=CHROMA_DIR)
        
        results = []
        
        # If document IDs are provided, convert to vector doc IDs
        doc_ids = []
        if rag_query.document_ids:
            with SessionLocal() as db:
                for doc_id in rag_query.document_ids:
                    vector_doc = db.query(VectorDocument).filter(
                        VectorDocument.original_document_id == doc_id
                    ).first()
                    
                    if vector_doc and vector_doc.is_indexed:
                        doc_ids.append(vector_doc.id)
        
        # Query each document or all documents if none specified
        if doc_ids:
            for doc_id in doc_ids:
                doc_results = vector_store.search_similar_chunks(
                    query=rag_query.query,
                    doc_id=doc_id,
                    limit=rag_query.limit
                )
                results.extend(doc_results)
                
            # Sort results by score across all documents
            results.sort(key=lambda x: x.get("score", 0) if x.get("score") is not None else 0, reverse=True)
            
            # Limit to the top results
            results = results[:rag_query.limit]
        else:
            # Query all documents
            results = vector_store.search_similar_chunks(
                query=rag_query.query,
                limit=rag_query.limit
            )
            
        return {
            "query": rag_query.query,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error querying vector store: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/vector-db/documents/{vector_doc_id}")
async def delete_vector_document(vector_doc_id: str):
    """Delete a document from the vector database"""
    if not CHROMA_AVAILABLE:
        raise HTTPException(status_code=400, detail="ChromaDB is not available")
    
    try:
        # Check if document exists
        with SessionLocal() as db:
            vector_doc = db.query(VectorDocument).filter(VectorDocument.id == vector_doc_id).first()
            
            if not vector_doc:
                raise HTTPException(status_code=404, detail=f"Vector document with ID {vector_doc_id} not found")
            
            # Delete from ChromaDB
            from document_processor import ChromaVectorStore
            vector_store = ChromaVectorStore(persist_directory=CHROMA_DIR)
            vector_store.delete_document(vector_doc_id)
            
            # Delete from database
            db.delete(vector_doc)
            db.commit()
            
        return {"status": "success", "message": f"Vector document {vector_doc_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting vector document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/{document_id}/process-semantic")
async def process_document_semantic(
    document_id: str, 
    request: AdvancedProcessDocumentRequest,
    background_tasks: BackgroundTasks
):
    """Process a document using semantic chunking and vector embeddings"""
    logger.info(f"Processing document semantically: {document_id} with request: {request}")
    
    try:
        # Get document content
        with SessionLocal() as db:
            doc = db.query(Document).filter(Document.id == document_id).first()
            
            if not doc:
                raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")
            
            if not doc.content:
                raise HTTPException(status_code=400, detail="Document has no content to process")
                
        # Initialize document processor with Chroma if requested
        chroma_dir = CHROMA_DIR if request.store_in_vectordb and CHROMA_AVAILABLE else None
        processor = DocumentProcessor(api_url="http://localhost:8000", chroma_persist_dir=chroma_dir)
        await processor.initialize()
        
        try:
            # Register websocket for progress
            async def progress_update(data):
                if document_id in processing_websockets:
                    try:
                        await processing_websockets[document_id].send_json(data)
                    except Exception as ws_error:
                        logger.error(f"Error sending WebSocket update: {ws_error}")

            # Process the document with semantic chunking
            start_time = time.time()
            vector_doc_id = None
            
            # Handle storing in vector DB if requested
            if request.store_in_vectordb and CHROMA_AVAILABLE:
                # Create vector document record
                with SessionLocal() as db:
                    vector_doc_id = str(uuid.uuid4())
                    vector_doc = VectorDocument(
                        id=vector_doc_id,
                        original_document_id=document_id,
                        is_indexed=False,
                        meta_info={
                            "original_filename": doc.filename,
                            "processing_type": request.processing_type,
                            "indexing_date": datetime.utcnow().isoformat()
                        }
                    )
                    db.add(vector_doc)
                    db.commit()
                
            # Process with all the advanced options
            result, processed_chunks = await processor.process_document(
                document_text=doc.content,
                prompt_template=request.prompt_template,
                chunk_size=request.chunk_size,
                overlap=request.overlap,
                model_id=request.llm_model_id,
                max_concurrent=request.max_concurrent,
                combine_prompt=request.combine_prompt,
                use_embeddings=request.use_embeddings,
                progress_callback=progress_update,
                document_id=vector_doc_id if vector_doc_id else None,
                metadata={
                    "original_document_id": document_id,
                    "filename": doc.filename,
                    "processing_type": request.processing_type
                }
            )
            
            processing_time = time.time() - start_time
            
            # Store processed result
            processed_doc_id = str(uuid.uuid4())

            print(f"result2: {result[0]}")
            
            with SessionLocal() as db:
                processed_doc = ProcessedDocument(
                    id=processed_doc_id,
                    original_document_id=document_id,
                    content=result[0],
                    processing_type=request.processing_type,
                    meta_info={
                        "original_document_name": doc.filename,
                        "processing_type": request.processing_type,
                        "processing_date": datetime.utcnow().isoformat(),
                        "chunks_processed": len(processed_chunks),
                        "processing_time_seconds": processing_time,
                        "vector_document_id": vector_doc_id,
                        "use_embeddings": request.use_embeddings,
                        "use_hierarchical_chunking": request.use_hierarchical_chunking
                    }
                )
                db.add(processed_doc)
                
                # Update vector document if created
                if vector_doc_id:
                    vector_doc = db.query(VectorDocument).filter(VectorDocument.id == vector_doc_id).first()
                    if vector_doc:
                        vector_doc.is_indexed = True
                        vector_doc.chunk_count = len(processed_chunks)
                        vector_doc.meta_info = {
                            **vector_doc.meta_info,
                            "indexed_at": datetime.utcnow().isoformat(),
                            "chunk_count": len(processed_chunks),
                            "processing_time_seconds": processing_time
                        }
                        
                db.commit()
            
            # Return response
            return {
                "status": "success",
                "result": result,
                "processing_info": {
                    "document_id": doc.id,
                    "document_name": doc.filename,
                    "chunks_processed": len(processed_chunks),
                    "processing_time_seconds": processing_time,
                    "processing_type": request.processing_type,
                    "model_id": request.llm_model_id or (llm_server.active_model["id"] if llm_server.active_model else None),
                    "processed_at": datetime.utcnow().isoformat(),
                    "vector_document_id": vector_doc_id,
                    "semantic_chunking": request.use_embeddings or request.use_hierarchical_chunking
                },
                "processed_document_id": processed_doc_id
            }
        except Exception as proc_error:
            logger.exception(f"Error in document processor: {proc_error}")
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(proc_error)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error processing document: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error processing document: {str(e)}")
    finally:
        if 'processor' in locals():
            await processor.close()

# The issue appears to be duplicated WebSocket route definitions
# Here's the corrected section for app.py:

@app.post("/rag/chat")
async def rag_chat_completion(
    session_id: str = Body(...),
    message: str = Body(...),
    document_ids: List[str] = Body(default=[]),
    rag_query: Optional[str] = Body(default=None),
    use_vector_search: bool = Body(default=True)
):
    """RAG-enhanced chat completion that uses vector search for relevant context"""
    if use_vector_search and not CHROMA_AVAILABLE:
        return {
            "status": "warning",
            "message": "Vector search requested but ChromaDB not available, falling back to regular chat",
            "response": await llm_server.chat_completion(session_id, message, document_ids)
        }
    
    try:
        # Initialize vector store if using vector search
        relevant_chunks = []
        if use_vector_search and CHROMA_AVAILABLE and rag_query:
            from document_processor import ChromaVectorStore
            vector_store = ChromaVectorStore(persist_directory=CHROMA_DIR)
            
            # If no query specified, use the message as the query
            query = rag_query if rag_query else message
            
            # Get vector doc IDs from original doc IDs if provided
            vector_doc_ids = []
            if document_ids:
                with SessionLocal() as db:
                    for doc_id in document_ids:
                        vector_doc = db.query(VectorDocument).filter(
                            VectorDocument.original_document_id == doc_id
                        ).first()
                        
                        if vector_doc and vector_doc.is_indexed:
                            vector_doc_ids.append(vector_doc.id)
            
            # Perform vector search for each document or across all documents
            if vector_doc_ids:
                for vector_doc_id in vector_doc_ids:
                    doc_chunks = vector_store.search_similar_chunks(
                        query=query,
                        doc_id=vector_doc_id,
                        limit=3
                    )
                    relevant_chunks.extend(doc_chunks)
            else:
                # Search across all documents
                relevant_chunks = vector_store.search_similar_chunks(
                    query=query,
                    limit=5
                )
                
            # Sort by relevance and limit
            if relevant_chunks:
                relevant_chunks.sort(key=lambda x: x.get("score", 0) if x.get("score") is not None else 0, reverse=True)
                relevant_chunks = relevant_chunks[:5]  # Limit to top 5 chunks
        
        # Prepare enhanced context
        enhanced_message = message
        
        if relevant_chunks:
            # Format chunks for RAG context
            context = "\n\n".join([
                f"Document: {chunk.get('metadata', {}).get('heading', 'Context chunk')}\n{chunk['text']}"
                for chunk in relevant_chunks
            ])
            
            # Add context to the user message
            enhanced_message = f"{message}\n\nREFERENCE CONTEXT:\n{context}"
        
        # Process with regular chat completion
        response = await llm_server.chat_completion(session_id, enhanced_message, document_ids)
        
        return {
            "response": response,
            "rag_info": {
                "vector_search_used": use_vector_search and CHROMA_AVAILABLE,
                "context_chunks_used": len(relevant_chunks),
                "chunks": [
                    {"id": chunk.get("id"), "text_preview": chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"]}
                    for chunk in relevant_chunks
                ] if relevant_chunks else []
            }
        }
    except Exception as e:
        logger.error(f"Error in RAG chat completion: {e}")
        # Fallback to regular chat
        try:
            response = await llm_server.chat_completion(session_id, message, document_ids)
            return {
                "status": "warning",
                "message": f"Error in RAG processing: {str(e)}, fell back to regular chat",
                "response": response
            }
        except Exception as chat_error:
            raise HTTPException(status_code=500, detail=f"Error in chat completion: {str(chat_error)}")

# Define the websocket routes only ONCE each
@app.websocket("/ws/vector-processing/{document_id}")
async def websocket_vector_processing(websocket: WebSocket, document_id: str):
    """WebSocket for real-time updates during vector document processing"""
    await websocket.accept()
    
    try:
        # Store websocket for progress updates
        processing_websockets[document_id] = websocket
        
        # Send initial connection confirmation
        await websocket.send_json({
            "status": "connected", 
            "message": "Vector processing updates will appear here",
            "document_id": document_id
        })
        
        # Check processing status
        with SessionLocal() as db:
            vector_doc = db.query(VectorDocument).filter(
                VectorDocument.original_document_id == document_id
            ).first()
            
            if vector_doc:
                await websocket.send_json({
                    "status": "info",
                    "message": f"Document {document_id} vector status: {'Indexed' if vector_doc.is_indexed else 'Processing'}",
                    "vector_document_id": vector_doc.id,
                    "is_indexed": vector_doc.is_indexed,
                    "chunk_count": vector_doc.chunk_count
                })
        
        # Wait for processing to complete or timeout
        for _ in range(600):  # 10 minutes timeout
            if document_id not in processing_websockets:
                break
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        if document_id in processing_websockets:
            del processing_websockets[document_id]
    finally:
        # Cleanup
        if document_id in processing_websockets:
            del processing_websockets[document_id]

@app.websocket("/ws/rag-chat/{session_id}")
async def websocket_rag_chat(websocket: WebSocket, session_id: str):
    """WebSocket for real-time RAG chat processing"""
    await websocket.accept()
    
    chat_websockets = {}  # Dictionary to hold active chat websockets
    
    try:
        # Store websocket for updates
        chat_websockets[session_id] = websocket
        
        # Send initial connection confirmation
        await websocket.send_json({
            "status": "connected", 
            "message": "RAG chat connection established",
            "session_id": session_id
        })
        
        # Listen for messages
        while True:
            data = await websocket.receive_json()
            
            # Handle different message types
            if data.get("type") == "message":
                # Process a new chat message with RAG
                message = data.get("message", "")
                document_ids = data.get("document_ids", [])
                use_vector_search = data.get("use_vector_search", True)
                
                # Send acknowledgment
                await websocket.send_json({
                    "status": "processing",
                    "message": "Processing your message with RAG",
                    "received_message": message[:100] + "..." if len(message) > 100 else message
                })
                
                # Use the existing RAG chat endpoint for processing
                try:
                    # Check if using vector search
                    relevant_chunks = []
                    if use_vector_search and CHROMA_AVAILABLE:
                        # Send intermediate update
                        await websocket.send_json({
                            "status": "searching",
                            "message": "Searching for relevant document chunks"
                        })
                        
                        # Perform vector search similar to the rag_chat_completion function
                        from document_processor import ChromaVectorStore
                        vector_store = ChromaVectorStore(persist_directory=CHROMA_DIR)
                        
                        # Get vector doc IDs
                        vector_doc_ids = []
                        with SessionLocal() as db:
                            for doc_id in document_ids:
                                vector_doc = db.query(VectorDocument).filter(
                                    VectorDocument.original_document_id == doc_id
                                ).first()
                                
                                if vector_doc and vector_doc.is_indexed:
                                    vector_doc_ids.append(vector_doc.id)
                        
                        # Perform search
                        for vector_doc_id in vector_doc_ids:
                            doc_chunks = vector_store.search_similar_chunks(
                                query=message,
                                doc_id=vector_doc_id,
                                limit=3
                            )
                            relevant_chunks.extend(doc_chunks)
                            
                        # Sort by relevance
                        if relevant_chunks:
                            relevant_chunks.sort(key=lambda x: x.get("score", 0) if x.get("score") is not None else 0, reverse=True)
                            relevant_chunks = relevant_chunks[:5]
                            
                            # Send update with found chunks
                            await websocket.send_json({
                                "status": "chunks_found",
                                "message": f"Found {len(relevant_chunks)} relevant chunks",
                                "chunks": [
                                    {
                                        "id": chunk.get("id"),
                                        "text_preview": chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"],
                                        "score": chunk.get("score", 0)
                                    }
                                    for chunk in relevant_chunks
                                ]
                            })
                    
                    # Prepare enhanced message with context if available
                    enhanced_message = message
                    if relevant_chunks:
                        # Format chunks for RAG context
                        context = "\n\n".join([
                            f"Document: {chunk.get('metadata', {}).get('heading', 'Context chunk')}\n{chunk['text']}"
                            for chunk in relevant_chunks
                        ])
                        
                        # Add context to the user message
                        enhanced_message = f"{message}\n\nREFERENCE CONTEXT:\n{context}"
                    
                    # Send update that we're generating the response
                    await websocket.send_json({
                        "status": "generating",
                        "message": "Generating response with retrieved context"
                    })
                    
                    # Process with chat completion
                    response = await llm_server.chat_completion(session_id, enhanced_message, document_ids)
                    
                    # Send the response
                    await websocket.send_json({
                        "status": "response",
                        "message": "Response generated",
                        "response": response,
                        "rag_info": {
                            "vector_search_used": use_vector_search and CHROMA_AVAILABLE,
                            "context_chunks_used": len(relevant_chunks)
                        }
                    })
                    
                except Exception as e:
                    logger.error(f"Error in WebSocket RAG chat: {e}")
                    # Send error message
                    await websocket.send_json({
                        "status": "error",
                        "message": f"Error processing message: {str(e)}",
                        "error": str(e)
                    })
                    
                    # Try fallback to regular chat
                    try:
                        response = await llm_server.chat_completion(session_id, message, document_ids)
                        await websocket.send_json({
                            "status": "fallback_response",
                            "message": "Fell back to regular chat",
                            "response": response
                        })
                    except Exception as chat_error:
                        await websocket.send_json({
                            "status": "critical_error",
                            "message": f"Failed in fallback chat: {str(chat_error)}",
                            "error": str(chat_error)
                        })
            
            elif data.get("type") == "ping":
                # Simple ping to keep connection alive
                await websocket.send_json({"type": "pong", "time": time.time()})
                
    except WebSocketDisconnect:
        # Clean up
        if session_id in chat_websockets:
            del chat_websockets[session_id]
        logger.info(f"WebSocket RAG chat connection closed for session {session_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket RAG chat: {e}")
    finally:
        # Final cleanup
        if session_id in chat_websockets:
            del chat_websockets[session_id]


# Health check endpoint with CUDA information
@app.get("/health")
async def health_check():
    """Health check endpoint with detailed CUDA information"""
    import platform
    
    cuda_info = {}
    if torch.cuda.is_available():
        cuda_info = {
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB",
            "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB",
            "max_memory_allocated": f"{torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB"
        }
        
        # Add all devices
        cuda_info["devices"] = []
        for i in range(torch.cuda.device_count()):
            cuda_info["devices"].append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_allocated": f"{torch.cuda.memory_allocated(i) / 1024**2:.2f} MB",
                "memory_reserved": f"{torch.cuda.memory_reserved(i) / 1024**2:.2f} MB"
            })
    
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "active_model": llm_server.active_model["id"] if llm_server.active_model else None,
        "system_info": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_info": cuda_info,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "pdf_processing_available": PDF_PROCESSING_AVAILABLE
        }
    }


# CUDA memory management endpoint (for advanced users)
@app.post("/cuda/clear-cache")
async def clear_cuda_cache():
    """Clear CUDA memory cache"""
    if not torch.cuda.is_available():
        return {"status": "error", "message": "CUDA not available"}
        
    try:
        # Get stats before clearing
        before_allocated = torch.cuda.memory_allocated() / 1024**2
        before_reserved = torch.cuda.memory_reserved() / 1024**2
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Get stats after clearing
        after_allocated = torch.cuda.memory_allocated() / 1024**2
        after_reserved = torch.cuda.memory_reserved() / 1024**2
        
        return {
            "status": "success", 
            "message": "CUDA cache cleared",
            "memory_before": {
                "allocated_mb": f"{before_allocated:.2f}",
                "reserved_mb": f"{before_reserved:.2f}"
            },
            "memory_after": {
                "allocated_mb": f"{after_allocated:.2f}",
                "reserved_mb": f"{after_reserved:.2f}"
            },
            "difference": {
                "allocated_mb": f"{before_allocated - after_allocated:.2f}",
                "reserved_mb": f"{before_reserved - after_reserved:.2f}"
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to clear CUDA cache: {str(e)}"}


# Add GPU stats endpoint
@app.get("/gpu/stats")
async def get_gpu_stats():
    """Get detailed GPU statistics"""
    try:
        # Import local gpu_stats module if available
        try:
            from gpu_stats import get_combined_gpu_stats
            stats = get_combined_gpu_stats()
            return stats
        except ImportError:
            # Fallback to simplified stats if the module is not available
            if not torch.cuda.is_available():
                return {"error": "CUDA not available"}
                
            stats = {
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "current_device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
                "memory_allocated": f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB",
                "memory_reserved": f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB"
            }
            return stats
    except Exception as e:
        return {"error": f"Failed to get GPU stats: {str(e)}"}

@app.get("/debug/models")
async def debug_models():
    """Debug endpoint to check available models and their status"""
    return {
        "available_models": list(llm_server.available_models.keys()),
        "active_model": llm_server.active_model["id"] if llm_server.active_model else None,
        "cuda_available": torch.cuda.is_available() if 'torch' in sys.modules else False,
        "transformers_available": TRANSFORMERS_AVAILABLE
    }

@app.get("/debug/document/{document_id}")
async def debug_document(document_id: str):
    """Debug endpoint to check document content"""
    with SessionLocal() as db:
        doc = db.query(Document).filter(Document.id == document_id).first()
        
        if not doc:
            return {"error": f"Document with ID {document_id} not found"}
            
        return {
            "id": doc.id,
            "filename": doc.filename,
            "content_length": len(doc.content) if doc.content else 0,
            "meta_info": doc.meta_info,
            "has_content": doc.content is not None and len(doc.content) > 0,
            "content_preview": doc.content[:500] + "..." if doc.content and len(doc.content) > 500 else doc.content
        }

# Device mapping for model loading
@app.post("/models/{model_id:path}/load-with-mapping")
async def load_model_with_mapping(
    model_id: str, 
    device_map: Dict = Body(default={"device_map": "auto"})
):
    """Load a model with custom device mapping"""
    if not TRANSFORMERS_AVAILABLE:
        return {"status": "error", "message": "Transformers library not available"}
        
    if not torch.cuda.is_available():
        return {"status": "error", "message": "CUDA not available"}
    
    # Strip the "models/" prefix if it exists
    if model_id.startswith("models/"):
        model_id = model_id[7:]  # Remove "models/" prefix
        logger.info(f"Stripped 'models/' prefix from model ID for mapping")
        
    if model_id not in llm_server.available_models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
    try:
        # Clear CUDA cache before loading
        torch.cuda.empty_cache()
        
        with llm_server.model_lock:
            if llm_server.available_models[model_id]["is_loaded"]:
                logger.info(f"Model {model_id} already loaded, unloading first")
                # Reset model to unloaded state
                llm_server.available_models[model_id]["is_loaded"] = False
                if llm_server.active_model and llm_server.active_model["id"] == model_id:
                    llm_server.active_model = None
            
            model_info = llm_server.available_models[model_id]
            logger.info(f"Loading model {model_id} with custom device mapping")
            
            # Use the path directly
            model_path = model_info["path"]

            is_phi_model = "phi" in model_id.lower()
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
            
            # Load model with custom device mapping
            if is_phi_model:
                logger.info(f"Special handling for Phi model without device_map")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if llm_server.parameters.use_fp16 else None,
                    low_cpu_mem_usage=True
                )
                # Manually move to CUDA
                model = model.to("cuda")
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if llm_server.parameters.use_fp16 else None,
                    low_cpu_mem_usage=True,
                    device_map=device_map.get("device_map", "auto")
                )
            
            # Create pipeline
            active_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=llm_server.parameters.max_length,
                temperature=llm_server.parameters.temperature,
                top_p=llm_server.parameters.top_p,
                top_k=llm_server.parameters.top_k,
                repetition_penalty=llm_server.parameters.repetition_penalty,
                no_repeat_ngram_size=llm_server.parameters.no_repeat_ngram_size
            )
            
            # Update model info
            model_info["is_loaded"] = True
            model_info["tokenizer"] = tokenizer
            model_info["model"] = model
            model_info["pipeline"] = active_pipeline
            model_info["loaded_at"] = datetime.utcnow()
            model_info["cuda_enabled"] = torch.cuda.is_available()
            model_info["device_map"] = device_map.get("device_map", "auto")
            
            llm_server.active_model = model_info
            llm_server.active_pipeline = active_pipeline
            
        return {
            "status": "success", 
            "message": f"Model {model_id} loaded successfully with device mapping: {device_map.get('device_map', 'auto')}"
        }
    except Exception as e:
        # Try to clear CUDA cache on error
        try:
            torch.cuda.empty_cache()
        except:
            pass
        logger.error(f"Failed to load model {model_id} with device mapping: {e}")
        return {"status": "error", "message": f"Failed to load model: {str(e)}"}


# Statistics endpoints
@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    with SessionLocal() as db:
        document_count = db.query(Document).count()
        session_count = db.query(ChatSession).count()
        message_count = db.query(ChatMessage).count()
        
    return {
        "document_count": document_count,
        "session_count": session_count,
        "message_count": message_count,
        "available_models": len(llm_server.available_models),
        "loaded_models": sum(1 for model in llm_server.available_models.values() if model.get("is_loaded", False)),
        "cuda_enabled": torch.cuda.is_available()
    }

@app.websocket("/ws/documents/{document_id}/process")
async def websocket_document_processing(websocket: WebSocket, document_id: str):
    await websocket.accept()
    
    try:
        # Store websocket for progress updates
        processing_websockets[document_id] = websocket
        
        # Send initial connection confirmation
        await websocket.send_json({"status": "connected", "message": "Processing updates will appear here"})
        
        # Wait for processing to complete or timeout
        for _ in range(300):  # 5 minutes timeout (300 seconds)
            if document_id not in processing_websockets:
                break
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        if document_id in processing_websockets:
            del processing_websockets[document_id]
    finally:
        # Cleanup
        if document_id in processing_websockets:
            del processing_websockets[document_id]

# Main entry point
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)