# app/api/models/llm.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from services.llm_service import LLMStatus # Import Enum

class ModelLoadRequest(BaseModel):
    # --- NEW: Optional backend type selection ---
    backend_type: Optional[Literal["local", "ollama", "vllm", "instructlab"]] = Field(
        default=None,
        description="Specify the backend type to use. If None, uses the server's configured default."
    )
    # --- model_name_or_path is now optional ---
    model_name_or_path: Optional[str] = Field(
        default=None, # Default to None, will use server default if not provided
        description="Identifier for the model (HF ID, path for 'local'; model name for API backends). If None, uses server's default model."
    )
    # Device/Quantization remain optional, only relevant for 'local'
    device: Optional[str] = Field(default=None, description="Local Only: Device ('cuda', 'cpu'). Overrides server default if backend is 'local'.") # Corrected Line 18
    quantization: Optional[str] = Field(default=None, description="Local Only: Quantization ('8bit', '4bit'). Overrides server default if backend is 'local'.") # Corrected Line 19

class BackendDetails(BaseModel):
    # Example structure, might vary based on backend.get_status_dict() output
    # For local:
    active_model: Optional[str] = None
    load_config: Optional[Dict[str, Any]] = None
    max_model_length: Optional[int] = None
    # For API:
    backend_url: Optional[str] = None
    # active_model is already top-level

    class Config:
        extra = 'allow' # Allow extra fields from backend status dicts


class ModelStatusResponse(BaseModel):
    backend_type: Optional[str] = None
    active_model: Optional[str] = None # Model name active on the backend
    status: str # Use the string value (e.g., "Ready", "Loading")
    last_error: Optional[str] = None
    generation_config: Optional[Dict[str, Any]] = None
    backend_details: Optional[BackendDetails] = None # Specifics from backend

class ModelUpdateRequest(BaseModel):
    # This remains the same, updates the global config
    max_new_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    repetition_penalty: Optional[float] = Field(default=None, ge=1.0)

    def get_updates(self) -> Dict[str, Any]:
        return self.dict(exclude_unset=True, exclude_none=True)


class AvailableModelsResponse(BaseModel):
    # Keep this as before
    local_model_paths: List[str] = Field(description="Absolute paths to models found in the dedicated local_models directory.")
    cached_hub_models: List[str] = Field(description="Model IDs found in the Hugging Face Hub cache directory.")