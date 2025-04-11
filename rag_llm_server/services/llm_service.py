# services/llm_service.py
import time
import logging
import asyncio
import gc
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers import BitsAndBytesConfig # For quantization

from core.config import settings

from .llm_backends import (
    LLMBackendBase,
    LocalTransformersBackend,
    OllamaBackend,
    VLLMBackend,
    InstructLabBackend
)

DEFAULT_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- LLM Status Enum ---
class LLMStatus(Enum):
    INACTIVE = "Inactive" # No backend configured/active
    LOADING = "Loading"   # Only for local backend
    CONFIGURING = "Configuring" # For API backends (setting model)
    READY = "Ready"       # Backend is configured and ready
    FAILED = "Failed"     # Loading/Configuration failed
    UNLOADING = "Unloading" # Only for local backend


# --- LLM State ---
llm_state: Dict[str, Any] = {
    "backend_instance": None, # Holds the active backend object (Local, Ollama, etc.)
    "backend_type": None,     # Which type is active ('local', 'ollama', ...)
    "active_model": None,     # Name/path of the model active on the backend
    "status": LLMStatus.INACTIVE,
    "config": { # Global generation config, passed to backend
        "max_new_tokens": settings.DEFAULT_LLM_MAX_NEW_TOKENS,
        "temperature": settings.DEFAULT_LLM_TEMPERATURE,
        "top_p": settings.DEFAULT_LLM_TOP_P,
        "top_k": settings.DEFAULT_LLM_TOP_K,
        "repetition_penalty": 1.15 # Keep this from previous fix
    },
    "last_error": None,
}

# --- Helper Function to Scan Local Models ---
def list_local_models() -> List[str]:
    """Scans the DEDICATED LOCAL_MODELS_DIR for potential model directories."""
    local_models = []
    models_dir = settings.LOCAL_MODELS_DIR
    logger.info(f"Scanning for local models in: {models_dir}")
    if models_dir.is_dir():
        for item in models_dir.iterdir():
            if item.is_dir():
                has_config = (item / "config.json").is_file()
                has_pytorch_bin = list(item.glob("*.bin"))
                has_safetensors = list(item.glob("*.safetensors"))
                if has_config and (has_pytorch_bin or has_safetensors):
                    logger.debug(f"Found potential local model directory: {item.name}")
                    local_models.append(str(item.resolve()))
    logger.info(f"Found {len(local_models)} potential models in {models_dir}.")
    return local_models

def list_cached_hub_models() -> List[str]:
    """
    Scans the configured Hugging Face Hub cache directory AND the default
    cache directory for downloaded models.
    """
    hub_models: Set[str] = set() # Use a set to automatically handle duplicates
    # Directories to scan
    scan_dirs = {settings.HUGGINGFACE_HUB_CACHE.resolve(), DEFAULT_HF_CACHE.resolve()}

    for cache_dir in scan_dirs:
        logger.info(f"Scanning for cached Hub models in: {cache_dir}")
        if not cache_dir.is_dir():
            logger.warning(f"Cache directory not found or not a directory: {cache_dir}")
            continue

        for item in cache_dir.iterdir():
            if item.is_dir() and item.name.startswith("models--"):
                snapshots_dir = item / "snapshots"
                if snapshots_dir.is_dir() and any(snapshots_dir.iterdir()):
                    model_id = item.name.removeprefix("models--").replace("--", "/")
                    logger.debug(f"Found potential cached Hub model: {model_id} (from {item.name}) in {cache_dir}")
                    hub_models.add(model_id) # Add to set

    logger.info(f"Found {len(hub_models)} unique potential cached models across specified cache directories.")
    return sorted(list(hub_models)) # Return sorted list

# ... (_load_model_task, get_llm_status, update_llm_config, generate_text) ...
# NOTE: The _load_model_task should work fine with these Hub IDs without modification

# --- Internal Unload Function ---
async def _unload_current_backend():
    """Calls unload on the current backend instance if it exists."""
    if llm_state["backend_instance"] is not None:
        logger.info(f"Unloading current backend ({llm_state['backend_type']})...")
        try:
            await llm_state["backend_instance"].unload()
        except Exception as e:
             logger.error(f"Error during backend unload: {e}", exc_info=True)
        finally:
            llm_state["backend_instance"] = None
            llm_state["backend_type"] = None
            llm_state["active_model"] = None
            llm_state["status"] = LLMStatus.INACTIVE
            llm_state["last_error"] = None

# --- Core Loading Function (Runs in Background) ---
# This now specifically targets the LocalTransformersBackend instance

async def _load_local_model_task(
    backend: LocalTransformersBackend,
    model_id: str,
    device: str,
    quantization: Optional[str]
):
    """Loads a model and tokenizer into the LocalTransformersBackend instance."""
    task_start_time = time.time()
    logger.info(f"[Local Loader] Starting task for '{model_id}' on device '{device}' q: '{quantization}'...")

    llm_state["status"] = LLMStatus.LOADING
    llm_state["last_error"] = None
    backend.model_name_or_path = model_id

    try:
        load_start_time = time.time()
        quantization_config = None
        # Determine device map strategy for AutoModel
        # Use 'auto' for multi-gpu or quantization, explicit device for single GPU full precision
        device_map = "auto" if quantization or torch.cuda.device_count() > 1 else device
        if device == "cpu":
             device_map = None # Don't use device_map for CPU

        logger.info(f"[Local Loader] Effective device_map strategy: '{device_map}'")

        if quantization:
            # ... (Quantization config logic - same as before) ...
              if quantization == "8bit" and torch.cuda.is_available():
                 quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                 logger.info("[Local Loader] Applying 8-bit quantization.")
              elif quantization == "4bit" and torch.cuda.is_available():
                  quantization_config = BitsAndBytesConfig(
                     load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                     bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                 )
                  logger.info("[Local Loader] Applying 4-bit quantization (nf4).")
              else:
                 logger.warning(f"[Local Loader] Quantization '{quantization}' not supported/applicable. Loading full precision.")
                 quantization = None

        # --- Load Tokenizer ---
        logger.info(f"[Local Loader] Loading tokenizer for '{model_id}'...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True,
            cache_dir=str(settings.HUGGINGFACE_HUB_CACHE.resolve())
        )
        logger.info("[Local Loader] Tokenizer loaded.")

        # --- Load Model ---
        logger.info(f"[Local Loader] Loading model '{model_id}'...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map, # Use determined device_map
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if device == "cuda" and not quantization else None, # Use float16 on GPU if not quantizing to bitsandbytes native types
            trust_remote_code=True,
            cache_dir=str(settings.HUGGINGFACE_HUB_CACHE.resolve())
        )
        logger.info("[Local Loader] Model loaded successfully.")

        load_time = time.time() - load_start_time
        logger.info(f"[Local Loader] Successfully loaded model '{model_id}' in {load_time:.2f}s.")

        max_len = getattr(model.config, "max_position_embeddings", None) \
                  or getattr(tokenizer, "model_max_length", None) \
                  or 1024 # Default fallback
        logger.info(f"[Local Loader] Determined model max length: {max_len}")

        # --- Update the specific backend instance ---
        backend.model = model # Store model
        backend.tokenizer = tokenizer
        backend.model_name_or_path = model_id
        backend.load_config = {"device": device, "quantization": quantization}
        backend.max_model_length = max_len

        # --- Update global state ---
        llm_state["status"] = LLMStatus.READY
        llm_state["active_model"] = model_id
        llm_state["last_error"] = None

    except Exception as e:
        # ... (Error handling - clear backend.model, backend.tokenizer) ...
        error_message = f"[Local Loader] Failed to load model '{model_id}': {type(e).__name__}: {e}"
        logger.error(error_message, exc_info=True)
        backend.model = None # Ensure refs are cleared on error
        backend.tokenizer = None
        llm_state["status"] = LLMStatus.FAILED
        llm_state["last_error"] = error_message
        llm_state["active_model"] = None
    finally:
        task_duration = time.time() - task_start_time
        logger.info(f"[Local Loader] Task for '{model_id}' finished in {task_duration:.2f}s. Status: {llm_state['status'].value}")

# --- Public Service Functions ---
def get_llm_status() -> Dict[str, Any]:
    """Returns the current status and configuration of the active LLM backend."""
    # status_copy = {
    #     "backend_type": llm_state["backend_type"],
    #     "active_model": llm_state["active_model"],
    #     "status": llm_state["status"].value,
    #     "generation_config": llm_state["config"],
    #     "last_error": llm_state["last_error"],
    #     "backend_details": None
    # }
    # if llm_state["backend_instance"] is not None:
    #     # Get specific details from the active backend
    #     status_copy["backend_details"] = llm_state["backend_instance"].get_status_dict()

    # return status_copy

    """Returns the current status and configuration of the active LLM backend."""
    backend_instance = llm_state.get("backend_instance") # Get instance safely
    backend_details = None
    if backend_instance is not None:
        try:
            # Call get_status_dict only if instance exists
            backend_details = backend_instance.get_status_dict()
        except Exception as e:
            # Catch potential errors during get_status_dict itself
            logger.error(f"Error getting backend details from instance: {e}", exc_info=True)
            backend_details = {"error": "Failed to retrieve backend details"}

    status_copy = {
        "backend_type": llm_state.get("backend_type"), # Use .get for safety
        "active_model": llm_state.get("active_model"),
        "status": llm_state.get("status", LLMStatus.INACTIVE).value, # Default to INACTIVE if status key somehow missing
        "generation_config": llm_state.get("config"),
        "last_error": llm_state.get("last_error"),
        "backend_details": backend_details # Use the safely retrieved details
    }
    return status_copy

def update_llm_config(new_config: Dict[str, Any]) -> Dict[str, Any]:
    """Updates the global generation configuration."""
    # Config applies globally, passed to whichever backend is active
    allowed_keys = {"max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty"}
    updated = False
    for key, value in new_config.items():
        if key in allowed_keys:
            if key in llm_state["config"] and llm_state["config"][key] != value:
                 logger.info(f"Updating global LLM generation config: {key} = {value}")
                 llm_state["config"][key] = value
                 updated = True
        else:
             logger.warning(f"Ignoring unsupported config key: {key}")
    if not updated:
         logger.info("No changes applied to LLM config.")
    return llm_state["config"]



# Now delegates to the active backend, running in executor if needed
async def generate_text(prompt: str) -> Optional[str]:
    """Generates text using the currently active LLM backend."""
    backend = llm_state.get("backend_instance")
    current_status = llm_state.get("status")

    if current_status != LLMStatus.READY or backend is None:
        logger.error(f"Cannot generate text: LLM backend not ready (Status: {current_status.value if current_status else 'N/A'}).")
        return None

    config = llm_state["config"] # Get current global config

    try:
        # Local backend's generate is blocking, API backends' generate is async
        if isinstance(backend, LocalTransformersBackend):
            logger.info("Running local generation in executor...")
            loop = asyncio.get_running_loop()
            # LocalBackend.generate is sync, run it in executor
            result = await loop.run_in_executor(None, backend.generate, prompt, config) # Pass config
        elif isinstance(backend, (OllamaBackend, VLLMBackend, InstructLabBackend)):
            logger.info(f"Running API generation via {llm_state['backend_type']} backend...")
            # API backend generate methods are already async
            result = await backend.generate(prompt, config) # Pass config
        else:
             logger.error(f"Unknown backend type {type(backend)} cannot generate.")
             result = None

        return result

    except Exception as e:
        logger.error(f"Error during call to backend generate method: {e}", exc_info=True)
        return None

# --- Function to initiate backend loading/configuration ---
# This replaces the direct call logic in the API endpoint
async def set_active_backend(
    backend_type: str,
    model_name_or_path: str,
    device: Optional[str] = None, # Primarily for local
    quantization: Optional[str] = None # Primarily for local
):
    """Sets the active LLM backend, unloading previous one if necessary."""
    logger.info(f"Request to set active backend: type='{backend_type}', model='{model_name_or_path}'")

    # 1. Unload existing backend first
    await _unload_current_backend()

    # 2. Create and configure the new backend instance
    new_backend: Optional[LLMBackendBase] = None
    llm_state["status"] = LLMStatus.CONFIGURING # Or LOADING for local
    llm_state["backend_type"] = backend_type
    llm_state["active_model"] = model_name_or_path # Tentative
    llm_state["last_error"] = None

    try:
        if backend_type == "local":
            llm_state["status"] = LLMStatus.LOADING # Specific status for local
            local_backend_instance = LocalTransformersBackend()
            llm_state["backend_instance"] = local_backend_instance # Store instance before background task

            # Run the actual loading in background executor
            loop = asyncio.get_running_loop()
            final_device = device or settings.LLM_DEVICE
            final_quant = quantization # Uses request value directly, None if not provided
            if quantization is None and settings.DEFAULT_LLM_QUANTIZATION is not None:
                 final_quant = settings.DEFAULT_LLM_QUANTIZATION # Apply default if needed

            logger.info(f"Scheduling local model load task: model='{model_name_or_path}', device='{final_device}', quantization='{final_quant}'")
            # Use wrapper to run the async task in executor
            def local_load_wrapper():
                 asyncio.run(_load_local_model_task(local_backend_instance, model_name_or_path, final_device, final_quant))
            loop.run_in_executor(None, local_load_wrapper)
            # Status will be updated to READY/FAILED by the background task
            return # Return early, loading happens in background

        elif backend_type == "ollama":
            if not settings.OLLAMA_BASE_URL:
                raise ValueError("OLLAMA_BASE_URL not configured.")
            new_backend = OllamaBackend(settings.OLLAMA_BASE_URL, model_name_or_path)
            # TODO: Optionally ping Ollama API here to verify model exists?

        elif backend_type == "vllm":
             if not settings.VLLM_BASE_URL:
                 raise ValueError("VLLM_BASE_URL not configured.")
             new_backend = VLLMBackend(settings.VLLM_BASE_URL, model_name_or_path)
             # TODO: Ping vLLM?

        elif backend_type == "instructlab":
            if not settings.INSTRUCTLAB_BASE_URL:
                raise ValueError("INSTRUCTLAB_BASE_URL not configured.")
            new_backend = InstructLabBackend(settings.INSTRUCTLAB_BASE_URL, model_name_or_path)
            # TODO: Ping InstructLab?

        else:
            raise ValueError(f"Unsupported LLM_BACKEND_TYPE: {backend_type}")

        # If we reached here, it's an API backend
        llm_state["backend_instance"] = new_backend
        llm_state["active_model"] = model_name_or_path # Confirm model
        llm_state["status"] = LLMStatus.READY # API backends are 'ready' once configured
        logger.info(f"Backend '{backend_type}' configured and ready with model '{model_name_or_path}'.")

    except Exception as e:
        error_message = f"Failed to configure backend '{backend_type}' with model '{model_name_or_path}': {e}"
        logger.error(error_message, exc_info=True)
        llm_state["status"] = LLMStatus.FAILED
        llm_state["last_error"] = error_message
        llm_state["backend_instance"] = None
        llm_state["backend_type"] = None
        llm_state["active_model"] = None
        # Re-raise? Or just let status reflect failure? Let status reflect.
