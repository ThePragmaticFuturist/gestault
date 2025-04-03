# Step 10: Optional External LLM Server Integration #

Okay, integrating external LLM inference servers like Ollama and vLLM alongside local loading requires adding a layer of abstraction. We need to configure which backend to use and then have our service call the appropriate interaction logic.

Here's a plan:

1.  **Configuration:** Add settings to choose the backend type (`local`, `ollama`, `vllm`) and specify base URLs for API backends.
2.  **Abstraction Layer:** Define a base class or common structure for different LLM backends.
3.  **Backend Implementations:** Create classes for `LocalTransformersBackend`, `OllamaBackend`, `VLLMBackend` (and potentially `InstructLabBackend` later).
4.  **Refactor `llm_service.py`:** Modify it to manage the *active* backend instance based on configuration, rather than just the local pipeline. The `generate_text` function will delegate to the active backend.
5.  **Update `load_model` Logic:** The `/load` endpoint will now configure the *model* for the *currently active backend type*. For API backends, this might just mean setting the model name to use in API calls. For local, it triggers the actual loading.
6.  **Install `httpx`:** For making asynchronous HTTP requests to Ollama/vLLM.

Let's implement this step-by-step.

**Step 1: Install `httpx`**

```bash
pip install httpx
```

**Step 2: Update Configuration (`core/config.py`)**

Add settings for backend selection and URLs.

```python
# core/config.py
import os
import torch
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
import tempfile
from typing import Optional, List, Any, Literal # Import Literal

# ... (get_default_device function) ...

class Settings(BaseSettings):
    # ... (General, DB, Embedding settings) ...

    # --- LLM Settings ---
    LLM_BACKEND_TYPE: Literal["local", "ollama", "vllm", "instructlab"] = Field(
        default="local",
        description="The type of LLM backend to use ('local', 'ollama', 'vllm', 'instructlab')."
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
    # Local specific settings
    LOCAL_MODELS_DIR: Path = BASE_DIR / "local_models"
    LLM_DEVICE: str = Field(default_factory=get_default_device)
    DEFAULT_LLM_QUANTIZATION: Optional[str] = Field(default=None)

    # Generation parameters (used by all backends, potentially mapped)
    DEFAULT_LLM_MAX_NEW_TOKENS: int = Field(default=512)
    DEFAULT_LLM_TEMPERATURE: float = Field(default=0.7)
    DEFAULT_LLM_TOP_P: float = Field(default=0.9)
    DEFAULT_LLM_TOP_K: int = Field(default=50) # Note: Not used by standard OpenAI Chat API

    # ... (ChromaDB, Document Processing, RAG/Chat settings) ...

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore'
        # Allow reading Literal values case-insensitively if needed
        # case_sensitive = False

settings = Settings()

# ... (Ensure directories exist) ...

# --- Print relevant config ---
print(f"--- Configuration Loaded ---")
# ... (existing prints) ...
print(f"LLM Backend Type: {settings.LLM_BACKEND_TYPE}") # Print backend type
print(f"Default LLM Identifier: {settings.DEFAULT_MODEL_NAME_OR_PATH}")
if settings.LLM_BACKEND_TYPE == 'ollama':
    print(f"Ollama URL: {settings.OLLAMA_BASE_URL}")
if settings.LLM_BACKEND_TYPE == 'vllm':
    print(f"vLLM URL: {settings.VLLM_BASE_URL}")
print(f"--------------------------")

```
*   Added `LLM_BACKEND_TYPE` using `Literal` for validation.
*   Added base URLs for Ollama, vLLM, InstructLab.
*   Renamed `DEFAULT_MODEL_NAME` to `DEFAULT_MODEL_NAME_OR_PATH` for clarity.

**Step 3: Create Abstraction & Backends (`services/llm_backends.py`)**

Create a new file `services/llm_backends.py`.

```python
# services/llm_backends.py
import logging
import abc # Abstract Base Classes
from typing import Dict, Any, Optional, AsyncGenerator

import httpx # For async API calls
from core.config import settings

logger = logging.getLogger(__name__)

# --- Base Class ---
class LLMBackendBase(abc.ABC):
    """Abstract base class for different LLM backends."""

    @abc.abstractmethod
    async def generate(self, prompt: str, config: Dict[str, Any]) -> Optional[str]:
        """Generates text based on the prompt and configuration."""
        pass

    @abc.abstractmethod
    def get_status_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representing the backend's current status."""
        pass

    async def unload(self):
        """Optional method to clean up resources (e.g., unload local model)."""
        logger.info(f"Unload called for {self.__class__.__name__}, default implementation does nothing.")
        pass # Default does nothing


# --- Local Transformers Backend ---
# We'll integrate the existing logic from llm_service into this class later
class LocalTransformersBackend(LLMBackendBase):
    def __init__(self):
        # This backend will hold the pipeline, tokenizer, etc.
        # These will be populated by a dedicated loading function later
        self.pipeline = None
        self.tokenizer = None
        self.model_name_or_path: Optional[str] = None
        self.load_config: Dict[str, Any] = {}
        self.max_model_length: Optional[int] = None
        logger.info("LocalTransformersBackend initialized (inactive).")

    async def generate(self, prompt: str, config: Dict[str, Any]) -> Optional[str]:
        # This will contain the refined pipeline call logic from generate_text
        if self.pipeline is None or self.tokenizer is None:
            logger.error("Local pipeline/tokenizer not loaded.")
            return None

        try:
            logger.info("Generating text with local Transformers pipeline...")
            generation_keys = {"max_new_tokens", "temperature", "top_p", "top_k"}
            generation_kwargs = {k: config[k] for k in generation_keys if k in config}
            generation_kwargs["do_sample"] = True
            generation_kwargs["num_return_sequences"] = 1
            generation_kwargs["repetition_penalty"] = config.get("repetition_penalty", 1.15) # Example: get from config

            logger.debug(f"Passing generation kwargs to pipeline: {generation_kwargs}")

            # NOTE: Pipeline call itself is blocking CPU/GPU
            # Needs to be run in executor by the caller (llm_service.generate_text)
            outputs = self.pipeline(prompt, **generation_kwargs)

            full_generated_text = outputs[0]['generated_text']
            logger.debug(f"LLM Raw Output:\n{full_generated_text}")

            # Use the prompt structure marker defined in sessions.py
            response_marker = "\n### RESPONSE:" # Ensure this matches sessions.py
            prompt_marker_pos = prompt.rfind(response_marker)

            if prompt_marker_pos != -1:
                 response_start_pos = prompt_marker_pos + len(response_marker)
                 if full_generated_text.startswith(prompt[:response_start_pos]):
                      result = full_generated_text[response_start_pos:].strip()
                 else:
                      logger.warning("Generated text didn't start with the expected prompt prefix. Using full generated text.")
                      result = full_generated_text.strip()
            else:
                 logger.warning("Response marker not found in original prompt. Using full generated text.")
                 result = full_generated_text.strip()

            if not result:
                 logger.warning("Extraction resulted in an empty string.")

            logger.info("Local LLM text generation complete.")
            return result

        except Exception as e:
            logger.error(f"Local LLM text generation failed: {e}", exc_info=True)
            return None

    def get_status_dict(self) -> Dict[str, Any]:
         return {
             "active_model": self.model_name_or_path,
             "load_config": self.load_config,
             "max_model_length": self.max_model_length,
             # Add any other relevant local status info
         }

    async def unload(self):
        """Unloads the local model and clears memory."""
        logger.info(f"Unloading local model '{self.model_name_or_path}'...")
        # Keep references to delete them explicitly
        pipeline_to_del = self.pipeline
        model_to_del = getattr(pipeline_to_del, 'model', None) if pipeline_to_del else None
        tokenizer_to_del = self.tokenizer # Use self.tokenizer

        self.pipeline = None
        self.tokenizer = None
        self.model_name_or_path = None
        self.load_config = {}
        self.max_model_length = None

        del pipeline_to_del
        del model_to_del
        del tokenizer_to_del

        import gc, torch # Import locally for unload
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Local model unloaded and memory cleared.")


# --- Ollama Backend ---
class OllamaBackend(LLMBackendBase):
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/generate"
        self.model_name = model_name
        # Use a shared client for potential connection pooling
        self.client = httpx.AsyncClient(timeout=120.0) # Increase timeout
        logger.info(f"OllamaBackend initialized: URL='{self.base_url}', Model='{self.model_name}'")

    async def generate(self, prompt: str, config: Dict[str, Any]) -> Optional[str]:
        logger.info(f"Generating text via Ollama backend (model: {self.model_name})...")
        # Map standard config keys to Ollama options
        options = {
            "temperature": config.get("temperature", settings.DEFAULT_LLM_TEMPERATURE),
            "top_k": config.get("top_k", settings.DEFAULT_LLM_TOP_K),
            "top_p": config.get("top_p", settings.DEFAULT_LLM_TOP_P),
            "num_predict": config.get("max_new_tokens", settings.DEFAULT_LLM_MAX_NEW_TOKENS),
            # Add other Ollama options if needed (e.g., stop sequences)
            # "stop": ["\n###"]
        }
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False, # Request non-streaming response
            "options": options,
            # "raw": True # If prompt includes special formatting Ollama might remove
        }
        logger.debug(f"Ollama Request Payload: {payload}")

        try:
            response = await self.client.post(self.api_url, json=payload)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            result_data = response.json()
            generated_text = result_data.get("response")
            if generated_text:
                logger.info("Ollama generation successful.")
                logger.debug(f"Ollama Response: {generated_text[:500]}...")
                return generated_text.strip()
            else:
                logger.error(f"Ollama response missing 'response' field. Full response: {result_data}")
                return None
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API request failed (HTTP {e.response.status_code}): {e.response.text}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Ollama API request failed (Network/Connection Error): {e}")
            return None
        except Exception as e:
            logger.error(f"Ollama generation failed unexpectedly: {e}", exc_info=True)
            return None

    def get_status_dict(self) -> Dict[str, Any]:
        return {
            "active_model": self.model_name,
            "backend_url": self.base_url,
            # Add check if Ollama server is reachable? Maybe later.
        }

    async def unload(self):
        # Close the httpx client when unloading
        await self.client.aclose()
        logger.info("OllamaBackend httpx client closed.")

# --- vLLM Backend (OpenAI Compatible) ---
class VLLMBackend(LLMBackendBase):
    def __init__(self, base_url: str, model_name: str):
        # vLLM's OpenAI endpoint is usually at /v1
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/v1/chat/completions" # Use Chat Completion endpoint
        self.model_name = model_name # This is passed in the payload
        self.client = httpx.AsyncClient(timeout=120.0)
        logger.info(f"VLLMBackend initialized: URL='{self.base_url}', Model='{self.model_name}' (using Chat API)")

    async def generate(self, prompt: str, config: Dict[str, Any]) -> Optional[str]:
        logger.info(f"Generating text via vLLM backend (model: {self.model_name})...")
        # Format prompt for Chat Completion API
        messages = [{"role": "user", "content": prompt}]

        # Map config keys to OpenAI parameters
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": config.get("temperature", settings.DEFAULT_LLM_TEMPERATURE),
            "top_p": config.get("top_p", settings.DEFAULT_LLM_TOP_P),
            "max_tokens": config.get("max_new_tokens", settings.DEFAULT_LLM_MAX_NEW_TOKENS),
            "stream": False,
            # top_k is not standard for OpenAI Chat API
            # Add other supported params if needed (e.g., stop)
        }
        logger.debug(f"vLLM Request Payload: {payload}")

        try:
            response = await self.client.post(self.api_url, json=payload)
            response.raise_for_status()

            result_data = response.json()
            # Extract content from the first choice's message
            if result_data.get("choices") and len(result_data["choices"]) > 0:
                 message = result_data["choices"][0].get("message")
                 if message and message.get("content"):
                      generated_text = message["content"]
                      logger.info("vLLM generation successful.")
                      logger.debug(f"vLLM Response: {generated_text[:500]}...")
                      return generated_text.strip()

            logger.error(f"vLLM response missing expected structure. Full response: {result_data}")
            return None

        except httpx.HTTPStatusError as e:
            logger.error(f"vLLM API request failed (HTTP {e.response.status_code}): {e.response.text}")
            return None
        except httpx.RequestError as e:
            logger.error(f"vLLM API request failed (Network/Connection Error): {e}")
            return None
        except Exception as e:
            logger.error(f"vLLM generation failed unexpectedly: {e}", exc_info=True)
            return None

    def get_status_dict(self) -> Dict[str, Any]:
         return {
             "active_model": self.model_name,
             "backend_url": self.base_url,
         }

    async def unload(self):
        await self.client.aclose()
        logger.info("VLLMBackend httpx client closed.")


# --- InstructLab Backend (Placeholder) ---
class InstructLabBackend(LLMBackendBase):
    def __init__(self, base_url: str, model_name: str):
        # Assuming similar OpenAI compatibility for now
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/v1/chat/completions" # Adjust if needed
        self.model_name = model_name
        self.client = httpx.AsyncClient(timeout=120.0)
        logger.info(f"InstructLabBackend initialized: URL='{self.base_url}', Model='{self.model_name}'")

    async def generate(self, prompt: str, config: Dict[str, Any]) -> Optional[str]:
        # Implement similarly to VLLMBackend, adjust payload/response parsing if needed
        logger.warning("InstructLabBackend generate() not fully implemented yet.")
        # --- Placeholder Implementation (like VLLM) ---
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model_name, "messages": messages,
            "temperature": config.get("temperature", settings.DEFAULT_LLM_TEMPERATURE),
            "top_p": config.get("top_p", settings.DEFAULT_LLM_TOP_P),
            "max_tokens": config.get("max_new_tokens", settings.DEFAULT_LLM_MAX_NEW_TOKENS),
            "stream": False,
        }
        try:
            response = await self.client.post(self.api_url, json=payload)
            response.raise_for_status()
            result_data = response.json()
            if result_data.get("choices") and result_data["choices"][0].get("message"):
                return result_data["choices"][0]["message"].get("content", "").strip()
            return None
        except Exception as e:
             logger.error(f"InstructLab generation failed: {e}", exc_info=True)
             return None
        # --- End Placeholder ---


    def get_status_dict(self) -> Dict[str, Any]:
         return {
             "active_model": self.model_name,
             "backend_url": self.base_url,
         }

    async def unload(self):
        await self.client.aclose()
        logger.info("InstructLabBackend httpx client closed.")

```
*   Creates `LLMBackendBase` ABC.
*   Moves the refined local generation logic into `LocalTransformersBackend`. Adds an `unload` method.
*   Implements `OllamaBackend` using `httpx` to call `/api/generate`.
*   Implements `VLLMBackend` using `httpx` to call the OpenAI-compatible `/v1/chat/completions` endpoint.
*   Adds a placeholder `InstructLabBackend`.
*   Each backend implements `generate` and `get_status_dict`. API backends also implement `unload` to close the `httpx` client.

**Step 4: Refactor `services/llm_service.py`**

This service now acts as a manager for the *active* backend.

```python
# services/llm_service.py
import time
import logging
import asyncio
import gc
import torch # Keep torch for CUDA checks
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

# Transformers needed only if local backend is used
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig

from core.config import settings
# Import the backend classes
from .llm_backends import (
    LLMBackendBase,
    LocalTransformersBackend,
    OllamaBackend,
    VLLMBackend,
    InstructLabBackend
)

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
# Stores the ACTIVE backend instance and global config
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


# --- Helper Functions (Scanning - keep as they are) ---
# ... list_local_models() ...
# ... list_cached_hub_models() ...


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


# --- Local Model Loading Task (Modified) ---
# This now specifically targets the LocalTransformersBackend instance
async def _load_local_model_task(
    backend: LocalTransformersBackend, # Pass the backend instance
    model_id: str,
    device: str,
    quantization: Optional[str]
):
    """Loads a model into the LocalTransformersBackend instance."""
    task_start_time = time.time()
    logger.info(f"[Local Loader] Starting task for '{model_id}' on device '{device}' q: '{quantization}'...")

    # No need to unload here, unload is handled before calling this task if needed

    llm_state["status"] = LLMStatus.LOADING # Update global status
    llm_state["last_error"] = None
    backend.model_name_or_path = model_id # Tentative assignment

    try:
        load_start_time = time.time()
        quantization_config = None
        pipeline_device_map = device if device != "cpu" else None
        if quantization:
             # ... (Quantization config logic - same as before) ...
              if quantization == "8bit" and torch.cuda.is_available():
                 quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                 logger.info("[Local Loader] Applying 8-bit quantization.")
                 pipeline_device_map = "auto"
              elif quantization == "4bit" and torch.cuda.is_available():
                  quantization_config = BitsAndBytesConfig(
                     load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                     bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                 )
                  logger.info("[Local Loader] Applying 4-bit quantization (nf4).")
                  pipeline_device_map = "auto"
              else: # Fallback
                 logger.warning(f"[Local Loader] Quantization '{quantization}' not supported/applicable. Loading full precision.")
                 quantization = None

        logger.info(f"[Local Loader] Loading tokenizer for '{model_id}'...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True,
            cache_dir=str(settings.HUGGINGFACE_HUB_CACHE.resolve())
        )
        logger.info("[Local Loader] Tokenizer loaded.")

        logger.info(f"[Local Loader] Initializing pipeline: model='{model_id}', device_map='{pipeline_device_map}', q_config={'Set' if quantization_config else 'None'}")
        pipeline_instance = pipeline(
            task="text-generation", model=model_id, tokenizer=tokenizer,
            device_map=pipeline_device_map,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            quantization_config=quantization_config, trust_remote_code=True,
        )
        load_time = time.time() - load_start_time
        logger.info(f"[Local Loader] Successfully loaded model '{model_id}' in {load_time:.2f}s.")

        max_len = getattr(pipeline_instance.model.config, "max_position_embeddings", None) \
                  or getattr(tokenizer, "model_max_length", None) \
                  or 1024
        logger.info(f"[Local Loader] Determined model max length: {max_len}")

        # --- Update the specific backend instance ---
        backend.pipeline = pipeline_instance
        backend.tokenizer = tokenizer
        backend.model_name_or_path = model_id # Confirm final name
        backend.load_config = {"device": device, "quantization": quantization}
        backend.max_model_length = max_len
        # --- Update global state ---
        llm_state["status"] = LLMStatus.READY
        llm_state["active_model"] = model_id
        llm_state["last_error"] = None

    except Exception as e:
        error_message = f"[Local Loader] Failed to load model '{model_id}': {type(e).__name__}: {e}"
        logger.error(error_message, exc_info=True)
        # Clear backend state
        backend.pipeline = None
        backend.tokenizer = None
        backend.model_name_or_path = None
        # Update global state
        llm_state["status"] = LLMStatus.FAILED
        llm_state["last_error"] = error_message
        llm_state["active_model"] = None # Clear model name on failure

    finally:
        task_duration = time.time() - task_start_time
        logger.info(f"[Local Loader] Task for '{model_id}' finished in {task_duration:.2f}s. Status: {llm_state['status'].value}")


# --- Public Service Functions ---

def get_llm_status() -> Dict[str, Any]:
    """Returns the current status and configuration of the active LLM backend."""
    status_copy = {
        "backend_type": llm_state["backend_type"],
        "active_model": llm_state["active_model"],
        "status": llm_state["status"].value,
        "generation_config": llm_state["config"],
        "last_error": llm_state["last_error"],
        "backend_details": None
    }
    if llm_state["backend_instance"] is not None:
        # Get specific details from the active backend
        status_copy["backend_details"] = llm_state["backend_instance"].get_status_dict()

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

# --- The core generation function ---
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

```
*   Imports backend classes.
*   `llm_state` now holds `backend_instance`, `backend_type`, `active_model`. Status enum updated.
*   `_unload_current_backend`: Helper to call `unload` on the current backend instance and reset state.
*   `_load_local_model_task`: Now takes the specific `LocalTransformersBackend` instance to update. Updates state on success/failure.
*   `get_llm_status`: Returns status including backend type and details from the active backend instance.
*   `generate_text`: Now the main entry point. Checks status, gets the active backend instance, gets global config, and calls `backend.generate(prompt, config)`. Crucially, it uses `run_in_executor` *only* if the backend is `LocalTransformersBackend`, otherwise it directly `await`s the API backend's async `generate`.
*   `set_active_backend`: New function orchestrating the switch. Unloads previous, creates the requested backend instance (or schedules local load), updates state.

**Step 5: Update API Models (`app/api/models/llm.py`)**

Adjust models to reflect the new state structure.

```python
# app/api/models/llm.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
# from services.llm_service import LLMStatus # No longer need Enum here

class ModelLoadRequest(BaseModel):
    # This name is now relative to the *configured backend type*
    model_name_or_path: str = Field(..., description="Identifier for the model (HF ID, local path for 'local' backend; model name for 'ollama'/'vllm').")
    # These are now only relevant for 'local' backend type
    device: Optional[str] = Field(default=None, description="Local Only: Device ('cuda', 'cpu'). Overrides server default if backend is 'local'.")
    quantization: Optional[str] = Field(default=None, description="Local Only: Quantization ('8bit', '4bit'). Overrides server default if backend is 'local'.")

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

```
*   `ModelLoadRequest`: Clarified fields, device/quantization are local-only now.
*   `ModelStatusResponse`: Updated fields to match new `llm_state` structure, added `backend_details`.
*   `ModelUpdateRequest`: Added `repetition_penalty`.

**Step 6: Update API Endpoint (`app/api/endpoints/models.py`)**

Refactor the `/load` endpoint to use the new `set_active_backend` service function.

```python
# app/api/endpoints/models.py
import logging
import asyncio
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks, status

# Import models and service functions
from app.api.models.llm import (
    ModelLoadRequest, ModelStatusResponse, ModelUpdateRequest, AvailableModelsResponse
)
# Import new service function and updated status getter
from services.llm_service import (
    list_local_models, list_cached_hub_models,
    set_active_backend, # <-- NEW import
    get_llm_status, update_llm_config,
    llm_state, LLMStatus # Keep state/enum for checking status
)
from core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# ... (@router.get("/available") remains the same) ...

@router.post(
    "/load",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Set active LLM backend and model", # Updated summary
)
async def load_or_set_model( # Renamed function for clarity
    load_request: ModelLoadRequest,
    background_tasks: BackgroundTasks, # Keep for potential future use, though less direct now
):
    """
    Sets the active LLM backend based on server config (LLM_BACKEND_TYPE)
    and configures it to use the specified model.
    For 'local' backend, initiates background loading.
    For API backends ('ollama', 'vllm', 'instructlab'), configures immediately.
    Unloads any previously active backend first.
    """
    current_status = llm_state["status"]
    # Prevent interfering with ongoing local load/unload
    if current_status == LLMStatus.LOADING or current_status == LLMStatus.UNLOADING:
         raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot change model/backend: Task already in progress (Status: {current_status.value}). Please wait."
        )

    backend_type = settings.LLM_BACKEND_TYPE # Get configured backend type
    model_id = load_request.model_name_or_path
    device = load_request.device # Pass along if provided (for local)
    quantization = load_request.quantization # Pass along if provided (for local)

    logger.info(f"Request received to activate model '{model_id}' on backend '{backend_type}'.")

    # Use the service function to handle the logic
    # This function is async because it might await the unload of the previous backend
    try:
        await set_active_backend(
            backend_type=backend_type,
            model_name_or_path=model_id,
            device=device,
            quantization=quantization
        )
    except ValueError as e: # Catch configuration errors from set_active_backend
         logger.error(f"Configuration error setting backend: {e}")
         raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error setting backend: {e}", exc_info=True)
        # Status should reflect failure from set_active_backend
        raise HTTPException(status_code=500, detail="Failed to set active backend.")

    # For API backends, status is READY immediately. For local, it's LOADING.
    final_status = get_llm_status() # Get updated status
    if final_status["status"] == LLMStatus.FAILED.value:
         return {
            "message": f"Backend configuration for '{model_id}' failed. Check server logs and status endpoint.",
            "status": final_status
         }
    elif final_status["status"] == LLMStatus.LOADING.value:
          return {
            "message": f"Local model loading initiated for '{model_id}'. Check status endpoint.",
            "status": final_status
         }
    else: # Should be READY for API backends
        return {
            "message": f"Backend '{backend_type}' configured with model '{model_id}'. Status: {final_status['status']}",
            "status": final_status
        }


# ... (@router.get("/status") remains the same) ...
# ... (@router.get("/config") remains the same) ...
# ... (@router.put("/config") remains the same) ...

```
*   Imports `set_active_backend`.
*   The `/load` endpoint now gets the `LLM_BACKEND_TYPE` from `settings`.
*   It calls `await set_active_backend(...)`.
*   It returns different messages based on whether it scheduled a background task (local) or configured an API backend immediately.

**Step 7: Update `main.py` Shutdown**

Ensure the new `unload` mechanism is called.

```python
# app/main.py
# ... imports ...
from services.system_service import shutdown_nvml
# Import the specific service function
from services.llm_service import _unload_current_backend # Import unload function

@app.on_event("shutdown")
async def shutdown_event():
    """Disconnect DB, shutdown NVML, unload LLM backend."""
    print("--> Event: shutdown")
    await disconnect_db()

    print("--> Shutting down NVML...")
    try:
         shutdown_nvml()
    except Exception as e:
         print(f"--> WARNING: Error during NVML shutdown: {e}")

    # --- ADDED: Unload active LLM backend ---
    print("--> Unloading active LLM backend...")
    try:
        # Use the dedicated unload function from llm_service
        await _unload_current_backend()
        print("--> LLM backend unloaded.")
    except Exception as e:
        print(f"--> WARNING: Error during LLM backend unload: {e}")
    # --- END ADDED ---

    print("--> Event: shutdown complete.")

# ... rest of main.py ...
```

**Testing the Multi-Backend Setup:**

1.  **Configure:** Set `LLM_BACKEND_TYPE` in `.env` or environment variables.
    *   **Local:** `LLM_BACKEND_TYPE=local`.
    *   **Ollama:** `LLM_BACKEND_TYPE=ollama`, `OLLAMA_BASE_URL=http://your_ollama_host:11434`. Make sure Ollama is running and has the model you want to use (e.g., `ollama run llama3`).
    *   **vLLM:** `LLM_BACKEND_TYPE=vllm`, `VLLM_BASE_URL=http://your_vllm_host:8000`. Make sure vLLM is running with the desired model.
2.  **Restart Server:** `uvicorn app.main:app --reload ...`
3.  **Load/Set Model:** Use `POST /api/v1/models/load`.
    *   For `local`, provide HF ID or path (e.g., `{"model_name_or_path": "gpt2"}`). Status should go to `LOADING` then `READY`/`FAILED`.
    *   For `ollama`, provide the model name Ollama uses (e.g., `{"model_name_or_path": "llama3"}`). Status should go to `READY` immediately (or `FAILED` if misconfigured).
    *   For `vllm`, provide the model name vLLM expects (often the path it was loaded with, e.g., `{"model_name_or_path": "/path/to/model/on/vllm/server"}`). Status `READY`/`FAILED`.
4.  **Check Status:** Use `GET /api/v1/models/status` to verify the backend type, active model, and status.
5.  **Chat:** Create a session and send messages. The request should be routed to the currently configured backend. Check logs to confirm.
6.  **Switch Backends:** Change the `LLM_BACKEND_TYPE` config, restart the server, load a relevant model, and test chat again.

This provides a flexible architecture to switch between local inference and various dedicated inference servers. Remember the concurrency warning about modifying config for summarization if using the local backend.

***

## How to use the `POST /api/v1/models/load` endpoint to connect to an online LLM service like Ollama or vLLM using this architecture. ##

The key is understanding that the **type** of backend (local, ollama, vllm) is determined by the **server's configuration** (`settings.LLM_BACKEND_TYPE`), while the `POST /load` request tells the server **which specific model** to use *on that configured backend*.

Here's the procedure:

**1. Configure the Server:**

*   **Choose Backend Type:** Decide which backend you want to use (e.g., `ollama`).
*   **Set Environment Variables (or `.env` file):** Before starting the server, you need to tell it which backend type to use and provide the necessary URL.
    *   **For Ollama:**
        ```bash
        # Set environment variables before running uvicorn
        export LLM_BACKEND_TYPE="ollama"
        export OLLAMA_BASE_URL="http://localhost:11434" # Or your Ollama server URL
        # Now start the server
        uvicorn app.main:app --reload ...
        ```
        Or, add these lines to your `.env` file:
        ```.env
        LLM_BACKEND_TYPE=ollama
        OLLAMA_BASE_URL=http://localhost:11434
        ```
    *   **For vLLM:**
        ```bash
        # Set environment variables
        export LLM_BACKEND_TYPE="vllm"
        export VLLM_BASE_URL="http://localhost:8000" # Or your vLLM server URL
        # Now start the server
        uvicorn app.main:app --reload ...
        ```
        Or, add these lines to your `.env` file:
        ```.env
        LLM_BACKEND_TYPE=vllm
        VLLM_BASE_URL=http://localhost:8000
        ```
*   **Restart Server:** If the server was already running, **restart it** so it picks up the new configuration from the environment variables or `.env` file. Check the startup logs to confirm it's using the correct `LLM Backend Type`.

**2. Use the `POST /api/v1/models/load` Endpoint:**

*   Now that the server knows it should be talking to Ollama (or vLLM), you use the `/load` endpoint to specify *which model on that remote server* you want to interact with.
*   **Request Body:** You need to send a JSON body containing the `model_name_or_path`.
    *   **`model_name_or_path`:** This value **must be the name of the model as recognized by the remote service**.
        *   For **Ollama:** This is the model tag you use with `ollama run` or `ollama pull` (e.g., `"llama3"`, `"mistral"`, `"codellama:7b"`).
        *   For **vLLM:** This often corresponds to the Hugging Face identifier or the path used when starting the vLLM server (e.g., `"meta-llama/Llama-2-7b-chat-hf"`, `"mistralai/Mistral-7B-Instruct-v0.1"`). Check your vLLM server's documentation or startup logs.
    *   **`device` and `quantization`:** These fields in the request body are **ignored** when the server's `LLM_BACKEND_TYPE` is set to `ollama`, `vllm`, or `instructlab`. They only apply to the `local` backend. You can omit them or leave them as `null`.

*   **Example Request (for Ollama, assuming server configured with `LLM_BACKEND_TYPE=ollama`):**
    ```bash
    curl -X 'POST' \
      'http://localhost:YOUR_SERVER_PORT/api/v1/models/load' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "model_name_or_path": "llama3"
    }'
    ```

*   **Example Request (for vLLM, assuming server configured with `LLM_BACKEND_TYPE=vllm`):**
    ```bash
    curl -X 'POST' \
      'http://localhost:YOUR_SERVER_PORT/api/v1/models/load' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.1"
    }'
    ```

**What Happens on the Server:**

1.  The `load_or_set_model` endpoint receives the request.
2.  It checks the *server's* configured `settings.LLM_BACKEND_TYPE`.
3.  It calls `set_active_backend` with the type (e.g., "ollama") and the `model_name_or_path` from the request (e.g., "llama3").
4.  `set_active_backend` unloads any previous backend.
5.  It creates an instance of the appropriate backend class (e.g., `OllamaBackend(url, "llama3")`).
6.  It stores this instance in `llm_state["backend_instance"]`.
7.  It updates the global `llm_state` (`backend_type`, `active_model`, `status` to `READY`).
8.  The API returns a 202 response confirming the backend is configured.

Now, when you send messages to a chat session, the `generate_text` function in `llm_service` will see the active backend is `OllamaBackend` (or `VLLMBackend`) and will call *its* `generate` method, which in turn makes the API call to the external Ollama/vLLM server using the configured model name (`llama3` in the example).
