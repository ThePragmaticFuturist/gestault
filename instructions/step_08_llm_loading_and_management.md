# Step 8: LLM Loading and Management #

## THE BIG IDEA: *Separation of Concerns + Modularity + Async I/O*

You’re building a **modular, asynchronous, retrieval-augmented LLM system** that:

- Keeps core responsibilities separated (e.g., session management, document storage, LLM orchestration)
- Treats the **LLM as a pluggable backend**, not the monolith
- Runs efficiently, even with heavy tasks like inference and search
- Enables **streaming**, **load balancing**, and **model swapping** down the line

---

## Analogy: Your Server is Like a Brain with Working Memory

- **LLM = language cortex** (fluid, creative, but not “aware”)
- **Chroma = long-term memory** (fast recall of document facts)
- **Session DB = short-term memory** (tracks what we just talked about)
- **LLM Service = brainstem** (manages energy, model switching, device use)
- **Prompt Constructor = executive function** (organizes all inputs into a coherent thought)

---
   
[Reasoning behind the code](reasoning/r8.md)
   
Okay, let's create the LLM loading and management API. This will involve:

1.  Defining configurations for models.
2.  Creating a service to handle the state and loading logic (in the background).
3.  Building the API endpoints.

1.  **Install Dependencies:**
    We need `transformers` and its dependencies, including `torch`. `accelerate` helps with device mapping and optimizations, and `bitsandbytes` enables easy quantization (optional but often needed for large models).

    ```bash
    pip install transformers torch torchvision torchaudio accelerate bitsandbytes sentencepiece
    ```
    *   `transformers`: The core Hugging Face library.
    *   `torch`, `torchvision`, `torchaudio`: PyTorch backend.
    *   `accelerate`: Optimizes model loading and execution across devices/multiple GPUs.
    *   `bitsandbytes`: For 8-bit and 4-bit quantization (requires a compatible environment, often Linux with NVIDIA GPU).
    *   `sentencepiece`: A tokenizer library used by many models (like Llama, T5).

2.  **Update Configuration (`core/config.py`):**
    Add settings related to LLM loading and local storage.

    ```python
    # core/config.py
    import os
    import torch # Import torch to check cuda availability easily
    from pydantic_settings import BaseSettings
    from pydantic import Field
    from pathlib import Path
    import tempfile
    from typing import Optional, List, Any

    BASE_DIR = Path(__file__).resolve().parent.parent

    # --- Helper function to determine default device ---
    def get_default_device():
        if torch.cuda.is_available():
            return "cuda"
        # Add checks for MPS (Apple Silicon) or other accelerators if needed
        # elif torch.backends.mps.is_available():
        #     return "mps"
        else:
            return "cpu"

    class Settings(BaseSettings):
        # ... (General, DB, Embedding settings) ...

        # --- NEW: LLM Settings ---
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
        DEFAULT_LLM_TOP_K: int = Field(default=50, description="Default top_k for LLM generation.")

        # ... (ChromaDB, Document Processing settings) ...
        # ... (RAG/Chat settings) ...

        class Config:
            env_file = '.env'
            env_file_encoding = 'utf-8'
            extra = 'ignore'

    settings = Settings()

    # --- Ensure necessary directories exist ---
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.HUGGINGFACE_HUB_CACHE.mkdir(parents=True, exist_ok=True)
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True) # Create local models dir
    _ = settings.SQLITE_DATABASE_URL
    _ = settings.CHROMA_PERSIST_PATH

    # --- Print relevant config ---
    print(f"--- Configuration Loaded ---")
    print(f"SQLite DB URL: {settings.SQLITE_DATABASE_URL}")
    print(f"ChromaDB Path: {settings.CHROMA_PERSIST_PATH}")
    print(f"Upload Dir: {settings.UPLOAD_DIR}")
    print(f"Embedding Model: {settings.EMBEDDING_MODEL_NAME} on {settings.EMBEDDING_DEVICE}")
    print(f"Local Models Dir: {settings.LOCAL_MODELS_DIR}") # Print new dir
    print(f"LLM Device: {settings.LLM_DEVICE}") # Print LLM device
    print(f"--------------------------")

    ```
    *   `LOCAL_MODELS_DIR`: Where the app will look for local models.
    *   `LLM_DEVICE`: Uses a helper function `get_default_device` to set "cuda" if available, otherwise "cpu".
    *   `DEFAULT_LLM_QUANTIZATION`: Sets a default quantization level (None, "8bit", "4bit"). Requires `bitsandbytes`.
    *   Added defaults for common generation parameters (`max_new_tokens`, `temperature`, `top_p`, `top_k`).

3.  **Create LLM Service (`services/llm_service.py`):**
    This service manages the LLM state and loading process.

    ```python
    # services/llm_service.py
    import time
    import logging
    import asyncio
    import gc
    from enum import Enum
    from pathlib import Path
    from typing import Optional, Dict, Any, List

    import torch
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
    from transformers import BitsAndBytesConfig # For quantization

    from core.config import settings

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # --- LLM Status Enum ---
    class LLMStatus(Enum):
        NOT_LOADED = "Not Loaded"
        LOADING = "Loading"
        LOADED = "Loaded"
        FAILED = "Failed"
        UNLOADING = "Unloading" # Added status for unloading


    # --- LLM State ---
    # Global state dictionary to hold LLM info
    # Avoids issues with module reloading during development compared to simple global vars
    llm_state: Dict[str, Any] = {
        "pipeline": None,
        "model_name_or_path": None,
        "status": LLMStatus.NOT_LOADED,
        "config": { # Holds the *generation* config
            "max_new_tokens": settings.DEFAULT_LLM_MAX_NEW_TOKENS,
            "temperature": settings.DEFAULT_LLM_TEMPERATURE,
            "top_p": settings.DEFAULT_LLM_TOP_P,
            "top_k": settings.DEFAULT_LLM_TOP_K,
        },
        "load_error": None,
        "load_config": {}, # Holds the config used *during loading* (device, quantization)
        "max_model_length": None
    }

    # --- Helper Function to Scan Local Models ---
    def list_local_models() -> List[str]:
        """Scans the LOCAL_MODELS_DIR for potential model directories."""
        local_models = []
        models_dir = settings.LOCAL_MODELS_DIR
        if models_dir.is_dir():
            for item in models_dir.iterdir():
                # Basic check: is it a directory and does it contain common HF files?
                if item.is_dir():
                    # Look for common files indicating a Hugging Face model directory
                    has_config = (item / "config.json").is_file()
                    has_pytorch_bin = list(item.glob("*.bin")) # pytorch_model.bin or similar
                    has_safetensors = list(item.glob("*.safetensors"))

                    if has_config and (has_pytorch_bin or has_safetensors):
                        local_models.append(str(item.resolve())) # Store absolute path
                # TODO: Add checks for specific file types like .gguf later if needed
        return local_models


    # --- Core Loading Function (Runs in Background) ---
    async def _load_model_task(model_id: str, device: str, quantization: Optional[str]):
        """The actual model loading logic, designed to run in an executor."""
        global llm_state
        task_start_time = time.time()
        logger.info(f"Starting model loading task for '{model_id}' on device '{device}' with quantization '{quantization}'...")

        # --- Unload previous model if exists ---
        if llm_state["pipeline"] is not None or llm_state["status"] == LLMStatus.LOADED:
            logger.info(f"Unloading previous model '{llm_state['model_name_or_path']}'...")
            llm_state["status"] = LLMStatus.UNLOADING
            previous_pipeline = llm_state.pop("pipeline", None)
            previous_model = getattr(previous_pipeline, 'model', None) if previous_pipeline else None
            previous_tokenizer = getattr(previous_pipeline, 'tokenizer', None) if previous_pipeline else None

            llm_state["model_name_or_path"] = None
            llm_state["pipeline"] = None

            # Explicitly delete objects and clear cache
            del previous_pipeline
            del previous_model
            del previous_tokenizer
            gc.collect() # Run garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache() # Clear GPU cache
            logger.info("Previous model unloaded and memory cleared.")
        # --- End Unload ---


        # --- Set Loading Status ---
        # --- Set Loading Status ---
        llm_state["status"] = LLMStatus.LOADING
        llm_state["model_name_or_path"] = model_id
        llm_state["load_error"] = None
        llm_state["load_config"] = {"device": device, "quantization": quantization}
        llm_state["tokenizer"] = None # Reset tokenizer
        llm_state["pipeline"] = None # Reset pipeline
        llm_state["max_model_length"] = None # Reset max length

        try:
            load_start_time = time.time()
            quantization_config = None
            pipeline_device_map = device if device != "cpu" else None
            if quantization:
                 if quantization == "8bit" and torch.cuda.is_available():
                     quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                     logger.info("Applying 8-bit quantization.")
                     pipeline_device_map = "auto"
                 elif quantization == "4bit" and torch.cuda.is_available():
                      quantization_config = BitsAndBytesConfig(
                         load_in_4bit=True,
                         bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                     )
                      logger.info("Applying 4-bit quantization (nf4).")
                      pipeline_device_map = "auto"
                 else:
                     logger.warning(f"Quantization '{quantization}' requested but not supported or CUDA not available. Loading in full precision.")
                     quantization = None
                     llm_state["load_config"]["quantization"] = None
    
             # --- Load Tokenizer explicitly first ---
            logger.info(f"Loading tokenizer for '{model_id}'...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=str(settings.HUGGINGFACE_HUB_CACHE.resolve())
            )
            logger.info("Tokenizer loaded.")
    
            # --- Load Model/Pipeline ---
            logger.info(f"Initializing pipeline with: model='{model_id}', device_map='{pipeline_device_map}', quantization_config={'Set' if quantization_config else 'None'}")
            # Pass the loaded tokenizer to the pipeline
            new_pipeline = pipeline(
                task="text-generation",
                model=model_id,
                tokenizer=tokenizer, # <-- Pass loaded tokenizer
                device_map=pipeline_device_map,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                quantization_config=quantization_config,
                trust_remote_code=True,
            )
    
            load_time = time.time() - load_start_time
            logger.info(f"Successfully loaded model '{model_id}' in {load_time:.2f} seconds.")
    
            # --- Get and store max length ---
            max_len = getattr(new_pipeline.model.config, "max_position_embeddings", None) \
                      or getattr(new_pipeline.tokenizer, "model_max_length", None) \
                      or 1024 # Fallback if not found (common for gpt2)
            logger.info(f"Determined model max length: {max_len}")

            llm_state["pipeline"] = new_pipeline
            llm_state["tokenizer"] = tokenizer # Store tokenizer
            llm_state["status"] = LLMStatus.LOADED
            llm_state["load_error"] = None
            llm_state["max_model_length"] = max_len # Store max length
        
            llm_state["config"] = { # Reset generation config
                "max_new_tokens": settings.DEFAULT_LLM_MAX_NEW_TOKENS,
                "temperature": settings.DEFAULT_LLM_TEMPERATURE,
                "top_p": settings.DEFAULT_LLM_TOP_P,
                "top_k": settings.DEFAULT_LLM_TOP_K,
            }
    
        except Exception as e:
            error_message = f"Failed to load model '{model_id}': {type(e).__name__}: {e}"
            logger.error(error_message, exc_info=True)
            llm_state["status"] = LLMStatus.FAILED
            llm_state["load_error"] = error_message
            llm_state["pipeline"] = None
            llm_state["model_name_or_path"] = None
            llm_state["tokenizer"] = None
            llm_state["max_model_length"] = None
        finally:
            task_duration = time.time() - task_start_time
            logger.info(f"Model loading task for '{model_id}' finished in {task_duration:.2f} seconds. Status: {llm_state['status'].value}")

    # --- Public Service Functions ---
    def get_llm_status() -> Dict[str, Any]:
        """Returns the current status and configuration of the LLM."""
        # Return a copy to prevent direct modification of the state dict
        status_copy = llm_state.copy()
        status_copy["status"] = llm_state["status"].value # Return enum value string
        status_copy.pop("pipeline", None) # Don't return the actual pipeline object
        return status_copy

    def update_llm_config(new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Updates the generation configuration for the loaded LLM."""
        if llm_state["status"] != LLMStatus.LOADED:
            raise ValueError("Cannot update config: No LLM is currently loaded.")

        # Validate and update allowed config keys
        allowed_keys = {"max_new_tokens", "temperature", "top_p", "top_k"} # Add more if needed
        updated = False
        for key, value in new_config.items():
            if key in allowed_keys:
                # Basic type validation could be added here
                if key in llm_state["config"] and llm_state["config"][key] != value:
                     logger.info(f"Updating LLM config: {key} = {value}")
                     llm_state["config"][key] = value
                     updated = True
            else:
                 logger.warning(f"Ignoring unsupported config key: {key}")

        if not updated:
             logger.info("No changes applied to LLM config.")

        return llm_state["config"] # Return the current config


    # --- Placeholder for Generation (will be used by chat endpoint later) ---
    def generate_text(prompt: str) -> Optional[str]:
        """Generates text using the loaded LLM pipeline."""
        if llm_state["status"] != LLMStatus.LOADED or llm_state["pipeline"] is None:
            logger.error("LLM not loaded, cannot generate text.")
            return None # Or raise an exception

        pipeline = llm_state["pipeline"]
        config = llm_state["config"]

        try:
            logger.info("Generating text with LLM...")
            # Use current config for generation
            # Note: The pipeline might ignore some parameters depending on the model/task
            # We pass only the ones the pipeline likely uses. Check pipeline docs.
            # 'do_sample=True' is usually needed for temperature/top_p/top_k to have effect
            outputs = pipeline(
                prompt,
                max_new_tokens=config["max_new_tokens"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                top_k=config["top_k"],
                do_sample=True, # Sample to use temp, top_p, top_k
                num_return_sequences=1 # Generate one response
            )
            # Extract the generated text (structure might vary slightly)
            generated_text = outputs[0]['generated_text']

            # Remove the prompt from the beginning if the pipeline includes it
            # This behaviour varies between models/pipelines
            if generated_text.startswith(prompt):
                # Be careful with this if prompt could be part of the response
                 result = generated_text[len(prompt):].strip()
            else:
                 result = generated_text.strip() # Assume it's only the new text

            logger.info("LLM text generation complete.")
            return result

        except Exception as e:
            logger.error(f"LLM text generation failed: {e}", exc_info=True)
            return None

    ```
    *   **State Management:** Uses a dictionary `llm_state` to store the pipeline object, name, status (Enum `LLMStatus`), generation config, loading config, and error messages.
    *   **`list_local_models`:** Scans the configured directory for potential model folders.
    *   **`_load_model_task`:**
        *   **Crucially designed to run in the background (e.g., via `run_in_executor`).**
        *   Handles **unloading** the previous model safely (deleting objects, clearing CUDA cache).
        *   Sets status to `LOADING`.
        *   Configures quantization using `BitsAndBytesConfig` if requested and possible.
        *   Uses `transformers.pipeline("text-generation", ...)` for loading. Specifies `device_map` or relies on `accelerate`. Includes `torch_dtype`. Sets `trust_remote_code=True` (use cautiously).
        *   Handles successful load (updates state to `LOADED`, stores pipeline).
        *   Handles errors (updates state to `FAILED`, stores error message).
    *   **`get_llm_status`:** Returns a copy of the state (excluding the pipeline object itself).
    *   **`update_llm_config`:** Allows changing generation parameters (temp, tokens, etc.) for the *currently loaded* model without reloading.
    *   **`generate_text`:** A placeholder function (that we'll call from the chat endpoint later) to demonstrate using the loaded pipeline with its current configuration. It includes basic logic to extract only the newly generated text.

4.  **Create LLM API Models (`app/api/models/llm.py`):**

    ```python
    # app/api/models/llm.py
    from pydantic import BaseModel, Field
    from typing import List, Optional, Dict, Any
    from services.llm_service import LLMStatus # Import Enum

    class ModelLoadRequest(BaseModel):
        model_name_or_path: str = Field(..., description="Identifier for the model (Hugging Face ID or absolute local path).")
        device: Optional[str] = Field(default=None, description="Device to load on ('cuda', 'cpu', 'auto'). Overrides server default.")
        quantization: Optional[str] = Field(default=None, description="Quantization level ('8bit', '4bit'). Overrides server default. Requires bitsandbytes & CUDA.")

    class ModelStatusResponse(BaseModel):
        model_name_or_path: Optional[str] = None
        status: str # Use the string value of the Enum
        load_error: Optional[str] = None
        load_config: Optional[Dict[str, Any]] = None
        generation_config: Optional[Dict[str, Any]] = None

    class ModelUpdateRequest(BaseModel):
        # Define fields for settings we allow updating
        max_new_tokens: Optional[int] = Field(default=None, ge=1)
        temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
        top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
        top_k: Optional[int] = Field(default=None, ge=1)

        # Method to get non-None values easily
        def get_updates(self) -> Dict[str, Any]:
            return self.dict(exclude_unset=True, exclude_none=True)

    class AvailableModelsResponse(BaseModel):
        local_models: List[str]
        # Add other sources later if needed (e.g., recently_used_hf)
    ```

5.  **Create LLM API Endpoints (`app/api/endpoints/models.py`):**

    ```python
    # app/api/endpoints/models.py
    import logging
    import asyncio # Needed for run_in_executor
    from typing import Dict, Any

    from fastapi import APIRouter, HTTPException, BackgroundTasks, status

    # Import models and service functions
    from app.api.models.llm import (
        ModelLoadRequest,
        ModelStatusResponse,
        ModelUpdateRequest,
        AvailableModelsResponse
    )
    from services.llm_service import (
        list_local_models,
        _load_model_task, # The core loading function
        get_llm_status,
        update_llm_config,
        llm_state, # Access state directly for checking status before background task
        LLMStatus
    )
    from core.config import settings

    logger = logging.getLogger(__name__)
    router = APIRouter()

    @router.get(
        "/available",
        response_model=AvailableModelsResponse,
        summary="List available local models",
    )
    async def get_available_models():
        """
        Scans the configured local models directory and returns a list of found models.
        """
        try:
            local_models = list_local_models()
            return AvailableModelsResponse(local_models=local_models)
        except Exception as e:
            logger.error(f"Error listing local models: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to list local models.")

    @router.post(
        "/load",
        status_code=status.HTTP_202_ACCEPTED, # Accepted for background processing
        summary="Load an LLM (async)",
    )
    async def load_model(
        load_request: ModelLoadRequest,
        background_tasks: BackgroundTasks,
    ):
        """
        Initiates loading of a specified LLM in the background.
        Unloads any previously loaded model first.
        """
        # Prevent starting a new load if one is already in progress
        current_status = llm_state["status"]
        if current_status == LLMStatus.LOADING or current_status == LLMStatus.UNLOADING:
             raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot load model: Task already in progress (Status: {current_status.value}). Please wait."
            )

        model_id = load_request.model_name_or_path
        # Use request device/quantization or fallback to defaults from settings
        device = load_request.device or settings.LLM_DEVICE
        quantization = load_request.quantization # Allow None explicitly, default handled by setting/None
        # If default is set and request is None, use default. If request is explicitly None, keep None.
        if load_request.quantization is None and settings.DEFAULT_LLM_QUANTIZATION is not None:
             quantization = settings.DEFAULT_LLM_QUANTIZATION


        logger.info(f"Received request to load model '{model_id}' with device='{device}', quantization='{quantization}'.")

        # Use run_in_executor to run the blocking I/O and CPU/GPU-bound task
        # in a separate thread, preventing it from blocking the main async event loop.
        loop = asyncio.get_running_loop()

        # Define the wrapper for run_in_executor
        def background_load_wrapper():
            # Need to create a new event loop for the thread if the task uses asyncio internally
            # (transformers doesn't typically need this for basic pipeline loading)
            # asyncio.run(_load_model_task(model_id, device, quantization)) # If task itself is async
            # Since _load_model_task is currently synchronous internally:
            asyncio.run(_load_model_task(model_id=model_id, device=device, quantization=quantization))


        # Schedule the blocking task in the default executor (ThreadPoolExecutor)
        # background_tasks.add_task doesn't work well for CPU/IO blocking tasks directly.
        # loop.run_in_executor(None, _load_model_task, model_id, device, quantization) # Arguments passed directly
        # Running an async function inside run_in_executor needs a wrapper like above or asyncio.to_thread
        try:
            # Use asyncio.to_thread for cleaner syntax if Python >= 3.9
            # asyncio.to_thread(_load_model_task, model_id=model_id, device=device, quantization=quantization)
            # For broader compatibility, use run_in_executor with the async wrapper
             loop.run_in_executor(None, background_load_wrapper) # Call the sync wrapper
             logger.info(f"Scheduled background loading task for model '{model_id}'.")
             # Immediately update status to LOADING (or maybe QUEUED?)
             # The task itself will set LOADING -> LOADED/FAILED
             # Let's set it here for immediate feedback
             llm_state["status"] = LLMStatus.LOADING
             llm_state["model_name_or_path"] = model_id # Tentative name
             llm_state["load_error"] = None

        except Exception as e:
             logger.error(f"Failed to schedule model loading task: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail="Failed to schedule model loading.")


        return {"message": f"Model loading initiated for '{model_id}'. Check status endpoint."}


    @router.get(
        "/status",
        response_model=ModelStatusResponse,
        summary="Get current LLM status",
    )
    async def get_model_status():
        """
        Returns the current status of the LLM, including loaded model name and config.
        """
        return get_llm_status()


    @router.get(
        "/config",
        response_model=Dict[str, Any],
        summary="Get current LLM generation configuration",
    )
    async def get_model_config():
        """
        Returns the current generation parameters (temperature, max_tokens, etc.).
        """
        status = get_llm_status()
        if status["status"] != LLMStatus.LOADED.value:
            raise HTTPException(status_code=404, detail="No model is currently loaded.")
        return status.get("generation_config", {})

    @router.put(
        "/config",
        response_model=Dict[str, Any],
        summary="Update LLM generation configuration",
    )
    async def update_model_config(update_request: ModelUpdateRequest):
        """
        Updates generation parameters for the currently loaded model.
        Does not require reloading the model.
        """
        if llm_state["status"] != LLMStatus.LOADED:
             raise HTTPException(status_code=404, detail="No model is currently loaded.")

        updates = update_request.get_updates()
        if not updates:
             raise HTTPException(status_code=400, detail="No valid configuration parameters provided.")

        try:
            updated_config = update_llm_config(updates)
            return updated_config
        except ValueError as e: # Catch errors from service layer
             raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
             logger.error(f"Error updating LLM config: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail="Failed to update LLM configuration.")

    ```
    *   **Imports:** Include `asyncio`, models, service functions.
    *   **`/available`:** Calls `list_local_models` service function.
    *   **`/load`:**
        *   Checks if a load/unload is already in progress (returns 409 Conflict).
        *   Gets model ID, device, quantization from request or settings.
        *   **Crucially:** Uses `loop.run_in_executor(None, background_load_wrapper)` to run the potentially blocking `_load_model_task` in a separate thread from FastAPI's default thread pool. This prevents blocking the main server loop. The `background_load_wrapper` is needed because `_load_model_task` is `async`. (Note: `asyncio.to_thread` is cleaner if Python 3.9+ is guaranteed).
        *   Sets status tentatively to `LOADING` for immediate feedback.
        *   Returns `202 Accepted`.
    *   **`/status`:** Calls `get_llm_status` service function.
    *   **`/config` (GET):** Gets current generation config from status.
    *   **`/config` (PUT):** Takes update request, calls `update_llm_config` service function.

6.  **Integrate Models Router in `app/main.py`:**

    ```python
    # app/main.py
    # ... (imports)
    from app.api.endpoints import documents
    from app.api.endpoints import sessions
    from app.api.endpoints import models # <--- ADDED IMPORT

    # ... (FastAPI app setup) ...

    # --- Include API Routers ---
    app.include_router(
        documents.router,
        prefix="/api/v1/documents",
        tags=["Documents"]
    )
    app.include_router(
        sessions.router,
        prefix="/api/v1/sessions",
        tags=["Chat Sessions"]
    )
    # --- ADDED: Include models router ---
    app.include_router(
        models.router,
        prefix="/api/v1/models",
        tags=["LLM Management"] # New tag
    )
    # --- END ADDED ---

    # ... (rest of main.py) ...
    ```

7.  **Run and Test:**
    *   Restart server: `uvicorn app.main:app --reload ...`
    *   Go to `/docs`. You should see the "LLM Management" section.
    *   **Test `/available`:** Execute `GET /api/v1/models/available`. It should list directories in your `local_models` folder that look like HF models (if any).
    *   **Test `/status`:** Execute `GET /api/v1/models/status`. Initially, it should show `Not Loaded`.
    *   **Test `/load`:**
        *   Execute `POST /api/v1/models/load`.
        *   Provide a valid Hugging Face model ID (e.g., `"gpt2"`, `"distilgpt2"`, or maybe a small quantized one like `"TheBloke/Mistral-7B-Instruct-v0.1-GGUF"` if you add GGUF support later, or a path like `/path/to/your/local_models/gpt2` if you downloaded it). For now, let's try `gpt2` as it's small:
            ```json
            {
              "model_name_or_path": "gpt2"
            }
            ```
        *   You should get a `202 Accepted` response immediately.
    *   **Test `/status` again:** Execute `GET /api/v1/models/status` repeatedly. You should see the status change from `Not Loaded` -> `Loading` -> `Loaded` (or `Failed`). Monitor the server logs for detailed loading progress/errors from `_load_model_task`.
    *   **Test `/config` (GET):** Once loaded, execute `GET /api/v1/models/config`. You should see the default generation parameters.
    *   **Test `/config` (PUT):** Execute `PUT /api/v1/models/config` with a body like:
        ```json
        {
          "temperature": 0.9,
          "max_new_tokens": 100
        }
        ```
        You should get a `200 OK` with the updated config. Verify with `GET /api/v1/models/config` again.
    *   **Test Loading Another Model:** Try loading a different model (e.g., `distilgpt2`) using `POST /load`. Check the logs for messages about unloading the previous model and loading the new one. Check the status updates.

**Summary:**

We now have an API for managing LLMs:
*   Listing local models.
*   Asynchronously loading models (from HF Hub or local path) using `transformers` pipelines.
*   Support for basic quantization (`bitsandbytes`).
*   Safe unloading/replacement of models.
*   Checking the loading status and configuration.
*   Updating generation parameters for the loaded model.

**Next Step:**

With the LLM management API in place, the clear next step is to **replace the placeholder LLM call** in the `add_message_to_session` endpoint (`app/api/endpoints/sessions.py`) with a call to our new `services.llm_service.generate_text` function. 

**Modify `app/api/endpoints/sessions.py`:**

We need to import the `generate_text` function and the `LLMStatus` enum from `llm_service`, call `generate_text` instead of the placeholder, and handle potential errors.

```python
# app/api/endpoints/sessions.py
import uuid
import logging
import datetime
from typing import List, Optional, Dict, Any
import asyncio # Keep asyncio import

from fastapi import APIRouter, HTTPException, Depends, status

# Import models and db access
from app.api.models.chat import (
    SessionCreateRequest,
    SessionMetadataResponse,
    MessageCreateRequest,
    ChatMessageResponse
)
from db.database import database, get_chroma_client
from db.models import sessions_table, chat_messages_table, documents_table
from core.config import settings
# Import embedding and search functions/models if needed directly
from services.embedding_service import generate_embeddings
# --- MODIFIED IMPORTS ---
# Import LLM service functions and status
from services.llm_service import (
    generate_text,
    get_llm_status,
    LLMStatus,
    summarize_text_with_query,
    llm_state
)
# --- END MODIFIED IMPORTS ---
from app.api.endpoints.documents import SearchQuery
import sqlalchemy

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Helper Function (Keep as is) ---
# ... get_session_or_404 ...

# --- Session Endpoints (Keep as they are) ---
# ... create_session, list_sessions, get_session, delete_session ...

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
):
    """
    Handles user message, RAG, LLM call, and stores conversation turn.
    """
    if message_data.role != "user":
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only 'user' role messages can be posted by the client.")

    session = await get_session_or_404(session_id)
    user_message_content = message_data.content
    now = datetime.datetime.now(datetime.timezone.utc)
    chroma_client = get_chroma_client()

    # --- Store User Message (Keep as is) ---
    try:
        insert_user_message_query = chat_messages_table.insert().values(
            session_id=session_id, timestamp=now, role="user", content=user_message_content, metadata=None,
        ).returning(chat_messages_table.c.id)
        user_message_id = await database.execute(insert_user_message_query)
        logger.info(f"[Session:{session_id}] Stored user message (ID: {user_message_id}).")
        update_session_query = sessions_table.update().where(sessions_table.c.id == session_id).values(last_updated_at=now)
        await database.execute(update_session_query)
    except Exception as e:
        logger.error(f"[Session:{session_id}] Failed to store user message: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to store user message.")

    # --- Perform RAG Search (Keep as is) ---
    rag_context = ""
    rag_chunk_ids = []
    rag_document_ids_in_session = session.get("rag_document_ids")
    if rag_document_ids_in_session:
        logger.info(f"[Session:{session_id}] Performing RAG search within documents: {rag_document_ids_in_session}")
        try:
            doc_collection = chroma_client.get_collection(settings.DOCUMENT_COLLECTION_NAME)
            query_embedding = generate_embeddings([user_message_content])
            if not query_embedding or not query_embedding[0]:
                raise ValueError("Failed to generate query embedding for RAG.")
            chroma_where_filter = {"document_id": {"$in": rag_document_ids_in_session}}
            results = doc_collection.query(
                query_embeddings=query_embedding, n_results=settings.RAG_TOP_K,
                where=chroma_where_filter, include=['documents', 'metadatas', 'distances']
            )
            retrieved_docs = results.get('documents', [[]])[0]
            retrieved_metadatas = results.get('metadatas', [[]])[0]
            retrieved_ids = results.get('ids', [[]])[0]
            if retrieved_docs:
                rag_chunk_ids = retrieved_ids
                context_parts = []
                for i, doc_text in enumerate(retrieved_docs):
                    metadata = retrieved_metadatas[i] if i < len(retrieved_metadatas) else {}
                    source_info = f"Source(Doc: {metadata.get('document_id', 'N/A')}, Chunk: {metadata.get('chunk_index', 'N/A')})"
                    context_parts.append(f"{source_info}:\n{doc_text}")
                rag_context = "\n\n---\n\n".join(context_parts)
                logger.info(f"[Session:{session_id}] Retrieved {len(retrieved_docs)} chunks for RAG context.")
                logger.debug(f"[Session:{session_id}] RAG Context:\n{rag_context[:500]}...")
        except Exception as e:
            logger.error(f"[Session:{session_id}] RAG search failed: {e}", exc_info=True)
            rag_context = "[RAG search failed]"
    else:
         logger.info(f"[Session:{session_id}] No RAG document IDs associated. Skipping RAG search.")

    # --- Retrieve Chat History (Keep as is) ---
    chat_history_str = ""
    try:
        history_limit = settings.CHAT_HISTORY_LENGTH * 2
        history_query = chat_messages_table.select().where(
            chat_messages_table.c.session_id == session_id
        ).order_by(chat_messages_table.c.timestamp.desc()).limit(history_limit)
        recent_messages = await database.fetch_all(history_query)
        recent_messages.reverse()
        if recent_messages:
             history_parts = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_messages]
             chat_history_str = "\n".join(history_parts)
             logger.info(f"[Session:{session_id}] Retrieved last {len(recent_messages)} messages for history.")
             logger.debug(f"[Session:{session_id}] Chat History:\n{chat_history_str[-500:]}...")
    except Exception as e:
        logger.error(f"[Session:{session_id}] Failed to retrieve chat history: {e}", exc_info=True)
        chat_history_str = "[Failed to retrieve history]"

    # --- Construct Prompt (Keep as is) ---
    prompt_for_llm = f"""CONTEXT:
{rag_context if rag_context else "No RAG context available."}

CHAT HISTORY:
{chat_history_str if chat_history_str else "No history available."}

USER QUERY:
{user_message_content}

ASSISTANT RESPONSE:"""
    logger.debug(f"[Session:{session_id}] Constructed prompt (first/last 200 chars):\n{prompt_for_llm[:200]}...\n...{prompt_for_llm[-200:]}")


    # --- ### MODIFIED SECTION: Call LLM Service ### ---

    assistant_response_content = None
    assistant_message_metadata = {
        "prompt_preview": prompt_for_llm[:200] + "...",
        "rag_chunks_retrieved": rag_chunk_ids,
        "llm_call_error": None # Add field for potential errors
    }

    # Check LLM status before attempting generation
    llm_status_data = get_llm_status()
    tokenizer = llm_state.get("tokenizer") # Get tokenizer object from state
    max_length = llm_state.get("max_model_length")
    model_name = llm_status_data.get("active_model", "N/A")

    if llm_status_data["status"] != LLMStatus.READY.value: # Check for READY status
        error_detail = f"LLM not ready (Status: {llm_status_data['status']}). Please load/configure a model first."
        logger.warning(f"[Session:{session_id}] {error_detail}")
        assistant_response_content = f"[ERROR: {error_detail}]"
        # Need to initialize metadata here if erroring out early
        assistant_message_metadata = {
            "prompt_preview": "[ERROR: LLM not ready]",
            "rag_chunks_retrieved": rag_chunk_ids, # Keep RAG info if available
            "llm_call_error": error_detail
        }
        llm_ready = False # Flag to skip generation
        # No prompt needed if we can't generate
        prompt_for_llm = "[ERROR: LLM/Tokenizer not ready for prompt construction]"
    # --- Handle potential missing tokenizer/max_length needed for truncation ---
    # These are primarily needed if we need to truncate context/history for the prompt
    # API backends might handle truncation server-side, but good practice to check here
    elif not tokenizer or not max_length:
         # Only critical if context/history is long and needs truncation
         # For now, let's allow proceeding but log a warning if they are missing
         # (relevant if switching from local to API backend without restarting server maybe?)
         logger.warning("Tokenizer or max_length not found in llm_state, prompt truncation might be inaccurate if needed.")
         llm_ready = True # Allow attempt, API backend might handle length
         # Prompt construction continues below...
    else:
        llm_ready = True # Flag to indicate LLM is ready for prompt construction
        # --- Define Prompt Components ---
        # Estimate tokens needed for instruction, markers, query, and response buffer
        # This is an approximation; actual token count depends on specific words.
        instruction_text = "You are a helpful AI assistant. Respond to the user's query based on the provided context and chat history."
        user_query_text = user_message_content
        # Reserve tokens for instruction, markers, query, and some buffer for the response itself
        # Let's reserve ~150 tokens for non-context/history parts + response buffer
        # And also account for the model's requested max_new_tokens
        generation_config = llm_status_data.get("generation_config", {})
        max_new_tokens = generation_config.get("max_new_tokens", settings.DEFAULT_LLM_MAX_NEW_TOKENS)
        reserved_tokens = 150 + max_new_tokens
        max_context_history_tokens = max_length - reserved_tokens
        logger.info(f"Max Length: {max_length}, Reserved: {reserved_tokens}, Available for Context/History: {max_context_history_tokens}")

        # --- Tokenize and Truncate Components ---
        # 1. RAG Context: Truncate if needed
        rag_context_tokens = []
        if rag_context and rag_context != "[RAG search failed]":
            rag_context_tokens = tokenizer.encode(f"\n### CONTEXT:\n{rag_context}", add_special_tokens=False)
            if len(rag_context_tokens) > max_context_history_tokens:
                logger.warning(f"Truncating RAG context ({len(rag_context_tokens)} tokens) to fit limit.")
                rag_context_tokens = rag_context_tokens[:max_context_history_tokens]
                # Optional: Decode back to string to see truncated context? Might be slow.
                # rag_context = tokenizer.decode(rag_context_tokens) # Use truncated context text

        available_tokens_for_history = max_context_history_tokens - len(rag_context_tokens)

        # 2. Chat History: Truncate from the OLDEST messages if needed
        chat_history_tokens = []
        if chat_history_str:
            # Start with full history prompt part
            full_history_prompt = f"\n### CHAT HISTORY:\n{chat_history_str}"
            chat_history_tokens = tokenizer.encode(full_history_prompt, add_special_tokens=False)
            if len(chat_history_tokens) > available_tokens_for_history:
                 logger.warning(f"Truncating chat history ({len(chat_history_tokens)} tokens) to fit limit.")
                 # Simple truncation: keep only the last N tokens
                 chat_history_tokens = chat_history_tokens[-available_tokens_for_history:]
                 # More robust: Re-tokenize messages one by one from recent to old until limit hit? Too complex for now.

        # --- Assemble Final Prompt (using tokens is safer, but harder to format) ---
        # For simplicity, let's assemble with potentially truncated *strings* derived from tokens
        # NOTE: Decoding tokens back to strings might introduce slight changes / tokenization artifacts.
        # A more advanced method would keep everything as token IDs until the final input.

        final_prompt_parts = []
        final_prompt_parts.append("### INSTRUCTION:")
        final_prompt_parts.append(instruction_text)

        if rag_context_tokens:
             # Decode the potentially truncated RAG context tokens
             decoded_rag_context_header = tokenizer.decode(tokenizer.encode("\n### CONTEXT:", add_special_tokens=False))
             decoded_rag_context_body = tokenizer.decode(rag_context_tokens[len(tokenizer.encode(decoded_rag_context_header, add_special_tokens=False)):])
             final_prompt_parts.append(decoded_rag_context_header + decoded_rag_context_body)


        if chat_history_tokens:
            # Decode the potentially truncated chat history tokens
            decoded_hist_header = tokenizer.decode(tokenizer.encode("\n### CHAT HISTORY:", add_special_tokens=False))
            decoded_hist_body = tokenizer.decode(chat_history_tokens[len(tokenizer.encode(decoded_hist_header, add_special_tokens=False)):])
            final_prompt_parts.append(decoded_hist_header + decoded_hist_body)


        final_prompt_parts.append("\n### USER QUERY:")
        final_prompt_parts.append(user_query_text)
        final_prompt_parts.append("\n### RESPONSE:")

        prompt_for_llm = "\n".join(final_prompt_parts)

        # Final Token Check (Optional but recommended)
        final_tokens = tokenizer.encode(prompt_for_llm)
        final_token_count = len(final_tokens)
        logger.info(f"Constructed final prompt with {final_token_count} tokens (Max allowed for input: {max_length - max_new_tokens}).")
        if final_token_count >= max_length:
            # This shouldn't happen often with the truncation, but as a safeguard:
            logger.error(f"FATAL: Final prompt ({final_token_count} tokens) still exceeds model max length ({max_length}). Truncating forcefully.")
            # Forcefully truncate the token list (might cut mid-word)
            truncated_tokens = final_tokens[:max_length - 5] # Leave a tiny buffer
            prompt_for_llm = tokenizer.decode(truncated_tokens)
            logger.debug(f"Forcefully truncated prompt:\n{prompt_for_llm}")


    logger.debug(f"[Session:{session_id}] Constructed prompt for LLM (Token Count: {final_token_count if 'final_token_count' in locals() else 'N/A'}):\n{prompt_for_llm[:200]}...\n...{prompt_for_llm[-200:]}")

    if llm_ready:
        try:
            logger.info(f"[Session:{session_id}] Sending prompt to LLM '{model_name}' via backend '{llm_status_data['backend_type']}'...") 
            loop = asyncio.get_running_loop()
            llm_response = await generate_text(prompt_for_llm)
            if llm_response is None:
                raise Exception("LLM generation failed or returned None.")
            assistant_response_content = llm_response
            logger.info(f"[Session:{session_id}] Received response from LLM.")
            logger.debug(f"[Session:{session_id}] LLM Response:\n{assistant_response_content[:500]}...")

        except Exception as e:
            error_detail = f"LLM generation failed: {type(e).__name__}"
            logger.error(f"[Session:{session_id}] {error_detail}: {e}", exc_info=True)
            # Ensure response content is set on error if it wasn't already
            if assistant_response_content is None:
                 assistant_response_content = f"[ERROR: {error_detail}]"
            assistant_message_metadata["llm_call_error"] = f"{error_detail}: {e}"

    # Ensure metadata is defined even if llm was not ready
    if not llm_ready:
        # We already set assistant_response_content and metadata earlier
        pass
    elif 'assistant_message_metadata' not in locals():
         # Initialize metadata if generation path didn't run due to other issues
          assistant_message_metadata = {
             "prompt_preview": prompt_for_llm[:200] + "...",
             "rag_chunks_retrieved": rag_chunk_ids,
             "llm_call_error": "Unknown error before storing message"
         }

    # --- Store Assistant Message (Using actual or error response) ---
    try:
        assistant_timestamp = datetime.datetime.now(datetime.timezone.utc)
        insert_assistant_message_query = chat_messages_table.insert().values(
            session_id=session_id,
            timestamp=assistant_timestamp,
            role="assistant",
            content=assistant_response_content, # Use the actual or error content
            metadata=assistant_message_metadata, # Include metadata with potential error
        ).returning(chat_messages_table.c.id, *[c for c in chat_messages_table.c])

        new_assistant_message_row = await database.fetch_one(insert_assistant_message_query)
        if not new_assistant_message_row:
            raise Exception("Failed to retrieve assistant message after insert.")

        logger.info(f"[Session:{session_id}] Stored assistant message (ID: {new_assistant_message_row['id']}).")

        # Update session timestamp
        update_session_query_after_assist = sessions_table.update().where(sessions_table.c.id == session_id).values(
             last_updated_at=assistant_timestamp
        )
        await database.execute(update_session_query_after_assist)

        # Return the structured assistant message response
        return ChatMessageResponse.parse_obj(dict(new_assistant_message_row))

    except Exception as e:
        logger.error(f"[Session:{session_id}] Failed to store assistant message: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to store assistant response.")


# --- List Messages Endpoint (Keep as is) ---
# ... list_messages_in_session ...

```

**Explanation of Changes:**

1.  **Imports:** Added imports for `generate_text`, `get_llm_status`, and `LLMStatus` from `services.llm_service`. Kept `asyncio`.
2.  **LLM Status Check:** Before attempting generation, it now calls `get_llm_status()` to check if the model `status` is `LOADED`. If not, it sets an error message as the response content and records the error in the message metadata.
3.  **Run LLM Call in Executor:**
    *   The call to `generate_text(prompt_for_llm)` is wrapped in `await loop.run_in_executor(None, generate_text, prompt_for_llm)`.
    *   This is **critical** because the LLM generation (`pipeline(...)` call inside `generate_text`) is CPU/GPU-bound and potentially long-running. Running it in an executor prevents it from blocking FastAPI's main async event loop, allowing the server to remain responsive to other requests.
4.  **Error Handling:** A `try...except` block surrounds the `run_in_executor` call. If `generate_text` returns `None` (indicating internal failure) or if `run_in_executor` raises an exception, it catches the error, logs it, sets an error message as the `assistant_response_content`, and records the error details in the `assistant_message_metadata`.
5.  **Store Response:** The code to store the assistant message now uses the `assistant_response_content` variable, which will contain either the actual LLM response or an error message. The `assistant_message_metadata` (including potential errors) is also stored.

**How to Test:**

1.  Restart the server: `uvicorn app.main:app --reload ...`
2.  **Load a Model:** Use `POST /api/v1/models/load` to load an LLM (e.g., `gpt2`). Wait for the status (`GET /api/v1/models/status`) to become `Loaded`.
3.  **Create a Session:** Create a session, optionally linking valid document IDs.
4.  **Send a Message:** Use `POST /api/v1/sessions/{session_id}/messages` to send a user message.
5.  **Observe Response:** This time, the response should contain text generated by the actual `gpt2` (or whichever model you loaded) based on the prompt (including RAG context if applicable). It won't be just the placeholder.
6.  **Check Logs:** The server logs should show the call to the LLM service (`Sending prompt to LLM...`) and the received response (`Received response from LLM.`).
7.  **Check DB:** The `chat_messages` table should contain the real assistant response. The metadata field might show `llm_call_error: null` if successful.
8.  **Test Failure Cases:**
    *   Try sending a message *without* loading an LLM first. The response should contain the `[ERROR: LLM not ready...]` message.
    *   Try loading a very large model that might fail (e.g., due to memory). Send a message and see if the `[ERROR: LLM generation failed...]` message appears in the response.

**Summary:**

We have now connected the chat endpoint to the LLM service. User messages trigger RAG search, prompt construction, and a call to the loaded LLM (running asynchronously in an executor thread) to generate a response. The conversation, including the real LLM output or any errors encountered, is stored.


## List `local_models` directory and the *actual Hugging Face cache* where models downloaded via `transformers` (or `huggingface_hub`) are stored. ##  

Here's how we can implement that:

1.  **Identify the HF Cache Directory:** We already have `settings.HUGGINGFACE_HUB_CACHE` defined in our `core/config.py`. This is the directory we need to scan.

2.  **Understand HF Cache Structure:** Models downloaded from the Hub are typically stored in subdirectories like:
    *   `models--gpt2`
    *   `models--bert-base-uncased`
    *   `models--google--flan-t5-large` (using `--` as separator for organization/model name)
    Inside these, there's often a `snapshots/<commit_hash>/` structure containing the actual files. We just need to identify the top-level model directories.

3.  **Implement Scanning Logic in `services/llm_service.py`:**
    We'll add a new function to specifically scan the Hub cache and parse the directory names back into model IDs.

    ```python
    # services/llm_service.py
    import time
    import logging
    import asyncio
    import gc
    from enum import Enum
    from pathlib import Path
    from typing import Optional, Dict, Any, List

    import torch
    # ... (other imports: pipeline, AutoModelForCausalLM, etc.)

    from core.config import settings

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # ... (LLMStatus Enum, llm_state dictionary) ...


    # --- Helper Function to Scan Dedicated Local Models Dir ---
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

    # --- NEW: Helper Function to Scan HF Hub Cache ---
    def list_cached_hub_models() -> List[str]:
        """Scans the Hugging Face Hub cache directory for downloaded models."""
        hub_models = []
        cache_dir = settings.HUGGINGFACE_HUB_CACHE
        logger.info(f"Scanning for cached Hub models in: {cache_dir}")

        if not cache_dir.is_dir():
            logger.warning(f"Hugging Face cache directory not found: {cache_dir}")
            return []

        for item in cache_dir.iterdir():
            # Look for directories starting with 'models--'
            if item.is_dir() and item.name.startswith("models--"):
                # Check if it contains snapshots (a good indicator it's a model cache)
                snapshots_dir = item / "snapshots"
                if snapshots_dir.is_dir() and any(snapshots_dir.iterdir()): # Check if snapshots is not empty
                    # Parse directory name back to model ID
                    # Example: 'models--google--flan-t5-large' -> 'google/flan-t5-large'
                    # Example: 'models--gpt2' -> 'gpt2'
                    model_id = item.name.removeprefix("models--").replace("--", "/")
                    logger.debug(f"Found potential cached Hub model: {model_id} (from {item.name})")
                    hub_models.append(model_id)

        logger.info(f"Found {len(hub_models)} potential cached models in {cache_dir}.")
        return sorted(list(set(hub_models))) # Return sorted unique list

    # ... (_load_model_task, get_llm_status, update_llm_config, generate_text) ...
    # NOTE: The _load_model_task should work fine with these Hub IDs without modification

    ```
    *   Renamed the original `list_local_models` slightly for clarity (scanning the *dedicated* directory).
    *   Added `list_cached_hub_models` function.
    *   It iterates through the configured `HUGGINGFACE_HUB_CACHE`.
    *   Looks for directories starting with `models--`.
    *   Checks for a non-empty `snapshots` subdirectory inside as confirmation.
    *   Parses the directory name (e.g., `models--google--flan-t5-large`) back into the standard Hub ID format (`google/flan-t5-large`) using `removeprefix` and `replace`.
    *   Returns a sorted, unique list of found model IDs.

4.  **Update API Model (`app/api/models/llm.py`):**
    Modify `AvailableModelsResponse` to include the list of cached Hub models.

    ```python
    # app/api/models/llm.py
    from pydantic import BaseModel, Field
    from typing import List, Optional, Dict, Any
    # ... (other models: ModelLoadRequest, etc.)

    class AvailableModelsResponse(BaseModel):
        local_model_paths: List[str] = Field(description="Absolute paths to models found in the dedicated local_models directory.")
        cached_hub_models: List[str] = Field(description="Model IDs found in the Hugging Face Hub cache directory.")
    ```
    *   Renamed `local_models` to `local_model_paths` for clarity (since they are paths).
    *   Added `cached_hub_models`.

5.  **Update API Endpoint (`app/api/endpoints/models.py`):**
    Modify the `get_available_models` endpoint to call both scanning functions.

    ```python
    # app/api/endpoints/models.py
    # ... (imports) ...
    # Make sure both service functions are imported
    from services.llm_service import (
        list_local_models,
        list_cached_hub_models, # <-- Import the new function
        _load_model_task,
        get_llm_status,
        update_llm_config,
        llm_state,
        LLMStatus
    )
    # ... (other imports) ...

    router = APIRouter()

    @router.get(
        "/available",
        response_model=AvailableModelsResponse,
        summary="List available local and cached Hub models", # Updated summary
    )
    async def get_available_models():
        """
        Scans the configured local models directory and the Hugging Face cache
        directory and returns lists of found models.
        """
        try:
            local_paths = list_local_models()
            hub_ids = list_cached_hub_models()
            return AvailableModelsResponse(
                local_model_paths=local_paths,
                cached_hub_models=hub_ids
            )
        except Exception as e:
            logger.error(f"Error listing available models: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to list available models.")

    # ... (rest of the endpoints: /load, /status, /config) ...
    ```
    *   Calls both `list_local_models` and `list_cached_hub_models`.
    *   Populates the updated `AvailableModelsResponse`.

6.  **Run and Test:**
    *   Restart the server: `uvicorn app.main:app --reload ...`
    *   Ensure you have previously loaded at least one model from the Hub (like `gpt2`) so it exists in the cache (`~/.cache/huggingface/hub` or your configured path).
    *   Execute `GET /api/v1/models/available` via the `/docs` page or `curl`.
    *   Check the response. It should now contain two lists:
        *   `local_model_paths`: Listing any models found in your specific `./local_models` folder (if any).
        *   `cached_hub_models`: Listing the IDs (like `"gpt2"`, `"all-MiniLM-L6-v2"`, etc.) found in the central HF cache.

This provides a much more comprehensive view of the models readily available to the server.

## UNDERLYING THEORIES

### 1. **Retrieval-Augmented Generation (RAG) Theory**

**Problem:** LLMs are stateless and expensive to train on your own data.  
**Solution:** Instead of retraining or fine-tuning, **retrieve relevant data**, inject it into the prompt, then generate.

This leverages:
- **Semantic Search** (via ChromaDB + vector embeddings)
- **LLM Prompt Engineering** to mix knowledge and context
- A hybrid: *structured memory + generative language*

🔍 **Core idea:** LLMs don’t need to “know” everything—they just need a good context window.

---

### 2. **Microservices + Domain-Driven Design (DDD)**

Even though you're not using Docker/Kubernetes (yet), the design treats the system as **composable services**:

| Concern | Module / Service |
|--------|------------------|
| Configs / Devices | `core/config.py` |
| Embedding Generation | `services/embedding_service.py` |
| LLM Management | `services/llm_service.py` |
| Chat Sessions | `endpoints/sessions.py` |
| Document Management | `endpoints/documents.py` |

**Core idea:** Make every part swappable and testable.

---

### 3. **Asynchronous Design with Background Tasks**

**Problem:** LLM loading and generation are blocking (slow and resource-heavy).  
**Solution:** Use Python’s `asyncio.run_in_executor()` to run long tasks *off the main thread*.

This enables:
- Scalability: Handle more users without freezing
- Parallelism: Load models or generate while serving other users
- Streaming or live-response systems later

---

### 4. **Stateful Inference with Stateless Interfaces**

You persist **chat sessions and message history** in a **SQL DB**, which gives you:

- Reproducibility (LLMs are stateless but logs aren’t)
- Context-aware LLM prompts (via `CHAT_HISTORY_LENGTH`)
- Support for features like summaries, auditing, etc.

**Core idea:** APIs are stateless, but you create the illusion of memory using a database.

---

### 5. **Dynamic Model Loading & Configuration**

By building a **live-configurable LLM layer**, you support:

- Switching between models (e.g., GPT-2, Mistral, Falcon)
- Running on different hardware (CPU, GPU)
- Tuning generation parameters (`temperature`, `top_k`, etc.)

**Core idea:** Treat models as hot-swappable plugins—not hardcoded.

---

### 6. **Human-in-the-Loop and Observability**

All messages, configs, prompts, and responses are stored in the DB. This:
- Enables **post-hoc debugging**
- Supports building dashboards or observability tooling
- Enables **user trust** and traceability for enterprise cases

You’re designing with **AI observability and explainability** in mind.

---

## THEORY-IN-PRACTICE SUMMARY

| Principle | How it shows up |
|----------|----------------|
| **Separation of Concerns** | Dedicated services and endpoints |
| **Modularity** | Pluggable models, configs, quantization |
| **Asynchronous Patterns** | Executors for heavy loading/generation |
| **Data Locality** | RAG keeps data in local vector DB |
| **Stateless Interfaces + Stateful DB** | Prompts feel stateful, backed by persistent history |
| **LLM Orchestration Layer** | Makes your app model-agnostic and future-proof |
| **Scalability & Resilience** | Allows error capture, retries, and task isolation |
