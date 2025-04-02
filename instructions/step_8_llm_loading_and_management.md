**Step 8: LLM Loading and Management**

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
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
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
        "load_config": {} # Holds the config used *during loading* (device, quantization)
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
        llm_state["status"] = LLMStatus.LOADING
        llm_state["model_name_or_path"] = model_id
        llm_state["load_error"] = None
        llm_state["load_config"] = {"device": device, "quantization": quantization} # Store load config


        try:
            load_start_time = time.time()
            quantization_config = None
            model_kwargs = {"device_map": device if device != "cpu" else None} # Use device_map for non-CPU

            if quantization:
                if quantization == "8bit" and torch.cuda.is_available():
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    logger.info("Applying 8-bit quantization.")
                    # device_map='auto' might be better with quantization
                    model_kwargs["device_map"] = "auto"
                elif quantization == "4bit" and torch.cuda.is_available():
                     quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16, # Or bfloat16 if supported
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                     logger.info("Applying 4-bit quantization (nf4).")
                     model_kwargs["device_map"] = "auto"
                else:
                    logger.warning(f"Quantization '{quantization}' requested but not supported or CUDA not available. Loading in full precision.")
                    quantization = None # Reset quantization if not applied
                    llm_state["load_config"]["quantization"] = None

            # Use pipeline for simplicity, assumes text-generation task
            # For more control, load Model and Tokenizer separately
            new_pipeline = pipeline(
                "text-generation",
                model=model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32, # Use float16 on GPU
                quantization_config=quantization_config,
                trust_remote_code=True, # Be cautious with this setting
                model_kwargs=model_kwargs,
                # Use configured cache path for transformers models
                cache_dir=str(settings.HUGGINGFACE_HUB_CACHE.resolve())
            )

            # If not using device_map='auto' or pipeline places model on wrong device:
            # if device != "cpu" and hasattr(new_pipeline, 'model'):
            #      if str(new_pipeline.model.device) != device:
            #          logger.info(f"Moving model explicitly to {device}")
            #          new_pipeline.model.to(device)


            load_time = time.time() - load_start_time
            logger.info(f"Successfully loaded model '{model_id}' in {load_time:.2f} seconds.")

            # Update state upon successful loading
            llm_state["pipeline"] = new_pipeline
            llm_state["status"] = LLMStatus.LOADED
            llm_state["load_error"] = None
            # Reset generation config to defaults when loading a new model
            llm_state["config"] = {
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
            llm_state["model_name_or_path"] = None # Clear model name on failure

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
