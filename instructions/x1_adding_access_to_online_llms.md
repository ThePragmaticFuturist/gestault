OK. as we add new features and functionality to the gestalt, they will be contained in single "steps" as the core of the system is already established and these will ultimately affect code across mutiple files. 

Next stage. How do I connect this RAG LLM server framework to an online LLM like Google Gemini 2.5? While my "vibe coding" peer is creating the additional code to Gemini 1.5 Flash, the same principles can be used to add the requisite classes and data structures for almost any online LLM. 

***   

Here's a thinking process for adding Google Gemini support:

1.  **Understand the Goal:** The user wants to add Google Gemini (specifically mentioning 2.5, but let's aim for the general Gemini API) as another selectable backend, similar to how Ollama and vLLM were added.

2.  **Identify Key Requirements for Gemini:**
    *   **API Endpoint:** What's the URL for Gemini API calls? (Need to consult Google AI documentation). It's likely `https://generativelanguage.googleapis.com/...`
    *   **Authentication:** How does Google authenticate API calls? (API Keys are standard for Google Cloud/AI services).
    *   **Client Library:** Is there an official Python client library from Google? (`google-generativeai`). Using the official library is almost always better than raw HTTP requests.
    *   **API Request/Response Format:** How is the prompt sent? What's the structure of the response? (Likely involves sending content/parts and receiving generated content). Need to map our internal prompt/config to Gemini's format.
    *   **Model Names:** What are the identifiers for Gemini models (e.g., `gemini-1.5-pro-latest`, `gemini-1.5-flash-latest`)?

3.  **Consult Google AI Documentation:**
    *   Search for "Google AI Python SDK", "Gemini API Python", "Gemini API Authentication".
    *   Find Quickstarts and API reference documentation.
    *   Key findings:
        *   Library: `google-generativeai`
        *   Authentication: API Key, typically set via `GOOGLE_API_KEY` environment variable or passed during client initialization.
        *   Client Initialization: `genai.configure(api_key=...)`, `model = genai.GenerativeModel(...)`
        *   Generation Method: `model.generate_content(...)`
        *   Request Format: Takes text prompt directly or structured `Content`/`Part` objects.
        *   Response Format: A `GenerateContentResponse` object with `text` attribute (for simple cases) or `parts`.
        *   Model Names: `gemini-1.5-pro-latest`, `gemini-1.5-flash-latest`, etc.
        *   Configuration Mapping: Need to map `temperature`, `top_p`, `top_k`, `max_output_tokens` (equivalent to `max_new_tokens`) to `genai.GenerationConfig`.

4.  **Plan the Implementation Steps:** Follow the pattern used for Ollama/vLLM:
    *   **Configuration (`core/config.py`):** Add `LLM_BACKEND_TYPE="gemini"`, add `GOOGLE_API_KEY` setting.
    *   **Install Library:** `pip install google-generativeai`.
    *   **Backend Class (`services/llm_backends.py`):** Create `GeminiBackend(LLMBackendBase)`.
        *   `__init__`: Takes `api_key` and `model_name`. Configures `genai` and initializes `genai.GenerativeModel`.
        *   `generate`: Maps internal config to `genai.GenerationConfig`, calls `model.generate_content(prompt, generation_config=...)`, extracts text from the response. Needs to handle potential exceptions. Should be `async def` as `generate_content` is likely async (or check docs - *Update: `generate_content` is synchronous, so the backend `generate` can be sync, and `llm_service` will put it in executor*). *Correction: The `google-generativeai` library often provides both sync and async methods. Let's check if an async client/method exists. Okay, `generate_content_async` exists! Perfect. We'll make the method `async`.*
        *   `get_status_dict`: Returns basic info (model name).
        *   `unload`: No explicit unload needed for API client usually.
    *   **Refactor `llm_service.py`:**
        *   Add `"gemini"` to `LLMStatus` (or just handle it as an API type). *No, update the backend type Literal.*
        *   Add `GeminiBackend` to imports.
        *   Update `set_active_backend` to handle `backend_type == "gemini"`, retrieving the API key from settings and instantiating `GeminiBackend`.
        *   Update `generate_text` to correctly identify `GeminiBackend` as an API backend (it already does if we inherit correctly and make `generate` async).
    *   **Update API Models (`app/api/models/llm.py`):** Add `"gemini"` to the `Literal` type for `backend_type` in `ModelLoadRequest`.
    *   **Update `requirements.txt`**.
    *   **Update Documentation/`.env` example**.

5.  **Write the Code (Iterative Refinement):**

    *   **Config:** Add `GOOGLE_API_KEY` to `Settings`. Add `gemini` to `LLM_BACKEND_TYPE` Literal.
    *   **Install:** Add `google-generativeai` to `requirements.txt`.
    *   **Backend Class:**
        *   Import `google.generativeai as genai`.
        *   In `__init__`: Check if API key is provided. Call `genai.configure(api_key=...)`. Instantiate `genai.GenerativeModel(model_name=...)`. Store the model object.
        *   In `generate`:
            *   Make it `async def`.
            *   Map config keys (`temperature`, `top_p`, `top_k`, `max_new_tokens`) to `genai.GenerationConfig` object keys (`temperature`, `top_p`, `top_k`, `max_output_tokens`). Handle potential missing keys gracefully using defaults.
            *   Wrap the call in `try...except` to catch Google API errors (`google.api_core.exceptions`).
            *   Call `await self.model.generate_content_async(prompt, generation_config=generation_config)`.
            *   Extract `response.text`. Handle cases where `response.text` might not exist (check response structure/safety features).
        *   `get_status_dict`: Return `{"active_model": self.model_name}`.
    *   **LLM Service:**
        *   Import `GeminiBackend`.
        *   In `set_active_backend`: Add `elif backend_type == "gemini":`. Check `settings.GOOGLE_API_KEY`. Instantiate `GeminiBackend`. Set status to `READY`.
        *   In `generate_text`: Ensure the `elif` includes `GeminiBackend` for direct `await`.
    *   **API Models:** Add `gemini` to the `backend_type` Literal in `ModelLoadRequest`.
    *   **Main (`app/main.py`):** Add check for `GOOGLE_API_KEY` in startup *if* Gemini is the *default* backend? Or just let `set_active_backend` handle the check when Gemini is requested. Let `set_active_backend` handle it to avoid forcing API key setup if the user only uses local models.

6.  **Testing Plan:**
    *   Set `GOOGLE_API_KEY` in `.env`.
    *   Restart server.
    *   Use `POST /load` with `backend_type="gemini"` and a valid Gemini model name (e.g., `gemini-1.5-flash-latest`).
    *   Check status (`GET /status`).
    *   Send a chat message. Verify logs show Gemini backend being called and check the response.
    *   Test error handling (e.g., invalid API key, invalid model name).

7.  **Review and Refine:** Check code for clarity, error handling, logging, and adherence to the established patterns. Ensure the API key isn't logged.

***   

Okay, connecting to managed online LLM services like Google's Gemini API follows the same multi-backend pattern we established. We'll add a new backend implementation for Gemini.

**Prerequisites:**

1.  **Google Cloud Project & API Key:**
    *   You need a Google Cloud project with the "Generative Language API" (or Vertex AI API, depending on the specific Gemini model/endpoint you target) enabled.
    *   You need an API Key associated with that project. You can create one in the Google Cloud Console under "APIs & Services" > "Credentials". **Treat this key like a password!**
    *   Reference: [https://ai.google.dev/docs/setup](https://ai.google.dev/docs/setup)
2.  **Google AI Python Library:** Install the official library.

**Implementation Steps:**

**1. Install Google AI Library**

```bash
pip install google-generativeai
```

**2. Update Configuration (`core/config.py`)**

Add `"gemini"` to the backend type literal and add a setting for the API key.

```python
# core/config.py
# ... (imports: Optional, List, Any, Literal) ...

class Settings(BaseSettings):
    # ... (General, DB, Embedding settings) ...

    # --- LLM Settings ---
    # Add 'gemini' to the allowed types
    LLM_BACKEND_TYPE: Literal["local", "ollama", "vllm", "instructlab", "gemini"] = Field(
        default="local",
        description="The default LLM backend ('local', 'ollama', 'vllm', 'instructlab', 'gemini')."
    )
    # Base URLs (Gemini uses API Key, not base URL usually)
    OLLAMA_BASE_URL: Optional[str] = Field(default="http://localhost:11434", ...)
    VLLM_BASE_URL: Optional[str] = Field(default=None, ...)
    INSTRUCTLAB_BASE_URL: Optional[str] = Field(default=None, ...)

    # --- NEW: Google API Key ---
    GOOGLE_API_KEY: Optional[str] = Field(
        default=None,
        description="API Key for Google AI (Gemini). Loaded from .env or environment."
    )
    # --------------------------

    DEFAULT_MODEL_NAME_OR_PATH: str = Field(
        default="gpt2",
        description="Default model identifier (HF ID, path for 'local'; model name for API backends like 'gemini-1.5-flash-latest' for 'gemini')." # Updated example
    )
    # ... (Local settings, Generation parameters) ...

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore'

settings = Settings()

# ... (Ensure directories exist) ...
# ... (Print config - maybe add check for GOOGLE_API_KEY if gemini is default?) ...
if settings.LLM_BACKEND_TYPE == 'gemini' and not settings.GOOGLE_API_KEY:
    print("WARNING: LLM_BACKEND_TYPE is 'gemini' but GOOGLE_API_KEY is not set!")
# ...
print(f"--------------------------")

```

**3. Create Gemini Backend (`services/llm_backends.py`)**

Add a new class `GeminiBackend`.

```python
# services/llm_backends.py
import logging
import abc
from typing import Dict, Any, Optional, AsyncGenerator

import httpx
from core.config import settings

# --- ADD Google AI Imports ---
try:
    import google.generativeai as genai
    google_ai_available = True
except ImportError:
    google_ai_available = False
    genai = None # Define genai as None if import fails
# --- END ADD ---


logger = logging.getLogger(__name__)

# --- Base Class (unchanged) ---
# ... LLMBackendBase ...

# --- Other Backends (unchanged) ---
# ... LocalTransformersBackend ...
# ... OllamaBackend ...
# ... VLLMBackend ...
# ... InstructLabBackend ...


# --- NEW: Gemini Backend ---
class GeminiBackend(LLMBackendBase):
    def __init__(self, api_key: str, model_name: str):
        if not google_ai_available or genai is None:
             raise ImportError("Attempted to initialize GeminiBackend, but 'google-generativeai' library is not installed. Please run 'pip install google-generativeai'.")

        self.model_name = model_name
        self.api_key = api_key
        try:
            # Configure the API key globally for the library (common pattern)
            genai.configure(api_key=self.api_key)
            # Create the specific model instance
            self.model = genai.GenerativeModel(model_name=self.model_name)
            logger.info(f"GeminiBackend initialized: Model='{self.model_name}' configured.")
        except Exception as e:
            logger.error(f"Failed to configure Gemini or initialize model '{self.model_name}': {e}", exc_info=True)
            raise ValueError(f"Gemini configuration/initialization failed: {e}") from e

    async def generate(self, prompt: str, config: Dict[str, Any]) -> Optional[str]:
        logger.info(f"Generating text via Gemini backend (model: {self.model_name})...")

        # Map internal config keys to Gemini GenerationConfig keys
        gemini_config = genai.types.GenerationConfig(
            # candidate_count=1, # Default is 1
            # stop_sequences=["\n###"], # Example stop sequence
            max_output_tokens=config.get("max_new_tokens", settings.DEFAULT_LLM_MAX_NEW_TOKENS),
            temperature=config.get("temperature", settings.DEFAULT_LLM_TEMPERATURE),
            top_p=config.get("top_p", settings.DEFAULT_LLM_TOP_P),
            top_k=config.get("top_k", settings.DEFAULT_LLM_TOP_K), # Note: top_k >= 1
        )
        # Ensure top_k is valid for Gemini (must be >= 1)
        if gemini_config.top_k is not None and gemini_config.top_k < 1:
            logger.warning(f"Gemini requires top_k >= 1. Received {gemini_config.top_k}, setting to 1.")
            gemini_config.top_k = 1

        logger.debug(f"Gemini GenerationConfig: {gemini_config}")

        try:
            # Use the asynchronous generation method
            response = await self.model.generate_content_async(
                contents=prompt, # Send the full prompt string
                generation_config=gemini_config,
                # safety_settings=... # Optional: configure safety settings
            )

            # Extract text, handling potential blocks or missing text
            if response.text:
                 generated_text = response.text
                 logger.info("Gemini generation successful.")
                 logger.debug(f"Gemini Response: {generated_text[:500]}...")
                 return generated_text.strip()
            else:
                 # Log details if text is missing (e.g., blocked by safety filters)
                 logger.warning(f"Gemini response did not contain text. Prompt feedback: {response.prompt_feedback}. Finish reason: {getattr(response.candidates[0], 'finish_reason', 'N/A')}")
                 # Consider checking response.candidates[0].content.parts if text is missing
                 return "[ERROR: Gemini response empty or blocked]"

        except Exception as e:
            # Catch potential google API errors or other issues
            logger.error(f"Gemini generation failed unexpectedly: {e}", exc_info=True)
            # Example specific error check:
            # if isinstance(e, google.api_core.exceptions.PermissionDenied):
            #     return "[ERROR: Gemini API key invalid or lacks permissions]"
            return f"[ERROR: Gemini generation failed - {type(e).__name__}]"

    def get_status_dict(self) -> Dict[str, Any]:
         return {
             "active_model": self.model_name,
             # Can add more details if needed, e.g., check API reachability
         }

    async def unload(self):
        # No explicit unload/cleanup needed for the genai library typically
        logger.info("GeminiBackend unload called (no action needed).")
        pass # Nothing specific to unload

```
*   Conditionally imports `google.generativeai`.
*   `__init__`: Takes API key, model name. Calls `genai.configure` and creates `genai.GenerativeModel`. Raises `ImportError` if library missing, `ValueError` on config fail.
*   `generate`: Maps our config to `genai.GenerationConfig`. Calls `model.generate_content_async`. Extracts `response.text`. Includes basic error handling and checks if the response was empty/blocked.
*   `get_status_dict`: Returns model name.
*   `unload`: Does nothing, as there's no persistent connection to close like `httpx`.

**4. Update LLM Service (`services/llm_service.py`)**

Add the `GeminiBackend` to the imports and the `set_active_backend` logic.

```python
# services/llm_service.py
# ... (imports) ...
from .llm_backends import (
    LLMBackendBase, LocalTransformersBackend, OllamaBackend,
    VLLMBackend, InstructLabBackend, GeminiBackend # <-- ADD GeminiBackend
)

# ... (LLMStatus, llm_state, helper functions, _unload_current_backend, _load_local_model_task) ...

# --- Public Service Functions ---
# ... (get_llm_status, update_llm_config - unchanged) ...

# --- Generate Text (Ensure GeminiBackend is checked) ---
async def generate_text(prompt: str) -> Optional[str]:
    # ... (get backend, status check) ...
    config = llm_state["config"]

    try:
        if isinstance(backend, LocalTransformersBackend):
            # ... (run local in executor) ...
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, backend.generate, prompt, config)
        # --- ADD GeminiBackend to the async call list ---
        elif isinstance(backend, (OllamaBackend, VLLMBackend, InstructLabBackend, GeminiBackend)):
            logger.info(f"Running API generation via {llm_state['backend_type']} backend...")
            result = await backend.generate(prompt, config)
        # --- END ADD ---
        else:
             logger.error(f"Unknown backend type {type(backend)} cannot generate.")
             result = None
        return result
    except Exception as e:
        # ... (error handling) ...
        return None


# --- Function to initiate backend loading/configuration ---
async def set_active_backend(
    backend_type: str,
    model_name_or_path: str,
    device: Optional[str] = None,
    quantization: Optional[str] = None
):
    # ... (logging, unload previous backend) ...

    # 2. Create and configure the new backend instance
    new_backend: Optional[LLMBackendBase] = None
    llm_state["status"] = LLMStatus.CONFIGURING # Default for API backends
    llm_state["backend_type"] = backend_type
    llm_state["active_model"] = model_name_or_path # Tentative
    llm_state["last_error"] = None

    try:
        if backend_type == "local":
            # ... (local backend loading logic - unchanged) ...
            return # Return early for local background load

        elif backend_type == "ollama":
            # ... (ollama logic - unchanged) ...
             if not settings.OLLAMA_BASE_URL: raise ValueError(...)
             new_backend = OllamaBackend(settings.OLLAMA_BASE_URL, model_name_or_path)

        elif backend_type == "vllm":
             # ... (vllm logic - unchanged) ...
              if not settings.VLLM_BASE_URL: raise ValueError(...)
              new_backend = VLLMBackend(settings.VLLM_BASE_URL, model_name_or_path)

        elif backend_type == "instructlab":
            # ... (instructlab logic - unchanged) ...
             if not settings.INSTRUCTLAB_BASE_URL: raise ValueError(...)
             new_backend = InstructLabBackend(settings.INSTRUCTLAB_BASE_URL, model_name_or_path)

        # --- ADD Gemini Backend Logic ---
        elif backend_type == "gemini":
             if not settings.GOOGLE_API_KEY:
                 raise ValueError("GOOGLE_API_KEY not configured in settings/environment.")
             # Initialization happens here, might raise error if API key invalid etc.
             new_backend = GeminiBackend(settings.GOOGLE_API_KEY, model_name_or_path)
        # --- END ADD ---

        else:
            raise ValueError(f"Unsupported LLM_BACKEND_TYPE: {backend_type}")

        # --- Update state for successfully configured API backend ---
        llm_state["backend_instance"] = new_backend
        llm_state["active_model"] = model_name_or_path
        llm_state["status"] = LLMStatus.READY # API backends are ready once instance created
        logger.info(f"Backend '{backend_type}' configured and ready with model '{model_name_or_path}'.")

    except Exception as e:
        # ... (Error handling - unchanged) ...

```
*   Imports `GeminiBackend`.
*   Adds check for `GeminiBackend` in `generate_text` to use `await`.
*   Adds `elif backend_type == "gemini":` block in `set_active_backend` which checks for the API key and instantiates `GeminiBackend`.

**5. Update API Model (`app/api/models/llm.py`)**

Add `"gemini"` to the `Literal` type for `backend_type` in `ModelLoadRequest`.

```python
# app/api/models/llm.py
# ... (imports: Literal) ...

class ModelLoadRequest(BaseModel):
    backend_type: Optional[Literal["local", "ollama", "vllm", "instructlab", "gemini"]] = Field(...) # Added "gemini"
    # ... (rest of ModelLoadRequest) ...

# ... (rest of the file) ...
```

**6. Update `requirements.txt`**

Add the new library:
```txt
# requirements.txt
# ... (other requirements) ...
google-generativeai
```
Run `pip install -r requirements.txt` again.

**7. Update `.env` Example**

Add the `GOOGLE_API_KEY` variable:
```.env
# Server Configuration
# LLM_BACKEND_TYPE=gemini # Example: Set default to Gemini
DEFAULT_MODEL_NAME_OR_PATH=gemini-1.5-flash-latest # Example Gemini model

# API Keys / URLs
# GOOGLE_API_KEY="AIz...YourApiKeyHere" # Uncomment and set if using Gemini
# OLLAMA_BASE_URL=http://localhost:11434
# VLLM_BASE_URL=http://localhost:8000

# Optional: Hugging Face Hub Token (for local backend downloads)
# HUGGING_FACE_HUB_TOKEN="hf_YourAccessTokenHere"
```

**Testing:**

1.  Add your Google API Key to your `.env` file or set it as an environment variable (`export GOOGLE_API_KEY=...`).
2.  Restart the Uvicorn server.
3.  Use `POST /api/v1/models/load` with the following body:
    ```json
    {
      "backend_type": "gemini",
      "model_name_or_path": "gemini-1.5-flash-latest" // Or gemini-1.5-pro-latest
    }
    ```
4.  Check `GET /api/v1/models/status`. It should show `backend_type: "gemini"`, `active_model: "gemini-1.5-flash-latest"`, `status: "Ready"`.
5.  Create a chat session and send a message.
6.  Verify the logs show the `GeminiBackend` being used and check the response quality.

You can now select Google Gemini as a backend for your RAG server!
