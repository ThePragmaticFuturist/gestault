# Solution: Add Startup Login Check #

Ensure the Hugging Face login (which stores a token locally) is done *before* the server potentially tries to load gated models. The best place is at the very beginning of the FastAPI startup event.

**1. Modify `app/main.py`:**

Add the check at the beginning of the `startup_event` function.

```python
# app/main.py
import uvicorn
from fastapi import FastAPI
import datetime
import sys # Import sys for exiting
import os

# --- ADD Hugging Face Hub import ---
from huggingface_hub import get_token, login # Import get_token
# --- END ADD ---

# Adjust path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"Project root added to sys.path: {project_root}")

# Import routers
# ... (document, sessions, models, system imports - keep diagnostic prints if desired) ...
print("-" * 20)
print("Attempting to import 'documents' endpoint module...")
try:
    from app.api.endpoints import documents
    print("SUCCESS: Imported 'documents' module.")
    print(f"  -> Type: {type(documents)}")
    print(f"  -> Has 'router' attribute: {hasattr(documents, 'router')}")
except ImportError as e:
    print(f"FAILED to import 'documents': {e}")
    documents = None # Ensure variable exists but is None
except Exception as e:
    print(f"FAILED to import 'documents' with unexpected error: {e}")
    documents = None

print("-" * 20)
print("Attempting to import 'sessions' endpoint module...")
try:
    from app.api.endpoints import sessions
    print("SUCCESS: Imported 'sessions' module.") # <<< **** WATCH FOR THIS ****
    print(f"  -> Type: {type(sessions)}")
    print(f"  -> Has 'router' attribute: {hasattr(sessions, 'router')}") # <<< **** AND THIS ****
except ImportError as e:
    print(f"FAILED to import 'sessions': {e}") # <<< **** OR THIS ****
    sessions = None
except Exception as e:
    print(f"FAILED to import 'sessions' with unexpected error: {e}")
    sessions = None

print("-" * 20)
print("Attempting to import 'models' endpoint module...")
try:
    from app.api.endpoints import models
    print("SUCCESS: Imported 'models' module.") # <<< **** WATCH FOR THIS ****
    print(f"  -> Type: {type(models)}")
    print(f"  -> Has 'router' attribute: {hasattr(models, 'router')}") # <<< **** AND THIS ****
except ImportError as e:
    print(f"FAILED to import 'models': {e}") # <<< **** OR THIS ****
    models = None
except Exception as e:
    print(f"FAILED to import 'models' with unexpected error: {e}")
    models = None

print("-" * 20)
print("Attempting to import 'system' endpoint module...")
try:
    from app.api.endpoints import system
    print("SUCCESS: Imported 'system' module.") # <<< **** WATCH FOR THIS ****
    print(f"  -> Type: {type(system)}")
    print(f"  -> Has 'router' attribute: {hasattr(system, 'router')}") # <<< **** AND THIS ****
except ImportError as e:
    print(f"FAILED to import 'system': {e}") # <<< **** OR THIS ****
    system = None
except Exception as e:
    print(f"FAILED to import 'system' with unexpected error: {e}")
    system = None

print("-" * 20)

# Continue with other imports
from core.config import settings
from db.database import connect_db, disconnect_db # Keep create/get calls out of imports if possible
from services.embedding_service import get_embedding_model # For pre-loading
from services.system_service import shutdown_nvml # For shutdown
from services.llm_service import _unload_current_backend # For shutdown

# --- Core Application Setup ---
# (Run DB/Chroma setup *before* FastAPI app instance if they are needed immediately)
# This ensures tables/collections exist before startup event tries to connect/use them.
try:
    print("--- Initializing database schema and collections ---")
    # Import creation functions here to run them early
    from db.database import create_metadata_tables, get_or_create_collections
    create_metadata_tables()
    get_or_create_collections()
    print("--- Database schema and collections initialization complete ---")
except Exception as e:
    print(f"FATAL: Failed to initialize database/ChromaDB structures: {e}", file=sys.stderr)
    sys.exit(1)


print("Creating FastAPI app instance...")
app = FastAPI(
    title=settings.APP_NAME,
    description="API server for managing RAG documents, interacting with LLMs, and managing chat sessions.",
    version="0.1.0",
)
print("FastAPI app instance created.")


# --- Event Handlers ---
@app.on_event("startup")
async def startup_event():
    print("--- Running Server Startup ---")

    # --- Check Hugging Face Login ---
    print("--> Checking Hugging Face token...")
    try:
        hf_token = get_token()
        if hf_token is None:
            print("\n" + "="*60)
            print("ERROR: Hugging Face token not found.")
            print("Please log in using the Hugging Face CLI before starting the server:")
            print("  huggingface-cli login")
            print("You may need read/write access depending on the models.")
            print("Server startup aborted.")
            print("="*60 + "\n")
            sys.exit(1) # Exit with error code
        else:
            # Optionally, you could add a whoami() call here to verify the token is valid
            # try:
            #     from huggingface_hub import whoami
            #     user_info = whoami()
            #     print(f"--> Hugging Face token found and verified for user: {user_info.get('name')}")
            # except Exception as auth_err:
            #     print(f"\n{'='*60}")
            #     print(f"ERROR: Found Hugging Face token but failed to verify: {auth_err}")
            #     print("The token might be invalid or expired. Try logging in again:")
            #     print("  huggingface-cli login")
            #     print(f"{'='*60}\n")
            #     sys.exit(1)
            print("--> Hugging Face token found. Login appears successful.")

    except ImportError:
         print("\n" + "="*60)
         print("WARNING: `huggingface_hub` library not found.")
         print("Cannot check login status. Loading gated models will likely fail.")
         print("Please install it: pip install huggingface_hub")
         print("="*60 + "\n")
         # Decide whether to exit or just warn
         # sys.exit(1) # Exit if login is strictly required
    except Exception as e:
         print(f"\n{'='*60}")
         print(f"ERROR: An unexpected error occurred checking Hugging Face token: {e}")
         print("Server startup aborted.")
         print(f"{'='*60}\n")
         sys.exit(1)
    # --- End Check ---


    # --- Original Startup Steps ---
    print("--> Connecting to database...")
    await connect_db()

    print("--> Pre-loading embedding model...")
    try:
        get_embedding_model() # Trigger embedding model load
        print("--> Embedding model loading initiated (check logs for completion).")
    except Exception as e:
        print(f"--> ERROR: Failed to pre-load embedding model during startup: {e}")

    print("--> Event: startup complete.")


@app.on_event("shutdown")
async def shutdown_event():
    """Disconnect DB, shutdown NVML, unload LLM backend."""
    # ... (shutdown logic remains the same) ...
    print("--> Event: shutdown")
    await disconnect_db()
    print("--> Shutting down NVML...")
    try:
         shutdown_nvml()
    except Exception as e:
         print(f"--> WARNING: Error during NVML shutdown: {e}")
    print("--> Unloading active LLM backend...")
    try:
        await _unload_current_backend()
        print("--> LLM backend unloaded.")
    except Exception as e:
        print(f"--> WARNING: Error during LLM backend unload: {e}")
    print("--> Event: shutdown complete.")


# --- Basic Root Endpoint ---
# ... (root endpoint) ...

# --- Include API Routers ---
# ... (include documents, sessions, models, system routers - use the diagnostic versions if helpful) ...
print("-" * 20)
if documents and hasattr(documents, 'router'):
    print("Including 'documents' router...")
    app.include_router(
        documents.router,
        prefix="/api/v1/documents",
        tags=["Documents"]
    )
    print("SUCCESS: Included 'documents' router.")
else:
    print("SKIPPING 'documents' router inclusion (module or router attribute not found).")

print("-" * 20)
if sessions and hasattr(sessions, 'router'): # Check module and attribute exist
    print("Including 'sessions' router...") # <<< **** WATCH FOR THIS ****
    app.include_router(
        sessions.router,
        prefix="/api/v1/sessions",
        tags=["Chat Sessions"]
    )
    print("SUCCESS: Included 'sessions' router.")
else:
    print("SKIPPING 'sessions' router inclusion (module or router attribute not found).") # <<< **** OR THIS ****

print("-" * 20)
if models and hasattr(models, 'router'): # Check module and attribute exist
    print("Including 'models' router...") # <<< **** WATCH FOR THIS ****
    app.include_router(
        models.router,
        prefix="/api/v1/models",
        tags=["LLM Management"]
    )
    print("SUCCESS: Included 'models' router.")
else:
    print("SKIPPING 'models' router inclusion (module or router attribute not found).") # <<< **** OR THIS ****

print("-" * 20)
if system and hasattr(system, 'router'): # Check module and attribute exist
    print("Including 'system' router...") # <<< **** WATCH FOR THIS ****
    app.include_router(
        system.router,
        prefix="/api/v1/system",
        tags=["System Status"]
    )
    print("SUCCESS: Included 'system' router.")
else:
    print("SKIPPING 'system' router inclusion (module or router attribute not found).") # <<< **** OR THIS ****
print("-" * 20)


# --- Running the Server ---
if __name__ == "__main__":
    # ... (uvicorn.run call remains the same) ...
    print("Starting server via __main__ block...")
    uvicorn.run(
        "app.main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=True, # Reload might interfere slightly with sys.exit, but okay for dev
        reload_dirs=[
            str(settings.BASE_DIR / 'app'),
            str(settings.BASE_DIR / 'core'),
            str(settings.BASE_DIR / 'db'),
            str(settings.BASE_DIR / 'services')
        ]
    )

print("--- End of main.py ---") # Should not be reached when run with uvicorn normally
```

**Explanation:**

1.  **Import `get_token` and `sys`:** Added necessary imports.
2.  **Check in `startup_event`:** Placed the check right at the beginning of the startup function.
3.  **Call `get_token()`:** Retrieves the locally stored token (if any).
4.  **Check if `None`:** If `get_token()` returns `None`, it means the user isn't logged in via `huggingface-cli login`.
5.  **Error Message & Exit:** If `None`, print a clear error explaining the problem and how to fix it (`huggingface-cli login`). Crucially, `sys.exit(1)` is called to terminate the server startup.
6.  **Success Message:** If a token is found, print a confirmation. (Added optional `whoami` check for robustness).
7.  **Error Handling:** Added `try...except` around the check to catch potential `ImportError` (if `huggingface_hub` isn't installed) or other unexpected issues.

**How to Use:**

1.  **Log Out (Optional for Testing):** If you are currently logged in, you can log out using `huggingface-cli logout`.
2.  **Run the Server:** Start the server `uvicorn app.main:app ...`.
3.  **Observe Startup:**
    *   If you are **logged out**, the server should print the error message about needing to log in and then exit. It won't proceed to connect to the database or load models.
    *   If you are **logged in**, the server should print "Hugging Face token found..." and continue the startup process as normal.
4.  **Log In:** If prompted, run `huggingface-cli login` in your terminal and follow the instructions (usually involves pasting a token from the Hugging Face website).
5.  **Restart Server:** Start the server again. It should now pass the check.
6.  **Load Llama 3 (Correctly):** Use the *correct* model identifier and ensure you've accepted the license terms on the Hugging Face website for that model. Example request to load the 8B instruct model using the `local` backend:
    ```json
    {
      "model_name_or_path": "meta-llama/Meta-Llama-3-8B-Instruct"
    }
    ```

This check ensures that the necessary authentication prerequisite is met before the server fully starts, preventing confusing errors later when trying to access gated models.

####################################

***pull token from .env file***

Okay, yes, we can definitely modify the server to read the Hugging Face token from the `.env` file instead of solely relying on the `huggingface-cli login` cache. This is a common and often preferred method for server deployments or CI/CD environments.

The `huggingface_hub` and `transformers` libraries are actually designed to **automatically look for the token in the `HUGGING_FACE_HUB_TOKEN` environment variable** if they can't find one in the default cache location (`~/.cache/huggingface/token`) or if one isn't passed explicitly.

So, the main steps are:

1.  Add `HUGGING_FACE_HUB_TOKEN` as an optional setting in `core/config.py` so `pydantic-settings` loads it from `.env`.
2.  Modify the startup check in `app/main.py` to verify that *either* `get_token()` finds a file token *or* the `HUGGING_FACE_HUB_TOKEN` environment variable is set.
3.  Update `.gitignore` and provide instructions for the `.env` file.

**Step 1: Modify `core/config.py`**

Add the `HUGGING_FACE_HUB_TOKEN` field to your `Settings` class.

```python
# core/config.py
import os
import torch
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
import tempfile
from typing import Optional, List, Any, Literal

# ... (get_default_device function) ...

class Settings(BaseSettings):
    # ... (General, DB, Embedding settings) ...

    # --- LLM Settings ---
    LLM_BACKEND_TYPE: Literal["local", "ollama", "vllm", "instructlab"] = Field(default="local", ...)
    OLLAMA_BASE_URL: Optional[str] = Field(default="http://localhost:11434", ...)
    VLLM_BASE_URL: Optional[str] = Field(default=None, ...)
    INSTRUCTLAB_BASE_URL: Optional[str] = Field(default=None, ...)
    DEFAULT_MODEL_NAME_OR_PATH: str = Field(default="gpt2", ...)

    # --- NEW: Optional HF Token from Environment ---
    HUGGING_FACE_HUB_TOKEN: Optional[str] = Field(
        default=None,
        description="Optional Hugging Face Hub token. If set, overrides login via CLI. Loaded from .env or environment variable."
    )
    # ---------------------------------------------

    LOCAL_MODELS_DIR: Path = BASE_DIR / "local_models"
    LLM_DEVICE: str = Field(default_factory=get_default_device)
    DEFAULT_LLM_QUANTIZATION: Optional[str] = Field(default=None)

    DEFAULT_LLM_MAX_NEW_TOKENS: int = Field(default=512)
    DEFAULT_LLM_TEMPERATURE: float = Field(default=0.7)
    DEFAULT_LLM_TOP_P: float = Field(default=0.9)
    DEFAULT_LLM_TOP_K: int = Field(default=50)

    # ... (Other settings) ...

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore'

settings = Settings()

# ... (Ensure directories exist) ...
# ... (Print config) ...
# Optionally print if token is found via environment
# Be careful NOT to print the token itself!
# print(f"Hugging Face Token via Env: {'Set' if settings.HUGGING_FACE_HUB_TOKEN else 'Not Set'}")
print(f"--------------------------")
```

**Step 2: Modify Startup Check in `app/main.py`**

Update the `startup_event` to check both `get_token()` and the environment variable (accessed via `settings`).

```python
# app/main.py
import uvicorn
from fastapi import FastAPI
import datetime
import sys
import os

# Import get_token (still useful for checking file)
from huggingface_hub import get_token

# ... (other imports: routers, config, db, services) ...
from core.config import settings # Import settings object

# ... (path setup, router imports, diagnostic prints) ...
# ... (Database/Chroma initialization) ...
# ... (FastAPI app = FastAPI(...) definition) ...

# --- Event Handlers ---
@app.on_event("startup")
async def startup_event():
    print("--- Running Server Startup ---")

    # --- Check Hugging Face Login / Token ---
    print("--> Checking Hugging Face token...")
    auth_ok = False
    source = "Not found"
    try:
        # Check 1: Environment Variable / .env file via settings
        env_token = settings.HUGGING_FACE_HUB_TOKEN
        if env_token:
            print("--> Found Hugging Face token via environment variable/'.env'.")
            # Optional: Add verification using the env token if needed
            # from huggingface_hub import HfApi
            # try:
            #     api = HfApi(token=env_token)
            #     user_info = api.whoami()
            #     print(f"--> Token verified for user: {user_info.get('name')}")
            #     auth_ok = True
            #     source = "Environment Variable"
            # except Exception as auth_err:
            #      print(f"--> WARNING: Token found via env var but failed verification: {auth_err}")
            #      # Decide if verification failure is critical
            # Let's assume token presence is enough for now
            auth_ok = True
            source = "Environment Variable / .env"

        # Check 2: Cached Token File (only if env var not found)
        if not auth_ok:
            print("--> No token found via environment, checking cached login file...")
            file_token = get_token() # Checks ~/.cache/huggingface/token
            if file_token:
                print("--> Found Hugging Face token via cached login file ('huggingface-cli login').")
                # Verification can be done implicitly when libraries use it
                auth_ok = True
                source = "Cached Login File"

        # If neither method found a token
        if not auth_ok:
            print("\n" + "="*60)
            print("ERROR: Hugging Face token not found.")
            print("Please provide a token using one of these methods:")
            print("  1. Log in using the Hugging Face CLI: `huggingface-cli login`")
            print("  2. Set the HUGGING_FACE_HUB_TOKEN in your environment variables or .env file.")
            print("You may need read/write access depending on the models.")
            print("Server startup aborted.")
            print("="*60 + "\n")
            sys.exit(1) # Exit with error code
        else:
             print(f"--> Hugging Face token found successfully (Source: {source}).")

    except ImportError:
         print("\n" + "="*60)
         print("WARNING: `huggingface_hub` library not found.")
         print("Cannot check token status. Loading gated models will likely fail.")
         print("Please install it: pip install huggingface_hub")
         print("="*60 + "\n")
         # sys.exit(1) # Optionally exit
    except Exception as e:
         print(f"\n{'='*60}")
         print(f"ERROR: An unexpected error occurred checking Hugging Face token: {e}")
         print("Server startup aborted.")
         print(f"{'='*60}\n")
         sys.exit(1)
    # --- End Check ---

    # --- Original Startup Steps ---
    # ... (connect_db, load embedding model) ...
    print("--> Event: startup complete.")

# ... (@app.on_event("shutdown") remains the same) ...
# ... (Root endpoint remains the same) ...
# ... (Include routers remains the same) ...
# ... (if __name__ == "__main__": remains the same) ...
```

**Explanation of Changes in `startup_event`:**

1.  Imports `settings`.
2.  First checks `settings.HUGGING_FACE_HUB_TOKEN` (which would have loaded the value from the environment variable or `.env`).
3.  If the environment token is found, `auth_ok` is set to `True`. (Optional verification commented out for simplicity).
4.  If *no* environment token is found, *then* it checks the cached file using `get_token()`.
5.  If either method succeeds, it prints a success message indicating the source.
6.  If *neither* method finds a token, it prints the error explaining *both* methods (CLI login or `.env`/environment variable) and exits.

**Step 3: Update `.gitignore` and `.env` Example**

*   **`.gitignore`:** Ensure `.env` is listed. Add a comment if desired.
    ```gitignore
    # ... other ignores ...
    .env # Ignore environment variables file (contains secrets!)
    # ...
    ```
*   **`.env` Example:** Add the `HUGGING_FACE_HUB_TOKEN` variable (commented out by default).
    ```.env
    # Server Configuration
    LLM_BACKEND_TYPE=local # Or ollama, vllm, etc.
    DEFAULT_MODEL_NAME_OR_PATH=gpt2 # Or meta-llama/Meta-Llama-3-8B-Instruct, etc.

    # API Backend URLs (Uncomment and set if using these backend types)
    # OLLAMA_BASE_URL=http://localhost:11434
    # VLLM_BASE_URL=http://localhost:8000
    # INSTRUCTLAB_BASE_URL=http://localhost:XXXX

    # Optional: Hugging Face Hub Token
    # Provide this if you need to access private/gated models and are NOT using 'huggingface-cli login'.
    # Get your token from https://huggingface.co/settings/tokens
    # HUGGING_FACE_HUB_TOKEN="hf_YourAccessTokenHere"

    # Other settings (Database paths, Embedding model, etc. usually derived in config.py)
    # SERVER_PORT=8001 # Example override
    ```

**How it Works Now:**

*   If you define `HUGGING_FACE_HUB_TOKEN` in your `.env` file or as an environment variable, `settings.HUGGING_FACE_HUB_TOKEN` will have that value. The startup check will find it and proceed. The `transformers`/`huggingface_hub` libraries will automatically pick up this environment variable for authentication when downloading models.
*   If `HUGGING_FACE_HUB_TOKEN` is *not* set in the environment/`.env`, the startup check will then look for the token file created by `huggingface-cli login`. If found, it proceeds. The libraries will use the file token.
*   If neither is found, the server startup fails with instructions.

This provides a flexible way to manage the token for both local development (`huggingface-cli login`) and server deployments (`.env`/environment variable).
