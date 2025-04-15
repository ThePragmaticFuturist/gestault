# app/main.py
import uvicorn
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

import datetime
import sys # Make sure sys is imported
import os

from huggingface_hub import get_token, login, logout # Import get_token

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"Project root added to sys.path: {project_root}")
# print(f"Current sys.path: {sys.path}") # Optional: Uncomment for more detail

# Continue with other imports
from core.config import settings
from db.database import connect_db, disconnect_db, create_metadata_tables, get_or_create_collections
from services.embedding_service import get_embedding_model

from services.system_service import shutdown_nvml
from services.llm_service import _unload_current_backend, set_active_backend # Import unload function

# --- Core Application Setup ---
print("Creating FastAPI app instance...")
app = FastAPI(
    title=settings.APP_NAME,
    description="API server for managing RAG documents, interacting with LLMs, and managing chat sessions. Created by Ken Hubbell, The Pragmatic Futurist LLC.",
    version="0.1.0",
)
print("FastAPI app instance created.")

# --- ADD CORS MIDDLEWARE CONFIGURATION ---
# DefineAh allowed origins (adjust for development/production)
# For development, allowing localhost ports is common.
# For production, list your specific frontend domain(s).
origins = [
    "http://localhost",         # Allow requests from base localhost
    "http://localhost:5173",    # Default Vite dev port
    "http://127.0.0.1:5173",    # Alternative localhost address
    # Add your deployed frontend URL here for production, e.g.:
    # "https://your-frontend-domain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # List of origins allowed
    allow_credentials=True, # Allow cookies (if, the classic CORS error! This is a standard security feature built into web browsers.
    # Same-Origin Policy: By default, web browsers restrict web pages (like your React app running on `http://localhost:5173 needed in the future)
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], # Explicitly allow POST and OPTIONS
    allow_headers=["Content-Type", "Authorization", "Accept", "Origin"], # Explicitly allow Content-Type and others
)
print(f"CORS Middleware enabled for origins: {origins}")

# --- Explicit Router Imports with Diagnostics ---
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

# --- End Explicit Router Imports ---

# --- Include Routers with Diagnostics ---
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
# --- End Include Routers ---


# --- Basic Root Endpoint ---
@app.get("/")
async def read_root():
    """ Root endpoint providing basic server information. """
    return {
        "message": f"Welcome to the {settings.APP_NAME}!",
        "status": "running",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

# --- Event Handlers ---
@app.on_event("startup")
async def startup_event():
    print("--> Event: startup")

    # --- Check Hugging Face Login ---
    print("--> Checking Hugging Face token...")
    auth_ok = False
    source = "Not found"
    try:
        # Check 1: Environment Variable / .env file via settings
        env_token = settings.HUGGING_FACE_HUB_TOKEN
        if env_token:
            print("--> Found Hugging Face token via environment variable/'.env'.")
            
            try:
                # Attempt login using the environment token
                # write_permission maybe needed depending on use case
                login(token=env_token, add_to_git_credential=False)
                print("--> Programmatic login successful using environment token.")
                auth_ok = True
                source = "Environment Variable (Programmatic Login)"
                login_attempted = True
            except Exception as login_err:
                print(f"--> WARNING: Programmatic login with env token failed: {login_err}")
                # Decide: fallback to checking file token or fail? Let's fallback for now.

            auth_ok = True
            source = "Environment Variable / .env"

        # Check 2: Cached Token File (only if env var not found)
        if not auth_ok:
            print("--> No token found via environment, checking cached login file...")
            file_token = get_token() # Checks ~/.cache/huggingface/token
            if file_token:
                print("--> Found Hugging Face token via cached login file ('huggingface-cli login').")
                
                try:
                    # Attempt login using the environment token
                    # write_permission maybe needed depending on use case
                    login(token=file_token, add_to_git_credential=False)
                    print("--> Programmatic login successful using environment token.")
                    auth_ok = True
                    source = "Cached Variable (Programmatic Login)"
                    login_attempted = True
                except Exception as login_err:
                    print(f"--> WARNING: Cached login with env token failed: {login_err}")
                    # Decide: fallback to checking file token or fail? Let's fallback for now.

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

    await connect_db()
    print("--> Pre-loading embedding model...")
    try:
        get_embedding_model()
        print("--> Embedding model loading initiated.")
    except Exception as e:
        print(f"--> ERROR: Failed to pre-load embedding model: {e}")

    # --- 4. Load Default LLM Model ---
    print(f"--> Attempting to load default LLM backend/model on startup...")
    print(f"    Backend Type: {settings.LLM_BACKEND_TYPE}")
    print(f"    Model: {settings.DEFAULT_MODEL_NAME_OR_PATH}")
    try:
        # Call the service function to set the default backend and model
        # Use settings for default values, request overrides are None here
        await set_active_backend(
            backend_type=settings.LLM_BACKEND_TYPE,
            model_name_or_path=settings.DEFAULT_MODEL_NAME_OR_PATH,
            device=settings.LLM_DEVICE, # Pass configured default device (relevant for local)
            quantization=settings.DEFAULT_LLM_QUANTIZATION # Pass default quant (relevant for local)
        )
        # Note: For 'local' backend, this schedules the load. Status check is needed later.
        # For API backends, this configures it and sets status to READY if successful.
        print(f"--> Default LLM backend ({settings.LLM_BACKEND_TYPE}) setup initiated/scheduled.")
        # You could add a short sleep and status check here if you MUST wait for local load
        # await asyncio.sleep(1) # Example wait
        # status = get_llm_status()
        # print(f"--> Post-initiation LLM status: {status.get('status')}")

    except ValueError as ve: # Catch config errors from set_active_backend
        print(f"--> ERROR: Configuration error loading default LLM: {ve}")
        print("    Server will start, but no LLM will be active initially.")
    except Exception as e:
        print(f"--> ERROR: Unexpected error loading default LLM during startup: {e}", exc_info=True)
        print("    Server will start, but no LLM will be active initially.")

    print("--> Event: startup complete.")

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

    # --- Unload active LLM backend ---
    print("--> Unloading active LLM backend...")
    try:
        # Use the dedicated unload function from llm_service
        await _unload_current_backend()
        print("--> LLM backend unloaded.")
    except Exception as e:
        print(f"--> WARNING: Error during LLM backend unload: {e}")

    print("--> Event: shutdown complete.")

# --- Running the Server ---
if __name__ == "__main__":
    print("Starting server via __main__ block...")
    uvicorn.run(
        "app.main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=True,
        reload_dirs=[
            str(settings.BASE_DIR / 'app'),
            str(settings.BASE_DIR / 'core'),
            str(settings.BASE_DIR / 'db'),
            str(settings.BASE_DIR / 'services')
        ]
    )

print("--- End of main.py ---") # Should not be reached when run with uvicorn normally
