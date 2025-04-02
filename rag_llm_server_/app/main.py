# app/main.py
import uvicorn
from fastapi import FastAPI
import datetime
import sys # Make sure sys is imported
import os

from huggingface_hub import get_token, login # Import get_token

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"Project root added to sys.path: {project_root}")
# print(f"Current sys.path: {sys.path}") # Optional: Uncomment for more detail

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

# Continue with other imports
from core.config import settings
from db.database import connect_db, disconnect_db, create_metadata_tables, get_or_create_collections
from services.embedding_service import get_embedding_model

from services.system_service import shutdown_nvml
from services.llm_service import _unload_current_backend # Import unload function

# --- Core Application Setup ---
print("Creating FastAPI app instance...")
app = FastAPI(
    title=settings.APP_NAME,
    description="API server for managing RAG documents, interacting with LLMs, and managing chat sessions.",
    version="0.1.0",
)
print("FastAPI app instance created.")

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

    await connect_db()
    print("--> Pre-loading embedding model...")
    try:
        get_embedding_model()
        print("--> Embedding model loading initiated.")
    except Exception as e:
        print(f"--> ERROR: Failed to pre-load embedding model: {e}")
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