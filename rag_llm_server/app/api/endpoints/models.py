# app/api/endpoints/models.py
import logging
import asyncio # Needed for run_in_executor
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks, status

# Import models and service functions
# Import models and service functions
from app.api.models.llm import (
    ModelLoadRequest, 
    ModelStatusResponse, 
    ModelUpdateRequest, 
    AvailableModelsResponse
)
# Import new service function and updated status getter
from services.llm_service import (
    list_local_models, 
    list_cached_hub_models,
    set_active_backend, # <-- NEW import
    get_llm_status, 
    update_llm_config,
    llm_state, 
    LLMStatus # Keep state/enum for checking status
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

@router.post(
    "/load",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Set active LLM backend and model",
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

    # Use requested type, otherwise fall back to server's configured default type
    backend_type = load_request.backend_type or settings.LLM_BACKEND_TYPE
    # Use requested model, otherwise fall back to server's configured default model
    model_id = load_request.model_name_or_path or settings.DEFAULT_MODEL_NAME_OR_PATH

    # Ensure a valid model ID is determined
    if not model_id:
         raise HTTPException(
            status_code=400,
            detail="No model_name_or_path provided in request and no DEFAULT_MODEL_NAME_OR_PATH configured on server."
        )
         
    device = load_request.device # Pass along if provided (for local)
    quantization = load_request.quantization # Pass along if provided (for local)

    logger.info(f"Request received to activate model '{model_id}' on effective backend '{backend_type}'.")
    logger.debug(f" (Source: Request Type='{load_request.backend_type}', Server Default Type='{settings.LLM_BACKEND_TYPE}')")
    logger.debug(f" (Source: Request Model='{load_request.model_name_or_path}', Server Default Model='{settings.DEFAULT_MODEL_NAME_OR_PATH}')")

    # Use the service function to handle the logic
    # This function is async because it might await the unload of the previous backend
    try:
        await set_active_backend(
            backend_type=backend_type,
            model_name_or_path=model_id,
            device=device, # Only used if backend_type is 'local'
            quantization=quantization # Only used if backend_type is 'local'
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
    if status["status"] != LLMStatus.READY.value:
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
    if llm_state["status"] != LLMStatus.READY:
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
