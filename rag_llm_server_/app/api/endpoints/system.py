# app/api/endpoints/system.py
import logging
from fastapi import APIRouter, HTTPException

# Import models and service functions
from app.api.models.system import SystemStatusResponse
from services.system_service import get_full_system_status

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get(
    "/status",
    response_model=SystemStatusResponse,
    summary="Get current server system status",
)
async def get_system_status_endpoint():
    """
    Retrieves current system status including CPU, RAM, Disk,
    and GPU information (if available).
    """
    try:
        # Calls are synchronous but generally fast enough for status checks.
        # For very high frequency polling, consider running in executor.
        status_data = get_full_system_status()
        # Use Pydantic's parse_obj for validation before returning
        return SystemStatusResponse.parse_obj(status_data)
    except Exception as e:
        logger.error(f"Error retrieving system status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve system status.")