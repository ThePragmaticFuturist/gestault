# Step 9: System Status API #

Okay, let's implement the System Status API. We'll use the `psutil` library, which is excellent for gathering cross-platform system information, and optionally `pynvml` for detailed NVIDIA GPU stats if available.
   
[Reasoning behind the code](reasoning/r9.md)
   
1.  **Install Dependencies:**
    ```bash
    pip install psutil pynvml
    ```
    *   `psutil`: For CPU, RAM, Disk, Temp (platform-dependent), etc.
    *   `pynvml`: For NVIDIA GPU details (will work only if NVIDIA drivers and NVML library are installed).

2.  **Define API Models (`app/api/models/system.py`):**
    Create a new file for system status models.

    ```python
    # app/api/models/__init__.py
    # (Ensure this file exists)
    ```

    ```python
    # app/api/models/system.py
    from pydantic import BaseModel, Field
    from typing import List, Optional, Dict, Any

    # --- Helper function for GB conversion ---
    def bytes_to_gb(num_bytes: Optional[int]) -> Optional[float]:
        if num_bytes is None:
            return None
        return round(num_bytes / (1024 ** 3), 2)

    class CPUStatus(BaseModel):
        usage_percent: float = Field(..., description="Current overall CPU utilization percentage.")
        core_count_logical: int = Field(..., description="Number of logical CPU cores (including hyperthreading).")
        core_count_physical: int = Field(..., description="Number of physical CPU cores.")
        current_frequency_mhz: Optional[float] = Field(None, description="Current CPU frequency in MHz.")
        max_frequency_mhz: Optional[float] = Field(None, description="Maximum CPU frequency in MHz.")

    class MemoryStatus(BaseModel):
        total_gb: Optional[float] = Field(..., description="Total physical RAM in GB.")
        available_gb: Optional[float] = Field(..., description="Available RAM in GB.")
        used_gb: Optional[float] = Field(..., description="Used RAM in GB.")
        used_percent: float = Field(..., description="RAM usage percentage.")

    class DiskUsage(BaseModel):
        path: str = Field(..., description="Mount point of the disk partition.")
        total_gb: Optional[float] = Field(..., description="Total space on the partition in GB.")
        used_gb: Optional[float] = Field(..., description="Used space on the partition in GB.")
        free_gb: Optional[float] = Field(..., description="Free space on the partition in GB.")
        used_percent: float = Field(..., description="Disk usage percentage for the partition.")

    class GPUStatus(BaseModel):
        id: int = Field(..., description="GPU index.")
        name: str = Field(..., description="GPU model name.")
        driver_version: Optional[str] = Field(None, description="NVIDIA driver version.")
        memory_total_mb: int = Field(..., description="Total GPU memory in MB.")
        memory_used_mb: int = Field(..., description="Used GPU memory in MB.")
        memory_free_mb: int = Field(..., description="Free GPU memory in MB.")
        utilization_gpu_percent: Optional[int] = Field(None, description="GPU utilization percentage.")
        temperature_celsius: Optional[int] = Field(None, description="GPU temperature in Celsius.")

    class SystemStatusResponse(BaseModel):
        cpu: CPUStatus
        memory: MemoryStatus
        disks: List[DiskUsage]
        gpus: Optional[List[GPUStatus]] = Field(None, description="List of NVIDIA GPU statuses (if available).")
        temperatures_celsius: Optional[Dict[str, Any]] = Field(None, description="Sensor temperatures (platform dependent).")

    ```
    *   Defines structures for CPU, Memory, Disk (list), GPU (list, optional), and the overall system status.
    *   Includes a helper to convert bytes to GB for readability.

3.  **Create System Service (`services/system_service.py`):**
    This service gathers the data.

    ```python
    # services/system_service.py
    import logging
    import shutil # For disk usage
    from typing import Dict, Any, Optional, List

    import psutil

    # Attempt to import pynvml, but don't fail if it's not installed or unusable
    try:
        import pynvml
        pynvml_available = True
        try:
            pynvml.nvmlInit()
            logger.info("pynvml initialized successfully.")
            # Optionally get driver version once
            # driver_version = pynvml.nvmlSystemGetDriverVersion()
        except pynvml.NVMLError as e:
            logger.warning(f"pynvml found but failed to initialize: {e}. GPU monitoring disabled.")
            pynvml_available = False
    except ImportError:
        logger.info("pynvml not found. NVIDIA GPU monitoring will be unavailable.")
        pynvml_available = False
    except Exception as e:
        logger.warning(f"An unexpected error occurred during pynvml import/init: {e}")
        pynvml_available = False

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # --- Helper function for GB conversion ---
    def bytes_to_gb(num_bytes: Optional[int]) -> Optional[float]:
        if num_bytes is None:
            return None
        return round(num_bytes / (1024 ** 3), 2)

    def bytes_to_mb(num_bytes: Optional[int]) -> Optional[int]:
         if num_bytes is None:
            return None
         return round(num_bytes / (1024 ** 2))


    # --- Core Data Gathering Functions ---

    def get_cpu_status() -> Dict[str, Any]:
        """Gathers CPU information using psutil."""
        usage = psutil.cpu_percent(interval=0.1) # Short interval for responsiveness
        logical_cores = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)

        freq = psutil.cpu_freq()
        current_freq = getattr(freq, 'current', None)
        max_freq = getattr(freq, 'max', None)

        return {
            "usage_percent": usage,
            "core_count_logical": logical_cores,
            "core_count_physical": physical_cores,
            "current_frequency_mhz": current_freq,
            "max_frequency_mhz": max_freq
        }

    def get_memory_status() -> Dict[str, Any]:
        """Gathers RAM information using psutil."""
        mem = psutil.virtual_memory()
        return {
            "total_gb": bytes_to_gb(mem.total),
            "available_gb": bytes_to_gb(mem.available),
            "used_gb": bytes_to_gb(mem.used),
            "used_percent": mem.percent
        }

    def get_disk_status() -> List[Dict[str, Any]]:
        """Gathers disk usage for relevant partitions using psutil."""
        disk_info = []
        # Define partitions you care about, or use psutil.disk_partitions() to get all
        # Example: Checking root ('/'), home ('/home'), and data ('/data') if they exist
        partitions_to_check = ['/']
        # Add more relevant paths if needed, e.g., where models or data are stored
        # partitions_to_check.append(str(settings.DATA_DIR.parent)) # Example: project root disk
        # partitions_to_check.append(str(settings.HUGGINGFACE_HUB_CACHE.parent)) # Cache disk

        # Get all partitions and check the relevant ones
        all_partitions = psutil.disk_partitions(all=False) # all=False avoids CD/floppy etc.
        checked_paths = set()

        for part in all_partitions:
             mountpoint = part.mountpoint
             # Check if this mountpoint covers any path we care about
             relevant = any(mountpoint == p or p.startswith(mountpoint + '/') for p in partitions_to_check)

             # Or simply include common/root partitions
             # relevant = mountpoint in ['/', '/home', '/var', '/data'] # Adjust as needed

             # Let's just check the root '/' for simplicity now
             if mountpoint == '/' and mountpoint not in checked_paths:
                 try:
                     usage = psutil.disk_usage(mountpoint)
                     disk_info.append({
                         "path": mountpoint,
                         "total_gb": bytes_to_gb(usage.total),
                         "used_gb": bytes_to_gb(usage.used),
                         "free_gb": bytes_to_gb(usage.free),
                         "used_percent": usage.percent
                     })
                     checked_paths.add(mountpoint)
                 except Exception as e:
                     logger.warning(f"Could not get disk usage for {mountpoint}: {e}")

        # You might want to explicitly add usage for specific directories like data/models
        # using shutil.disk_usage if they reside on different partitions not caught above.
        # Example:
        # try:
        #     data_usage = shutil.disk_usage(settings.DATA_DIR)
        #     # ... append if path not already checked ...
        # except FileNotFoundError:
        #     pass

        return disk_info

    def get_gpu_status() -> Optional[List[Dict[str, Any]]]:
        """Gathers NVIDIA GPU information using pynvml, if available."""
        if not pynvml_available:
            return None

        gpu_info = []
        try:
            num_devices = pynvml.nvmlDeviceGetCount()
            driver_version = pynvml.nvmlSystemGetDriverVersion() # Get driver version once

            for i in range(num_devices):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except pynvml.NVMLError_NotSupported:
                     temperature = None # Temp reading might not be supported

                gpu_info.append({
                    "id": i,
                    "name": name,
                    "driver_version": driver_version, # Return bytes as string
                    "memory_total_mb": bytes_to_mb(mem_info.total),
                    "memory_used_mb": bytes_to_mb(mem_info.used),
                    "memory_free_mb": bytes_to_mb(mem_info.free),
                    "utilization_gpu_percent": utilization.gpu,
                    "temperature_celsius": temperature
                })
        except pynvml.NVMLError as e:
            logger.error(f"Failed to get GPU info via pynvml: {e}")
            # May need re-initialization if error occurred?
            # For simplicity, just return None or partial info if preferred
            return None # Indicate GPU info unavailable due to error
        except Exception as e:
             logger.error(f"Unexpected error getting GPU info: {e}", exc_info=True)
             return None

        return gpu_info

    def get_temperature_status() -> Optional[Dict[str, Any]]:
        """Gathers sensor temperatures using psutil (platform dependent)."""
        temps = {}
        if hasattr(psutil, "sensors_temperatures"):
            try:
                all_temps = psutil.sensors_temperatures()
                if not all_temps:
                     return None # No sensors found/accessible

                # Process the temps dictionary (structure varies by OS)
                # Example: Select specific sensors like 'coretemp' on Linux
                for name, entries in all_temps.items():
                    temp_list = []
                    for entry in entries:
                        temp_list.append({
                            "label": entry.label or f"Sensor_{len(temp_list)}",
                            "current": entry.current,
                            "high": entry.high,
                            "critical": entry.critical,
                        })
                    if temp_list: # Only add if we found entries
                        temps[name] = temp_list

                return temps if temps else None # Return None if processing yielded nothing

            except Exception as e:
                logger.warning(f"Could not read sensor temperatures: {e}. This may require specific permissions or setup.")
                return None
        else:
            logger.info("psutil.sensors_temperatures not available on this platform.")
            return None

    def get_full_system_status() -> Dict[str, Any]:
        """Gathers all system status components."""
        status = {
            "cpu": get_cpu_status(),
            "memory": get_memory_status(),
            "disks": get_disk_status(),
            "gpus": get_gpu_status(),
            "temperatures_celsius": get_temperature_status()
        }
        return status

    # --- Optional: Shutdown NVML ---
    # Could be called during FastAPI shutdown event
    def shutdown_nvml():
        if pynvml_available:
            try:
                 pynvml.nvmlShutdown()
                 logger.info("pynvml shut down successfully.")
            except pynvml.NVMLError as e:
                 logger.warning(f"Error during pynvml shutdown: {e}")

    ```
    *   Imports `psutil` and conditionally imports `pynvml`, handling `ImportError` and initialization errors.
    *   Includes helper functions for byte conversion.
    *   `get_cpu_status`, `get_memory_status`, `get_disk_status`, `get_gpu_status`, `get_temperature_status`: Each function focuses on one component using `psutil` or `pynvml`.
    *   GPU/Temp functions handle potential errors or unavailability gracefully.
    *   `get_full_system_status`: Orchestrates calls to the individual functions.
    *   Added `shutdown_nvml` (we'll hook this into the FastAPI shutdown later).

4.  **Create System API Endpoints (`app/api/endpoints/system.py`):**

    ```python
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

    ```
    *   Simple router with one endpoint `/status`.
    *   Calls the `get_full_system_status` service function.
    *   Uses `parse_obj` to validate the collected data against the `SystemStatusResponse` model.

5.  **Integrate System Router and NVML Shutdown in `app/main.py`:**

    ```python
    # app/main.py
    # ... (imports) ...
    from app.api.endpoints import documents, sessions, models
    from app.api.endpoints import system # <--- ADDED IMPORT

    # Import service functions for shutdown
    from services.system_service import shutdown_nvml
    from services.llm_service import llm_state, LLMStatus # For potential LLM unload on shutdown

    # ... (FastAPI app setup) ...

    # --- Event Handlers ---
    @app.on_event("startup")
    async def startup_event():
        # ... (existing startup logic: connect_db, load embedding model) ...
        print("--> Event: startup complete.")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Disconnect DB, shutdown NVML, potentially unload models."""
        print("--> Event: shutdown")
        await disconnect_db()

        # --- ADDED: Shutdown NVML ---
        print("--> Shutting down NVML...")
        try:
             # This call is synchronous, fine for shutdown
             shutdown_nvml()
        except Exception as e:
             print(f"--> WARNING: Error during NVML shutdown: {e}")
        # --- END ADDED ---

        # --- Optional: Unload LLM model on shutdown? ---
        # If you want to release GPU memory explicitly on shutdown
        # if llm_state.get("status") == LLMStatus.LOADED:
        #     print("--> Unloading LLM model...")
        #     # Need an unload function or run parts of _load_model_task's unload logic
        #     # unload_llm_model() # Assume such a sync function exists
        # --- End Optional Unload ---

        print("--> Event: shutdown complete.")


    # --- Basic Root Endpoint ---
    # ... (root endpoint) ...

    # --- Include API Routers ---
    app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
    app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["Chat Sessions"])
    app.include_router(models.router, prefix="/api/v1/models", tags=["LLM Management"])
    # --- ADDED: Include system router ---
    app.include_router(
        system.router,
        prefix="/api/v1/system",
        tags=["System Status"] # New tag
    )
    # --- END ADDED ---

    # ... (if __name__ == "__main__": block) ...
    ```
    *   Import the `system` router and the `shutdown_nvml` function.
    *   Include the `system.router` with a prefix and tag.
    *   In the `shutdown` event handler, add a call to `shutdown_nvml()` within a try-except block.

6.  **Run and Test:**
    *   Restart the server: `uvicorn app.main:app --reload ...`
    *   Check the startup logs. You should see messages about `pynvml` initialization (or warnings if it failed/not found).
    *   Go to `/docs`. You should see the new "System Status" section.
    *   Execute `GET /api/v1/system/status`.
    *   Examine the response. It should contain CPU, Memory, Disk information.
    *   If you have an NVIDIA GPU and drivers installed correctly, the `gpus` field should be populated. Otherwise, it will likely be `null`.
    *   The `temperatures_celsius` field may or may not be populated depending on your OS, hardware sensors, and permissions.

**Summary:**

We have added a comprehensive System Status API:
*   Monitors CPU usage and frequency.
*   Monitors RAM usage.
*   Monitors Disk usage for key partitions.
*   Monitors NVIDIA GPU utilization, memory, and temperature (conditionally, using `pynvml`).
*   Attempts to monitor system temperatures (conditionally, using `psutil`).
*   Provides a single endpoint (`/api/v1/system/status`) to retrieve all metrics.
*   Handles potential library issues (e.g., `pynvml` not installed) gracefully.
*   Ensures NVML resources are cleaned up on server shutdown.

This completes the final major feature planned for our RAG LLM server! We now have capabilities for document management, RAG search, LLM interaction, chat session management, and system monitoring.
