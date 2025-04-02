# services/system_service.py
import logging
import shutil # For disk usage
from typing import Dict, Any, Optional, List

import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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