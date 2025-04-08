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