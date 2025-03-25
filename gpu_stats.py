#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility for collecting GPU statistics with CUDA 11.8 support.
"""

import subprocess
import json
import os
import sys
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger("gpu_stats")

def get_nvidia_smi_output() -> Optional[str]:
    """
    Run nvidia-smi command and return the output as a string.
    Returns None if nvidia-smi is not available or fails.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,power.draw", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.warning(f"Failed to run nvidia-smi: {e}")
        return None

def parse_nvidia_smi_output(output: str) -> List[Dict[str, Any]]:
    """
    Parse the output of nvidia-smi command into a list of dictionaries.
    Each dictionary contains information about a GPU.
    """
    if not output:
        return []
        
    gpus = []
    for line in output.strip().split("\n"):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 9:
            gpu_info = {
                "index": int(parts[0]),
                "name": parts[1],
                "temperature": float(parts[2]),
                "gpu_utilization": float(parts[3]),
                "memory_utilization": float(parts[4]),
                "memory_total": float(parts[5]),
                "memory_used": float(parts[6]),
                "memory_free": float(parts[7]),
                "power_draw": float(parts[8]) if parts[8] else None
            }
            gpus.append(gpu_info)
    return gpus

def get_torch_cuda_info() -> Dict[str, Any]:
    """
    Get CUDA information from PyTorch.
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {
                "cuda_available": False,
                "reason": "CUDA not available in PyTorch"
            }
            
        info = {
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "devices": []
        }
        
        # Get per-device information
        for i in range(info["device_count"]):
            device_info = {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
                "memory_allocated": float(torch.cuda.memory_allocated(i) / (1024**2)),  # MB
                "memory_reserved": float(torch.cuda.memory_reserved(i) / (1024**2)),    # MB
            }
            info["devices"].append(device_info)
            
        return info
    except ImportError:
        return {
            "cuda_available": False,
            "reason": "PyTorch not installed"
        }
    except Exception as e:
        return {
            "cuda_available": False,
            "reason": str(e)
        }

def get_combined_gpu_stats() -> Dict[str, Any]:
    """
    Get combined GPU statistics from nvidia-smi and PyTorch.
    """
    # Get nvidia-smi stats
    nvidia_smi_output = get_nvidia_smi_output()
    nvidia_gpus = parse_nvidia_smi_output(nvidia_smi_output) if nvidia_smi_output else []
    
    # Get PyTorch stats
    torch_info = get_torch_cuda_info()
    
    # Combine the information
    result = {
        "timestamp": os.popen("date").read().strip(),
        "nvidia_smi_available": bool(nvidia_smi_output),
        "pytorch_cuda_available": torch_info.get("cuda_available", False),
        "cuda_version": torch_info.get("cuda_version"),
        "device_count": torch_info.get("device_count", len(nvidia_gpus)),
        "gpus": []
    }
    
    # If nvidia-smi is available, use it as the primary source and enhance with PyTorch info
    if nvidia_gpus:
        for nvidia_gpu in nvidia_gpus:
            gpu_info = nvidia_gpu.copy()
            
            # Find matching PyTorch device
            if torch_info.get("cuda_available", False):
                for torch_device in torch_info.get("devices", []):
                    if torch_device["index"] == nvidia_gpu["index"]:
                        gpu_info.update({
                            "memory_allocated_mb": torch_device["memory_allocated"],
                            "memory_reserved_mb": torch_device["memory_reserved"],
                            "capability": torch_device.get("capability")
                        })
                        break
            
            result["gpus"].append(gpu_info)
    # If nvidia-smi is not available but PyTorch CUDA is, use PyTorch info
    elif torch_info.get("cuda_available", False):
        for torch_device in torch_info.get("devices", []):
            gpu_info = {
                "index": torch_device["index"],
                "name": torch_device["name"],
                "memory_allocated_mb": torch_device["memory_allocated"],
                "memory_reserved_mb": torch_device["memory_reserved"],
                "capability": torch_device.get("capability")
            }
            result["gpus"].append(gpu_info)
    
    return result

if __name__ == "__main__":
    # If run directly, print GPU stats as JSON
    stats = get_combined_gpu_stats()
    print(json.dumps(stats, indent=2))