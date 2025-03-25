#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fixed run script for CUDA-optimized LLM server.
Specifically tuned for Python 3.10 and CUDA 12.4.
"""

import os
import sys
import logging
import argparse
import uvicorn
import shutil

# Check and rename app files if needed
if not os.path.exists("app.py") and os.path.exists("app_fixed.py"):
    shutil.copy2("app_fixed.py", "app.py")
    print("Using fixed version of app.py with SQLAlchemy keywords fixed")

# Ensure PyTorch is initialized with proper CUDA settings
try:
    import torch
    if torch.cuda.is_available():
        # Print CUDA info
        print(f"CUDA is available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Current CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        
        # Optimize for CUDA 12.4
        torch.cuda.empty_cache()
        # Enable TF32 precision (faster on Ampere GPUs)
        if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
    else:
        print("CUDA is not available, using CPU")
except ImportError:
    print("PyTorch not installed, CUDA optimizations skipped")

# Ensure necessary directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run the CUDA-Optimized LLM Server (Fixed Version)')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
parser.add_argument('--debug', action='store_true', help='Run in debug mode with auto-reload')
parser.add_argument('--log-level', type=str, default='info', 
                    choices=['debug', 'info', 'warning', 'error', 'critical'],
                    help='Logging level')
parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage even if available')
parser.add_argument('--fp16', action='store_true', help='Enable half precision (FP16) for faster inference')
parser.add_argument('--max-memory', type=str, help='Set maximum GPU memory usage (e.g. "4GiB")')
parser.add_argument('--check-db', action='store_true', help='Check and reset database if needed')

args = parser.parse_args()

# Check and reset database if requested
if args.check_db and os.path.exists("llm_server.db"):
    if args.debug:
        print("Debug mode: Renaming existing database for fresh start")
        if os.path.exists("llm_server.db.bak"):
            os.remove("llm_server.db.bak")
        os.rename("llm_server.db", "llm_server.db.bak")
        print("Renamed llm_server.db to llm_server.db.bak")

# Set environment variables for GPU configuration
if args.no_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("GPU usage disabled via --no-gpu flag")

if args.fp16:
    os.environ["USE_FP16"] = "1"
    print("Using FP16 (half precision) for inference")

if args.max_memory:
    os.environ["GPU_MAX_MEMORY"] = args.max_memory
    print(f"Setting maximum GPU memory to {args.max_memory}")

# Configure logging
log_level = getattr(logging, args.log_level.upper())
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_server.log"),
        logging.StreamHandler()
    ]
)

# Print startup message
print("="*50)
print("Starting CUDA-Optimized LLM Server (Fixed Version)")
print(f"Host: {args.host}")
print(f"Port: {args.port}")
print(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")
print(f"Log level: {args.log_level.upper()}")

# Print CUDA configuration
try:
    if torch.cuda.is_available() and not args.no_gpu:
        print(f"CUDA enabled: Using {torch.cuda.device_count()} GPU(s)")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"FP16 (half precision): {args.fp16}")
    else:
        print("CUDA disabled: Using CPU only")
except NameError:
    print("PyTorch not available")
print("="*50)

#################### Add these functions to run.py before the main block

def check_and_install_dependencies():
    """Check for required dependencies and offer to install missing ones"""
    dependencies = {
        "chromadb": "Vector database for semantic document storage",
        "docling": "Hierarchical document chunking library",
        "docling_core": "Core components for document processing",
        "sklearn": "For clustering in semantic chunking"
    }
    
    missing = []
    for package, description in dependencies.items():
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing.append((package, description))
    
    if missing:
        print("\nSome optional dependencies are missing:")
        for package, description in missing:
            print(f"  - {package}: {description}")
        
        install = input("\nWould you like to install the missing dependencies? (y/n): ")
        if install.lower() == 'y':
            import subprocess
            for package, _ in missing:
                print(f"Installing {package}...")
                subprocess.call([sys.executable, "-m", "pip", "install", package])
            print("Dependencies installed!")
        else:
            print("Skipping dependency installation. Some features may be limited.")

def setup_chroma_environment():
    """Set up ChromaDB environment"""
    try:
        import chromadb
        
        # Create Chroma directory
        chroma_dir = "./chroma_db"
        os.makedirs(chroma_dir, exist_ok=True)
        
        # Test Chroma connection
        client = chromadb.PersistentClient(path=chroma_dir)
        print(f"ChromaDB initialized at {chroma_dir}")
        
        return True
    except ImportError:
        print("ChromaDB not installed - vector search features will be disabled")
        return False
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        return False

# Inside the main block, add these lines before starting the server:
if __name__ == "__main__":
    try:
        # Check dependencies
        check_and_install_dependencies()
        
        # Set up ChromaDB environment
        chroma_available = setup_chroma_environment()
        if chroma_available:
            print("✓ Vector search capabilities enabled")
        
        # Run the server with higher timeout for CUDA model loading
        uvicorn.run(
            "app:app", 
            host=args.host, 
            port=args.port, 
            reload=args.debug,
            reload_excludes=["*.log", "*.db", "uploads/*", "embeddings/*", "chroma_db/*"],
            log_level=args.log_level.lower(),
            timeout_keep_alive=120
        )
    except Exception as e:
        print(f"Error starting server: {e}")
        # Try to clear CUDA memory if possible
        try:
            if 'torch' in sys.modules and torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("CUDA memory cache cleared")
        except:
            pass