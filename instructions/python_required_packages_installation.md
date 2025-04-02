**Python Requiremd Packages Installation**

Here is a `requirements.txt` file listing all the core Python packages required for this RAG LLM server.

It's highly recommended to **pin the versions** for reproducible builds. You can generate a fully versioned file by activating your virtual environment and running `pip freeze > requirements.txt`. However, this list includes the necessary packages based on our steps:

```txt
# requirements.txt

# --- Core Web Framework & Server ---
fastapi
uvicorn[standard] # Includes extras like websockets, watchfiles for reload

# --- Configuration ---
pydantic-settings
python-dotenv

# --- Databases ---
sqlalchemy          # Core SQL toolkit
databases[sqlite]   # Async database access + sqlite driver (aiosqlite)
chromadb            # Vector database

# --- Document Processing ---
python-docx         # For .docx files
pypdf               # For .pdf files
beautifulsoup4      # For HTML/XML parsing
lxml                # Parser often used with BeautifulSoup
langchain-text-splitters # For text chunking algorithms
python-magic        # For file type detection (requires libmagic system library)
aiofiles            # For async file operations

# --- Embedding Model ---
sentence-transformers # Easy interface for embedding models
# torch, torchvision, torchaudio are dependencies, usually installed together
torch
torchvision
torchaudio

# --- LLM Loading & Interaction ---
transformers        # Hugging Face core library for models/pipelines
accelerate          # Optimizes model loading/execution across devices
bitsandbytes        # For 8-bit/4-bit quantization (optional, often needs Linux/NVIDIA)
sentencepiece       # Tokenizer used by many models

# --- API Clients (for External LLMs) ---
httpx               # Async HTTP client for Ollama/vLLM APIs

# --- System Status ---
psutil              # Cross-platform system metrics (CPU, RAM, Disk)
pynvml              # For NVIDIA GPU monitoring (optional, requires NVIDIA drivers/setup)

```

**How to Use:**

1.  Save the content above into a file named `requirements.txt` in the root directory of your project (`rag_llm_server/`).
2.  Ensure you have a virtual environment created and activated.
3.  Run: `pip install -r requirements.txt`

**Important Notes:**

*   **Pinning Versions:** As mentioned, for production or sharing, replace this list with the output of `pip freeze` to lock down exact versions and ensure reproducibility.
*   **`python-magic`:** Requires the `libmagic` library to be installed on your operating system (e.g., `sudo apt-get install libmagic1` on Debian/Ubuntu, `brew install libmagic` on macOS).
*   **`bitsandbytes`:** Primarily works on Linux with compatible NVIDIA GPUs and CUDA setup. Installation might fail or the library might not be usable on other systems (like macOS or Windows without WSL).
*   **`pynvml`:** Requires NVIDIA drivers and the NVML library to be properly installed and accessible on your system to report GPU stats. It will install but fail at runtime if the NVIDIA environment isn't set up.
*   **`torch`:** The command `pip install torch torchvision torchaudio` usually installs the appropriate version (CPU or CUDA) for your system based on PyTorch's detection. If you have specific CUDA requirements, consult the official PyTorch installation guide.

This file should cover all the functionality built into the server.
