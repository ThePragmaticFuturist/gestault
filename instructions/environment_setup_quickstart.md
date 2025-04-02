**Environment Setup Quick Start Guide**

This guide assumes you have the server code available (e.g., cloned from a Git repository).

**Prerequisites:**

*   **Python:** Version 3.10 or later recommended.
*   **Git:** To clone the repository (if applicable).
*   **Conda (Recommended):** Miniconda or Anaconda for managing environments, especially with CUDA. [https://www.anaconda.com/download](https://www.anaconda.com/download)
*   **(Optional) NVIDIA GPU:** Required for local GPU acceleration.
*   **(Optional) NVIDIA Driver:** If using GPU, ensure the latest compatible driver is installed. Verify with `nvidia-smi`.
*   **(Optional) CUDA Toolkit & cuDNN:** Only needed if installing PyTorch via `pip` for GPU (`venv` method). Not usually needed if using `conda`.
*   **(Optional) `libmagic`:** System library required by `python-magic`.
    *   Debian/Ubuntu: `sudo apt-get update && sudo apt-get install -y libmagic1`
    *   macOS: `brew install libmagic`
    *   Other Linux: Use your package manager (e.g., `yum`, `dnf`).

**Steps:**

1.  **Get the Code:**
    ```bash
    # If using git:
    # git clone <repository_url>
    cd rag_llm_server_extended # Navigate to the project root directory
    ```

2.  **Create Conda Environment:**
    ```bash
    conda create -n rag_server_env python=3.10
    conda activate rag_server_env
    ```

3.  **Install PyTorch with CUDA (If using GPU):**
    *   Go to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).
    *   Select: Your OS, Conda, Python 3.10, desired CUDA version (≤ version shown by `nvidia-smi`).
    *   **Copy and run the generated `conda install ...` command.** Example (check website for current command!):
        ```bash
        # Example for CUDA 12.1
        conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
        ```

4.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Hugging Face Authentication:** Choose **one** method:
    *   **Method A (CLI Login):** Run this in your terminal and follow prompts (requires a HF account):
        ```bash
        huggingface-cli login
        ```
    *   **Method B (Environment/`.env`):**
        *   Create a file named `.env` in the project root directory (`rag_llm_server_extended/`).
        *   Get an access token (with read permissions) from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
        *   Add the following line to your `.env` file, replacing the placeholder:
          ```.env
          HUGGING_FACE_HUB_TOKEN="hf_YourAccessTokenHere"
          ```
        *   **Important:** Ensure `.env` is listed in your `.gitignore` file to avoid committing your token.

6.  **Configure Server (`.env` file):**
    *   Edit the `.env` file (create if using Method B above).
    *   Set `LLM_BACKEND_TYPE` (e.g., `local`, `ollama`). Default is `local`.
    *   Set `DEFAULT_MODEL_NAME_OR_PATH` (e.g., `gpt2`, `meta-llama/Meta-Llama-3-8B-Instruct`). Default is `gpt2`.
    *   If using `ollama` or `vllm`, uncomment and set the corresponding `_BASE_URL` variable.
    *   *(Optional)* Set `HUGGING_FACE_HUB_TOKEN` if using Method B in Step 5.
    *   *(Optional)* Override `SERVER_PORT` if needed.

7.  **Run the Server:**
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app --reload-dir core --reload-dir db --reload-dir services
    ```
    *   Use `--host 0.0.0.0` to access from other machines on your network, or `127.0.0.1` for local access only.
    *   Adjust `--port` if needed.
    *   `--reload` is useful for development. Remove for production.

8.  **Verify:**
    *   Check the terminal logs for successful startup, including the HF token check.
    *   Open your browser to `http://localhost:8000/docs` (or the configured host/port). You should see the FastAPI documentation for all API endpoints (Documents, Chat Sessions, LLM Management, System Status).
