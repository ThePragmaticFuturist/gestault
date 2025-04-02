**CUDA Environment Setup Guide**

Setting up a Python virtual environment that correctly utilizes your NVIDIA GPU via CUDA involves a few critical steps, primarily ensuring compatibility between your NVIDIA driver, the CUDA toolkit version, and the deep learning framework (like PyTorch, which `transformers` uses).

**Using `conda` (Recommended for CUDA Environments):**

`conda` is generally recommended for managing environments with complex binary dependencies like CUDA libraries because it can manage non-Python packages alongside Python ones, often simplifying the process.

**Prerequisites:**

1.  **NVIDIA GPU:** You need a CUDA-capable NVIDIA GPU.
2.  **NVIDIA Driver:** You **must** have the appropriate NVIDIA proprietary driver installed on your system (Linux or Windows). This is the most critical step.
    *   **Check:** Open a terminal or command prompt and run `nvidia-smi`. If this command runs successfully and shows your GPU details, you have a driver installed. Note the `CUDA Version` listed in the top right – this is the *maximum* CUDA version your *driver* supports, not necessarily the version you need to install.
    *   **Install/Update:** If `nvidia-smi` fails or the driver is very old, download and install the latest appropriate driver for your GPU model and OS from the [NVIDIA Driver Downloads page](https://www.nvidia.com/Download/index.aspx). Reboot after installation if required.
3.  **Miniconda or Anaconda:** You need `conda` installed. Miniconda is a lighter-weight option. Download and install it from the [official Anaconda site](https://www.anaconda.com/download#installation) if you haven't already. Follow their OS-specific instructions.

**Setup Steps using `conda`:**

1.  **Open Terminal/Anaconda Prompt:** Use your standard terminal (Linux/macOS) or the "Anaconda Prompt" (Windows).

2.  **Create a New Conda Environment:** Choose a name (e.g., `cuda_env`) and a Python version (3.10 or 3.11 are good choices currently).
    ```bash
    conda create -n cuda_env python=3.10
    ```
    Press `y` to proceed when prompted.

3.  **Activate the Environment:**
    ```bash
    conda activate cuda_env
    ```
    Your terminal prompt should now start with `(cuda_env)`.

4.  **Install PyTorch with CUDA Support (CRITICAL STEP):**
    *   **Go to the official PyTorch website:** [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    *   **Use the Configuration Tool:** Select your OS, choose "Conda" as the package manager, choose your desired Python version (matching the one used in `conda create`), and select the **CUDA Version** you want to target.
        *   **Which CUDA Version?** Check the output of `nvidia-smi` again. While it shows the *maximum* supported version, PyTorch often works well with slightly older, stable CUDA toolkit versions (e.g., 11.8, 12.1). The PyTorch website will show available pre-compiled versions. Pick one that is less than or equal to the version shown by `nvidia-smi`. For newer GPUs/drivers, CUDA 12.1+ is common. For slightly older ones, 11.8 might be offered.
    *   **Copy the Command:** The website will generate a specific `conda install ...` command. **Use the command provided by the PyTorch website.** It will look something like this (***this is an EXAMPLE, use the one from the website!***):
        ```bash
        # Example for CUDA 11.8
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

        # Example for CUDA 12.1
        # conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
        ```
        *   `-c pytorch -c nvidia`: Tells conda to look in the official PyTorch and NVIDIA channels for these packages.
        *   `pytorch-cuda=11.8` (or `12.1`): This special package pulls in the necessary CUDA runtime libraries managed by conda, avoiding the need for a separate system-wide CUDA toolkit installation *for running PyTorch*.
    *   **Run the command** in your activated environment. Agree to the installation (`y`). This might take some time as it downloads PyTorch and CUDA components.

5.  **Install Other Dependencies:** Now install the rest of the packages needed for your server using `pip` (it's fine to use `pip` inside a conda environment):
    ```bash
    pip install -r requirements.txt
    ```
    (Make sure your `requirements.txt` file is up-to-date from the previous step). Or install manually:
    ```bash
    pip install fastapi uvicorn[standard] pydantic-settings python-dotenv sqlalchemy "databases[sqlite]" chromadb python-docx pypdf beautifulsoup4 lxml langchain-text-splitters python-magic aiofiles sentence-transformers transformers accelerate bitsandbytes sentencepiece httpx psutil pynvml
    ```

6.  **Verify CUDA Integration:**
    *   Start a Python interpreter within the activated environment: `python`
    *   Run the following commands:
        ```python
        import torch

        # Check if CUDA is available to PyTorch
        cuda_available = torch.cuda.is_available()
        print(f"PyTorch CUDA available: {cuda_available}")

        if cuda_available:
            # Get the number of GPUs
            num_gpus = torch.cuda.device_count()
            print(f"Number of GPUs available: {num_gpus}")

            # Get the name of the primary GPU
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU Name: {gpu_name}")

            # Optional: Check CUDA version PyTorch was compiled with
            pytorch_cuda_version = torch.version.cuda
            print(f"PyTorch compiled with CUDA Version: {pytorch_cuda_version}")
        else:
            print("CUDA not available to PyTorch. Check installation steps and driver compatibility.")

        exit() # Exit the Python interpreter
        ```
    *   You **must** see `PyTorch CUDA available: True` for GPU acceleration to work.

**Using `venv` (Standard Python Virtual Environment):**

This method is possible but often trickier, as `venv` doesn't manage CUDA libraries. It relies on finding a system-wide CUDA Toolkit installation.

**Prerequisites:**

1.  **NVIDIA GPU & Driver:** Same as above. Run `nvidia-smi`.
2.  **System-wide CUDA Toolkit Installation:** You generally need to have the full NVIDIA CUDA Toolkit installed separately on your system. Download it from the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive). Choose a version compatible with your driver *and* the PyTorch version you intend to install. Make sure its `bin` directory is in your system's `PATH` and libraries are findable (e.g., `LD_LIBRARY_PATH` on Linux).
3.  **cuDNN Installation:** You usually also need the corresponding cuDNN library installed and placed correctly within your CUDA Toolkit installation directories. Download from [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive).

**Setup Steps using `venv`:**

1.  **Open Terminal/Command Prompt.**
2.  **Create Virtual Environment:**
    ```bash
    python -m venv cuda_venv
    ```
3.  **Activate Environment:**
    *   Linux/macOS: `source cuda_venv/bin/activate`
    *   Windows: `.\cuda_venv\Scripts\activate`
4.  **Install PyTorch with CUDA (CRITICAL STEP):**
    *   Go to the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    *   Select your OS, choose **"Pip"** as the package manager, Python version, and the CUDA version matching your **system-wide CUDA Toolkit installation**.
    *   Copy the specific `pip install torch ...` command provided. It will likely include an `--index-url` or `--extra-index-url` pointing to PyTorch's download servers for CUDA builds. ***This command is crucial. Do not just run `pip install torch`***. Example (use the one from the website!):
        ```bash
        # Example for CUDA 11.8 (check PyTorch site for current URL!)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

        # Example for CUDA 12.1
        # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```
    *   Run the command in your activated environment.
5.  **Install Other Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Or install manually with `pip install ...`.
6.  **Verify CUDA Integration:** Same verification steps as in the `conda` section (run `python`, import `torch`, check `torch.cuda.is_available()`, etc.).

**Troubleshooting:**

*   **`torch.cuda.is_available()` is `False`:**
    *   Most likely cause: Incompatibility between NVIDIA Driver version, CUDA Toolkit version (either system-wide for `venv` or the `pytorch-cuda` package for `conda`), and the installed PyTorch version. Double-check compatibility and use the command from the PyTorch website.
    *   Ensure the environment is activated correctly.
    *   (venv specific): Check `PATH` and `LD_LIBRARY_PATH` environment variables point correctly to your system CUDA installation.
*   **`nvidia-smi` works but PyTorch doesn't see CUDA:** Usually a version mismatch or installation issue with the PyTorch CUDA components. Reinstall PyTorch using the exact command from their website.
*   **Environment Issues:** If things get messy, sometimes it's easiest to deactivate (`conda deactivate`), remove the environment (`conda env remove -n cuda_env`), and start fresh.

Using `conda` significantly simplifies managing the CUDA library dependencies alongside PyTorch, making it the generally preferred route for GPU-accelerated Python environments.
