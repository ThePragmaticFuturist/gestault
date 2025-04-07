# Citations or Links to Relevant Resources #

This acknowledges the great open-source libraries and common patterns leveraged in this project.

**1. Core Framework & Server**

*   **FastAPI:** Used as the main asynchronous web framework for building the API. Its design heavily influenced the endpoint structure, request/response handling, background tasks, and dependency injection concepts (though we used global instances for simplicity in some cases).
    *   **Citation/URL:** [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
    *   **Note:** FastAPI itself builds upon Starlette (ASGI toolkit) and Pydantic (data validation).
*   **Uvicorn:** The ASGI server used to run the FastAPI application. The `[standard]` extra includes `watchfiles` for the `--reload` functionality.
    *   **Citation/URL:** [https://www.uvicorn.org/](https://www.uvicorn.org/) / [https://github.com/encode/uvicorn](https://github.com/encode/uvicorn)

**2. Configuration**

*   **Pydantic (& `pydantic-settings`):** Used for defining configuration models (`Settings` class) and validating environment variables/`.env` file settings.
    *   **Citation/URL:** [https://docs.pydantic.dev/](https://docs.pydantic.dev/)
*   **`python-dotenv`:** Used implicitly by `pydantic-settings` to load variables from a `.env` file into environment variables.
    *   **Citation/URL:** [https://github.com/theskumar/python-dotenv](https://github.com/theskumar/python-dotenv)

**3. Databases**

*   **SQLAlchemy (Core):** Used to define the structure of the SQLite tables (`sessions`, `chat_messages`, `documents`, `document_chunks`) using its schema definition tools (`Table`, `Column`, `MetaData`).
    *   **Citation/URL:** [https://www.sqlalchemy.org/](https://www.sqlalchemy.org/) / [https://docs.sqlalchemy.org/en/20/core/](https://docs.sqlalchemy.org/en/20/core/)
*   **`databases` library:** Used for performing asynchronous database operations (connect, disconnect, execute, fetch) against the SQLite database, integrating well with FastAPI's async nature.
    *   **Citation/URL:** [https://github.com/encode/databases](https://github.com/encode/databases)
    *   **Note:** Relies on `aiosqlite` for the async SQLite driver (`databases[sqlite]`).
*   **ChromaDB:** Used as the vector database for storing and searching document chunk embeddings. We used `chromadb.PersistentClient` and methods like `get_or_create_collection`, `add`, and `query`.
    *   **Citation/URL:** [https://www.trychroma.com/](https://www.trychroma.com/) / [https://docs.trychroma.com/](https://docs.trychroma.com/) / [https://github.com/chroma-core/chroma](https://github.com/chroma-core/chroma)

**4. Document Processing**

*   **`python-docx`:** Used to extract text content from `.docx` files.
    *   **Citation/URL:** [https://python-docx.readthedocs.io/en/latest/](https://python-docx.readthedocs.io/en/latest/)
*   **`pypdf`:** Used to extract text content from `.pdf` files.
    *   **Citation/URL:** [https://pypdf.readthedocs.io/en/stable/](https://pypdf.readthedocs.io/en/stable/)
*   **`beautifulsoup4` (Beautiful Soup) & `lxml`:** Used to parse and extract text content from HTML files. `lxml` is a fast parser often used with BeautifulSoup.
    *   **Citation/URL (BS4):** [https://www.crummy.com/software/BeautifulSoup/bs4/doc/](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
    *   **Citation/URL (lxml):** [https://lxml.de/](https://lxml.de/)
*   **`python-magic`:** Used for reliable file type detection based on file content (magic numbers) rather than just extensions.
    *   **Citation/URL:** [https://github.com/ahupp/python-magic](https://github.com/ahupp/python-magic)
    *   **Note:** Requires system-level `libmagic` library.
*   **`aiofiles`:** Used for performing asynchronous file reading/writing operations, particularly useful within FastAPI's async environment (though we used standard `shutil` for simplicity in the final upload code).
    *   **Citation/URL:** [https://github.com/Tinche/aiofiles](https://github.com/Tinche/aiofiles)
*   **`langchain-text-splitters`:** Used specifically for the `RecursiveCharacterTextSplitter` algorithm to chunk documents into smaller pieces.
    *   **Citation/URL:** [https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter) (LangChain documentation)

**5. Embedding Model**

*   **`sentence-transformers`:** Used as a high-level interface to load and use sentence embedding models (like `all-MiniLM-L6-v2`) from the Hugging Face Hub via the `SentenceTransformer` class and its `.encode()` method.
    *   **Citation/URL:** [https://www.sbert.net/](https://www.sbert.net/) / [https://github.com/UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers)
*   **PyTorch:** The underlying deep learning framework required by `sentence-transformers` and `transformers`. Handled CUDA device management (`torch.cuda.is_available`, tensors).
    *   **Citation/URL:** [https://pytorch.org/](https://pytorch.org/)

**6. LLM Loading & Interaction**

*   **`transformers` (Hugging Face):** The core library used for loading LLMs (via `pipeline`, `AutoTokenizer`, `AutoModelForCausalLM`), tokenizing input, and generating text (`pipeline()` call, `model.generate()`).
    *   **Citation/URL:** [https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)
*   **`accelerate` (Hugging Face):** Used implicitly or explicitly (e.g., `device_map='auto'`) by `transformers` to handle efficient model loading and placement across devices (CPU, GPU, multiple GPUs).
    *   **Citation/URL:** [https://huggingface.co/docs/accelerate/index](https://huggingface.co/docs/accelerate/index)
*   **`bitsandbytes`:** Used for enabling 8-bit and 4-bit model quantization via `transformers` `BitsAndBytesConfig` to reduce memory usage.
    *   **Citation/URL:** [https://github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
*   **`sentencepiece`:** A tokenizer library dependency required by many Hugging Face models.
    *   **Citation/URL:** [https://github.com/google/sentencepiece](https://github.com/google/sentencepiece)
*   **`httpx`:** Used as the asynchronous HTTP client library to interact with external LLM API endpoints (Ollama, vLLM).
    *   **Citation/URL:** [https://www.python-httpx.org/](https://www.python-httpx.org/)

**7. System Status**

*   **`psutil`:** Used to gather cross-platform system information like CPU usage/frequency, RAM usage, disk usage, and potentially temperatures.
    *   **Citation/URL:** [https://github.com/giampaolo/psutil](https://github.com/giampaolo/psutil) / [https://psutil.readthedocs.io/en/latest/](https://psutil.readthedocs.io/en/latest/)
*   **`pynvml`:** Used to get detailed NVIDIA GPU information (memory, utilization, temperature) by interfacing with the NVIDIA Management Library (NVML).
    *   **Citation/URL:** [https://github.com/gpuopenanalytics/pynvml](https://github.com/gpuopenanalytics/pynvml) / [https://pypi.org/project/pynvml/](https://pypi.org/project/pynvml/)
    *   **Note:** Relies on NVIDIA drivers being installed.

**8. Core Concepts & Patterns**

*   **RAG (Retrieval-Augmented Generation):** The fundamental architecture combining retrieval (from ChromaDB) with generation (by the LLM).
    *   **Citation/URL (Seminal Paper):** Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Nogueira, G., ... & Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *arXiv preprint arXiv:2005.11401*. [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
    *   **Citation/URL (Explanations):** Hugging Face Blog posts, LangChain documentation on RAG.
*   **FastAPI Background Tasks:** The pattern used for offloading document processing (chunking, embedding) after the initial upload request returned.
    *   **Citation/URL:** [https://fastapi.tiangolo.com/tutorial/background-tasks/](https://fastapi.tiangolo.com/tutorial/background-tasks/)
*   **Asyncio `run_in_executor`:** Used to run blocking, CPU/GPU-bound code (local LLM inference, summarization) in a separate thread pool without blocking the main FastAPI async event loop.
    *   **Citation/URL:** [https://docs.python.org/3/library/asyncio-eventloop.html#executing-code-in-thread-or-process-pools](https://docs.python.org/3/library/asyncio-eventloop.html#executing-code-in-thread-or-process-pools)
*   **Hugging Face Hub Cache Structure:** Understanding the `models--<org>--<model>` and `snapshots/<hash>` structure was used for scanning the cache. Documented implicitly via library usage.
    *   **Citation/URL:** [https://huggingface.co/docs/huggingface_hub/guides/manage-cache](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)
*   **OpenAI-Compatible API:** The structure used for interacting with vLLM (and potentially InstructLab) follows the conventions established by OpenAI's API for chat completions.
    *   **Citation/URL:** [https://platform.openai.com/docs/api-reference/chat](https://platform.openai.com/docs/api-reference/chat) / vLLM documentation on OpenAI compatibility.

This list covers the primary external libraries and core concepts. Standard Python libraries (`os`, `datetime`, `logging`, `uuid`, `shutil`, `json`, `typing`, `abc`, `asyncio`, etc.) are generally not cited unless a very specific, non-obvious pattern was directly copied. The overall structure and integration logic represent the collaborative development process.
