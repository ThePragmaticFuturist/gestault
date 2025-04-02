# Glossary of Terms #

*   **API (Application Programming Interface):** A set of rules and protocols allowing different software applications to communicate with each other. In our case, the way clients interact with the server (e.g., upload documents, start chats).
*   **ASGI (Asynchronous Server Gateway Interface):** A standard interface between async-capable Python web servers (like Uvicorn) and async web frameworks (like FastAPI).
*   **Background Task:** A task (like document processing) initiated by an API request but executed by the server *after* the initial response has been sent to the client.
*   **Backend (LLM):** Refers to the actual service or library handling LLM inference (e.g., local Transformers, Ollama API, vLLM API).
*   **`bitsandbytes`:** A library enabling quantization (e.g., 8-bit, 4-bit) for deep learning models, primarily on NVIDIA GPUs, to reduce memory usage.
*   **ChromaDB:** An open-source vector database used to store and efficiently search vector embeddings based on similarity.
*   **Chunking:** The process of splitting large documents into smaller, overlapping text segments for embedding and retrieval.
*   **Conda:** A cross-platform package and environment manager, particularly useful for managing complex dependencies like CUDA.
*   **Context Window:** The maximum number of tokens an LLM can process as input at one time. Prompts exceeding this limit cause errors or degraded performance.
*   **CUDA:** A parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs).
*   **`.env` file:** A text file used to store environment variables (like configuration settings or API keys) for an application, typically loaded on startup.
*   **Executor Thread:** A separate thread (managed by `asyncio.run_in_executor`) used to run blocking, synchronous code (like local LLM inference) without halting the main asynchronous event loop.
*   **FastAPI:** A modern, high-performance Python web framework for building APIs, built on Starlette and Pydantic, with native async support.
*   **Hugging Face Hub:** A platform hosting a vast collection of pre-trained models, datasets, and tools for machine learning.
*   **Hugging Face Token:** An access key used to authenticate requests to the Hugging Face Hub, required for downloading private/gated models or uploading.
*   **Keyword Search:** Searching for documents based on the literal presence of specific words or phrases (implemented here using SQL `LIKE`).
*   **LLM (Large Language Model):** A deep learning model trained on vast amounts of text data, capable of understanding and generating human-like text (e.g., GPT-2, Llama 3, Granite).
*   **NVML (NVIDIA Management Library):** A C-based API for monitoring and managing NVIDIA GPU devices. `pynvml` is the Python binding.
*   **Ollama:** A tool for easily running open-source LLMs locally via a simple API.
*   **Pipeline (Hugging Face):** A high-level abstraction in the `transformers` library that simplifies common NLP tasks like text generation by bundling model loading, tokenization, inference, and post-processing.
*   **Prompt Engineering:** The process of designing effective input prompts to guide an LLM towards generating the desired output.
*   **`psutil`:** A cross-platform Python library for retrieving information on running processes and system utilization (CPU, memory, disks, network, sensors).
*   **Pydantic:** A Python library for data validation and settings management using Python type hints. Used extensively by FastAPI.
*   **Quantization:** The process of reducing the precision (number of bits) used to represent model weights and activations (e.g., from 32-bit float to 8-bit or 4-bit integer), significantly reducing memory usage and potentially increasing speed, sometimes at the cost of accuracy.
*   **RAG (Retrieval-Augmented Generation):** An AI architecture that combines information retrieval (fetching relevant documents/data) with large language model generation to produce more knowledgeable, accurate, and context-aware responses.
*   **Semantic Search:** Searching for documents based on meaning and conceptual similarity, typically using vector embeddings and nearest-neighbor search in a vector database.
*   **Sentence Transformers:** A Python library built on `transformers` and PyTorch/TensorFlow, providing an easy way to compute dense vector embeddings (semantic representations) for sentences, paragraphs, and images.
*   **SQLAlchemy (Core):** A Python SQL toolkit that provides tools for defining database schemas and constructing SQL queries programmatically, without necessarily using the full Object-Relational Mapper (ORM).
*   **SQLite:** A C-language library implementing a self-contained, serverless, zero-configuration, transactional SQL database engine. Used here for storing metadata.
*   **Tokenizer:** A component that converts raw text into a sequence of tokens (numbers) that an LLM can understand, and vice-versa.
*   **Transformers (Library):** The core Hugging Face library providing access to thousands of pre-trained models (including LLMs) and tools for NLP tasks.
*   **Uvicorn:** An ASGI server implementation used to run asynchronous Python web applications like those built with FastAPI or Starlette.
*   **Vector Database:** A specialized database designed to efficiently store and query high-dimensional vectors based on similarity metrics (like cosine similarity or Euclidean distance). ChromaDB is an example.
*   **Vector Embedding:** A dense numerical vector representation of a piece of data (like text) where similar items have vectors that are close together in the vector space.
*   **vLLM:** An open-source library designed for fast and efficient LLM inference and serving, often providing an OpenAI-compatible API endpoint.
*   **VRAM (Video RAM):** The dedicated memory present on a GPU, crucial for loading model weights and performing computations for deep learning tasks.
