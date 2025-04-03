# The Most Logical Starting Point: The Basic Server Structure #

Before we can handle documents, models, or chats, we need the fundamental web server framework in place. This will be the skeleton upon which we'll hang all the other features. We'll use **FastAPI** for this, as it's modern, fast, asynchronous (great for I/O-bound tasks like network requests and potentially waiting for LLMs), and has excellent built-in documentation features.
  
[Reasoning behind the code](reasoning/r1.md)  
  
**Step 1: Project Setup and Basic FastAPI App**

1.  **Create Project Directory:**
    ```bash
    mkdir rag_llm_server
    cd rag_llm_server
    ```

2.  **Set up a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate it:
    # Windows:
    # venv\Scripts\activate
    # macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install Initial Dependencies:**
    ```bash
    pip install fastapi uvicorn[standard]
    ```
    *   `fastapi`: The web framework.
    *   `uvicorn`: An ASGI server to run our FastAPI application. `[standard]` includes useful extras like websocket support (which we might use later) and better performance.

4.  **Create the Main Application File (`main.py`):**

    ```python
    # main.py
    import uvicorn
    from fastapi import FastAPI
    import datetime
    import os

    # --- Core Application Setup ---
    # Create the main FastAPI application instance
    app = FastAPI(
        title="RAG LLM Server",
        description="API server for managing RAG documents, interacting with LLMs, and managing chat sessions.",
        version="0.1.0",
    )

    # --- Basic Root Endpoint ---
    @app.get("/")
    async def read_root():
        """
        Root endpoint providing basic server information.
        """
        return {
            "message": "Welcome to the RAG LLM Server!",
            "status": "running",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

    # --- Placeholder for Future Routers ---
    # We will add routers for different functionalities (documents, models, sessions, system) later.
    # Example: from app.api.endpoints import documents
    #          app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])

    # --- Running the Server (for direct execution) ---
    if __name__ == "__main__":
        # Get port from environment variable or default to 8000
        port = int(os.environ.get("PORT", 8000))
        print(f"Starting server on http://localhost:{port}")
        # Uvicorn is used to run the FastAPI application
        # --reload enables auto-reloading during development when code changes
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

    ```

5.  **Create a `.gitignore` file:**
    ```
    # .gitignore
    venv/
    __pycache__/
    *.pyc
    *.pyo
    .env
    *.db
    *.db-journal
    chroma_db_data/ # Placeholder for ChromaDB data directory
    local_models/ # Placeholder for local LLM model files
    uploaded_docs/ # Placeholder for temporarily stored uploads
    *.log

    # IDE specific files
    .idea/
    .vscode/
    *.swp
    ```

6.  **Run the Server:**
    ```bash
    uvicorn main:app --reload
    ```
    *   `main`: Refers to the `main.py` file.
    *   `app`: Refers to the `FastAPI` instance named `app` inside `main.py`.
    *   `--reload`: Tells uvicorn to automatically restart the server when it detects changes in the code files. Very handy for development.

7.  **Access the Server:**
    *   Open your web browser and go to `http://localhost:8000` (or `http://127.0.0.1:8000`). You should see the JSON response:
        ```json
        {
          "message": "Welcome to the RAG LLM Server!",
          "status": "running",
          "timestamp": "..."
        }
        ```
    *   Go to `http://localhost:8000/docs`. FastAPI automatically generates interactive API documentation (Swagger UI).
    *   Go to `http://localhost:8000/redoc`. FastAPI also provides alternative documentation (ReDoc).

**Explanation:**

1.  **Imports:** We import `FastAPI` to create our web application, `uvicorn` to run it programmatically (though we mostly use the command line), `datetime` for the timestamp, and `os` to potentially read environment variables later (like the port).
2.  **`FastAPI(...)`:** This line creates the main application object. We give it a `title`, `description`, and `version`, which are used in the automatic API documentation.
3.  **`@app.get("/")`:** This is a decorator provided by FastAPI. It tells the application that the function below (`read_root`) should handle GET requests made to the root path (`/`).
4.  **`async def read_root():`:** We define the function as `async`. FastAPI is built on ASGI (Asynchronous Server Gateway Interface), and using `async` allows FastAPI to handle requests concurrently without blocking, which is crucial for performance, especially when dealing with I/O operations like network calls, database access, or waiting for LLM responses.
5.  **Return Value:** The function returns a Python dictionary. FastAPI automatically converts this dictionary into a JSON response.
6.  **`if __name__ == "__main__":`:** This standard Python construct allows the script to be run directly. When you execute `python main.py`, this block runs. We use it here to start the `uvicorn` server programmatically, although running `uvicorn main:app --reload` from the terminal is more common during development. We added getting the port from an environment variable as good practice.
7.  **Project Structure (Initial):** Right now, it's just `main.py` and `requirements.txt`. As we add features, we'll create subdirectories (like `app`, `core`, `services`, `models`, `db`) to keep the code organized.

**Next Steps:**

We have the basic server running. The next logical step is to set up the configuration management and initialize our database connections (SQLite for metadata/sessions and ChromaDB for vectors). This is foundational because almost all other features will depend on configuration settings and database access.
