**Architecture and Data Flows of the RAG LLM server**

**1. High-Level Server Architecture Diagram**



![This diagram shows the main subsystems and their primary interactions.](instructions/system architecture.png)

```
graph TD
    subgraph Client_Side
        User[User/Client Application]
    end

    subgraph FastAPI_Server [FastAPI Server (Python/Uvicorn)]
        A[API Endpoints<br/>(FastAPI Routers)]
        B[Configuration<br/>(core/config.py)]
        C[Background Tasks<br/>(FastAPI)]
        D[Services Layer]
        E[Database Layer<br/>(SQLAlchemy Core + Databases Lib)]
        F[Vector DB Client<br/>(ChromaDB Client)]
        G[LLM Loader/Manager<br/>(Local/API Handling)]
        H[Embedding Loader<br/>(SentenceTransformer)]
    end

    subgraph Storage
        SQL[(SQLite DB<br/>Metadata, Sessions, Chunks)]
        VDB[(ChromaDB<br/>Vector Embeddings)]
    end

    subgraph External_Services
        ExtLLM[Ollama/vLLM/InstructLab API<br/>(Optional)]
        HFModels[Hugging Face Hub<br/>(Model Downloads)]
    end

    subgraph System_Monitoring
        SysMon[System Resources<br/>(psutil, pynvml)]
    end

    User -- HTTP API Requests --> A
    A -- Uses --> B
    A -- Triggers --> C
    A -- Calls --> D
    A -- Reads/Writes --> E
    A -- Reads/Writes --> F
    A -- Manages --> G
    A -- Monitors --> SysMon

    C -- Calls --> D[Document Processor, Embedding]

    D -- Uses --> B
    D -- Reads/Writes --> E
    D -- Reads/Writes --> F
    D -- Uses --> G
    D -- Uses --> H
    D -- Reads --> SysMon

    E -- Interacts with --> SQL
    F -- Interacts with --> VDB

    G -- Loads/Interacts --> HFModels
    G -- Interacts (Optional) --> ExtLLM
    G -- Uses --> H[If loading local embedding model needs HF]

    H -- Loads --> HFModels

    style FastAPI_Server fill:#f9f,stroke:#333,stroke-width:2px
    style Storage fill:#ccf,stroke:#333,stroke-width:1px
    style External_Services fill:#ccf,stroke:#333,stroke-width:1px
    style System_Monitoring fill:#cfc,stroke:#333,stroke-width:1px
```

**Explanation:**

*   The **User** interacts with the **API Endpoints** built with FastAPI.
*   These endpoints use **Configuration**, trigger **Background Tasks**, and call various functions in the **Services Layer**.
*   The **Services Layer** contains the core logic (Document Processing, Embedding, LLM, System Status).
*   Services interact with the **Database Layer** (for SQLite) and the **Vector DB Client** (for ChromaDB).
*   The **LLM Loader/Manager** handles loading models either locally (potentially downloading from **Hugging Face Hub**) or configuring connections to **External LLM APIs**.
*   The **Embedding Loader** loads the sentence transformer model, likely from the **Hugging Face Hub**.
*   **System Monitoring** uses libraries to check underlying system resources.

---

**2. Component Diagrams & Data Flows**

**2.1. Document Ingestion Flow**

*   **Trigger:** `POST /api/v1/documents/upload` with file data.

```
sequenceDiagram
    participant Client
    participant API (FastAPI /documents)
    participant SQLite (documents table)
    participant BackgroundTask (document_processor)
    participant TextExtractor (docx, pdf, etc.)
    participant TextChunker (langchain)
    participant SQLite (chunks table)
    participant EmbeddingService
    participant ChromaDB

    Client ->>+ API: Upload File (e.g., doc.pdf)
    API ->> API: Save to temp file (/tmp/...)
    API ->>+ SQLite: INSERT document (id, filename, status='pending')
    SQLite -->>- API: Confirm Insert
    API -->>- Client: 202 Accepted (doc_id)
    API ->> BackgroundTask: Schedule process_document_upload(temp_path, doc_id, filename)

    Note over BackgroundTask: Runs asynchronously

    BackgroundTask ->>+ SQLite: UPDATE document SET status='processing' WHERE id=doc_id
    SQLite -->>- BackgroundTask: Confirm Update
    BackgroundTask ->>+ TextExtractor: extract_text(temp_path)
    TextExtractor -->>- BackgroundTask: extracted_text
    BackgroundTask ->>+ SQLite: UPDATE document SET status='chunking' WHERE id=doc_id
    SQLite -->>- BackgroundTask: Confirm Update
    BackgroundTask ->>+ TextChunker: chunk_text(extracted_text)
    TextChunker -->>- BackgroundTask: List[chunk_text]
    BackgroundTask ->>+ SQLite: INSERT MANY chunks (chunk_id, doc_id, index, text)
    SQLite -->>- BackgroundTask: Confirm Insert
    BackgroundTask ->>+ SQLite: UPDATE document SET status='embedding', chunk_count=N WHERE id=doc_id
    SQLite -->>- BackgroundTask: Confirm Update
    BackgroundTask ->>+ EmbeddingService: generate_embeddings(List[chunk_text])
    EmbeddingService -->>- BackgroundTask: List[embedding_vector]
    BackgroundTask ->>+ ChromaDB: collection.add(ids=chunk_ids, embeddings=vectors, documents=chunks, metadatas=...)
    ChromaDB -->>- BackgroundTask: Confirm Add
    BackgroundTask ->>+ SQLite: UPDATE document SET status='completed', vector_count=N WHERE id=doc_id
    SQLite -->>- BackgroundTask: Confirm Update
    BackgroundTask ->> BackgroundTask: Delete temp file

    alt On Error during processing
        BackgroundTask ->>+ SQLite: UPDATE document SET status='failed', error_message=... WHERE id=doc_id
        SQLite -->>- BackgroundTask: Confirm Update
        BackgroundTask ->> BackgroundTask: Delete temp file
    end

```

**2.2. Semantic Search Flow**

*   **Trigger:** `POST /api/v1/documents/search/semantic` with query data.

```
sequenceDiagram
    participant Client
    participant API (FastAPI /documents)
    participant EmbeddingService
    participant ChromaDB

    Client ->>+ API: POST SearchQuery(query, top_k, filters)
    API ->>+ EmbeddingService: generate_embeddings([query_text])
    EmbeddingService -->>- API: List[query_vector]
    API ->> API: Build ChromaDB 'where' filter (optional)
    API ->>+ ChromaDB: collection.query(query_embeddings, n_results, where=filter, include=...)
    ChromaDB -->>- API: SearchResults (ids, distances, documents, metadatas)
    API ->> API: Format results into List[ChunkResult]
    API -->>- Client: 200 OK SearchResponse(results)

    alt Embedding/Query Error
        EmbeddingService -->> API: None / Exception
        API -->> Client: 500 Internal Server Error
    end
    alt ChromaDB Error
        ChromaDB -->> API: Exception
        API -->> Client: 500 Internal Server Error
    end
```

**2.3. LLM Loading Flow (Local Backend Example)**

*   **Trigger:** `POST /api/v1/models/load` with `model_name_or_path`. (Server `LLM_BACKEND_TYPE` is `local`).

```
sequenceDiagram
    participant Client
    participant API (FastAPI /models)
    participant LLMService (set_active_backend)
    participant LLMService (_unload_current_backend)
    participant LLMService (Background Executor)
    participant LLMService (_load_local_model_task)
    participant LocalBackend (Instance)
    participant TransformersLib

    Client ->>+ API: POST ModelLoadRequest(model_id, device?, quant?)
    API ->> API: Check if already Loading/Unloading (If yes, return 409)
    API ->>+ LLMService: await set_active_backend(type='local', model_id, ...)
    LLMService ->>+ LLMService (_unload_current_backend): await unload() [If previous backend exists]
    LLMService (_unload_current_backend) ->> LocalBackend: await self.unload() [If prev was local]
    LocalBackend -->> LLMService (_unload_current_backend): Cleanup done
    LLMService (_unload_current_backend) -->>- LLMService: Unload complete
    LLMService ->> LocalBackend: Create instance: backend = LocalTransformersBackend()
    LLMService ->> LLMService: Update global state (status=LOADING, backend_instance=backend)
    LLMService ->> LLMService (Background Executor): Schedule local_load_wrapper(backend, model_id, ...)
    LLMService -->>- API: Return early (Loading scheduled)
    API -->>- Client: 202 Accepted {"message": "Loading initiated..."}

    Note right of LLMService (Background Executor): Runs in separate thread
    LLMService (Background Executor) ->>+ LLMService (_load_local_model_task): asyncio.run(_load_local_model_task(backend, ...))
    LLMService (_load_local_model_task) ->>+ TransformersLib: AutoTokenizer.from_pretrained(model_id)
    TransformersLib -->>- LLMService (_load_local_model_task): tokenizer
    LLMService (_load_local_model_task) ->>+ TransformersLib: pipeline(model=model_id, tokenizer=...)
    TransformersLib -->>- LLMService (_load_local_model_task): pipeline_instance
    LLMService (_load_local_model_task) ->> LocalBackend: Update instance: backend.pipeline = ..., backend.tokenizer = ...
    LLMService (_load_local_model_task) ->> LLMService: Update global state (status=READY, active_model=...)
    LLMService (_load_local_model_task) -->>- LLMService (Background Executor): Task complete

    alt Loading Fails
        LLMService (_load_local_model_task) ->> LocalBackend: Clear instance attributes (pipeline=None, etc.)
        LLMService (_load_local_model_task) ->> LLMService: Update global state (status=FAILED, error=...)
        LLMService (_load_local_model_task) -->>- LLMService (Background Executor): Task failed
    end

```

**2.4. Chat Interaction Flow (with RAG & Summarization)**

*   **Trigger:** `POST /api/v1/sessions/{session_id}/messages` with user message.

```
sequenceDiagram
    participant Client
    participant API (FastAPI /sessions)
    participant SQLite (sessions, chat_messages)
    participant EmbeddingService
    participant ChromaDB
    participant LLMService (summarize/generate)
    participant LLMExecutor (Background Thread)
    participant ActiveBackend (Local/Ollama/vLLM)


    Client ->>+ API: POST user_message
    API ->>+ SQLite: Fetch session data (incl. rag_doc_ids)
    SQLite -->>- API: Session data
    API ->>+ SQLite: INSERT user_message
    SQLite -->>- API: Confirm insert, update session timestamp
    API ->>+ EmbeddingService: generate_embeddings([user_message])
    EmbeddingService -->>- API: query_embedding
    API ->>+ ChromaDB: query(query_embedding, filter=rag_doc_ids)
    ChromaDB -->>- API: retrieved_chunks (text, metadata)

    loop For Each Retrieved Chunk
        API ->>+ LLMExecutor: Schedule summarize_text_with_query(chunk_text, user_query)
        LLMExecutor ->>+ LLMService: summarize_text_with_query(...)
        LLMService ->>+ LLMService: generate_text(summary_prompt) [Uses temp config]
            Note over LLMService: This might call executor again if backend is local
            LLMService ->> ActiveBackend: generate(summary_prompt, temp_config)
            ActiveBackend -->> LLMService: summary_text / None
        LLMService -->>- LLMExecutor: summary_text / None
        LLMExecutor -->>- API: summary_text / None
        API ->> API: Append summary to context_parts
    end

    API ->> API: Combine summaries into rag_context
    API ->>+ SQLite: Fetch chat_history for session
    SQLite -->>- API: List[messages]
    API ->> API: Get tokenizer, max_length from llm_state
    API ->> API: Build final_prompt (Instruction, Summarized Context, History, Query)
    API ->> API: Check/Truncate final_prompt based on token limits

    API ->>+ LLMExecutor: Schedule generate_text(final_prompt)
    LLMExecutor ->>+ LLMService: generate_text(final_prompt) [Uses main config]
    LLMService ->> ActiveBackend: generate(final_prompt, main_config) [Direct await if API backend]
    ActiveBackend -->> LLMService: assistant_response_text / None
    LLMService -->>- LLMExecutor: assistant_response_text / None
    LLMExecutor -->>- API: assistant_response_text / None

    alt LLM Generation Fails
        API ->> API: Set assistant_response_content = "[ERROR...]"
        API ->> API: Record error in metadata
    end

    API ->>+ SQLite: INSERT assistant_message (content, metadata)
    SQLite -->>- API: Confirm insert, update session timestamp
    API -->>- Client: 201 Created (assistant_message details)

```

**2.5. System Status Flow**

*   **Trigger:** `GET /api/v1/system/status`

```
sequenceDiagram
    participant Client
    participant API (FastAPI /system)
    participant SystemService (get_full_system_status)
    participant psutil
    participant pynvml (Optional)

    Client ->>+ API: GET /status
    API ->>+ SystemService: get_full_system_status()
    SystemService ->>+ psutil: cpu_percent(), cpu_count(), cpu_freq()
    psutil -->>- SystemService: CPU data
    SystemService ->>+ psutil: virtual_memory()
    psutil -->>- SystemService: Memory data
    SystemService ->>+ psutil: disk_partitions(), disk_usage('/')
    psutil -->>- SystemService: Disk data
    opt pynvml Available
        SystemService ->>+ pynvml: nvmlDeviceGetCount(), GetHandle(), GetName(), GetMemoryInfo(), GetUtilization(), GetTemperature()
        pynvml -->>- SystemService: GPU data list
    end
    opt psutil Temp Sensors Available
        SystemService ->>+ psutil: sensors_temperatures()
        psutil -->>- SystemService: Temperature data
    end
    SystemService ->> SystemService: Assemble status dictionary
    SystemService -->>- API: status_dict
    API ->> API: Validate with SystemStatusResponse.parse_obj(status_dict)
    API -->>- Client: 200 OK SystemStatusResponse

    alt Error Getting Status
        SystemService/psutil/pynvml -->> API: Exception
        API -->> Client: 500 Internal Server Error
    end
```

These diagrams illustrate the key interactions and data movements within the server architecture.
