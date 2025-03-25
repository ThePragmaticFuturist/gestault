#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Document Processing Pipeline with Semantic Chunking and Chroma DB support

This utility processes documents by:
1. Using hierarchical or semantic chunking techniques
2. Applying a specified prompt to each chunk
3. Processing each prompt+chunk with the LLM server
4. Combining the results into a coherent response
5. Storing embeddings and chunks in Chroma DB for retrieval
"""

import os
import sys
import json
import aiohttp
import asyncio
import time
import numpy as np
import re
import logging
from typing import List, Dict, Any, Optional, Callable, Iterator, Union, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("document_processor")

# Try importing Chroma dependencies
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
    logger.info("ChromaDB available for vector storage")
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not installed - vector storage features will be disabled")

# Try importing docling for hierarchical chunking
try:
    import itertools
    from docling.document_converter import DocumentConverter
    from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
    from docling_core.transforms.chunker.base import BaseChunk
    DOCLING_AVAILABLE = True
    logger.info("Docling available for hierarchical chunking")
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning("Docling not installed - falling back to simpler chunking methods")

class ChromaVectorStore:
    """Chroma DB vector store for document chunks and embeddings"""
    
    def __init__(self, persist_directory: str = "chroma_db"):
        """Initialize Chroma DB client"""
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Please install it with 'pip install chromadb'")
            
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize default embedding function
        self.default_ef = embedding_functions.DefaultEmbeddingFunction()
        
        # Create or get collections
        self.documents_collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.default_ef
        )
        
        self.chunks_collection = self.client.get_or_create_collection(
            name="document_chunks",
            embedding_function=self.default_ef
        )
        
        logger.info(f"ChromaVectorStore initialized at {persist_directory}")
        
    def add_document(self, doc_id: str, content: str, metadata: Dict = None) -> None:
        """Add a full document to the store"""
        self.documents_collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[metadata or {}]
        )
        
    def add_chunks(self, doc_id: str, chunks: List[Dict[str, Any]]) -> None:
        """Add document chunks to the store"""
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        texts = [chunk.get("text", chunk.get("content", "")) for chunk in chunks]
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            metadata = {
                "document_id": doc_id,
                "chunk_index": i,
                "heading": chunk.get("title", ""),
            }
            # Add any additional metadata from the chunk
            if "metadata" in chunk:
                metadata.update(chunk["metadata"])
            metadatas.append(metadata)
        
        self.chunks_collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
    def search_similar_chunks(self, query: str, doc_id: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """Search for chunks similar to the query"""
        # Filter by document if specified
        where_filter = {"document_id": doc_id} if doc_id else None
        
        results = self.chunks_collection.query(
            query_texts=[query],
            n_results=limit,
            where=where_filter
        )
        
        # Format results
        formatted_results = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                formatted_results.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "id": results["ids"][0][i] if results["ids"] else f"result_{i}",
                    "score": results["distances"][0][i] if "distances" in results and results["distances"] else None
                })
        
        return formatted_results
        
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Retrieve a document by ID"""
        results = self.documents_collection.get(ids=[doc_id])
        
        if not results["documents"]:
            return None
            
        return {
            "id": doc_id,
            "content": results["documents"][0],
            "metadata": results["metadatas"][0] if results["metadatas"] else {}
        }
        
    def get_document_chunks(self, doc_id: str) -> List[Dict]:
        """Retrieve all chunks for a document"""
        results = self.chunks_collection.get(
            where={"document_id": doc_id}
        )
        
        formatted_results = []
        for i, doc in enumerate(results["documents"]):
            formatted_results.append({
                "id": results["ids"][i],
                "text": doc,
                "metadata": results["metadatas"][i] if results["metadatas"] else {}
            })
            
        return formatted_results
            
    def delete_document(self, doc_id: str) -> None:
        """Delete a document and its chunks"""
        # Delete the document
        try:
            self.documents_collection.delete(ids=[doc_id])
        except Exception as e:
            logger.warning(f"Error deleting document {doc_id}: {e}")
            
        # Delete all chunks for the document
        try:
            self.chunks_collection.delete(
                where={"document_id": doc_id}
            )
        except Exception as e:
            logger.warning(f"Error deleting chunks for document {doc_id}: {e}")


class DocumentProcessor:
    """Enhanced document processor with semantic chunking and Chroma integration."""
    
    def __init__(self, api_url: str = "http://localhost:8000", 
                 chroma_persist_dir: Optional[str] = "chroma_db"):
        """
        Initialize the document processor.
        
        Args:
            api_url: URL of the LLM server API
            chroma_persist_dir: Directory to persist Chroma DB, None to disable
        """
        self.api_url = api_url
        self.session = None
        self.initialized = False
        self.progress_callback = None
        self.document_embedding = None
        self.has_embedding = False
        
        # Initialize Chroma if available
        self.chroma_enabled = CHROMA_AVAILABLE and chroma_persist_dir is not None
        if self.chroma_enabled:
            try:
                self.vector_store = ChromaVectorStore(persist_directory=chroma_persist_dir)
                logger.info(f"Chroma vector store initialized at {chroma_persist_dir}")
            except Exception as e:
                logger.error(f"Error initializing Chroma DB: {e}")
                self.chroma_enabled = False
    
    async def initialize(self):
        """Initialize HTTP session."""
        if not self.initialized:
            self.session = aiohttp.ClientSession()
            self.initialized = True
            logger.info("DocumentProcessor initialized with session")
    
    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            self.initialized = False
            logger.info("DocumentProcessor session closed")
    
    def set_document_embedding(self, embedding):
        """Set document embedding for similarity-based processing"""
        self.document_embedding = embedding
        self.has_embedding = True
        
    def hierarchical_chunk_document(self, source: str) -> List[Dict[str, str]]:
        """Use docling for hierarchical document chunking"""
        if not DOCLING_AVAILABLE:
            logger.warning("Docling not available - falling back to regular chunking")
            # Fall back to regular chunking
            chunks = self.split_document(source, 2000, 200)
            return [{"title": f"Chunk {i+1}", "text": chunk} for i, chunk in enumerate(chunks)]
        
        try:
            def chunk_document_source(source_text: str) -> Iterator[BaseChunk]:
                """Read the document and perform a hierarchical chunking"""
                converter = DocumentConverter()
                chunks = HierarchicalChunker().chunk(converter.convert(source=source_text).document)
                return chunks

            def merge_chunks(chunks: Iterator[BaseChunk]) -> Iterator[Dict[str, str]]:
                """Merge chunks having the same headings"""
                prior_headings: Optional[List[str]] = None
                document: Dict[str, str] = {}
                
                for chunk in chunks:
                    text = chunk.text.replace('\r\n', '\n')
                    # Use first two heading levels
                    current_headings = chunk.meta.headings[:2] if hasattr(chunk.meta, 'headings') else []
                    
                    if prior_headings != current_headings:
                        if document:
                            yield document
                        prior_headings = current_headings
                        document = {'title': " - ".join(current_headings) if current_headings else "Untitled Section", 
                                   'text': text}
                    else:
                        document['text'] += f"\n\n{text}"
                
                if document:
                    yield document
            
            # Process the document
            chunks = chunk_document_source(source)
            documents = list(merge_chunks(chunks))
            
            logger.info(f"Hierarchical chunking produced {len(documents)} semantic sections")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in hierarchical chunking: {e}")
            # Fall back to regular chunking
            chunks = self.split_document(source, 2000, 200)
            return [{"title": f"Chunk {i+1}", "text": chunk} for i, chunk in enumerate(chunks)]
    
    def split_document(self, text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        """
        Split document text into overlapping chunks.
        
        Args:
            text: The document text to split
            chunk_size: Maximum size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            A list of text chunks
        """
        # Split by paragraphs first to maintain context
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If paragraph is too big, split it further
            if len(paragraph) > chunk_size:
                # Add any current chunk content first
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Split long paragraph into smaller pieces
                for i in range(0, len(paragraph), chunk_size - overlap):
                    chunk = paragraph[i:i + chunk_size]
                    if i + chunk_size >= len(paragraph):
                        current_chunk = chunk
                    else:
                        chunks.append(chunk)
            
            # Normal case: add paragraph to current chunk if it fits
            elif len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            
            # If adding paragraph exceeds chunk size, start a new chunk
            else:
                chunks.append(current_chunk)
                current_chunk = paragraph
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def split_document_semantic(self, text: str, chunk_size: int = 2000, overlap: int = 200) -> List[Dict[str, str]]:
        """
        Split document into semantic chunks using embeddings or hierarchical chunking.
        This creates more meaningful chunks than simple character-based chunking.
        """
        # Try hierarchical chunking first if docling is available
        if DOCLING_AVAILABLE:
            try:
                return self.hierarchical_chunk_document(text)
            except Exception as e:
                logger.warning(f"Hierarchical chunking failed: {e}, trying embedding-based chunking")
        
        # Fall back to embedding-based semantic chunking
        try:
            import torch
            from sklearn.cluster import KMeans
            
            # Split by paragraphs first
            paragraphs = text.split('\n\n')
            
            # If we have very few paragraphs, fall back to regular chunking
            if len(paragraphs) < 3:
                chunks = self.split_document(text, chunk_size, overlap)
                return [{"title": f"Chunk {i+1}", "text": chunk} for i, chunk in enumerate(chunks)]
            
            # Create embeddings (assuming we have access to an embedding model)
            if hasattr(self, 'embedding_model'):
                paragraph_embeddings = self.embedding_model.encode(paragraphs)
            elif self.has_embedding and hasattr(self, 'document_embedding'):
                # If we don't have direct access to the embedding model but have document embedding
                # We'll use a different approach
                logger.warning("No direct access to embedding model - using simplified clustering")
                chunks = self.split_document(text, chunk_size, overlap)
                return [{"title": f"Chunk {i+1}", "text": chunk} for i, chunk in enumerate(chunks)]
            else:
                # Fallback to regular chunking if no embedding capabilities
                chunks = self.split_document(text, chunk_size, overlap)
                return [{"title": f"Chunk {i+1}", "text": chunk} for i, chunk in enumerate(chunks)]
            
            # Determine number of clusters based on text length
            n_clusters = max(1, min(10, len(text) // chunk_size))
            
            # Use K-means to cluster paragraphs by semantic similarity
            kmeans = KMeans(n_clusters=n_clusters)
            clusters = kmeans.fit_predict(paragraph_embeddings)
            
            # Group paragraphs by cluster
            cluster_paragraphs = [[] for _ in range(n_clusters)]
            for i, cluster_id in enumerate(clusters):
                cluster_paragraphs[cluster_id].append(paragraphs[i])
            
            # Create chunks from each cluster
            semantic_chunks = []
            for i, cluster in enumerate(cluster_paragraphs):
                cluster_text = "\n\n".join(cluster)
                
                # If cluster is too large, split it further
                if len(cluster_text) <= chunk_size:
                    semantic_chunks.append({
                        "title": f"Topic Group {i+1}",
                        "text": cluster_text
                    })
                else:
                    # Split large clusters into smaller chunks
                    sub_chunks = self.split_document(cluster_text, chunk_size, overlap)
                    for j, sub_chunk in enumerate(sub_chunks):
                        semantic_chunks.append({
                            "title": f"Topic Group {i+1}.{j+1}",
                            "text": sub_chunk
                        })
            
            return semantic_chunks
            
        except Exception as e:
            logger.warning(f"Error in semantic chunking: {e}, falling back to regular chunking")
            chunks = self.split_document(text, chunk_size, overlap)
            return [{"title": f"Chunk {i+1}", "text": chunk} for i, chunk in enumerate(chunks)]

    def apply_prompt_template(self, chunk: str, prompt_template: str) -> str:
        """
        Apply a prompt template to a document chunk.
        
        Args:
            chunk: The document chunk
            prompt_template: The prompt template with {chunk} placeholder
            
        Returns:
            The formatted prompt with the chunk inserted
        """
        return prompt_template.format(chunk=chunk)

    async def process_chunk(self, chunk: Union[str, Dict], prompt_template: str, model_id: Optional[str] = None) -> str:
        """
        Process a single chunk with the LLM server.
        
        Args:
            chunk: The document chunk (string or dict with "text" key)
            prompt_template: The prompt template 
            model_id: Optional model ID to use
            
        Returns:
            The processed chunk result
        """
        if not self.session:
            await self.initialize()

        # Extract text if chunk is a dictionary
        chunk_text = chunk["text"] if isinstance(chunk, dict) else chunk
        chunk_title = chunk.get("title", "") if isinstance(chunk, dict) else ""
        
        if self.progress_callback:
            chunk_preview = chunk_text[:50] + "..." if len(chunk_text) > 50 else chunk_text
            await self.progress_callback({
                "stage": "processing_chunk",
                "message": f"Processing chunk: {chunk_title or chunk_preview}",
                "progress": None
            })
        
        try:
            # Format the prompt with the chunk
            formatted_prompt = prompt_template
            if "{chunk}" in prompt_template:
                formatted_prompt = prompt_template.format(chunk=chunk_text)
            elif "{text}" in prompt_template:
                formatted_prompt = prompt_template.format(text=chunk_text)
                
            # Add title context if available and not already in prompt
            if chunk_title and "{title}" not in formatted_prompt:
                if "title:" not in formatted_prompt.lower() and "heading:" not in formatted_prompt.lower():
                    formatted_prompt = f"Title: {chunk_title}\n\n{formatted_prompt}"

            # Use chat/generate endpoint for more consistent results
            async with self.session.post(
                f"{self.api_url}/chat/generate",
                json={"prompt": formatted_prompt},
                timeout=60
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error processing chunk: {error_text}")
                    return f"Error: {error_text}"
                
                result_json = await response.json()
                response_text = result_json.get("response", "No response received")
                
                # Clean the response text to avoid code artifacts
                cleaned_response = self._clean_response(response_text)
                return cleaned_response
            
        except Exception as e:
            logger.error(f"Exception processing chunk: {e}")
            return f"Error: {str(e)}"

    async def process_document(
        self, 
        document_text: str, 
        prompt_template: str,
        chunk_size: int = 2000,
        overlap: int = 200,
        model_id: Optional[str] = None,
        max_concurrent: int = 3,
        combine_prompt: Optional[str] = None,
        use_embeddings: bool = True,
        progress_callback: Optional[Callable] = None,
        document_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Tuple[str, List[Dict]]:
        """
        Process an entire document by chunks and combine results.
        
        Returns:
            Tuple of (combined result text, list of processed chunks)
        """
        
        self.progress_callback = progress_callback
        processed_chunks = []
        
        if progress_callback:
            await progress_callback({
                "stage": "chunking", 
                "progress": 10,
                "message": "Starting document chunking"
            })

        if not document_text:
            logger.error("Empty document text provided")
            return "Error: Empty document provided", []
        
        try:
            # Choose chunking method based on available tools and settings
            if use_embeddings and (hasattr(self, 'has_embedding') and self.has_embedding or DOCLING_AVAILABLE):
                chunks = self.split_document_semantic(document_text, chunk_size, overlap)
                logger.info(f"Split document into {len(chunks)} semantic chunks")
            else:
                basic_chunks = self.split_document(document_text, chunk_size, overlap)
                chunks = [{"title": f"Chunk {i+1}", "text": chunk} for i, chunk in enumerate(basic_chunks)]
                logger.info(f"Split document into {len(chunks)} regular chunks")
            
            if not chunks:
                return "Error: Document splitting produced no chunks", []
                
            # Store in Chroma if enabled and document_id is provided
            if self.chroma_enabled and document_id:
                try:
                    # Store the full document
                    self.vector_store.add_document(
                        doc_id=document_id,
                        content=document_text,
                        metadata=metadata or {}
                    )
                    
                    # Store the chunks
                    self.vector_store.add_chunks(
                        doc_id=document_id,
                        chunks=chunks
                    )
                    
                    logger.info(f"Stored document and {len(chunks)} chunks in Chroma DB")
                except Exception as e:
                    logger.error(f"Error storing document in Chroma: {e}")

            # Process chunks with semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)
            
            # Clean up the prompt templates - make sure they're properly formatted
            clean_prompt_template = prompt_template.strip()
            clean_combine_prompt = combine_prompt.strip() if combine_prompt else None
            
            async def process_with_rate_limit(chunk):
                async with semaphore:
                    result = await self.process_chunk(chunk, clean_prompt_template, model_id)
                    # Store processed chunk
                    processed_chunk = {
                        "title": chunk["title"] if isinstance(chunk, dict) else f"Chunk {len(processed_chunks)+1}",
                        "text": chunk["text"] if isinstance(chunk, dict) else chunk,
                        "processed_result": result
                    }
                    processed_chunks.append(processed_chunk)
                    return result
            
            # For the processing_chunk message:
            if self.progress_callback:
                await self.progress_callback({
                    "stage": "processing_chunks", 
                    "progress": 30,
                    "message": f"Processing chunks ({len(chunks)} total chunks)"
                })

            # Process all chunks concurrently (with rate limiting)
            start_time = time.time()
            chunk_results = await asyncio.gather(
                *[process_with_rate_limit(chunk) for chunk in chunks]
            )
            processing_time = time.time() - start_time
            
            # Check for errors and filter them out
            valid_results = [result for result in chunk_results if not result.startswith("Error")]
            
            if not valid_results:
                return "Error: All chunks failed processing. Please try again.", processed_chunks
            
            # Combine the results
            if len(valid_results) == 1:
                logger.info("Only one chunk processed, returning result directly")
                return valid_results[0], processed_chunks
            
            # If a combining prompt is provided, use it
            if self.progress_callback:
                await self.progress_callback({
                    "stage": "combining", 
                    "progress": 80,
                    "message": f"Combining {len(valid_results)} processed chunks"
                })
                
            if clean_combine_prompt:
                # Format chunks for combining
                chunk_list = "\n\n".join([
                    f"Chunk {i+1} ({processed_chunks[i]['title']}):\n{result.strip()}"
                    for i, result in enumerate(valid_results)
                ])
                
                final_prompt = clean_combine_prompt.format(chunks=chunk_list)
                logger.info(f"Using combine prompt to merge {len(valid_results)} chunks")
                combined_result = await self.process_chunk("", final_prompt, model_id)
                
                if not combined_result or combined_result.strip() == "":
                    logger.warning("Empty combined result, falling back to concatenation")
                    combined_result = "\n\n".join([
                        f"Section {i+1} ({processed_chunks[i]['title']}):\n{result}" 
                        for i, result in enumerate(valid_results)
                    ])
                
                return combined_result, processed_chunks
            
            # Default combination approach (fallback)
            logger.info(f"Using default concatenation to combine {len(valid_results)} chunks")
            return "\n\n".join([
                f"Section {i+1} ({processed_chunks[i]['title']}):\n{result}" 
                for i, result in enumerate(valid_results)
            ]), processed_chunks
            
        except Exception as e:
            logger.exception(f"Error in process_document: {e}")
            return f"Error processing document: {str(e)}", processed_chunks

    async def retrieve_similar_chunks(self, query: str, doc_id: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """Retrieve chunks similar to the query from Chroma DB"""
        if not self.chroma_enabled:
            return []
            
        try:
            return self.vector_store.search_similar_chunks(
                query=query, 
                doc_id=doc_id,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Error retrieving similar chunks: {e}")
            return []

    def _clean_response(self, text: str) -> str:
        """Clean the response text to remove unwanted artifacts."""
        # Remove any triple backticks and code block indicators
        text = re.sub(r'```[\w]*\n', '', text)
        text = re.sub(r'```', '', text)
        
        # Remove Python code artifacts that might appear
        patterns_to_remove = [
            r'def\s+\w+\(.*?\):.*?return.*?',
            r'print\(.*?\)',
            r'import.*?\n',
            r'from\s+.*?\s+import\s+.*?\n',
            r'""".*?"""',
            r"'''.*?'''"
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.DOTALL)
        
        # Clean up any excess whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        return text
        
# Example prompt templates
DEFAULT_CHUNK_PROMPT = """
Please analyze the following document chunk and extract the key information:

{chunk}

Provide a concise summary of the main points from this section. Ensure correct spelling of all words in the final response:
"""

DEFAULT_COMBINE_PROMPT = """
You have been provided with processed chunks from a document. Please synthesize these into a cohesive final response. Ensure correct spelling of all words in the final response:

{chunks}

Create a comprehensive and coherent response that combines all the information from these chunks. Verify all spelling and grammar are correct for your response.
"""