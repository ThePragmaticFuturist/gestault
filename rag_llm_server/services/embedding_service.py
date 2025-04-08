# services/embedding_service.py
import time
import logging
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer
from core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global variable to hold the loaded model ---
# Avoids reloading the model on every request/task
_embedding_model: Optional[SentenceTransformer] = None

def _load_embedding_model() -> SentenceTransformer:
    """Loads the Sentence Transformer model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model '{settings.EMBEDDING_MODEL_NAME}' onto device '{settings.EMBEDDING_DEVICE}'...")
        start_time = time.time()
        try:
            # Check for CUDA availability if requested
            if settings.EMBEDDING_DEVICE == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                settings.EMBEDDING_DEVICE = "cpu"

            _embedding_model = SentenceTransformer(
                settings.EMBEDDING_MODEL_NAME,
                device=settings.EMBEDDING_DEVICE,
                cache_folder=str(settings.HUGGINGFACE_HUB_CACHE.resolve()) # Use configured cache path
            )
            # You might want to warm up the model here if needed
            # _embedding_model.encode(["Warm-up sentence."])
            load_time = time.time() - start_time
            logger.info(f"Embedding model loaded successfully in {load_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{settings.EMBEDDING_MODEL_NAME}': {e}")
            # Depending on requirements, you might want to raise the exception
            # or handle it gracefully (e.g., disable embedding features)
            raise RuntimeError(f"Could not load embedding model: {e}") from e
    return _embedding_model

def get_embedding_model() -> SentenceTransformer:
    """Returns the loaded embedding model, loading it if necessary."""
    # This function ensures the model is loaded only once.
    return _load_embedding_model()

def generate_embeddings(texts: List[str]) -> Optional[List[List[float]]]:
    """
    Generates embeddings for a list of texts using the loaded model.
    Ensures the output is List[List[float]] for compatibility.
    """
    if not texts:
        return []
    try:
        model = get_embedding_model()
        logger.info(f"Generating embeddings for {len(texts)} text chunks...")
        start_time = time.time()

        # Generate embeddings (may return tensors or lists depending on settings/device)
        embeddings_output = model.encode(
            texts,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=False # Keep this, we'll convert tensor manually if needed
        )
        gen_time = time.time() - start_time

        # --- ### NEW: Explicit Conversion to List[List[float]] ### ---
        final_embeddings: List[List[float]] = []
        if isinstance(embeddings_output, list):
            if len(embeddings_output) > 0:
                if isinstance(embeddings_output[0], torch.Tensor):
                    # Input was List[Tensor], convert each tensor
                    logger.debug("Converting List[Tensor] output to List[List[float]].")
                    final_embeddings = [tensor.tolist() for tensor in embeddings_output]
                elif isinstance(embeddings_output[0], list):
                    # Already List[List[float]] (or similar)
                    logger.debug("Output is already List[List[...]], using as is.")
                    final_embeddings = embeddings_output
                else:
                    logger.error(f"Unexpected item type in embedding list: {type(embeddings_output[0])}")
                    return None
            # If empty list, final_embeddings remains []
        elif isinstance(embeddings_output, torch.Tensor):
             # Handle case where encode might return a single tensor for batch=1 or single input?
             logger.debug("Converting single Tensor output to List[List[float]].")
             # Need to wrap in another list to match ChromaDB expectation for multiple embeddings
             single_list = embeddings_output.tolist()
             # Check if it's already nested [[...]] or just [...]
             if isinstance(single_list[0], list):
                 final_embeddings = single_list
             else:
                 final_embeddings = [single_list] # Wrap the single list
        else:
             logger.error(f"Unexpected output type from model.encode: {type(embeddings_output)}")
             return None
        # --- ### END EXPLICIT CONVERSION ### ---


        logger.info(f"Generated and converted {len(final_embeddings)} embeddings in {gen_time:.2f} seconds.")
        return final_embeddings

    except Exception as e:
        logger.error(f"Error generating embeddings: {e}", exc_info=True)
        return None

# --- Optional: Pre-load model during startup ---
# You could call get_embedding_model() once when the app starts
# in main.py's startup event if you want the loading delay to happen then.