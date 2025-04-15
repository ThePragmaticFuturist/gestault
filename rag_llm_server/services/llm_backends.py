# services/llm_backends.py
import logging
import abc # Abstract Base Classes
from typing import Dict, Any, Optional, AsyncGenerator

import torch

from transformers import PreTrainedTokenizerBase, PreTrainedModel

import httpx # For async API calls
from core.config import settings

logger = logging.getLogger(__name__)

# --- ADD Google AI Imports ---
try:
    import google.generativeai as genai
    google_ai_available = True
except ImportError:
    google_ai_available = False
    genai = None # Define genai as None if import fails
# --- END ADD ---

# --- Base Class ---
class LLMBackendBase(abc.ABC):
    """Abstract base class for different LLM backends."""

    @abc.abstractmethod
    async def generate(self, prompt: str, config: Dict[str, Any]) -> Optional[str]:
        """Generates text based on the prompt and configuration."""
        pass

    @abc.abstractmethod
    def get_status_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representing the backend's current status."""
        pass

    async def unload(self):
        """Optional method to clean up resources (e.g., unload local model)."""
        logger.info(f"Unload called for {self.__class__.__name__}, default implementation does nothing.")
        pass # Default does nothing


# --- Local Transformers Backend ---
# We'll integrate the existing logic from llm_service into this class later
class LocalTransformersBackend(LLMBackendBase):
    def __init__(self):
        # This backend will hold the pipeline, tokenizer, etc.
        # These will be populated by a dedicated loading function later
        self.pipeline = None
        self.tokenizer = None
        self.model_name_or_path: Optional[str] = None
        self.load_config: Dict[str, Any] = {}
        self.max_model_length: Optional[int] = None
        logger.info("LocalTransformersBackend initialized (inactive).")

    def generate(self, prompt: str, config: Dict[str, Any]) -> Optional[str]:
        # This will contain the refined pipeline call logic from generate_text
        # if self.pipeline is None or self.tokenizer is None:
        #     logger.error("Local pipeline/tokenizer not loaded.")
        #     return None
        if self.model is None or self.tokenizer is None:
            logger.error("Local model/tokenizer not loaded.")
            return None

        try:
            # logger.info("Generating text with local Transformers pipeline...")
            # generation_keys = {"max_new_tokens", "temperature", "top_p", "top_k"}
            # generation_kwargs = {k: config[k] for k in generation_keys if k in config}
            # generation_kwargs["do_sample"] = True
            # generation_kwargs["num_return_sequences"] = 1
            # generation_kwargs["repetition_penalty"] = config.get("repetition_penalty", 1.15) # Example: get from config

            # logger.debug(f"Passing generation kwargs to pipeline: {generation_kwargs}")

            logger.info("Generating text with local model.generate()...")

            # 1. Tokenize the input prompt
            # Ensure padding side is set correctly if needed (often left for generation)
            # self.tokenizer.padding_side = "left" # Uncomment if model requires left padding
            inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # 2. Prepare generation arguments (filter from global config)
            generation_keys = {"max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty"}
            generation_kwargs = {k: config[k] for k in generation_keys if k in config}
            # Add other necessary generate() arguments
            generation_kwargs["do_sample"] = True
            # Add pad_token_id if tokenizer has it, otherwise use eos_token_id
            generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id


            logger.debug(f"Passing generation kwargs to model.generate: {generation_kwargs}")

            # 3. Call model.generate()
            # Ensure no blocking operations outside executor if needed (generate itself is blocking)
            with torch.no_grad(): # Disable gradient calculation for inference
                 outputs = self.model.generate(
                     **inputs,
                     **generation_kwargs
                 )

            # 4. Decode the output tokens, skipping prompt tokens and special tokens
            # outputs[0] contains the full sequence (prompt + generation)
            # input_ids_len = inputs['input_ids'].shape[1] # Length of the input prompt tokens
            # generated_token_ids = outputs[0][input_ids_len:] # Get only generated tokens
            # result = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

            # --- OR: Decode the full output and clean (simpler) ---
            full_generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.debug(f"LLM Raw Decoded Output (skip_special_tokens=True):\n{full_generated_text}")

            # Use the same prompt cleaning logic as before (find marker)
            response_marker = "\n### RESPONSE:"
            prompt_marker_pos = prompt.rfind(response_marker)
            # Clean the *original* prompt (remove marker part) before checking startswith
            prompt_base = prompt[:prompt_marker_pos] if prompt_marker_pos != -1 else prompt

            if full_generated_text.startswith(prompt_base):
                 # Try extracting after the *original* full prompt if marker logic fails
                 result = full_generated_text[len(prompt):].strip()
                 if not result and prompt_marker_pos != -1: # If removing full prompt failed, try marker again
                     response_start_pos = prompt_marker_pos + len(response_marker)
                     result = full_generated_text[response_start_pos:].strip()
                 logger.debug("Extracted response by removing prompt.")

            elif prompt_marker_pos != -1 and response_marker in full_generated_text:
                 # Maybe prompt isn't exactly at start, find marker in output
                  response_start_pos = full_generated_text.rfind(response_marker) + len(response_marker)
                  result = full_generated_text[response_start_pos:].strip()
                  logger.debug("Extracted response by finding marker in output.")
            else:
                 # Fallback: Assume output is only the response (less likely with generate)
                 logger.warning("Cannot determine prompt end in generated text. Using full decoded text.")
                 result = full_generated_text.strip()


            if not result:
                 logger.warning("Extraction resulted in an empty string.")

            logger.info("Local LLM text generation complete.")
            return result

        except Exception as e:
            logger.error(f"Local LLM text generation failed: {e}", exc_info=True)
            return None


    def get_status_dict(self) -> Dict[str, Any]:

        # --- ADD CHECK FOR self.model ---
         model_device = None
         # Check if model exists FIRST before trying to access attributes
         if self.model is not None:
              # Now check if device attribute exists (should always for loaded HF models)
              if hasattr(self.model, 'device'):
                   try:
                        # Safely get the device
                        model_device = str(self.model.device)
                   except Exception as e:
                        logger.warning(f"Could not determine model device even though model exists: {e}")
                        model_device = "Error fetching device"
              else:
                   # Should be rare, but handle case where model exists but no device attr
                    logger.warning("Model object exists but lacks 'device' attribute.")
                    model_device = "Unknown device"

         # --- End Check ---

         return {
             "active_model": self.model_name_or_path,
             "load_config": self.load_config,
             "max_model_length": self.max_model_length,
             "model_device": model_device, # Use the safely determined value
         }

    async def unload(self):
        """Unloads the local model and clears memory."""
        logger.info(f"Unloading local model '{self.model_name_or_path}'...")
        # Keep references to delete them explicitly
        model_to_del = self.model
        tokenizer_to_del = self.tokenizer

        self.model = None # Clear model ref
        self.tokenizer = None
        self.model_name_or_path = None
        self.load_config = {}
        self.max_model_length = None

        del model_to_del
        del tokenizer_to_del

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Local model unloaded and memory cleared.")


# --- Ollama Backend ---
class OllamaBackend(LLMBackendBase):
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/generate"
        self.model_name = model_name
        # Use a shared client for potential connection pooling
        self.client = httpx.AsyncClient(timeout=120.0) # Increase timeout
        logger.info(f"OllamaBackend initialized: URL='{self.base_url}', Model='{self.model_name}'")

    async def generate(self, prompt: str, config: Dict[str, Any]) -> Optional[str]:
        logger.info(f"Generating text via Ollama backend (model: {self.model_name})...")
        # Map standard config keys to Ollama options
        options = {
            "temperature": config.get("temperature", settings.DEFAULT_LLM_TEMPERATURE),
            "top_k": config.get("top_k", settings.DEFAULT_LLM_TOP_K),
            "top_p": config.get("top_p", settings.DEFAULT_LLM_TOP_P),
            "num_predict": config.get("max_new_tokens", settings.DEFAULT_LLM_MAX_NEW_TOKENS),
            # Add other Ollama options if needed (e.g., stop sequences)
            # "stop": ["\n###"]
        }
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False, # Request non-streaming response
            "options": options,
            # "raw": True # If prompt includes special formatting Ollama might remove
        }
        logger.debug(f"Ollama Request Payload: {payload}")

        try:
            response = await self.client.post(self.api_url, json=payload)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            result_data = response.json()
            generated_text = result_data.get("response")
            if generated_text:
                logger.info("Ollama generation successful.")
                logger.debug(f"Ollama Response: {generated_text[:500]}...")
                return generated_text.strip()
            else:
                logger.error(f"Ollama response missing 'response' field. Full response: {result_data}")
                return None
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API request failed (HTTP {e.response.status_code}): {e.response.text}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Ollama API request failed (Network/Connection Error): {e}")
            return None
        except Exception as e:
            logger.error(f"Ollama generation failed unexpectedly: {e}", exc_info=True)
            return None

    def get_status_dict(self) -> Dict[str, Any]:
        return {
            "active_model": self.model_name,
            "backend_url": self.base_url,
            # Add check if Ollama server is reachable? Maybe later.
        }

    async def unload(self):
        # Close the httpx client when unloading
        await self.client.aclose()
        logger.info("OllamaBackend httpx client closed.")

# --- vLLM Backend (OpenAI Compatible) ---
class VLLMBackend(LLMBackendBase):
    def __init__(self, base_url: str, model_name: str):
        # vLLM's OpenAI endpoint is usually at /v1
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/v1/chat/completions" # Use Chat Completion endpoint
        self.model_name = model_name # This is passed in the payload
        self.client = httpx.AsyncClient(timeout=120.0)
        logger.info(f"VLLMBackend initialized: URL='{self.base_url}', Model='{self.model_name}' (using Chat API)")

    async def generate(self, prompt: str, config: Dict[str, Any]) -> Optional[str]:
        logger.info(f"Generating text via vLLM backend (model: {self.model_name})...")
        # Format prompt for Chat Completion API
        messages = [{"role": "user", "content": prompt}]

        # Map config keys to OpenAI parameters
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": config.get("temperature", settings.DEFAULT_LLM_TEMPERATURE),
            "top_p": config.get("top_p", settings.DEFAULT_LLM_TOP_P),
            "max_tokens": config.get("max_new_tokens", settings.DEFAULT_LLM_MAX_NEW_TOKENS),
            "stream": False,
            # top_k is not standard for OpenAI Chat API
            # Add other supported params if needed (e.g., stop)
        }
        logger.debug(f"vLLM Request Payload: {payload}")

        try:
            response = await self.client.post(self.api_url, json=payload)
            response.raise_for_status()

            result_data = response.json()
            # Extract content from the first choice's message
            if result_data.get("choices") and len(result_data["choices"]) > 0:
                 message = result_data["choices"][0].get("message")
                 if message and message.get("content"):
                      generated_text = message["content"]
                      logger.info("vLLM generation successful.")
                      logger.debug(f"vLLM Response: {generated_text[:500]}...")
                      return generated_text.strip()

            logger.error(f"vLLM response missing expected structure. Full response: {result_data}")
            return None

        except httpx.HTTPStatusError as e:
            logger.error(f"vLLM API request failed (HTTP {e.response.status_code}): {e.response.text}")
            return None
        except httpx.RequestError as e:
            logger.error(f"vLLM API request failed (Network/Connection Error): {e}")
            return None
        except Exception as e:
            logger.error(f"vLLM generation failed unexpectedly: {e}", exc_info=True)
            return None

    def get_status_dict(self) -> Dict[str, Any]:
         return {
             "active_model": self.model_name,
             "backend_url": self.base_url,
         }

    async def unload(self):
        await self.client.aclose()
        logger.info("VLLMBackend httpx client closed.")


# --- InstructLab Backend (Placeholder) ---
class InstructLabBackend(LLMBackendBase):
    def __init__(self, base_url: str, model_name: str):
        # Assuming similar OpenAI compatibility for now
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/v1/chat/completions" # Adjust if needed
        self.model_name = model_name
        self.client = httpx.AsyncClient(timeout=120.0)
        logger.info(f"InstructLabBackend initialized: URL='{self.base_url}', Model='{self.model_name}'")

    async def generate(self, prompt: str, config: Dict[str, Any]) -> Optional[str]:
        # Implement similarly to VLLMBackend, adjust payload/response parsing if needed
        logger.warning("InstructLabBackend generate() not fully implemented yet.")
        # --- Placeholder Implementation (like VLLM) ---
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model_name, "messages": messages,
            "temperature": config.get("temperature", settings.DEFAULT_LLM_TEMPERATURE),
            "top_p": config.get("top_p", settings.DEFAULT_LLM_TOP_P),
            "max_tokens": config.get("max_new_tokens", settings.DEFAULT_LLM_MAX_NEW_TOKENS),
            "stream": False,
        }
        try:
            response = await self.client.post(self.api_url, json=payload)
            response.raise_for_status()
            result_data = response.json()
            if result_data.get("choices") and result_data["choices"][0].get("message"):
                return result_data["choices"][0]["message"].get("content", "").strip()
            return None
        except Exception as e:
             logger.error(f"InstructLab generation failed: {e}", exc_info=True)
             return None
        # --- End Placeholder ---


    def get_status_dict(self) -> Dict[str, Any]:
         return {
             "active_model": self.model_name,
             "backend_url": self.base_url,
         }

    async def unload(self):
        await self.client.aclose()
        logger.info("InstructLabBackend httpx client closed.")

# --- NEW: Gemini Backend ---
class GeminiBackend(LLMBackendBase):
    def __init__(self, api_key: str, model_name: str):
        if not google_ai_available or genai is None:
             raise ImportError("Attempted to initialize GeminiBackend, but 'google-generativeai' library is not installed. Please run 'pip install google-generativeai'.")

        self.model_name = model_name
        self.api_key = api_key
        try:
            # Configure the API key globally for the library (common pattern)
            genai.configure(api_key=self.api_key)
            # Create the specific model instance
            self.model = genai.GenerativeModel(model_name=self.model_name)
            logger.info(f"GeminiBackend initialized: Model='{self.model_name}' configured.")
        except Exception as e:
            logger.error(f"Failed to configure Gemini or initialize model '{self.model_name}': {e}", exc_info=True)
            raise ValueError(f"Gemini configuration/initialization failed: {e}") from e

    async def generate(self, prompt: str, config: Dict[str, Any]) -> Optional[str]:
        logger.info(f"Generating text via Gemini backend (model: {self.model_name})...")

        # Map internal config keys to Gemini GenerationConfig keys
        gemini_config = genai.types.GenerationConfig(
            # candidate_count=1, # Default is 1
            # stop_sequences=["\n###"], # Example stop sequence
            max_output_tokens=config.get("max_new_tokens", settings.DEFAULT_LLM_MAX_NEW_TOKENS),
            temperature=config.get("temperature", settings.DEFAULT_LLM_TEMPERATURE),
            top_p=config.get("top_p", settings.DEFAULT_LLM_TOP_P),
            top_k=config.get("top_k", settings.DEFAULT_LLM_TOP_K), # Note: top_k >= 1
        )
        # Ensure top_k is valid for Gemini (must be >= 1)
        if gemini_config.top_k is not None and gemini_config.top_k < 1:
            logger.warning(f"Gemini requires top_k >= 1. Received {gemini_config.top_k}, setting to 1.")
            gemini_config.top_k = 1

        logger.debug(f"Gemini GenerationConfig: {gemini_config}")

        try:
            # Use the asynchronous generation method
            response = await self.model.generate_content_async(
                contents=prompt, # Send the full prompt string
                generation_config=gemini_config,
                # safety_settings=... # Optional: configure safety settings
            )

            # Extract text, handling potential blocks or missing text
            if response.text:
                 generated_text = response.text
                 logger.info("Gemini generation successful.")
                 logger.debug(f"Gemini Response: {generated_text[:500]}...")
                 return generated_text.strip()
            else:
                 # Log details if text is missing (e.g., blocked by safety filters)
                 logger.warning(f"Gemini response did not contain text. Prompt feedback: {response.prompt_feedback}. Finish reason: {getattr(response.candidates[0], 'finish_reason', 'N/A')}")
                 # Consider checking response.candidates[0].content.parts if text is missing
                 return "[ERROR: Gemini response empty or blocked]"

        except Exception as e:
            # Catch potential google API errors or other issues
            logger.error(f"Gemini generation failed unexpectedly: {e}", exc_info=True)
            # Example specific error check:
            # if isinstance(e, google.api_core.exceptions.PermissionDenied):
            #     return "[ERROR: Gemini API key invalid or lacks permissions]"
            return f"[ERROR: Gemini generation failed - {type(e).__name__}]"

    def get_status_dict(self) -> Dict[str, Any]:
         return {
             "active_model": self.model_name,
             # Can add more details if needed, e.g., check API reachability
         }

    async def unload(self):
        # No explicit unload/cleanup needed for the genai library typically
        logger.info("GeminiBackend unload called (no action needed).")
        pass # Nothing specific to unload
