import os
import logging
from typing import List, Dict, Optional
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class GeminiService:
    """
    Service for handling Gemini AI API interactions.
    Based on user's existing implementation from LAMA project.
    """
    
    def __init__(self):
        self.logger = logger
        
        # Get API key from environment
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # Configuration defaults
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
        self.max_output_tokens = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "8192"))
        self.temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
        
        self.logger.info(">>>>>> Gemini Service initialized <<<<<<")
        self.logger.info(f"Using Gemini model: {self.model}")
        self.logger.info(f"Max output tokens: {self.max_output_tokens}")
        self.logger.info(f"Temperature: {self.temperature}")

    async def gemini_complete(self, prompt: str, **kwargs) -> str:
        """
        Complete a prompt using Gemini API.
        This is the function used by LightRAG for LLM calls.
        ROBUST VERSION - handles any calling pattern
        """
        try:
            # DEBUG: Log exactly what LightRAG is passing
            self.logger.info(f"DEBUG: gemini_complete called with prompt='{prompt[:50]}...', kwargs={list(kwargs.keys())}")
            
            # Initialize the GenAI Client with API Key
            client = genai.Client(api_key=self.api_key)

            # Handle all possible parameter variations that LightRAG might use
            system_prompt = kwargs.get('system_prompt', None)
            history_messages = kwargs.get('history_messages', [])
            
            # Some versions might use different parameter names
            if not history_messages:
                history_messages = kwargs.get('messages', [])
                
            # Ensure history_messages is always a list
            if not isinstance(history_messages, list):
                history_messages = []

            combined_prompt = ""
            if system_prompt:
                combined_prompt += f"{system_prompt}\n"

            # Add history if present
            for msg in history_messages:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    combined_prompt += f"{msg['role']}: {msg['content']}\n"

            # Finally, add the new user prompt
            combined_prompt += f"user: {prompt}"

            # Call the Gemini model
            response = client.models.generate_content(
                model=self.model,
                contents=[combined_prompt],
                config=types.GenerateContentConfig(
                    max_output_tokens=self.max_output_tokens,
                    temperature=self.temperature
                ),
            )

            # Return the response text
            if response and response.text:
                self.logger.info(f"Gemini API call successful, response length: {len(response.text)}")
                return response.text
            else:
                self.logger.warning("Received empty response from Gemini API")
                return "Error: Empty response from API"
                
        except Exception as e:
            self.logger.error(f"GEMINI ERROR: {e}")
            self.logger.error(f"Full traceback:", exc_info=True)
            return f"Error: {str(e)}"

    def validate_api_key(self) -> bool:
        """
        Validate that the API key is present and properly formatted.
        
        Returns:
            bool: True if API key appears valid
        """
        if not self.api_key:
            return False
        
        # Basic validation - Gemini API keys typically start with specific patterns
        if len(self.api_key) < 20:
            return False
            
        return True

    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the configured model.
        
        Returns:
            Dictionary with model configuration information
        """
        return {
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "api_key_configured": bool(self.api_key)
        } 