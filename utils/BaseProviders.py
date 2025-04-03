import os
import numpy as np
from PIL import Image
from typing import Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]')
logger = logging.getLogger(__name__)


class BaseLLMProvider:
    """Base class for LLM providers"""

    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = None):
        self.api_key = self._get_value(
            api_key, self._get_api_key_env_var, "API key")
        self.base_url = self._get_value(
            base_url, self._get_base_url_env_var, "base URL")
        self.model_name = self._get_value(
            model_name, self._get_model_name_env_var, "model name")

        self._initialize()

    def _get_value(self, provided_value, env_var_method, value_name):
        """Fetch value from argument or environment variable, with warning if missing."""
        value = provided_value or os.environ.get(env_var_method())
        if not value:
            logger.warning(
                f"No {value_name} provided for {self.__class__.__name__}", exc_info=True)
        return value

    def _get_api_key_env_var(self) -> str:
        """Return environment variable name for API key"""
        raise NotImplementedError

    def _get_base_url_env_var(self) -> str:
        """Return environment variable name for base URL"""
        raise NotImplementedError

    def _get_model_name_env_var(self) -> str:
        """Return environment variable name for model name"""
        raise NotImplementedError

    def _initialize(self):
        """Initialize the LLM client"""
        pass

    def process_text(self, text: str, prev_summary: str = None, prompt_template: str = None) -> Tuple[str, str, float]:
        """
        Process text chunk with LLM

        Args:
            text: Text to process
            prev_summary: Previous chunk summary for context
            prompt_template: Custom prompt template

        Returns:
            Tuple of (processed_text, summary, confidence)
        """
        raise NotImplementedError


class BaseVLMProvider:
    """Base class for VLM providers"""

    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = None):
        self.api_key = self._get_value(
            api_key, self._get_api_key_env_var, "API key")
        self.base_url = self._get_value(
            base_url, self._get_base_url_env_var, "base URL")
        self.model_name = self._get_value(
            model_name, self._get_model_name_env_var, "model name")

        self._initialize()

    def _get_value(self, provided_value, env_var_method, value_name):
        """Fetch value from argument or environment variable, with warning if missing."""
        value = provided_value or os.environ.get(env_var_method())
        if not value:
            logger.warning(
                f"No {value_name} provided for {self.__class__.__name__}", exc_info=True)
        return value

    def _get_api_key_env_var(self) -> str:
        """Return environment variable name for API key"""
        raise NotImplementedError

    def _get_base_url_env_var(self) -> str:
        """Return environment variable name for base URL"""
        raise NotImplementedError

    def _get_model_name_env_var(self) -> str:
        """Return environment variable name for model name"""
        raise NotImplementedError

    def _initialize(self):
        """Initialize the VLM client"""
        pass

    def process_image(self, image: Image.Image, prev_summary: str = None, prompt_template: str = None) -> Tuple[str, str, float]:
        """
        Process image with VLM

        Args:
            image: PIL Image to process
            prev_summary: Previous chunk summary for context
            prompt_template: Custom prompt template

        Returns:
            Tuple of (extracted_text, summary, confidence)
        """
        raise NotImplementedError
