from .BaseProviders import BaseLLMProvider, BaseVLMProvider
from typing import Tuple
import openai
import logging
import io
import base64
from PIL import Image
from .helpers import get_text_and_last_paragraph

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]')
logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider for text processing"""

    def _get_api_key_env_var(self):
        return "LLM_API_KEY"

    def _get_base_url_env_var(self) -> str:
        return "LLM_API_BASE_URL"

    def _get_model_name_env_var(self) -> str:
        return "LLM_MODEL_NAME"

    def _initialize(self):
        try:
            self.client = openai.OpenAI(
                api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            logger.error(
                "OpenAI package not installed. Install with 'pip install openai'", exc_info=True)
            raise

    def process_text(self, text: str, prev_summary: str = None, prompt_template: str = None) -> Tuple[str, str]:
        context = f"Previous context: {prev_summary}\n\n" if prev_summary else ""

        if not prompt_template:
            prompt_template = """
            {context}The following is a chunk of text from a PDF document:

            {text}

            Extract and clean the text while preserving meaning but fixing OCR or formatting issues. Identify document structure elements (headings, lists, tables, etc.) by listing their positions. Provide a brief summary in 200 words or less. Start directly with the cleaned text followed by structure elements and summary. And do not include any other text.Use proper Markdown syntax without unnecessary escape characters or formatting issues. Avoid wrapping the response in a fenced code block unless required. Ensure clean, well-structured output with appropriate headings, lists, bold, and italics for direct rendering in a Markdown viewer.
            """

        prompt = prompt_template.format(context=context, text=text)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts and processes text from PDF documents."},
                    {"role": "user", "content": prompt}
                ]
            )

            result = response.choices[0].message.content if hasattr(
                response, 'choices') else None

            if not result:
                logging.warning("Received None response from OpenAI")
                return text, "No content extracted"

            processed_text, summary = get_text_and_last_paragraph(result)

            if not processed_text:
                logger.error("Invalid response format from OpenAI")
                return text, "No content extracted"
            if not summary:
                summary = "No summary provided"

            return processed_text, summary

        except Exception as e:
            logger.error(
                f"Error processing text with OpenAI: {e}", exc_info=True)
            return text, "Error generating summary"


# Example VLM Provider implementations
class OpenAIVisionProvider(BaseVLMProvider):
    """OpenAI Vision API provider for image processing"""

    def _get_api_key_env_var(self):
        return "VLM_API_KEY"

    def _get_base_url_env_var(self) -> str:
        return "VLM_API_BASE_URL"

    def _get_model_name_env_var(self) -> str:
        return "VLM_MODEL_NAME"

    def _initialize(self):
        try:
            self.client = openai.OpenAI(
                api_key=self.api_key, base_url=self.base_url)
            self.base64 = base64
        except ImportError:
            logger.error(
                "OpenAI package not installed. Install with 'pip install openai'", exc_info=True)
            raise

    def process_image(self, image: Image.Image, prev_summary: str = None, prompt_template: str = None) -> Tuple[str, str]:
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = self.base64.b64encode(buffered.getvalue()).decode()

        context = f"Previous context: {prev_summary}\n\n" if prev_summary else ""

        if not prompt_template:
            prompt = f"""
            {context} This image is a page from a PDF document.

            Extract all text visible in the image while preserving structure and layout. Describe any images, tables, charts, graphs, diagrams, or non-text elements in detail, including their content, purpose, and visual characteristics. Then provide a brief summary of the page content. Start directly with the extracted text followed by your descriptions of visual elements and your summary in next paragraph.Use proper Markdown syntax without unnecessary escape characters or formatting issues. Avoid wrapping the response in a fenced code block unless required. Ensure clean, well-structured output with appropriate headings, lists, bold, and italics for direct rendering in a Markdown viewer.
            """
        else:
            prompt = prompt_template.format(context=context)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_str}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ]
            )

            # Check if response is valid before proceeding
            if response is None:
                logging.warning("Received None response from Vision API")
                return "", "No content extracted"

            # Handle content extraction from response
            result = response.choices[0].message.content if hasattr(
                response, 'choices') else None

            if not result:
                logger.error("Invalid response format from Vision API")
                return "", "No content extracted"

            processed_text, summary = get_text_and_last_paragraph(result)

            if not processed_text:
                logger.error("Invalid response format from Vision API")
                return "", "No content extracted"
            if not summary:
                summary = "No summary provided"

            return processed_text, summary

        except Exception as e:
            logger.error(
                f"Error processing image with Vision API: {str(e)}", exc_info=True)
            return "", f"Error: {str(e)}"
