import os
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Callable, Union, Any
import re
from dataclasses import dataclass
import logging
import json
from collections import Counter
import pytesseract
import google.generativeai as genai
import openai
import base64
import anthropic
import threading


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d')
logger = logging.getLogger(__name__)

# Define data structures


@dataclass
class ExtractedContent:
    """Class for storing extracted content and metadata"""
    text: str
    summary: str
    page_num: int
    confidence: float = 1.0
    has_images: bool = False
    structure_tags: Dict[str, List[Tuple[int, int]]] = None


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


# Example LLM Provider implementations
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

    def process_text(self, text: str, prev_summary: str = None, prompt_template: str = None) -> Tuple[str, str, float]:
        context = f"Previous context: {prev_summary}\n\n" if prev_summary else ""

        if not prompt_template:
            prompt_template = """
            {context}The following is a chunk of text from a PDF document:
            
            {text}
            
            Please perform the following tasks:
            1. Extract and clean the text, preserving the original meaning but fixing any OCR or formatting issues.
            2. Identify any document structure elements (headings, lists, tables, etc.)
            3. Provide a brief summary of this chunk (max 100 words)
            
            Respond in JSON format with the following structure:
            {{
                "processed_text": "the cleaned and processed text",
                "summary": "brief summary of the chunk",
                "structure": {{"headings": [[start_idx, end_idx]], "lists": [[start_idx, end_idx]], "tables": [[start_idx, end_idx]]}}
            }}
            
            IMPORTANT: Your response must be valid JSON.
            """

        prompt = prompt_template.format(context=context, text=text)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts and processes text from PDF documents. Always repond in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            processed_text = result.get("processed_text", text)
            summary = result.get("summary", "No summary provided")
            confidence = 0.95  # OpenAI doesn't provide confidence scores, using fixed value

            return processed_text, summary, confidence

        except Exception as e:
            logger.error(f"Error processing text with OpenAI: {e}", exc_info=True)
            return text, "Error generating summary", 0.5


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider for text processing"""

    def _get_api_key_env_var(self):
        return "ANTHROPIC_API_KEY"

    def _get_model_name_env_var(self) -> str:
        return "ANTHROPIC_MODEL_NAME"

    def _initialize(self):
        try:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            logger.error(
                "Anthropic package not installed. Install with 'pip install anthropic'", exc_info=True)
            raise

    def process_text(self, text: str, prev_summary: str = None, prompt_template: str = None) -> Tuple[str, str, float]:
        context = f"Previous context: {prev_summary}\n\n" if prev_summary else ""

        if not prompt_template:
            prompt_template = """
            {context}The following is a chunk of text from a PDF document:
            
            {text}
            
            Please perform the following tasks:
            1. Extract and clean the text, preserving the original meaning but fixing any OCR or formatting issues.
            2. Identify any document structure elements (headings, lists, tables, etc.)
            3. Provide a brief summary of this chunk (max 100 words)
            
            Respond in JSON format with the following structure:
            {{
                "processed_text": "the cleaned and processed text",
                "summary": "brief summary of the chunk",
                "structure": {{"headings": [[start_idx, end_idx]], "lists": [[start_idx, end_idx]], "tables": [[start_idx, end_idx]]}}
            }}
            """

        prompt = prompt_template.format(context=context, text=text)

        try:
            response = self.client.messages.create(
                model=self.model_name or "claude-3-opus-20240229",
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                system="You are a helpful assistant that extracts and processes text from PDF documents."
            )

            # Extract JSON from response
            json_match = re.search(
                r'{.*}', response.content[0].text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                result = {"processed_text": text,
                          "summary": "Error parsing response"}

            processed_text = result.get("processed_text", text)
            summary = result.get("summary", "No summary provided")
            confidence = 0.95  # Anthropic doesn't provide confidence scores, using fixed value

            return processed_text, summary, confidence

        except Exception as e:
            logger.error(f"Error processing text with Anthropic: {e}", exc_info=True)
            return text, "Error generating summary", 0.5


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

    def process_image(self, image: Image.Image, prev_summary: str = None, prompt_template: str = None) -> Tuple[str, str, float]:
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = self.base64.b64encode(buffered.getvalue()).decode('utf-8')

        context = f"Previous context: {prev_summary}\n\n" if prev_summary else ""

        if not prompt_template:
            prompt = f"""
            {context}This image is a page from a PDF document. Please:
            1. Extract all text visible in the image, preserving structure and layout
            
            Respond in JSON format with the following structure:
            {{
                "extracted_text": "all text from the image"
                "summary": "brief summary of the page content"
            }}
            
            IMPORTANT: Your response must be valid JSON.
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
                ],
                max_tokens=4096,
                response_format={"type": "json_object"}
            )
            
            # Check if response is valid before proceeding
            if not response:
                logging.warning("Received None response from Vision API")
                return "Error processing image", "No response", 0.5

            # Handle both OpenAI and Alibaba response formats
            try:
                if hasattr(response, 'choices') and response.choices:
                    content = response.choices[0].message.content 
                    content = content.strip()
                    
                    if isinstance(content, str):
                        result = json.loads(content)
                    else:
                        # Handle Alibaba format where content might be a dict
                        result = content
                else:
                    raise ValueError("Invalid response format")

                extracted_text = result.get("extracted_text", "")
                summary = result.get("summary", "No summary provided")

                # Just return the extracted text and placeholder values for the other fields
                return extracted_text, summary, 0.9

            except (json.JSONDecodeError, AttributeError, ValueError) as e:
                logger.error(f"Error parsing response: {e}", exc_info=True)
                logger.debug(f"Response content: {response}")
                return "Error processing image" if response else "No response", "Error parsing response", 0.5

        except Exception as e:
            logger.error(f"Error processing image with Vision API: {e}", exc_info=True)
            return "Error processing image", "Error generating summary", 0.5


class GoogleGeminiProvider(BaseVLMProvider):
    """Google Gemini Vision API provider for image processing"""

    def _get_api_key_env_var(self):
        return "GOOGLE_API_KEY"

    def _get_model_name_env_var(self) -> str:
        return "GOOGLE_MODEL_NAME"

    def _initialize(self):
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                self.model_name or 'gemini-pro-vision')
        except ImportError:
            logger.error(
                "Google Generative AI package not installed. Install with 'pip install google-generativeai'", exc_info=True)
            raise

    def process_image(self, image: Image.Image, prev_summary: str = None, prompt_template: str = None) -> Tuple[str, str, float]:
        context = f"Previous context: {prev_summary}\n\n" if prev_summary else ""

        if not prompt_template:
            prompt = f"""
            {context}This image is a page from a PDF document. Please:
            1. Extract all text visible in the image, preserving structure and layout
            2. Identify any tables, charts, or diagrams and describe their content
            3. Provide a brief summary of this page's content (max 100 words)
            
            Respond in JSON format with the following structure:
            {{
                "extracted_text": "all text from the image",
                "summary": "brief summary of the page content"
            }}
            """
        else:
            prompt = prompt_template.format(context=context)

        try:
            response = self.model.generate_content([prompt, image])

            # Extract JSON from response
            json_match = re.search(r'{.*}', response.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                result = {"extracted_text": "Error parsing the image",
                          "summary": "Error parsing response"}

            extracted_text = result.get("extracted_text", "")
            summary = result.get("summary", "No summary provided")
            confidence = 0.9  # Gemini doesn't provide confidence scores, using fixed value

            return extracted_text, summary, confidence

        except Exception as e:
            logger.error(f"Error processing image with Google Gemini: {e}", exc_info=True)
            return "Error processing image", "Error generating summary", 0.5


class PDFParser:
    """Main PDF parsing class with LLM and VLM integration"""

    def __init__(
        self,
        llm_provider: BaseLLMProvider = None,
        vlm_provider: BaseVLMProvider = None,
        chunk_size: int = 4000,
        chunk_overlap: int = 400,
        parallel_processing: bool = True,
        max_workers: int = 4,
        min_image_size: int = 100,  # Min image size in pixels to process
        ocr_fallback: bool = True,
        confidence_threshold: float = 0.7,
        structure_detection: bool = True
    ):
        self.llm_provider = llm_provider  # Language model provider
        self.vlm_provider = vlm_provider  # Vision model provider
        # chunk size is the size of the text to be processed at a time
        self.chunk_size = chunk_size
        # chunk overlap is the overlap between two chunks. This is useful for maintaining context
        self.chunk_overlap = chunk_overlap
        # parallel processing is used to process multiple pages at the same time
        self.parallel_processing = parallel_processing
        # max_workers is the number of workers to use for parallel processing
        self.max_workers = max_workers
        # min_image_size is the minimum size of an image to be processed
        self.min_image_size = min_image_size
        # ocr_fallback is used to enable OCR as a fallback for image processing
        self.ocr_fallback = ocr_fallback
        # confidence_threshold is the threshold for merging adjacent
        self.confidence_threshold = confidence_threshold
        # structure_detection is used to enable structure detection
        self.structure_detection = structure_detection

        # Initialize fallback OCR if needed
        if self.ocr_fallback:
            try:
                self.pytesseract = pytesseract
            except ImportError:
                logger.warning(
                    "Tesseract OCR not installed, OCR fallback disabled. Install with 'pip install pytesseract'", exc_info=True)
                self.ocr_fallback = False

    def _detect_has_images(self, doc) -> bool:
        """Detect if the PDF has images"""
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            if len(image_list) > 0:
                for img in image_list:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    if base_image:
                        image_bytes = base_image["image"]
                        image = Image.open(io.BytesIO(image_bytes))
                        if image.width > self.min_image_size and image.height > self.min_image_size:
                            return True
        return False

    def _extract_headers_footers(self, doc) -> Tuple[List[str], List[str]]:
        """
        Attempt to identify headers and footers for removal
        Returns (headers, footers) lists
        """
        headers = []
        footers = []

        if len(doc) >= 3:  # Need at least 3 pages to detect patterns
            # Sample the first few pages
            sample_pages = min(10, len(doc))

            # Get text from top and bottom of pages
            top_texts = []
            bottom_texts = []

            for page_num in range(sample_pages):
                page = doc[page_num]
                text = page.get_text()
                lines = text.split('\n')

                if len(lines) >= 2:
                    top_texts.append(lines[0].strip())
                    bottom_texts.append(lines[-1].strip())

            # Find repeating patterns

            top_counter = Counter(top_texts)
            bottom_counter = Counter(bottom_texts)

            # If a text appears in more than 70% of pages, consider it header/footer
            header_threshold = sample_pages * 0.7
            footer_threshold = sample_pages * 0.7

            for text, count in top_counter.items():
                if count >= header_threshold and text:
                    headers.append(text)

            for text, count in bottom_counter.items():
                if count >= footer_threshold and text:
                    footers.append(text)

        return headers, footers

    def _extract_semantic_chunks(self, text: str, headers: List[str], footers: List[str]) -> List[str]:
        """
        Extract semantic chunks from text with improved chunking logic
        """
        # Remove headers and footers
        for header in headers:
            text = text.replace(header, "")
        for footer in footers:
            text = text.replace(footer, "")

        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        # Try to identify natural sections using headings
        section_pattern = r'(?:\n|^)(#{1,6}|\d+\.\s+|\w+\.\s+|[A-Z][A-Z\s]+:)'
        sections = re.split(section_pattern, text)

        # If sections are too large, further split them
        chunks = []
        for section in sections:
            if len(section) > self.chunk_size:
                # Split by paragraphs
                paragraphs = re.split(r'\n\n+', section)
                current_chunk = ""

                for paragraph in paragraphs:
                    if len(current_chunk) + len(paragraph) + 2 <= self.chunk_size:
                        current_chunk += paragraph + "\n\n"
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = paragraph + "\n\n"

                if current_chunk:
                    chunks.append(current_chunk.strip())
            else:
                chunks.append(section.strip())

        # Add overlap between chunks if needed
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped_chunks = [chunks[0]]

            for i in range(1, len(chunks)):
                prev_chunk = chunks[i-1]
                current_chunk = chunks[i]

                if len(prev_chunk) > self.chunk_overlap:
                    overlap_text = prev_chunk[-self.chunk_overlap:]
                    overlapped_chunks.append(overlap_text + current_chunk)
                else:
                    overlapped_chunks.append(prev_chunk + current_chunk)

            chunks = overlapped_chunks

        # Remove empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]

        return chunks

    def _process_text_page(self, page_text: str, page_num: int, prev_summary: str = None) -> ExtractedContent:
        """Process a page of text using LLM"""
        if not self.llm_provider:
            return ExtractedContent(
                text=page_text,
                summary=f"Page {page_num + 1} content",
                page_num=page_num
            )

        processed_text, summary, confidence = self.llm_provider.process_text(
            page_text, prev_summary)

        return ExtractedContent(
            text=processed_text,
            summary=summary,
            page_num=page_num,
            confidence=confidence
        )

    def _process_image_page(self, page_image: Image.Image, page_num: int, prev_summary: str = None) -> ExtractedContent:
        """Process a page as image using VLM"""
        if not self.vlm_provider:
            # Fallback to OCR if available
            if self.ocr_fallback:
                try:
                    extracted_text = self.pytesseract.image_to_string(
                        page_image)
                    return ExtractedContent(
                        text=extracted_text,
                        summary=f"Page {page_num + 1} content (OCR fallback)",
                        page_num=page_num,
                        has_images=True,
                        confidence=0.7
                    )
                except Exception as e:
                    logger.error(f"OCR fallback failed: {e}", exc_info=True)

            return ExtractedContent(
                text=f"[Image content on page {page_num + 1}]",
                summary=f"Page {page_num + 1} contains image content",
                page_num=page_num,
                has_images=True,
                confidence=0
            )

        extracted_text, summary, confidence = self.vlm_provider.process_image(
            page_image, prev_summary)

        return ExtractedContent(
            text=extracted_text,
            summary=summary or f"Page {page_num + 1} content",
            page_num=page_num,
            has_images=True,
            confidence=confidence
        )

    def _render_page_to_image(self, page, dpi: int = 300) -> Image.Image:
        """Render a PDF page to a PIL Image"""
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    def _merge_adjacent_chunks(self, chunks: List[ExtractedContent], confidence_threshold: float = 0.7) -> List[ExtractedContent]:
        """Merge adjacent chunks with low confidence"""
        if len(chunks) <= 1:
            return chunks

        merged_chunks = []
        i = 0

        while i < len(chunks):
            current_chunk = chunks[i]

            # If current chunk has low confidence and not the last chunk
            if current_chunk.confidence < confidence_threshold and i < len(chunks) - 1:
                next_chunk = chunks[i+1]

                # Merge the chunks
                merged_chunk = ExtractedContent(
                    text=current_chunk.text + "\n\n" + next_chunk.text,
                    summary=f"Combined content from pages {current_chunk.page_num+1}-{next_chunk.page_num+1}",
                    page_num=current_chunk.page_num,
                    confidence=max(current_chunk.confidence,
                                   next_chunk.confidence),
                    has_images=current_chunk.has_images or next_chunk.has_images
                )

                merged_chunks.append(merged_chunk)
                i += 2  # Skip the next chunk since we merged it
            else:
                merged_chunks.append(current_chunk)
                i += 1

        return merged_chunks

    def _hierarchical_summarize(self, chunks: List[ExtractedContent]) -> str:
        """Create a hierarchical summary of the document"""
        if not self.llm_provider or len(chunks) <= 1:
            return chunks[0].summary if chunks else ""

        # Group chunks into groups of 5
        chunk_groups = [chunks[i:i+5] for i in range(0, len(chunks), 5)]
        group_summaries = []

        for group in chunk_groups:
            # Include both text and image summaries
            group_text = "\n".join(
                [f"Page {chunk.page_num+1} Summary: {chunk.summary}" for chunk in group if chunk.summary])

            # Only process if we have meaningful summaries
            if group_text.strip():
                prompt_template = """
                The following are summaries from pages of a document:
                
                {text}
                
                Please provide a cohesive summary that integrates these page summaries (max 150 words).
                """

                _, group_summary, _ = self.llm_provider.process_text(
                    group_text, prompt_template=prompt_template)
                group_summaries.append(group_summary)

        # If we have multiple group summaries, summarize them again
        if len(group_summaries) > 1:
            final_text = "\n\n".join(
                [f"Section {i+1} Summary: {summary}" for i, summary in enumerate(group_summaries)])

            prompt_template = """
            The following are summaries from sections of a document:
            
            {text}
            
            Please provide a comprehensive document summary that integrates these section summaries (max 250 words).
            """

            _, document_summary, _ = self.llm_provider.process_text(
                final_text, prompt_template=prompt_template)
            return document_summary

        return group_summaries[0] if group_summaries else ""

    def parse_pdf(self, pdf_path: str) -> Dict:
        """
        Parse a PDF file and extract content using LLM/VLM

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary with extracted content
        """
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Error opening PDF file: {e}", exc_info=True)
            return {"error": f"Failed to open PDF: {str(e)}"}

        # Store page count before processing
        total_pages = len(doc)

        # Detect if PDF has images
        has_images = self._detect_has_images(doc)
        logger.info(f"PDF contains images: {has_images}")

        # Extract headers and footers
        headers, footers = self._extract_headers_footers(doc)
        logger.info(f"Detected headers: {headers}")
        logger.info(f"Detected footers: {footers}")

        # Process pages
        all_extracted_content = []
        prev_summary = None

        def process_page(page_num):
            nonlocal prev_summary
            page = doc[page_num]

            if has_images:
                # Render page to image and process with VLM
                page_image = self._render_page_to_image(page)
                extracted_content = self._process_image_page(
                    page_image, page_num, prev_summary)
            else:
                # Extract text and process with LLM
                page_text = page.get_text()

                # Remove headers and footers
                for header in headers:
                    page_text = page_text.replace(header, "")
                for footer in footers:
                    page_text = page_text.replace(footer, "")

                extracted_content = self._process_text_page(
                    page_text, page_num, prev_summary)

            prev_summary = extracted_content.summary
            return extracted_content

        if self.parallel_processing:
            # Process pages in parallel
            page_indices = list(range(len(doc)))

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Process first page to get initial summary
                first_content = process_page(0)
                all_extracted_content.append(first_content)
                prev_summary = first_content.summary

                # Process remaining pages in parallel with updated context
                future_to_page = {executor.submit(
                    process_page, page_num): page_num for page_num in page_indices[1:]}

                for future in concurrent.futures.as_completed(future_to_page):
                    page_num = future_to_page[future]
                    try:
                        extracted_content = future.result()
                        all_extracted_content.append(extracted_content)
                    except Exception as e:
                        logger.error(f"Error processing page {page_num}: {e}", exc_info=True)
                        all_extracted_content.append(ExtractedContent(
                            text=f"[Error processing page {page_num + 1}]",
                            summary=f"Error on page {page_num + 1}",
                            page_num=page_num,
                            confidence=0
                        ))
        else:
            # Process pages sequentially
            for page_num in range(len(doc)):
                extracted_content = process_page(page_num)
                all_extracted_content.append(extracted_content)

        # Sort by page number
        all_extracted_content.sort(key=lambda x: x.page_num)

        # Merge adjacent low-confidence chunks
        if self.confidence_threshold < 1.0:
            all_extracted_content = self._merge_adjacent_chunks(
                all_extracted_content, self.confidence_threshold)

        # Create hierarchical summary
        document_summary = self._hierarchical_summarize(all_extracted_content)

        # Close the document
        doc.close()

        # Prepare result
        result = {
            "text": "\n\n".join([content.text for content in all_extracted_content]),
            "pages": [
                {
                    "page_num": content.page_num + 1,
                    "text": content.text,
                    "summary": content.summary,
                    "has_images": content.has_images,
                    "confidence": content.confidence
                }
                for content in all_extracted_content
            ],
            "summary": document_summary,
            "metadata": {
                "total_pages": total_pages,
                "contains_images": has_images
            }
        }

        return result


# Example usage function
def parse_pdf_with_custom_providers(
    pdf_path: str,
    llm_provider_name: str = "openai",
    vlm_provider_name: str = "openai",
    llm_api_key: str = None,
    vlm_api_key: str = None,
    parallel: bool = True,
    output_path: str = None
) -> Dict:
    """
    Parse a PDF with custom LLM and VLM providers

    Args:
        pdf_path: Path to the PDF file
        llm_provider_name: Name of the LLM provider ('openai', 'anthropic', None)
        vlm_provider_name: Name of the VLM provider ('openai', 'gemini', None)
        llm_api_key: API key for the LLM provider
        vlm_api_key: API key for the VLM provider
        parallel: Whether to process pages in parallel
        output_path: Path to save the output JSON file

    Returns:
        Dictionary with extracted content
    """

    # Initialize providers
    llm_provider = None
    vlm_provider = None

    if llm_provider_name == "openai":
        llm_provider = OpenAIProvider(api_key=llm_api_key)
    elif llm_provider_name == "anthropic":
        llm_provider = AnthropicProvider(api_key=llm_api_key)

    if vlm_provider_name == "openai":
        vlm_provider = OpenAIVisionProvider(api_key=vlm_api_key)
    elif vlm_provider_name == "gemini":
        vlm_provider = GoogleGeminiProvider(api_key=vlm_api_key)

    # Initialize PDF parser
    parser = PDFParser(llm_provider=llm_provider,
                       vlm_provider=vlm_provider, parallel_processing=parallel)

    # Parse the PDF
    result = parser.parse_pdf(pdf_path)

    # Save output to file if needed
    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)  # Save with pretty formatting

    return result


# Example usage
pdf_path = "./pdf/DF.pdf"
output_path = "sample_output.json"
result = parse_pdf_with_custom_providers(
    pdf_path, parallel=True, output_path=output_path)
print(result)
