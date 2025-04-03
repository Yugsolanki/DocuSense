import os
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io
import concurrent.futures
from typing import List, Dict, Tuple
import re
from dataclasses import dataclass
import logging
import json
from collections import Counter
import pytesseract
import requests
import tempfile
import gc
from utils.BaseProviders import BaseLLMProvider, BaseVLMProvider
from utils.OpenAIProviders import OpenAIProvider, OpenAIVisionProvider


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]')
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

    def _detect_has_images(self, doc, sample_size: int = 10) -> bool:
        """Detect if the PDF has images efficiently by sampling pages"""

        # If the document has few pages, check all
        if len(doc) <= sample_size:
            pages_to_check = range(len(doc))
        else:
            # Otherwise, check a sample of pages spread throughout the document
            stride = max(1, len(doc) // sample_size)
            pages_to_check = range(0, len(doc), stride)

        for page_num in pages_to_check:
            page = doc[page_num]
            image_list = page.get_images(full=True)
            if image_list:
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

        processed_text, summary = self.llm_provider.process_text(
            page_text, prev_summary)

        return ExtractedContent(
            text=processed_text,
            summary=summary,
            page_num=page_num
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

        extracted_text, summary = self.vlm_provider.process_image(
            page_image, prev_summary)

        return ExtractedContent(
            text=extracted_text,
            summary=summary or f"Page {page_num + 1} content",
            page_num=page_num,
            has_images=True
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

        if not chunks:
            return ""

        if not self.llm_provider:
            return chunks[0].summary if chunks else ""

        if len(chunks) <= 1:
            return chunks[0].summary

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
                
                Please provide a cohesive summary that integrates these page summaries (max 150 words). Start directly with the summary. And do not include any other text.Use proper Markdown syntax without unnecessary escape characters or formatting issues. Avoid wrapping the response in a fenced code block unless required. Ensure clean, well-structured output with appropriate headings, lists, bold, and italics for direct rendering in a Markdown viewer.
                """

                _, group_summary = self.llm_provider.process_text(
                    group_text, prompt_template=prompt_template)
                group_summaries.append(group_summary)

        # If we have multiple group summaries, summarize them again
        if len(group_summaries) > 1:
            final_text = "\n\n".join(
                [f"Section {i+1} Summary: {summary}" for i, summary in enumerate(group_summaries)])

            prompt_template = """ 
            The following are summaries from sections of a document:
            
            {text}
            
            Provide an integrated document summary combining these section summaries in 250 words or less. Start directly with the summary. And do not include any other text.Use proper Markdown syntax without unnecessary escape characters or formatting issues. Avoid wrapping the response in a fenced code block unless required. Ensure clean, well-structured output with appropriate headings, lists, bold, and italics for direct rendering in a Markdown viewer.
            """

            _, document_summary = self.llm_provider.process_text(
                final_text, prompt_template=prompt_template)
            return document_summary

        return group_summaries[0] if group_summaries else ""

    def parse_pdf(self, pdf_input: str) -> Dict:
        """
        Parse a PDF file or URL and extract content using LLM/VLM

        Args:
            pdf_input: Path to the PDF file or URL to the PDF

        Returns:
            Dictionary with extracted content
        """
        temp_file = None
        doc = None

        try:
            # Check if input is a URL
            if pdf_input.startswith(('http://', 'https://')):
                response = requests.get(pdf_input, stream=True)
                response.raise_for_status()
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix='.pdf')
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file.close()
                pdf_path = temp_file.name
            else:
                pdf_path = pdf_input

            doc = fitz.open(pdf_path)
            total_pages = len(doc)  # Store page count before any processing

            has_images = self._detect_has_images(doc)
            logger.info(f"PDF contains images: {has_images}")

            headers, footers = self._extract_headers_footers(doc)
            logger.info(f"Detected headers: {headers}")
            logger.info(f"Detected footers: {footers}")

            all_extracted_content = []
            sequential_pages = min(3, total_pages)
            prev_summary = None

            for page_num in range(sequential_pages):
                page = doc[page_num]
                if has_images:
                    page_image = self._render_page_to_image(page)
                    extracted_content = self._process_image_page(
                        page_image, page_num, prev_summary)
                else:
                    page_text = page.get_text()
                    for header in headers:
                        page_text = page_text.replace(header, "")
                    for footer in footers:
                        page_text = page_text.replace(footer, "")
                    extracted_content = self._process_text_page(
                        page_text, page_num, prev_summary)

                all_extracted_content.append(extracted_content)
                prev_summary = extracted_content.summary

            remaining_pages = list(range(sequential_pages, total_pages))
            if self.parallel_processing and remaining_pages:
                shared_context = prev_summary
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    def process_page_with_context(page_num):
                        page = doc[page_num]
                        if has_images:
                            page_image = self._render_page_to_image(page)
                            return self._process_image_page(page_image, page_num, shared_context)
                        else:
                            page_text = page.get_text()
                            for header in headers:
                                page_text = page_text.replace(header, "")
                            for footer in footers:
                                page_text = page_text.replace(footer, "")
                            return self._process_text_page(page_text, page_num, shared_context)

                    future_to_page = {executor.submit(process_page_with_context, page_num): page_num
                                      for page_num in remaining_pages}

                    for future in concurrent.futures.as_completed(future_to_page):
                        page_num = future_to_page[future]
                        try:
                            extracted_content = future.result()
                            all_extracted_content.append(extracted_content)
                        except Exception as e:
                            logger.error(
                                f"Error processing page {page_num}: {e}", exc_info=True)
                            all_extracted_content.append(ExtractedContent(
                                text=f"[Error processing page {page_num + 1}]",
                                summary=f"Error on page {page_num + 1}",
                                page_num=page_num,
                                confidence=0
                            ))
            else:
                for page_num in remaining_pages:
                    page = doc[page_num]
                    if has_images:
                        page_image = self._render_page_to_image(page)
                        extracted_content = self._process_image_page(
                            page_image, page_num, prev_summary)
                    else:
                        page_text = page.get_text()
                        for header in headers:
                            page_text = page_text.replace(header, "")
                        for footer in footers:
                            page_text = page_text.replace(footer, "")
                        extracted_content = self._process_text_page(
                            page_text, page_num, prev_summary)

                    all_extracted_content.append(extracted_content)
                    prev_summary = extracted_content.summary

            all_extracted_content.sort(key=lambda x: x.page_num)
            if self.confidence_threshold < 1.0:
                all_extracted_content = self._merge_adjacent_chunks(
                    all_extracted_content, self.confidence_threshold)

            document_summary = self._hierarchical_summarize(
                all_extracted_content)

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

        except Exception as e:
            logger.error(f"Error processing PDF: {e}", exc_info=True)
            return {"error": f"Failed to process PDF: {str(e)}"}

        finally:
            # Cleanup in finally block to ensure it always runs
            if doc is not None:
                doc.close()
            if temp_file is not None:
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    logger.warning(
                        f"Failed to delete temporary file: {e}", exc_info=True)
            gc.collect()

        return result


# Example usage function
def parse_pdf_with_custom_providers(
    pdf_url: str,
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

    if vlm_provider_name == "openai":
        vlm_provider = OpenAIVisionProvider(api_key=vlm_api_key)

    # Initialize PDF parser
    parser = PDFParser(llm_provider=llm_provider,
                       vlm_provider=vlm_provider, parallel_processing=parallel)

    # Parse the PDF
    result = parser.parse_pdf(pdf_url)

    # Save output to file if needed
    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)  # Save with pretty formatting

    return result
