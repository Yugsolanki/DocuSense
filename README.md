# DocuSense

DocuSense is a powerful PDF parsing and content extraction tool that leverages Large Language Models (LLMs) and Vision Language Models (VLMs) to extract, process, and summarize content from PDF documents. Whether your PDF contains text, images, or a combination of both, DocuSense provides a comprehensive solution for extracting meaningful information.

## Features

- **Text Extraction**: Extract and clean text from PDF documents, preserving the original meaning and structure.
- **Image Processing**: Extract text and content from images within PDFs using advanced vision models.
- **Document Summarization**: Generate hierarchical summaries of the document content.
- **Parallel Processing**: Process multiple pages in parallel for faster extraction.
- **Customizable Providers**: Support for multiple LLM and VLM providers, including OpenAI and any OpenAI API-compatible models.
- **OCR Fallback**: Fallback to OCR (Tesseract) for image processing when VLM is not available.
- **Header/Footer Detection**: Automatically detect and remove headers and footers from the document.
- **Confidence-Based Merging**: Merge adjacent chunks of text based on confidence scores for better context preservation.

## How It works

DocuSense follows a structured workflow to extract, process, and summarize content from PDF documents. Below is a high-level overview of the process:

#### Flow Chart
![Flow Chart](docs/flowchart.png)
*Flow chart illustrating the overall workflow of DocuSense.*

1. **PDF Parsing**: The PDF is loaded and split into individual pages.
2. **Text Extraction**: Text is extracted from each page using LLMs.
3. **Image Processing**: If images are detected, they are processed using VLMs or OCR (Tesseract) as a fallback.
4. **Content Summarization**: Extracted content is summarized hierarchically.
5. **Output Generation**: The final output is structured into text, summaries, and metadata.

#### Sequence Diagram
![Sequence Diagram](docs/sequence-diagram.png)
*Sequence diagram showing the interaction between components during PDF processing.*

## Installation

DocuSense can be installed using either `uv` or `pip`.

### Using `uv`

If you are using `uv`, you can install the required dependencies with the following command:

```bash
uv add PyMuPDF pillow numpy pytesseract openai
```

### Using `pip`

If you are using `pip`, you can install the required dependencies with the following command:

```bash
pip install PyMuPDF pillow numpy pytesseract openai
```

## Usage

### Basic Usage

To parse a PDF document and extract its content, you can use the `parse_pdf_with_custom_providers` function:

```python
from docusense import parse_pdf_with_custom_providers

pdf_path = "sample.pdf"
output_path = "sample_output.json"

result = parse_pdf_with_custom_providers(
    pdf_path,
    llm_provider_name="openai",
    vlm_provider_name="openai",
    parallel=True,
    output_path=output_path
)

print(result)
```

### Custom Providers

You can customize the LLM and VLM providers used for text and image processing:

```python
from docusense import PDFParser, OpenAIProvider, OpenAIVisionProvider

# Initialize providers
llm_provider = OpenAIProvider(api_key="your_openai_api_key")
vlm_provider = OpenAIVisionProvider(api_key="your_openai_api_key")

# Initialize PDF parser
parser = PDFParser(
    llm_provider=llm_provider,
    vlm_provider=vlm_provider,
    chunk_size=4000,
    chunk_overlap=400,
    parallel_processing=True,
    max_workers=4,
    min_image_size=100,
    ocr_fallback=True,
    confidence_threshold=0.7,
    structure_detection=True
)

# Parse the PDF
pdf_path = "sample.pdf"
result = parser.parse_pdf(pdf_path)

print(result)
```

## Configuration

### Environment Variables

You can configure the API keys and other settings using environment variables:

- `LLM_API_KEY`: API key for the LLM provider.
- `VLM_API_KEY`: API key for the VLM provider.
- `LLM_API_BASE_URL`: Base URL for the LLM provider (if applicable).
- `VLM_API_BASE_URL`: Base URL for the VLM provider (if applicable).
- `LLM_MODEL_NAME`: Model name for the LLM provider.
- `VLM_MODEL_NAME`: Model name for the VLM provider.

### Parameters

- `chunk_size`: The size of the text chunk to be processed at a time (default: 4000).
- `chunk_overlap`: The overlap between two chunks for maintaining context (default: 400).
- `parallel_processing`: Whether to process pages in parallel (default: True).
- `max_workers`: The number of workers to use for parallel processing (default: 4).
- `min_image_size`: The minimum size of an image to be processed (default: 100 pixels).
- `ocr_fallback`: Whether to enable OCR as a fallback for image processing (default: True).
- `confidence_threshold`: The threshold for merging adjacent chunks (default: 0.7).
- `structure_detection`: Whether to enable structure detection (default: True).

## Supported Providers

### LLM Providers

- **OpenAI**: Uses GPT models for text processing.
- **OpenAI API Compatible**: Supports any model that implements the OpenAI API protocol

### VLM Providers

- **OpenAI Vision**: Uses OpenAI's vision models for image processing.
- **OpenAI API Compatible**: Supports any vision model that implements the OpenAI API protocol

## Example Output

The output of the `parse_pdf` function is a dictionary containing the extracted content, summaries, and metadata:

```json
{
  "text": "Extracted text from the document...",
  "pages": [
    {
      "page_num": 1,
      "text": "Extracted text from page 1...",
      "summary": "Summary of page 1...",
      "has_images": false,
      "confidence": 0.95
    },
    {
      "page_num": 2,
      "text": "Extracted text from page 2...",
      "summary": "Summary of page 2...",
      "has_images": true,
      "confidence": 0.9
    }
  ],
  "summary": "Hierarchical summary of the entire document...",
  "metadata": {
    "total_pages": 2,
    "contains_images": true
  }
}
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.

## License

DocuSense is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) for PDF parsing.
- [OpenAI](https://openai.com/) for LLM and VLM APIs.
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for OCR fallback.

## Contact

For any questions or support, please open an issue on the GitHub repository.

---

**DocuSense** - Your intelligent document processing companion.
