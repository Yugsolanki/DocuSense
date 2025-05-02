from fastapi import FastAPI, HTTPException
import requests
from main2 import PDFParser
from main2 import OpenAIProvider, OpenAIVisionProvider
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define the request body
class PDFRequest(BaseModel):
    pdf_url: str
    title: str
    source_url: str
    scraped_at: str


@app.post("/parse-pdf/")
async def parse_pdf(PDFRequest: PDFRequest):
    """
    API to parse a PDF from a given URL.

    Args:
        pdf_url (str): URL of the PDF.

    Returns:
        dict: Extracted content as JSON.
    """
    if not PDFRequest.pdf_url:
        logger.error("PDF URL is required.")
        raise HTTPException(status_code=400, detail="PDF URL is required.")

    if not PDFRequest.pdf_url.startswith(('http://', 'https://', "data:application/pdf;base64,")):
        logger.error("Invalid PDF URL.")
        raise HTTPException(status_code=400, detail="Invalid PDF URL.")

    logger.info(f"Received: {PDFRequest}")

    # try:
    #     response = requests.get(PDFRequest.pdf_url)
    #     response.raise_for_status()
    # except requests.RequestException as e:
    #     logger.error(f"Error accessing PDF URL: {e}")
    #     raise HTTPException(
    #         status_code=400, detail=f"PDF URL is not accessible: {str(e)}")

    try:
        if PDFRequest.pdf_url.startswith('data:application/pdf;base64,'):
            import base64
            # Extract the base64 part after the comma
            base64_data = PDFRequest.pdf_url.split(',')[1]
            pdf_content = base64.b64decode(base64_data)
        else:
            response = requests.get(PDFRequest.pdf_url)
            response.raise_for_status()
            pdf_content = response.content
    except requests.RequestException as e:
        logger.error(f"Error accessing PDF URL: {e}")
        raise HTTPException(
            status_code=400, detail=f"PDF URL is not accessible: {str(e)}")
    except (base64.binascii.Error, ValueError) as e:
        logger.error(f"Error decoding base64 PDF: {e}")
        raise HTTPException(
            status_code=400, detail=f"Invalid base64 PDF data: {str(e)}")

    # Initialize providers (Ensure these are correctly implemented in 'main')
    llm_provider = OpenAIProvider()
    vlm_provider = OpenAIVisionProvider()

    # Initialize and use PDF Parser
    parser = PDFParser(llm_provider=llm_provider,
                       vlm_provider=vlm_provider, process_sequentially=True, ocr_fallback=True)

    result = parser.parse_pdf(PDFRequest.pdf_url)

    return {"status": "success", "data": result}
