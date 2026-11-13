import pymupdf as fitz
import io
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class PDFParser:
    def __init__(self):
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._ocr = RapidOCR()
            except ImportError:
                logger.warning("rapidocr-onnxruntime not installed. Scanned pages will not be OCR-processed.")
                self._ocr = False
        return self._ocr

    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from all pages of a PDF.
        Automatically falls back to OCR if digital text is empty (scanned image).
        """
        doc = fitz.open(pdf_path)
        pages_data = []

        logger.info(f"Parsing PDF: {pdf_path} (Total pages: {len(doc)})")

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            raw_text = page.get_text("text").strip()
            
            # If digital text exists and is meaningful
            if len(raw_text) > 40:
                pages_data.append({
                    "page_number": page_num,
                    "text": raw_text,
                    "is_ocr": False
                })
            else:
                # Page is likely an image scan, run OCR
                logger.info(f"Page {page_num} appears scanned. Running OCR...")
                ocr_text = self._ocr_page(page)
                pages_data.append({
                    "page_number": page_num,
                    "text": ocr_text,
                    "is_ocr": True
                })

        doc.close()
        return pages_data

    def _ocr_page(self, page) -> str:
        ocr_engine = self._get_ocr()
        if not ocr_engine:
            return ""

        try:
            # Render page to high-res image (pixmap, 2.0 zoom factor for ~150-200 DPI)
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            
            result, _ = ocr_engine(img_bytes)
            if not result:
                return ""
                
            # result is a list of [box, text, confidence]
            extracted_lines = [item[1] for item in result if item and len(item) > 1]
            return "\n".join(extracted_lines)
        except Exception as e:
            logger.error(f"Error during OCR on page: {e}")
            return ""

pdf_parser = PDFParser()
