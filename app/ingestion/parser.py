import pymupdf as fitz
import io
import os
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class PDFParser:
    def __init__(self, cache_dir: str = "./storage/cache_text"):
        self._ocr = None
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

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
        Caches extracted text to disk to avoid redundant OCR processing.
        """
        filename = os.path.basename(pdf_path)
        cache_file = os.path.join(self.cache_dir, f"{filename}.json")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                logger.info(f"Loaded parsed text for '{filename}' from cache ({len(cached_data)} pages).")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to read cache for '{filename}': {e}")

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
                logger.info(f"Page {page_num}/{len(doc)} appears scanned. Running OCR...")
                ocr_text = self._ocr_page(page)
                pages_data.append({
                    "page_number": page_num,
                    "text": ocr_text,
                    "is_ocr": True
                })

        doc.close()

        # Save to cache
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(pages_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Cached parsed text for '{filename}' to disk.")
        except Exception as e:
            logger.warning(f"Failed to write cache for '{filename}': {e}")

        return pages_data

    def _ocr_page(self, page) -> str:
        ocr_engine = self._get_ocr()
        if not ocr_engine:
            return ""

        try:
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            
            result, _ = ocr_engine(img_bytes)
            if not result:
                return ""
                
            extracted_lines = [item[1] for item in result if item and len(item) > 1]
            return "\n".join(extracted_lines)
        except Exception as e:
            logger.error(f"Error during OCR on page: {e}")
            return ""

pdf_parser = PDFParser()

