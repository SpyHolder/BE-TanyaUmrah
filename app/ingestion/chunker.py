import re
from typing import List, Dict, Any

class SmartChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits parsed pages into overlapping chunks while tracking page number
        and active pasal/bab section headers.
        """
        all_chunks = []
        current_section = "Umum"
        chunk_counter = 0

        for page_item in pages_data:
            page_num = page_item["page_number"]
            text = page_item["text"]

            if not text.strip():
                continue

            # Check if there is a new Pasal or BAB mentioned on this page
            section_matches = re.findall(r'((?:BAB\s+[IVXLCDM]+|Pasal\s+\d+)[^\n\.\:]*)', text, re.IGNORECASE)
            if section_matches:
                current_section = section_matches[0].strip()

            # Split text into paragraphs
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [text]

            current_chunk = ""
            for p in paragraphs:
                # If adding paragraph exceeds chunk_size, store current_chunk and start new with overlap
                if len(current_chunk) + len(p) > self.chunk_size and len(current_chunk) > 100:
                    chunk_counter += 1
                    all_chunks.append({
                        "chunk_index": chunk_counter,
                        "text": current_chunk.strip(),
                        "page": page_num,
                        "section": current_section
                    })
                    # Keep overlap from tail of current chunk
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                    current_chunk = overlap_text + "\n" + p
                else:
                    if current_chunk:
                        current_chunk += "\n" + p
                    else:
                        current_chunk = p

            if current_chunk.strip():
                chunk_counter += 1
                all_chunks.append({
                    "chunk_index": chunk_counter,
                    "text": current_chunk.strip(),
                    "page": page_num,
                    "section": current_section
                })

        return all_chunks

smart_chunker = SmartChunker()
