import re
from typing import Any, Dict, List


class SmartChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("Require chunk_size > chunk_overlap >= 0")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        section = "Umum"
        for page in pages_data:
            text = page["text"].strip()
            # Split at actual heading lines, not inline references to other pasal.
            headings = list(
                re.finditer(
                    r"(?im)^[ \t]*(?:BAB\s+[IVXLCDM]+\b|Pasal\s+\d+\b)[^\n]*", text
                )
            )
            boundaries = (
                [0] + [m.start() for m in headings if m.start() > 0] + [len(text)]
            )
            for lo, hi in zip(boundaries, boundaries[1:]):
                block = text[lo:hi].strip()
                if not block:
                    continue
                heading = re.match(
                    r"(?i)(?:BAB\s+[IVXLCDM]+\b|Pasal\s+\d+\b)[^\n]*", block
                )
                if heading:
                    section = heading.group(0)[:200].strip()
                start = 0
                while start < len(block):
                    end = min(start + self.chunk_size, len(block))
                    if end < len(block):
                        # Prefer a paragraph/word boundary without producing tiny chunks.
                        cut = max(
                            block.rfind("\n", start, end), block.rfind(" ", start, end)
                        )
                        if cut > start + self.chunk_size // 2:
                            end = cut
                    content = block[start:end].strip()
                    if content:
                        chunks.append(
                            {
                                "chunk_index": len(chunks) + 1,
                                "text": content,
                                "page": page["page_number"],
                                "section": section,
                            }
                        )
                    if end == len(block):
                        break
                    start = max(start + 1, end - self.chunk_overlap)
        return chunks


smart_chunker = SmartChunker()
