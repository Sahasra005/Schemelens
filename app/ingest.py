"""Turn PDFs into clean, page-aware chunks (text + tables)."""
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf  # PyMuPDF

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_CHARS


@dataclass
class Chunk:
    text: str
    source: str   # PDF file name
    page: int     # 1-based page number, used for citations
    kind: str     # "text" | "table" (Phase 3 adds "image")


def _norm(line: str) -> str:
    """Normalise a line so 'Portal 5' and 'Portal 6' look identical."""
    return re.sub(r"\d+", "#", line.strip().lower())


def find_repeated_lines(pages_lines: list[list[str]], min_pages: int = 6, ratio: float = 0.4) -> set[str]:
    """Headers/footers = lines that appear on a large share of pages."""
    if len(pages_lines) < min_pages:
        return set()
    counts: Counter = Counter()
    for lines in pages_lines:
        counts.update({_norm(l) for l in lines})
    limit = len(pages_lines) * ratio
    return {line for line, n in counts.items() if n >= limit}


def _cut_long(line: str, size: int) -> list[str]:
    """Split one very long line at word boundaries."""
    if len(line) <= size:
        return [line]
    parts, cur = [], ""
    for word in line.split():
        if cur and len(cur) + len(word) + 1 > size:
            parts.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        parts.append(cur)
    return parts


def split_text(lines: list[str], size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Pack whole lines into chunks; the overlap is made of whole trailing lines,
    so a chunk never starts or ends in the middle of a word."""
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for line in lines:
        for piece in _cut_long(line, size):
            if current and length + len(piece) + 1 > size:
                chunks.append("\n".join(current))
                carry, carried = [], 0
                for prev in reversed(current):
                    if carried + len(prev) + 1 > overlap:
                        break
                    carry.insert(0, prev)
                    carried += len(prev) + 1
                current, length = carry, carried
            current.append(piece)
            length += len(piece) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def rows_to_markdown(rows: list[list]) -> str:
    """Convert extracted table rows into a markdown table the LLM can read."""
    clean = [[(cell or "").replace("\n", " ").strip() for cell in row] for row in rows if row]
    if len(clean) < 2:
        return ""
    header, body = clean[0], clean[1:]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def parse_pdf(path: Path) -> list[Chunk]:
    doc = pymupdf.open(path)

    # pass 1: read every page's lines so repeated headers/footers can be found
    pages_lines = []
    for page in doc:
        lines = [re.sub(r"\s+", " ", l).strip() for l in page.get_text("text").splitlines()]
        pages_lines.append([l for l in lines if l])
    boilerplate = find_repeated_lines(pages_lines)

    # pass 2: build chunks
    chunks: list[Chunk] = []
    for page_num, lines in enumerate(pages_lines, start=1):
        kept = [l for l in lines if _norm(l) not in boilerplate]
        for piece in split_text(kept):
            if len(piece) >= MIN_CHUNK_CHARS:
                chunks.append(Chunk(piece, path.name, page_num, "text"))

        # tables are kept whole so eligibility rows are not cut apart
        try:
            for table in doc[page_num - 1].find_tables().tables:
                md = rows_to_markdown(table.extract())
                if md:
                    chunks.append(Chunk(md, path.name, page_num, "table"))
        except Exception as exc:
            print(f"  [warn] table detection failed on {path.name} p{page_num}: {exc}")
    doc.close()
    return chunks
