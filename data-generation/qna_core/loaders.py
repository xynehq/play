from pathlib import Path
from typing import List, Dict
import re

def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text).strip()
    return text

def read_pdf(path: Path) -> str:
    from pdfminer.high_level import extract_text  # lazy import
    return extract_text(str(path)) or ""

def read_docx(path: Path) -> str:
    from docx import Document  # lazy import
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_doc(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".pdf":
        return _normalize(read_pdf(path))
    if suf == ".docx":
        return _normalize(read_docx(path))
    if suf in {".txt", ".md"}:
        return _normalize(read_text(path))
    raise ValueError(f"Unsupported file type: {path.suffix}")
