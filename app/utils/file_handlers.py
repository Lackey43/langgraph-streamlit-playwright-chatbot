"""Multi-modal file processing utilities.
Supports: PDF (text + tables), Images (OCR or vision description), DOCX, TXT, CSV/JSON.
Returns structured summaries suitable for injection into LLM context.
"""
import logging
import base64
from io import BytesIO
from typing import List, Dict, Any, Tuple
from pathlib import Path

import pdfplumber
from docx import Document
from PIL import Image
import pytesseract
import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".docx", ".txt", ".csv", ".json", ".md"}

def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()

def process_uploaded_files(files: List[Any]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Process a list of Streamlit UploadedFile objects.
    Returns:
        - list of file metadata dicts (for state)
        - combined context string ready for LLM
    """
    processed = []
    context_parts = ["### Uploaded Documents & Files Context:\n"]
    
    for file in files:
        ext = get_file_extension(file.name)
        if ext not in SUPPORTED_EXTENSIONS:
            context_parts.append(f"⚠️ Unsupported file type: {file.name} (skipped)")
            continue
        
        try:
            file_bytes = file.getvalue()
            file_info = {
                "name": file.name,
                "type": ext,
                "size_kb": round(len(file_bytes) / 1024, 1),
                "summary": "",
                "excerpt": ""
            }
            
            if ext == ".pdf":
                text, tables = _extract_pdf(file_bytes)
                file_info["summary"] = f"PDF with {len(text)} chars text and {len(tables)} tables."
                file_info["excerpt"] = text[:2000] + ("..." if len(text) > 2000 else "")
                context_parts.append(f"**📄 {file.name}** (PDF):\n{file_info['excerpt']}\n")
                if tables:
                    context_parts.append(f"Tables found: {tables[0][:500]}...\n")
            
            elif ext in {".png", ".jpg", ".jpeg"}:
                description = _process_image(file_bytes, file.name)
                file_info["summary"] = "Image processed via OCR/vision."
                file_info["excerpt"] = description
                context_parts.append(f"**🖼️ {file.name}** (Image):\n{description}\n")
            
            elif ext == ".docx":
                text = _extract_docx(file_bytes)
                file_info["excerpt"] = text[:2500]
                context_parts.append(f"**📝 {file.name}** (DOCX):\n{file_info['excerpt']}\n")
            
            elif ext in {".txt", ".md"}:
                text = file_bytes.decode("utf-8", errors="ignore")
                file_info["excerpt"] = text[:3000]
                context_parts.append(f"**📃 {file.name}**:\n{file_info['excerpt']}\n")
            
            elif ext == ".csv":
                df = pd.read_csv(BytesIO(file_bytes))
                summary = f"CSV with {len(df)} rows and {len(df.columns)} columns. Columns: {list(df.columns)[:8]}"
                file_info["summary"] = summary
                file_info["excerpt"] = df.head(10).to_markdown()
                context_parts.append(f"**📊 {file.name}** (CSV):\n{summary}\nPreview:\n{file_info['excerpt']}\n")
            
            elif ext == ".json":
                import json
                data = json.loads(file_bytes.decode())
                file_info["summary"] = f"JSON object with {len(data) if isinstance(data, (list, dict)) else 'unknown'} items."
                file_info["excerpt"] = str(data)[:1500]
                context_parts.append(f"**🔧 {file.name}** (JSON):\n{file_info['excerpt']}\n")
            
            processed.append(file_info)
            
        except Exception as e:
            logger.error(f"Failed to process {file.name}: {e}")
            context_parts.append(f"❌ Error processing {file.name}: {str(e)[:100]}")
    
    full_context = "\n".join(context_parts)
    return processed, full_context

def _extract_pdf(file_bytes: bytes) -> Tuple[str, List[str]]:
    """Extract text and tables from PDF using pdfplumber."""
    text_parts = []
    tables = []
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages[:5]):  # Limit pages for speed
            page_text = page.extract_text() or ""
            text_parts.append(f"[Page {i+1}]\n{page_text}")
            page_tables = page.extract_tables()
            if page_tables:
                for t in page_tables:
                    tables.append(pd.DataFrame(t[1:], columns=t[0]).to_markdown())
    return "\n\n".join(text_parts), tables

def _extract_docx(file_bytes: bytes) -> str:
    doc = Document(BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs[:30])

def _process_image(file_bytes: bytes, filename: str) -> str:
    """
    Process image: Try pytesseract OCR first. 
    If vision enabled and LLM supports, caller can enhance with base64.
    For now returns OCR text or description note.
    """
    try:
        img = Image.open(BytesIO(file_bytes))
        # OCR
        ocr_text = pytesseract.image_to_string(img).strip()
        if ocr_text:
            return f"OCR extracted text from image:\n{ocr_text[:1500]}"
        else:
            return f"Image '{filename}' uploaded. No readable text via OCR. (Visual content would be analyzed by vision-capable LLM if enabled.)"
    except Exception as e:
        logger.warning(f"Image processing/OCR failed for {filename}: {e}")
        return f"Image file '{filename}' received. OCR failed; visual analysis available with multimodal LLM."

def image_to_base64(file_bytes: bytes) -> str:
    """Convert image bytes to base64 data URI for multimodal LLM messages."""
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    # Try to detect mime
    try:
        img = Image.open(BytesIO(file_bytes))
        mime = f"image/{img.format.lower()}" if img.format else "image/jpeg"
    except:
        mime = "image/jpeg"
    return f"data:{mime};base64,{b64}"
