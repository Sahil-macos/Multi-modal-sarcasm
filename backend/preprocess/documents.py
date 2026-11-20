"""Text extraction from documents: PDF, DOCX, TXT/MD/CSV/JSON/HTML, images."""
import re
from pathlib import Path

TEXT_EXT = {".txt", ".md", ".csv", ".json", ".log", ".tsv"}
HTML_EXT = {".html", ".htm"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
DOC_EXT = {".pdf", ".docx"} | TEXT_EXT | HTML_EXT | IMAGE_EXT


def extract_document(path) -> dict:
    path = Path(path)
    ext = path.suffix.lower()
    method = ext.lstrip(".")
    if ext == ".pdf":
        import pymupdf
        pages = []
        with pymupdf.open(path) as doc:
            for p in doc:
                pages.append(p.get_text("text"))
        text = "\n".join(pages)
        if not text.strip():  # scanned PDF -> OCR rendered pages
            text, method = _ocr_pdf(path), "pdf+ocr"
        meta = {"pages": len(pages)}
    elif ext == ".docx":
        import docx
        d = docx.Document(str(path))
        parts = [p.text for p in d.paragraphs]
        for t in d.tables:
            for row in t.rows:
                parts.append(" | ".join(c.text for c in row.cells))
        text, meta = "\n".join(parts), {"paragraphs": len(d.paragraphs)}
    elif ext in HTML_EXT:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
        text, meta = re.sub(r"<[^>]+>", " ", raw), {}
    elif ext in IMAGE_EXT:
        import numpy as np
        from PIL import Image
        from .ocr import ocr_image
        img = np.array(Image.open(path).convert("RGB"))
        text, meta, method = "\n".join(t for t, _ in ocr_image(img)), {}, "image+ocr"
    elif ext in TEXT_EXT:
        text, meta = path.read_text(encoding="utf-8", errors="ignore"), {}
    else:
        raise ValueError(f"Unsupported document type: {ext}")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return {"text": text, "method": method, **meta}


def _ocr_pdf(path, max_pages: int = 10) -> str:
    import numpy as np
    import pymupdf
    from .ocr import ocr_image
    out = []
    with pymupdf.open(path) as doc:
        for p in list(doc)[:max_pages]:
            pix = p.get_pixmap(dpi=150)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]
            out += [t for t, _ in ocr_image(img)]
    return "\n".join(out)
