"""
Step 1 – PDF Extraction & Clause Segmentation
----------------------------------------------
Extracts text from each agreement PDF, segments it into the smallest
meaningful legal provisions (articles / paragraphs / clauses), and saves
the result as a JSON file in data/raw/.

Usage:
    python -m src.extraction
"""

import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import io

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AGREE_DIR, AGREEMENTS, MIN_CLAUSE_CHARS, RAW_DIR


# ── PDF text extraction ────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract plain text from a PDF.
    Tries pdfplumber first; falls back to PyMuPDF if no text is found.
    """
    # -- pdfplumber pass --
    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        print(f"  [pdfplumber error] {pdf_path.name}: {e}")

    text = "\n".join(text_parts).strip()
    if text:
        return text

    # -- PyMuPDF fallback --
    try:
        doc = fitz.open(str(pdf_path))
        parts = []
        for page in doc:
            t = page.get_text("text")
            if t and t.strip():
                parts.append(t)
        doc.close()
        text = "\n".join(parts).strip()
        if text:
            print(f"  [PyMuPDF fallback used for {pdf_path.name}]", end="")
            return text
    except Exception as e:
        print(f"  [PyMuPDF error] {pdf_path.name}: {e}")

    # -- OCR fallback (for scanned PDFs) --
    print(f"\n  [OCR] Scanned PDF detected — running Tesseract OCR on {pdf_path.name} …", end="", flush=True)
    try:
        doc = fitz.open(str(pdf_path))
        ocr_parts = []
        # Limit to first 100 pages for performance; adjust if needed
        max_pages = min(doc.page_count, 100)
        for page_num in range(max_pages):
            page = doc[page_num]
            # Render at 200 DPI for good OCR quality
            mat  = fitz.Matrix(200/72, 200/72)
            pix  = page.get_pixmap(matrix=mat, alpha=False)
            img  = Image.open(io.BytesIO(pix.tobytes("png")))
            t    = pytesseract.image_to_string(img, config="--psm 6")
            if t.strip():
                ocr_parts.append(t)
            if (page_num + 1) % 20 == 0:
                print(f" {page_num+1}/{max_pages}", end="", flush=True)
        doc.close()
        text = "\n".join(ocr_parts).strip()
        if text:
            print(f" done ({len(text):,} chars)")
            return text
    except Exception as e:
        print(f"\n  [OCR error] {pdf_path.name}: {e}")

    return ""


# ── Clause segmentation ────────────────────────────────────────────────────────

# Patterns that indicate the start of a new legal provision
_ARTICLE_PATTERNS = [
    # "Article 1", "Article 1.1", "ARTICLE 1"
    r"^(ARTICLE|Article)\s+\d+[\.\d]*[:\s]",
    # "Chapter 1", "CHAPTER I"
    r"^(CHAPTER|Chapter)\s+[\dIVXivx]+[:\s]",
    # "Section 1", "Section A"
    r"^(SECTION|Section)\s+[\dA-Za-z]+[:\s]",
    # Numbered paragraphs at start of line: "1.", "1.1", "(a)"
    r"^\d+\.\s+[A-Z]",
    r"^\(\s*[a-z]\s*\)\s+",
    # Rule headers in Rules of Origin annexes
    r"^(Rule|RULE)\s+\d+",
    # Annex headers
    r"^(ANNEX|Annex)\s+",
]
_BOUNDARY_RE = re.compile(
    "|".join(f"(?:{pattern})" for pattern in _ARTICLE_PATTERNS),
    re.MULTILINE,
)


def _infer_chapter(text_block: str) -> str:
    """Try to extract a chapter/section name from a leading header."""
    lines = text_block.strip().splitlines()
    for line in lines[:3]:
        line = line.strip()
        if re.match(r"(Chapter|CHAPTER|Section|SECTION|Article|ARTICLE|Annex|ANNEX)", line):
            return line[:120]
    return ""


def _extract_article_id(text_block: str) -> str:
    for line in text_block.strip().splitlines()[:5]:
        line = line.strip()
        match = re.match(
            r"((?:Article|ARTICLE|Rule|RULE|Section|SECTION)\s+[\dA-Za-z\.]+)",
            line,
        )
        if match:
            return match.group(1)
    return ""


def _is_noise_block(text_block: str) -> bool:
    cleaned = text_block.strip()
    if not cleaned:
        return True

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    preview = " ".join(lines[:6]).upper()

    if "TABLE OF CONTENTS" in preview:
        return True
    if len(lines) <= 3 and all(
        re.fullmatch(r"[\dA-ZIVX\-\(\)\.:/ ]+", line) for line in lines
    ):
        return True
    if cleaned.count("...") > 3:
        return True
    if re.fullmatch(r"[\d\.\s]+", cleaned):
        return True
    return False


def _iter_provision_blocks(text: str) -> list[str]:
    matches = list(_BOUNDARY_RE.finditer(text))
    if not matches:
        return [text]

    blocks: list[str] = []
    if matches[0].start() > 0:
        blocks.append(text[: matches[0].start()])

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append(text[start:end])

    return blocks


def segment_provisions(
    raw_text: str,
    agreement: str,
    doc_type: str,
) -> list[dict]:
    """
    Split raw text into individual provisions.
    Returns a list of dicts with metadata + provision text.
    """
    # Normalise whitespace while keeping line breaks
    text = re.sub(r"\r\n?", "\n", raw_text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Split on article / section / rule headers while preserving the header line
    splits = _iter_provision_blocks(text)

    provisions = []
    article_no  = ""
    chapter     = ""
    para_idx    = 0

    for block in splits:
        if block is None:
            continue
        block = block.strip()
        if not block or len(block) < MIN_CLAUSE_CHARS or _is_noise_block(block):
            continue

        new_article = _extract_article_id(block)
        if new_article:
            article_no = new_article
        new_chapter = _infer_chapter(block)
        if new_chapter:
            chapter = new_chapter

        # Further split long blocks into paragraphs
        sub_blocks = re.split(r"\n\n+", block)
        for sub in sub_blocks:
            sub = sub.strip()
            if len(sub) < MIN_CLAUSE_CHARS or _is_noise_block(sub):
                continue

            para_idx += 1
            stored_text = sub[:1500]          # cap at documented 1500-char max
            provisions.append({
                "id":            f"{agreement}_{doc_type}_{para_idx:05d}",
                "agreement":     agreement,
                "doc_type":      doc_type,
                "chapter":       chapter,
                "article":       article_no,
                "paragraph_idx": para_idx,
                "text":          stored_text,
                "char_count":    len(stored_text),  # matches stored text, not original
            })

    return provisions


# ── Main extraction loop ───────────────────────────────────────────────────────

def run_extraction() -> list[dict]:
    """Extract and segment all agreements. Returns combined provision list."""
    all_provisions: list[dict] = []

    for agreement, filenames in AGREEMENTS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {agreement}  ({len(filenames)} file(s))")
        print("="*60)

        for fname in filenames:
            pdf_path = AGREE_DIR / fname
            if not pdf_path.exists():
                print(f"  [SKIP] Not found: {fname}")
                continue

            print(f"  Extracting: {fname}  ({pdf_path.stat().st_size / 1_000_000:.1f} MB) …", end="", flush=True)
            raw_text = extract_text_from_pdf(pdf_path)
            print(f"  {len(raw_text):,} chars")

            if not raw_text.strip():
                print(f"  [WARN] No text extracted from {fname}")
                continue

            # Infer document type from filename
            fname_lower = fname.lower()
            if "annex" in fname_lower:
                doc_type = "Annex"
            elif "protocol" in fname_lower:
                doc_type = "Protocol"
            elif "implementing" in fname_lower:
                doc_type = "Implementing Arrangement"
            elif "understanding" in fname_lower:
                doc_type = "Understanding"
            else:
                doc_type = "Main Agreement"

            provisions = segment_provisions(raw_text, agreement, doc_type)
            print(f"  → {len(provisions):,} provisions segmented")

            # Save per-file raw output
            out_file = RAW_DIR / f"{agreement}_{doc_type.replace(' ', '_')}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(provisions, f, ensure_ascii=False, indent=2)

            all_provisions.extend(provisions)

    # Save combined dataset
    combined_path = RAW_DIR / "all_provisions.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_provisions, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Total provisions extracted: {len(all_provisions):,}")
    print(f"   Saved to: {combined_path}")
    return all_provisions


if __name__ == "__main__":
    run_extraction()
