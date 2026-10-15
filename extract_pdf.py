import fitz
import pdfplumber
import json
import string
import collections
import re
from pathlib import Path

# Set up academic symbols and Greek letters to exclude from "non-alphanumeric junk"
ascii_punc = set(string.punctuation)
whitespace = set(" \n\r\t")
academic_symbols = set("°±≥≤αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚ prosecutionΛΜΝΞΟΠΡΣΤΥΦΧΨΩ−×÷√∞≈≠≡⊂⊃⊆⊇∈∉∩∪–—")
ALLOWED_NON_ALNUM = ascii_punc | whitespace | academic_symbols

SECTION_HEADERS = [
    "abstract", "introduction", "related work", "methods", "methodology",
    "implementation", "experimental results", "results", "discussion",
    "replicability", "datasets", "ethics", "limitations", "conclusion", "references", "appendix",
    "background", "evaluation"
]


def get_document_body_size(doc):
    """
    Computes the most frequent font size across the first few pages of the document.
    """
    all_spans = []
    for page in doc[:3]:
        d = page.get_text("dict")
        for b in d.get("blocks", []):
            if b.get("type") == 0:
                for l in b.get("lines", []):
                    for s in l.get("spans", []):
                        all_spans.append(s)
    if not all_spans:
        return 10.0
    sizes = [round(s.get("size", 10.0), 1) for s in all_spans]
    return collections.Counter(sizes).most_common(1)[0][0]


def is_block_header(text, spans, body_size):
    """
    Detects if a block is an academic section header using font styling and text structure heuristics.
    """
    text = text.strip()
    if not text or len(text) > 60:
        return False
    
    # Exclude non-header indicators (tables, figures, formulas, metadata)
    if any(sym in text for sym in ["%", "Latency", "F1", "Table ", "Figure ", "Emails:", "arXiv:"]):
        return False
        
    rest = text.lstrip("0123456789. \t")
    if not rest:
        return False
        
    # Exclude full sentences or bullet points (e.g., "1. We design...")
    if rest.lower().startswith(("we ", "our ", "this ", "the ", "in ", "for ")) and text.endswith("."):
        return False
    if text.endswith(".") and not any(text.lower().strip(" .0123456789") == h for h in SECTION_HEADERS):
        return False
        
    # Header text shouldn't contain multiple numbers scattered in body
    if sum(c.isdigit() for c in rest) > 1:
        return False
        
    # Word count of header should be concise
    words = text.split()
    if len(words) > 8:
        return False

    # 1. Check if text matches standard academic section names or starts with section numbering
    cleaned = text.lower().strip(" .0123456789")
    is_standard = any(cleaned == h or cleaned.startswith(h) for h in SECTION_HEADERS)
    has_numbering = bool(re.match(r'^(?:[0-9]+(?:\.[0-9]+)*|[A-Z]+(?:\.[0-9]+)*)\s*[A-Z]', text))
    
    if not (is_standard or has_numbering):
        return False
        
    # 2. Check formatting (if spans provided): must be bold or larger than body size
    if spans:
        has_bold = False
        has_larger = False
        for s in spans:
            font = s.get("font", "").lower()
            if "bold" in font or "medi" in font or s.get("flags", 0) & 4:
                has_bold = True
            if s.get("size", 0) >= body_size + 0.5:
                has_larger = True
        if not (has_bold or has_larger):
            return False
            
    return True


def sort_blocks_layout_aware(blocks, page_width, page_height, col_gap=15):
    """
    Sorts structured text blocks to preserve reading order in two-column academic layouts.
    Groups full-width elements and sorts columns independently within vertical bands.
    """
    col_divider = page_width / 2

    text_blocks = [b for b in blocks if b["text"].strip()]
    if not text_blocks:
        return []

    # 1. Classify blocks into full-width vs column blocks
    full_width_blocks = []
    column_blocks = []
    for b in text_blocks:
        x0, y0, x1, y1 = b["bbox"]
        if x0 < col_divider - col_gap and x1 > col_divider + col_gap:
            full_width_blocks.append(b)
        else:
            column_blocks.append(b)

    # 2. Create full-width bands (merge overlapping y-intervals)
    intervals = [(b["bbox"][1], b["bbox"][3]) for b in full_width_blocks]
    intervals.sort(key=lambda x: x[0])
    merged_intervals = []
    for y0, y1 in intervals:
        if not merged_intervals or y0 > merged_intervals[-1][1]:
            merged_intervals.append([y0, y1])
        else:
            merged_intervals[-1][1] = max(merged_intervals[-1][1], y1)

    # 3. Construct vertical bands
    bands = []
    curr_y = 0.0
    for y0, y1 in merged_intervals:
        if y0 > curr_y:
            bands.append({
                "type": "split",
                "y0": curr_y,
                "y1": y0,
                "blocks": []
            })
        bands.append({
            "type": "full",
            "y0": y0,
            "y1": y1,
            "blocks": []
        })
        curr_y = y1

    if curr_y < page_height:
        bands.append({
            "type": "split",
            "y0": curr_y,
            "y1": page_height,
            "blocks": []
        })

    # 4. Assign blocks to bands based on vertical midpoints
    for b in text_blocks:
        y0, y1 = b["bbox"][1], b["bbox"][3]
        ymid = (y0 + y1) / 2
        assigned = False
        for band in bands:
            if band["y0"] <= ymid < band["y1"]:
                band["blocks"].append(b)
                assigned = True
                break
        if not assigned and bands:
            bands[-1]["blocks"].append(b)

    # 5. Sort blocks within each band
    sorted_blocks = []
    for band in bands:
        if not band["blocks"]:
            continue
        if band["type"] == "full":
            band_blocks = sorted(band["blocks"], key=lambda x: (x["bbox"][1], x["bbox"][0]))
            sorted_blocks.extend(band_blocks)
        else:
            left_col = []
            right_col = []
            for b in band["blocks"]:
                x0, _, x1, _ = b["bbox"]
                xmid = (x0 + x1) / 2
                if xmid < col_divider:
                    left_col.append(b)
                else:
                    right_col.append(b)
            left_col.sort(key=lambda x: x["bbox"][1])
            right_col.sort(key=lambda x: x["bbox"][1])
            sorted_blocks.extend(left_col)
            sorted_blocks.extend(right_col)

    return sorted_blocks


def run_quality_check(text):
    """
    Computes quality metrics for extracted text: word count, char count, and junk ratio.
    """
    if not text:
        return {"word_count": 0, "char_count": 0, "junk_ratio": 0.0, "warnings": ["is_empty"]}

    words = text.split()
    word_count = len(words)
    char_count = len(text)

    non_ws_chars = [c for c in text if not c.isspace()]
    if not non_ws_chars:
        return {"word_count": 0, "char_count": char_count, "junk_ratio": 0.0, "warnings": ["is_empty"]}

    junk_chars = [c for c in non_ws_chars if not c.isalnum() and c not in ALLOWED_NON_ALNUM]
    junk_ratio = len(junk_chars) / len(non_ws_chars)

    warnings = []
    if word_count < 100:
        warnings.append("suspiciously_short")
    if junk_ratio > 0.30:
        warnings.append("high_junk_ratio")

    return {
        "word_count": word_count,
        "char_count": char_count,
        "junk_ratio": junk_ratio,
        "warnings": warnings
    }


def extract_page_with_pdfplumber(pdf_path, page_num):
    """
    Extracts text from a single page using pdfplumber as a fallback.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num - 1 < len(pdf.pages):
                page = pdf.pages[page_num - 1]
                return page.extract_text() or ""
    except Exception as e:
        print(f"Error in pdfplumber extraction on page {page_num}: {e}")
    return ""


def extract_with_pymupdf(pdf_path):
    """
    Extracts text and styled blocks page-by-page using PyMuPDF and layout-aware block sorting.
    """
    doc = fitz.open(pdf_path)
    body_size = get_document_body_size(doc)
    pages = []
    
    for page_num, page in enumerate(doc, start=1):
        page_width = page.rect.width
        page_height = page.rect.height
        
        d = page.get_text("dict")
        raw_blocks = []
        for b in d.get("blocks", []):
            if b.get("type") == 0:  # text block
                lines = b.get("lines", [])
                line_texts = []
                spans = []
                for l in lines:
                    line_str = "".join(s.get("text", "") for s in l.get("spans", [])).strip()
                    if line_str:
                        line_texts.append(line_str)
                    for s in l.get("spans", []):
                        spans.append(s)
                
                block_text = "\n".join(line_texts).strip()
                if block_text:
                    is_hdr = is_block_header(block_text, spans, body_size)
                    raw_blocks.append({
                        "bbox": b["bbox"],
                        "text": block_text,
                        "is_header": is_hdr,
                        "spans": spans
                    })

        sorted_blocks = sort_blocks_layout_aware(raw_blocks, page_width, page_height)
        page_text = "\n\n".join(b["text"] for b in sorted_blocks)
        
        blocks_data = [{"text": b["text"], "is_header": b["is_header"]} for b in sorted_blocks]
        pages.append({
            "page": page_num,
            "text": page_text,
            "blocks": blocks_data
        })
        
    doc.close()
    return pages, body_size


def extract_paper(pdf_path, out_dir=None):
    """
    Runs the full PDF extraction pipeline:
    - Extracts using PyMuPDF + layout-aware sorting + header detection.
    - Performs quality check.
    - Falls back to pdfplumber for messy pages.
    - Saves structured JSON and raw text files.
    """
    pdf_path = Path(pdf_path)
    if out_dir is None:
        out_dir = Path(__file__).parent / "extracted"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Primary extraction with PyMuPDF
    pymupdf_pages, body_size = extract_with_pymupdf(str(pdf_path))

    final_pages = []
    quality_warnings = []
    pages_report = []

    # 2. Run quality checks and apply pdfplumber fallback logic
    for p in pymupdf_pages:
        page_num = p["page"]
        text = p["text"]
        blocks = p["blocks"]
        metrics = run_quality_check(text)

        extractor_used = "pymupdf"
        final_text = text
        final_blocks = blocks
        final_metrics = metrics

        # If flagged for low quality, try pdfplumber
        if metrics["warnings"]:
            print(f"Page {page_num}: PyMuPDF extraction flagged: {metrics['warnings']}. Trying pdfplumber fallback...")
            plumber_text = extract_page_with_pdfplumber(str(pdf_path), page_num)
            plumber_metrics = run_quality_check(plumber_text)

            if not plumber_metrics["warnings"]:
                print(f"Page {page_num}: pdfplumber fallback succeeded. Using pdfplumber.")
                extractor_used = "pdfplumber_fallback"
                final_text = plumber_text
                final_metrics = plumber_metrics
                # Create virtual blocks for pdfplumber text
                paragraphs = [para.strip() for para in plumber_text.split("\n\n") if para.strip()]
                final_blocks = [
                    {"text": para, "is_header": is_block_header(para, [], body_size)}
                    for para in paragraphs
                ]
            else:
                print(f"Page {page_num}: pdfplumber also flagged: {plumber_metrics['warnings']}. Keeping PyMuPDF and flagging for manual review.")
                final_metrics = metrics
                if "manual_review_recommended" not in final_metrics["warnings"]:
                    final_metrics["warnings"].append("manual_review_recommended")

        for w in final_metrics["warnings"]:
            quality_warnings.append({
                "page": page_num,
                "warning": w,
                "extractor_used": extractor_used,
                "word_count": final_metrics["word_count"],
                "junk_ratio": final_metrics["junk_ratio"]
            })

        pages_report.append({
            "page": page_num,
            "word_count": final_metrics["word_count"],
            "char_count": final_metrics["char_count"],
            "junk_ratio": final_metrics["junk_ratio"],
            "warnings": final_metrics["warnings"],
            "extractor_used": extractor_used
        })

        final_pages.append({
            "page": page_num,
            "text": final_text,
            "extractor_used": extractor_used,
            "blocks": final_blocks
        })

    # 3. Compile overall quality report
    total_pages = len(final_pages)
    avg_words = sum(p["word_count"] for p in pages_report) / total_pages if total_pages > 0 else 0
    quality_report = {
        "source": pdf_path.name,
        "total_pages": total_pages,
        "average_words_per_page": round(avg_words, 1),
        "quality_warnings": quality_warnings,
        "pages_report": pages_report
    }

    full_text = "\n\n".join(p["text"] for p in final_pages)

    # 4. Save results to disk
    out_json = out_dir / f"{pdf_path.stem}.json"
    out_txt = out_dir / f"{pdf_path.stem}.txt"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "source": pdf_path.name,
            "quality_report": quality_report,
            "pages": final_pages
        }, f, indent=2)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"\nExtraction complete for {pdf_path.name}:")
    print(f"- Total Pages: {total_pages}")
    print(f"- Average Words/Page: {avg_words:.1f}")
    print(f"- Structured JSON saved: {out_json}")
    print(f"- Raw Text saved: {out_txt}")

    return final_pages, quality_report


if __name__ == "__main__":
    import sys
    test_pdf = r"C:\Users\supev\OneDrive\Desktop\Documents\research_major.pdf"
    if not Path(test_pdf).exists():
        print(f"Error: Test PDF not found at {test_pdf}")
        sys.exit(1)
    extract_paper(test_pdf)
