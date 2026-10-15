"""
Paper Mind - Chunking Module
------------------------------
Two interchangeable chunking strategies with a unified interface:
1. Fixed-size chunking (equal-sized word windows with overlap)
2. Section-aware chunking (splits on detected PDF section headers, with sub-chunking for long sections)
"""

import json
from pathlib import Path


def fixed_size_chunking(pages, chunk_size=150, overlap=30):
    """
    Splits full text into equal-sized word chunks, with overlap so
    sentences aren't harshly cut at chunk boundaries.
    
    Returns list of dicts:
    {"text": ..., "page": ..., "section": None, "strategy": "fixed"}
    """
    chunks = []
    for page in pages:
        page_num = page["page"]
        text = page["text"]
        words = text.split()
        
        if not words:
            continue
            
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:
                chunks.append({
                    "text": " ".join(chunk_words),
                    "page": page_num,
                    "section": None,
                    "strategy": "fixed"
                })
            i += (chunk_size - overlap)
            if i >= len(words):
                break
    return chunks


SECTION_HEADERS = [
    "abstract", "introduction", "related work", "methods", "methodology",
    "implementation", "experimental results", "results", "discussion",
    "replicability", "datasets", "ethics", "limitations", "conclusion", "references", "appendix",
    "background", "evaluation"
]


def section_aware_chunking(pages, max_section_words=500, overlap=50):
    """
    Splits text along detected section headers from PDF layout/font heuristics.
    If a section exceeds max_section_words, it is split into overlapping sub-chunks
    while retaining the section label on each sub-chunk.
    
    Returns list of dicts:
    {"text": ..., "page": ..., "section": ..., "strategy": "section_aware"}
    """
    chunks = []
    current_section = "Abstract"  # Default initial section
    buffer = []
    buffer_page = None

    def flush():
        nonlocal buffer, buffer_page, current_section
        if not buffer:
            return
        
        section_text = "\n\n".join(buffer).strip()
        buffer = []
        if len(section_text) < 20:
            return

        words = section_text.split()
        # If section exceeds max_section_words, sub-chunk with overlap
        if len(words) > max_section_words:
            i = 0
            while i < len(words):
                sub_words = words[i:i + max_section_words]
                if sub_words:
                    chunks.append({
                        "text": " ".join(sub_words),
                        "page": buffer_page,
                        "section": current_section,
                        "strategy": "section_aware"
                    })
                i += (max_section_words - overlap)
                if i >= len(words):
                    break
        else:
            chunks.append({
                "text": section_text,
                "page": buffer_page,
                "section": current_section,
                "strategy": "section_aware"
            })

    for page in pages:
        page_num = page["page"]
        blocks = page.get("blocks", [])

        if blocks:
            for b in blocks:
                text = b["text"].strip()
                if not text:
                    continue

                if b.get("is_header"):
                    flush()
                    # Clean multi-line header text (e.g., "1\nIntroduction" -> "1 Introduction")
                    cleaned_header = " ".join(text.split())
                    current_section = cleaned_header
                    buffer_page = page_num
                else:
                    buffer.append(text)
                    if buffer_page is None:
                        buffer_page = page_num
        else:
            # Fallback for plain text extraction
            lines = page["text"].split("\n")
            for line in lines:
                stripped = line.strip()
                lower = stripped.lower().strip(" .0123456789")
                is_header = any(lower == h or lower.startswith(h) for h in SECTION_HEADERS) and len(stripped) < 40

                if is_header:
                    flush()
                    current_section = stripped
                    buffer_page = page_num
                else:
                    if stripped:
                        buffer.append(stripped)
                        if buffer_page is None:
                            buffer_page = page_num

    flush()
    return chunks


def chunk_paper(pages, strategy="fixed", **kwargs):
    """
    Unified entry point for chunking papers based on configured strategy.
    
    strategy: 'fixed' | 'section_aware'
    """
    if strategy == "fixed":
        chunk_size = kwargs.get("chunk_size", 150)
        overlap = kwargs.get("overlap", 30)
        return fixed_size_chunking(pages, chunk_size=chunk_size, overlap=overlap)
    elif strategy == "section_aware":
        max_section_words = kwargs.get("max_section_words", 500)
        overlap = kwargs.get("overlap", 50)
        return section_aware_chunking(pages, max_section_words=max_section_words, overlap=overlap)
    else:
        raise ValueError(f"Unknown chunking strategy: '{strategy}'. Supported strategies: 'fixed', 'section_aware'.")


if __name__ == "__main__":
    extracted_json = Path(__file__).parent / "extracted" / "research_major.json"
    if not extracted_json.exists():
        print(f"Error: {extracted_json} not found. Run extract_pdf.py first.")
        exit(1)

    with open(extracted_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    pages = data["pages"]

    fixed = chunk_paper(pages, strategy="fixed", chunk_size=150, overlap=30)
    section = chunk_paper(pages, strategy="section_aware", max_section_words=500, overlap=50)

    print(f"=== Chunking Results for {data.get('source', 'paper')} ===")
    print(f"Fixed-size chunking:   {len(fixed)} chunks")
    print(f"Section-aware chunking: {len(section)} chunks")

    fixed_word_counts = [len(c["text"].split()) for c in fixed]
    section_word_counts = [len(c["text"].split()) for c in section]

    print("\n--- Statistics ---")
    print(f"Fixed:   Avg Words = {sum(fixed_word_counts)/len(fixed):.1f} | Min = {min(fixed_word_counts)} | Max = {max(fixed_word_counts)}")
    print(f"Section: Avg Words = {sum(section_word_counts)/len(section):.1f} | Min = {min(section_word_counts)} | Max = {max(section_word_counts)}")

    print("\n--- Sample Fixed Chunks ---")
    for i, c in enumerate(fixed[:3]):
        print(f"[Fixed Chunk {i+1}] (Page {c['page']}, {len(c['text'].split())} words) -> {c['text'][:120]}...")

    print("\n--- Sample Section-Aware Chunks ---")
    for i, c in enumerate(section[:6]):
        print(f"[Section: {c['section']}] (Page {c['page']}, {len(c['text'].split())} words) -> {c['text'][:120]}...")

    out_dir = Path(__file__).parent / "extracted"
    with open(out_dir / "chunks_fixed.json", "w", encoding="utf-8") as f:
        json.dump(fixed, f, indent=2)
    with open(out_dir / "chunks_section.json", "w", encoding="utf-8") as f:
        json.dump(section, f, indent=2)

    print(f"\nSaved:\n- {out_dir / 'chunks_fixed.json'}\n- {out_dir / 'chunks_section.json'}")
