---
sidebar_position: 16
title: Document Processing Pipelines
description: Parsing, chunking strategies, OCR, and table extraction — turning raw documents into something a model can use.
---

# Document Processing Pipelines

## What it is

Document processing pipelines are the pre-processing layer that transforms raw documents — PDFs, Word files, HTML pages, scanned images, spreadsheets — into clean, structured text that an LLM can reason over. They sit upstream of every RAG (retrieval-augmented generation: searching a document store and pasting the relevant results into the prompt) system and any application that ingests unstructured data.

The pipeline typically covers: format parsing (extract text from the source format), structure detection (identify headings, paragraphs, tables, lists), cleaning (remove noise, artifacts, and boilerplate), and chunking (split into appropriately sized pieces for indexing).

## The problem it solves

LLMs consume text. Real-world documents are not text — they are PDFs with complex layouts, scanned images, HTML pages with navigation boilerplate, spreadsheets with merged cells, and Word documents with tracked changes. The gap between "raw document" and "clean text a model can use" is large and full of failure modes.

Without a robust processing pipeline:
- PDF text extraction produces garbled output (columns merged, tables linearized incorrectly, headers repeated on every page)
- Scanned documents produce no text at all without OCR
- HTML pages include navigation menus, cookie banners, and ads alongside the actual content
- Tables extracted as plain text lose their structure and become nonsensical

The quality of your document processing directly determines the quality of your RAG retrieval: garbage in, garbage out.

## How it works under the hood

### Format-specific parsing

Different document formats require different parsers:

**PDF:** The most common and problematic format. PDFs can be: text-based (extractable), image-based (scanned, requires OCR — optical character recognition: software that reads text out of a picture of a page), or hybrid (text and images mixed).

The function below reads a PDF page by page and checks each page's "text coverage" — whether real extractable text came back at all, which is how you detect a scanned page before it silently produces empty or garbled content:

```python
import pypdf
from pathlib import Path

def extract_pdf_text(path: str) -> dict:
    reader = pypdf.PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        pages.append({
            "page_number": i + 1,
            "text": text.strip() if text else "",
            "has_text": bool(text and text.strip()),
        })
    return {
        "total_pages": len(reader.pages),
        "pages": pages,
        "text_coverage": sum(1 for p in pages if p["has_text"]) / len(pages),
    }

result = extract_pdf_text("document.pdf")
print(f"Text coverage: {result['text_coverage']:.0%}")
if result["text_coverage"] < 0.5:
    print("Warning: Low text coverage — document may be scanned. Run OCR.")
```

**HTML:** Strip markup and extract meaningful content, discarding navigation, footers, and ads:

```python
from bs4 import BeautifulSoup
import requests

def extract_html_content(url_or_html: str) -> dict:
    if url_or_html.startswith("http"):
        html = requests.get(url_or_html, timeout=10).text
    else:
        html = url_or_html

    soup = BeautifulSoup(html, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    # Extract title and main content
    title = soup.find("title")
    main = soup.find("main") or soup.find("article") or soup.find("body")

    return {
        "title": title.get_text(strip=True) if title else "",
        "text": main.get_text(separator="\n", strip=True) if main else "",
    }
```

**Word (.docx):** the function below walks every paragraph in the document, keeping non-empty ones and marking headings with markdown `#` symbols so the structure survives the conversion to plain text:

```python
from docx import Document as DocxDocument

def extract_docx_text(path: str) -> str:
    doc = DocxDocument(path)
    sections = []
    for para in doc.paragraphs:
        if para.text.strip():
            # Preserve heading structure
            prefix = "#" * para.style.font.size if "Heading" in para.style.name else ""
            sections.append(f"{prefix} {para.text}" if prefix else para.text)
    return "\n\n".join(sections)
```

### OCR for scanned documents

When PDF text extraction returns empty or garbled content, use OCR to extract text from the rendered page images:

```python
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

def ocr_pdf(path: str, dpi: int = 300) -> list[dict]:
    images = convert_from_path(path, dpi=dpi)
    pages = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image, lang="eng")
        pages.append({
            "page_number": i + 1,
            "text": text.strip(),
            "method": "ocr",
        })
    return pages

def smart_pdf_extract(path: str) -> list[dict]:
    """Use text extraction if available; fall back to OCR."""
    result = extract_pdf_text(path)
    if result["text_coverage"] >= 0.7:
        return result["pages"]
    else:
        print(f"Low text coverage ({result['text_coverage']:.0%}), using OCR...")
        return ocr_pdf(path)
```

### Table extraction

Tables in PDFs are notoriously difficult — text-based PDF parsers linearize rows and columns in ways that lose structure. Specialized table extractors use layout analysis — looking at where text is actually positioned on the page, not just the order it was written in, to figure out which words belong to which row and column:

```python
import pdfplumber

def extract_tables_from_pdf(path: str) -> list[dict]:
    tables = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()
            for j, table in enumerate(page_tables):
                if not table:
                    continue
                # Convert to markdown for LLM consumption
                header = table[0]
                rows = table[1:]
                md_rows = ["| " + " | ".join(str(c or "") for c in header) + " |",
                           "| " + " | ".join("---" for _ in header) + " |"]
                md_rows.extend("| " + " | ".join(str(c or "") for c in row) + " |" for row in rows)
                tables.append({
                    "page": i + 1,
                    "table_index": j,
                    "markdown": "\n".join(md_rows),
                    "rows": len(rows),
                    "cols": len(header),
                })
    return tables
```

### Structure-aware chunking

After extracting text, chunk it for indexing. The chunking strategy depends on document structure:

```python
import re

def structure_aware_chunk(text: str, max_chunk_tokens: int = 400) -> list[dict]:
    """Split text at heading boundaries, then by size if needed."""
    # Split at markdown-style headings
    heading_pattern = re.compile(r"^(#{1,3}\s+.+)$", re.MULTILINE)
    sections = re.split(heading_pattern, text)

    chunks = []
    current_heading = ""
    current_text = ""

    for part in sections:
        if heading_pattern.match(part):
            # Flush previous section
            if current_text.strip():
                chunks.extend(split_by_size(current_heading, current_text, max_chunk_tokens))
            current_heading = part
            current_text = ""
        else:
            current_text += part

    if current_text.strip():
        chunks.extend(split_by_size(current_heading, current_text, max_chunk_tokens))

    return chunks

def split_by_size(heading: str, text: str, max_tokens: int) -> list[dict]:
    """Further split a section if it exceeds max_tokens."""
    words = text.split()
    approx_tokens_per_word = 1.3
    max_words = int(max_tokens / approx_tokens_per_word)

    if len(words) <= max_words:
        return [{"heading": heading, "text": (heading + "\n\n" + text).strip()}]

    chunks = []
    for i in range(0, len(words), max_words):
        chunk_words = words[i:i + max_words]
        chunks.append({
            "heading": heading,
            "text": (heading + "\n\n" + " ".join(chunk_words)).strip() if i == 0
                    else " ".join(chunk_words),
        })
    return chunks
```

## Concrete example

A complete document ingestion pipeline for a RAG system that handles PDFs, HTML, and Word documents:

```python
import anthropic
from pathlib import Path
import pypdf
import pdfplumber
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
import re

client = anthropic.Anthropic()

def ingest_document(path_or_url: str) -> list[dict]:
    """
    Returns a list of chunks: {"text": str, "source": str, "page": int | None}
    """
    source = path_or_url
    ext = Path(path_or_url).suffix.lower() if not path_or_url.startswith("http") else ".html"

    if ext == ".pdf":
        reader = pypdf.PdfReader(path_or_url)
        all_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                all_text.append({"text": text.strip(), "page": i + 1})

        # Also extract tables
        with pdfplumber.open(path_or_url) as pdf:
            for i, page in enumerate(pdf.pages):
                for table in page.extract_tables() or []:
                    if table and table[0]:
                        header = table[0]
                        rows = ["| " + " | ".join(str(c or "") for c in r) + " |" for r in table]
                        all_text.append({"text": "\n".join(rows), "page": i + 1, "is_table": True})

        raw_text = "\n\n".join(t["text"] for t in all_text)

    elif ext in (".html", ".htm") or path_or_url.startswith("http"):
        import requests
        html = requests.get(path_or_url, timeout=10).text if path_or_url.startswith("http") else open(path_or_url).read()
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.find("body")
        raw_text = main.get_text(separator="\n", strip=True) if main else ""

    elif ext in (".docx",):
        doc = DocxDocument(path_or_url)
        raw_text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

    else:
        with open(path_or_url) as f:
            raw_text = f.read()

    # Clean common artifacts
    raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)
    raw_text = re.sub(r"[ \t]+", " ", raw_text)

    # Chunk
    words = raw_text.split()
    chunk_size = 350  # words (~450 tokens)
    overlap = 40      # words

    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunks.append({
            "text": " ".join(chunk_words),
            "source": source,
            "chunk_index": len(chunks),
        })

    return chunks

# Use with a RAG pipeline
chunks = ingest_document("company_policy.pdf")
print(f"Extracted {len(chunks)} chunks")
for chunk in chunks[:2]:
    print(f"Chunk {chunk['chunk_index']}: {chunk['text'][:100]}...")
```

## When to use it / when not to

#### Invest in document processing when

- Your corpus includes PDFs, scanned documents, or multi-format files
- Table data in documents is important (financial reports, technical specs, pricing sheets)
- Document structure (headings, sections) matters for chunking quality
- Your RAG accuracy is limited by retrieval quality rather than the LLM

#### Simpler approaches work when

- Your documents are clean, pre-structured text (markdown, plain text, clean HTML)
- All documents are from the same format and the extraction is reliable
- You're in an early prototype stage where extraction quality is secondary to testing the pipeline

#### The practical question

Look at your extracted text before indexing it. Literally read 10 chunks from your corpus. If they look clean and semantically coherent, your pipeline is working. If they have garbled text, broken tables, repeated headers, or cut-off sentences, your chunking or extraction needs improvement.

:::tip[My take]

The single biggest lever in document processing is not the chunking strategy or the embedding model — it's the quality of the text extraction. A perfectly chunked and embedded version of garbled PDF extraction still retrieves garbled content. Spend time here.

Evaluate your extraction quality on a representative sample before indexing the full corpus. Use the LLM itself: extract a page, ask Claude "Is this text coherent and complete? What information appears to be missing or garbled?" A quick manual spot-check of 10–20 pages reveals most extraction problems.

For PDFs with complex layouts (multi-column text, mixed text and images), consider using a document AI service (AWS Textract, Azure Document Intelligence, Google Document AI) rather than open-source parsers. The cost is higher but the extraction quality for complex layouts is dramatically better.

:::

## Main tools and libraries

| Tool | Use for |
|---|---|
| `pypdf` / `PyMuPDF` | Text extraction from text-based PDFs; fast and open-source |
| `pdfplumber` | Table extraction from PDFs; better layout awareness than pypdf |
| `pytesseract` + `pdf2image` | OCR pipeline for scanned PDFs |
| `python-docx` | Word document parsing |
| `BeautifulSoup` | HTML parsing and boilerplate removal |
| `unstructured` | Unified document processing library; handles PDF, HTML, DOCX, images |
| AWS Textract / Azure Document Intelligence | Managed document AI; best for complex layouts, handwriting, forms |
| Marker | High-quality PDF-to-markdown conversion using vision models |
| `docling` (IBM) | Document parsing with layout understanding and table reconstruction |

## Common failure modes and gotchas

**1. Assuming all PDFs are text-based.** PDFs produced from scanned images contain no extractable text — you get blank pages or encoding artifacts. Always check text coverage per page and route to OCR when coverage is low.

**2. Column merging in multi-column layouts.** Most PDF text extractors read text in document order, which merges columns horizontally instead of reading each column top-to-bottom. A two-column academic paper extracted naively produces alternating lines from each column. Use layout-aware extractors (PyMuPDF with layout analysis, or Marker) for multi-column documents.

**3. Table linearization.** A table extracted as plain text loses row and column relationships. `2024 | $1.2M | 12%` means nothing without the column headers. Always extract tables separately and convert to markdown or JSON before including in chunks.

**4. Header and footer repetition.** Many PDFs repeat the document title, company name, and page number on every page. These get extracted as text and pollute every chunk. Detect and strip repeated patterns across pages before chunking.

**5. Chunk boundary cuts mid-sentence.** Fixed-size character or word chunking cuts sentences arbitrarily. Use sentence-boundary-aware chunking or add overlap (50–100 words) to preserve context across boundaries.

**6. Encoding artifacts in extracted text.** Ligatures (fi, fl), special characters, and non-ASCII characters often appear as garbled sequences or question marks in extracted text. Run a unicode normalization step — collapsing different byte-representations of the same character down to one consistent form — and detect/log high artifact rates.

**7. Large documents indexed as too few chunks.** A 200-page report chunked into 20 large chunks is under-indexed — queries that should retrieve page 47 can't because the chunk containing page 47 also contains pages 45–50 and dilutes the relevance. Prefer smaller chunks (300–500 tokens) with overlap.

## Project ideas

**1. Extraction quality audit** — Take 10 representative documents from your corpus. Extract text from each using two methods (pypdf vs. pdfplumber vs. unstructured). Manually read the output for each and score: is the text coherent? Are tables preserved? Are headers garbled? Identify which method works best for your document types.

**2. Table extraction pipeline** — Find 5 PDFs with tables (financial reports, product specs, pricing sheets). Extract text with a standard parser, then extract tables specifically with pdfplumber. Index both. Test retrieval queries that require table data. Measure whether table-aware extraction improves retrieval on those queries.

**3. OCR vs. text extraction comparison** — Take a PDF that has mixed text and scanned sections. Extract with both text extraction and OCR. Compare word error rate on a page you manually transcribe as ground truth. Measure latency and cost difference. This makes the OCR/text tradeoff concrete.

**4. Chunking strategy ablation** — Index the same document corpus with three chunking strategies: (a) fixed-size 512-character chunks, (b) sentence-boundary-aware chunks, (c) structure-aware chunks (split at headings). Run 50 retrieval queries where you know ground-truth answers. Measure retrieval recall at k=5 for each strategy.

## Going deeper

#### Foundational reading

- Tesseract OCR documentation (github.com/tesseract-ocr/tesseract) — the reference for open-source OCR; explains language packs, confidence scoring, and performance tuning.
- `unstructured` documentation (docs.unstructured.io) — the most comprehensive open-source document processing library; covers 20+ file formats with layout understanding.

#### Tools

- Marker (github.com/VikParuchuri/marker) — high-quality PDF-to-markdown conversion using vision models; handles multi-column, tables, and equations significantly better than text-based parsers.
- `docling` (github.com/DS4SD/docling) — IBM's document understanding library; particularly strong on table structure reconstruction.
- AWS Textract documentation — managed OCR + table + form extraction; useful when open-source extraction quality is insufficient.
