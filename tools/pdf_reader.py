"""
PDF Reader Tools (Week 4, Day 4)
Provides whole-document overview extraction and single-page fetching using PyMuPDF (fitz).
"""

import os
try:
    import pymupdf as fitz  # Modern PyMuPDF import
except ImportError:
    import fitz  # Legacy fallback


def read_pdf(file_path: str) -> str:
    """
    Opens a PDF file, extracts metadata, and reads the first 3000 characters of text.
    Provides a document overview for the agent.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at path '{file_path}'."

    if not file_path.lower().endswith(".pdf"):
        return f"Error: Path '{file_path}' does not point to a valid .pdf file."

    try:
        with fitz.open(file_path) as doc:
            total_pages = len(doc)

            # Read metadata if available
            meta = doc.metadata or {}
            title = meta.get("title", "Unknown Title")
            author = meta.get("author", "Unknown Author")

            header = (
                f"--- PDF DOCUMENT OVERVIEW ---\n"
                f"File: {os.path.basename(file_path)}\n"
                f"Title: {title}\n"
                f"Author: {author}\n"
                f"Total Pages: {total_pages}\n"
                f"------------------------------\n\n"
            )

            extracted_text = ""
            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text().strip()

                # Skip empty or image-only pages
                if not text:
                    continue

                extracted_text += f"[Page {page_num + 1}]\n{text}\n\n"

                # Enforce 3000-character cap on document overview
                if len(extracted_text) >= 3000:
                    extracted_text = (
                        extracted_text[:3000]
                        + "\n\n... [TRUNCATED: Document exceeds 3000 characters. "
                        "Use read_pdf_page(file_path, page_number) to inspect specific pages in detail.]"
                    )
                    break

            if not extracted_text.strip():
                return (
                    f"{header}Warning: No text could be extracted from this PDF. "
                    f"The document may be a scanned image or contain protected content."
                )

            return header + extracted_text

    except Exception as e:
        return f"Error processing PDF '{file_path}': {str(e)}"


def read_pdf_page(file_path: str, page_number: int) -> str:
    """
    Extracts complete text from a single specific page of a PDF file.
    Note: page_number is 1-indexed (e.g., page_number=1 reads the first page).
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at path '{file_path}'."

    try:
        with fitz.open(file_path) as doc:
            total_pages = len(doc)

            # Convert 1-indexed human page number to 0-indexed PyMuPDF page index
            zero_indexed_page = page_number - 1

            if zero_indexed_page < 0 or zero_indexed_page >= total_pages:
                return (
                    f"Error: Invalid page number {page_number}. "
                    f"Document contains {total_pages} total pages (valid range: 1-{total_pages})."
                )

            page = doc[zero_indexed_page]
            text = page.get_text().strip()

            if not text:
                return (
                    f"--- Page {page_number} of {total_pages} ---\n"
                    f"[Warning: Page {page_number} contains no extractable text. It may be an image or blank page.]"
                )

            return f"--- Page {page_number} of {total_pages} ---\n{text}"

    except Exception as e:
        return f"Error reading page {page_number} of PDF '{file_path}': {str(e)}"
