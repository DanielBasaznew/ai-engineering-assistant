# rag/ingestion.py
import os
from pypdf import PdfReader

try:
    from rag.vector_store import get_collection, add_documents
    from rag.chunker import chunk_by_tokens, chunk_by_paragraph
except ImportError:
    try:
        from .vector_store import get_collection, add_documents
        from .chunker import chunk_by_tokens, chunk_by_paragraph
    except (ImportError, ValueError):
        from vector_store import get_collection, add_documents
        try:
            from chunker import chunk_by_tokens, chunk_by_paragraph
        except ImportError:
            from chunking import chunk_by_tokens, chunk_by_paragraph

BATCH_SIZE = 100  # Safe batch size to avoid memory strain or SQLite limits


def ingest_pdf(
    file_path: str,
    collection_name: str = "knowledge_base",
    chunk_size: int = 350,
    overlap: int = 50,
):
    """
    Extracts text from ALL PDF pages, concatenates into one continuous text,
    then applies paragraph-aware chunking across the entire document.
    Preserves page metadata by mapping chunk positions back to page boundaries.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")

    filename = os.path.basename(file_path)
    reader = PdfReader(file_path)
    collection = get_collection(collection_name)

    total_pages = len(reader.pages)
    print(f"\n[INGESTING PDF] '{filename}' ({total_pages} pages)...")

    # --- STEP 1: Concatenate all pages into one continuous text ---
    page_texts = []
    page_boundaries = []  # list of (start_char, end_char, page_number)
    running_length = 0

    for page_idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if not page_text or not page_text.strip():
            continue
        page_texts.append(page_text)
        page_boundaries.append((running_length, running_length + len(page_text), page_idx))
        running_length += len(page_text) + 2  # +2 for "\n\n" separator

    full_text = "\n\n".join(page_texts)

    # Trim References section if present to avoid citation noise hijacking vector search
    ref_marker = "\nReferences\n"
    ref_pos = full_text.find(ref_marker)
    if ref_pos != -1:
        print(f"[INFO] Trimming References section ({len(full_text) - ref_pos} chars) from '{filename}'.")
        full_text = full_text[:ref_pos]

    # --- STEP 2: Chunk the ENTIRE document with paragraph-aware chunking ---
    all_doc_chunks = chunk_by_paragraph(full_text, max_chunk_size=chunk_size, overlap=overlap)

    # --- STEP 3: Map each chunk back to its source page ---
    all_chunks = []
    all_metadatas = []
    all_ids = []

    for chunk_counter, chunk_text in enumerate(all_doc_chunks):
        # Find which page this chunk starts in
        chunk_start = full_text.find(chunk_text)
        if chunk_start == -1:
            chunk_start = full_text.find(chunk_text[:50])

        matched_page = 1  # default fallback
        if chunk_start != -1:
            for start_char, end_char, page_num in page_boundaries:
                if chunk_start >= start_char and chunk_start < end_char:
                    matched_page = page_num
                    break

        chunk_id = f"{filename}_p{matched_page}_c{chunk_counter}"
        all_chunks.append(chunk_text)
        all_metadatas.append(
            {
                "source": filename,
                "page": matched_page,
                "file_type": "pdf",
            }
        )
        all_ids.append(chunk_id)

    # --- STEP 4: Batch insertion into ChromaDB ---
    total_chunks = len(all_chunks)
    for i in range(0, total_chunks, BATCH_SIZE):
        batch_chunks = all_chunks[i : i + BATCH_SIZE]
        batch_meta = all_metadatas[i : i + BATCH_SIZE]
        batch_ids = all_ids[i : i + BATCH_SIZE]
        add_documents(collection, batch_chunks, batch_meta, batch_ids)

    print(f"[SUCCESS] Ingested {total_chunks} chunks from PDF '{filename}' into '{collection_name}'.\n")
    return total_chunks


def ingest_text_file(
    file_path: str,
    collection_name: str = "knowledge_base",
    max_chunk_size: int = 350,
    overlap: int = 50,
):
    """
    Ingests plain text files using paragraph-aware chunking.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found at: {file_path}")

    filename = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Strip Project Gutenberg boilerplate if present
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start_idx = full_text.find(start_marker)
    end_idx = full_text.find(end_marker)
    if start_idx != -1:
        start_idx = full_text.find("\n", start_idx) + 1
        if end_idx != -1:
            full_text = full_text[start_idx:end_idx].strip()
        else:
            full_text = full_text[start_idx:].strip()
        # Also strip Table of Contents if present
        toc_marker = " CONTENTS\n"
        toc_pos = full_text.find(toc_marker)
        if toc_pos != -1:
            story_pos = full_text.find("Letter 1\n\n", toc_pos)
            if story_pos != -1:
                full_text = full_text[:toc_pos] + full_text[story_pos:]
                print(f"[INFO] Stripped Table of Contents from '{filename}'.")

    collection = get_collection(collection_name)
    chunks = chunk_by_paragraph(full_text, max_chunk_size=max_chunk_size, overlap=overlap)

    metadatas = [{"source": filename, "page": 1, "file_type": "txt"} for _ in chunks]
    ids = [f"{filename}_c{i}" for i in range(len(chunks))]

    # Batch insertion
    total_chunks = len(chunks)
    for i in range(0, total_chunks, BATCH_SIZE):
        add_documents(
            collection,
            chunks[i : i + BATCH_SIZE],
            metadatas[i : i + BATCH_SIZE],
            ids[i : i + BATCH_SIZE],
        )

    print(f"[SUCCESS] Ingested {total_chunks} chunks from TXT '{filename}' into '{collection_name}'.\n")
    return total_chunks


if __name__ == "__main__":
    print("Ingestion engine initialized successfully.")
