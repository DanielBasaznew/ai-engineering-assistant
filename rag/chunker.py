# rag/chunker.py
import tiktoken


def get_encoder(model_name: str = "gpt-3.5-turbo"):
    """Returns the tiktoken encoding model for accurate token counting."""
    return tiktoken.encoding_for_model(model_name)


def chunk_by_tokens(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Fixed-size chunking using a sliding window of tokens.
    Uses tiktoken to ensure precise token counts.
    """
    encoder = get_encoder()
    tokens = encoder.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        # 1. Grab tokens from start up to start + chunk_size
        end = start + chunk_size
        chunk_tokens = tokens[start:end]

        # 2. Decode tokens back into plain text string
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append(chunk_text)

        # 3. Slide the window forward (chunk_size minus overlap)
        step = chunk_size - overlap
        if step <= 0:
            raise ValueError("chunk_size must be greater than overlap!")

        start += step

    return chunks


def chunk_by_paragraph(text: str, max_chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Paragraph-aware chunking. Groups full paragraphs together up to max_chunk_size tokens.
    Falls back to fixed-size token chunking for oversized paragraphs.
    """
    encoder = get_encoder()
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk_paragraphs = []
    current_tokens = 0

    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean:
            continue

        p_tokens = len(encoder.encode(p_clean))

        # Edge Case Safety Fallback: A single paragraph exceeds max token limit
        if p_tokens > max_chunk_size:
            # First, flush anything currently sitting in the buffer
            if current_chunk_paragraphs:
                chunks.append("\n\n".join(current_chunk_paragraphs))
                current_chunk_paragraphs = []
                current_tokens = 0

            # Use fixed token chunking on this giant paragraph
            sub_chunks = chunk_by_tokens(p_clean, chunk_size=max_chunk_size, overlap=overlap)
            chunks.extend(sub_chunks)
            continue

        # Standard case: append paragraph if it fits within token budget
        if current_tokens + p_tokens <= max_chunk_size:
            current_chunk_paragraphs.append(p_clean)
            current_tokens += p_tokens
        else:
            # Buffer full! Push current chunk and start a new one
            chunks.append("\n\n".join(current_chunk_paragraphs))

            # OVERLAP: carry the last paragraph into the new chunk if it fits
            last_paragraph = current_chunk_paragraphs[-1]
            last_p_tokens = len(encoder.encode(last_paragraph))
            if last_p_tokens + p_tokens <= max_chunk_size:
                current_chunk_paragraphs = [last_paragraph, p_clean]
                current_tokens = last_p_tokens + p_tokens
            else:
                current_chunk_paragraphs = [p_clean]
                current_tokens = p_tokens

    # Push any leftover paragraphs at the end
    if current_chunk_paragraphs:
        chunks.append("\n\n".join(current_chunk_paragraphs))

    return chunks


if __name__ == "__main__":
    sample_text = (
        "Paragraph 1: Machine learning is a field of study in artificial intelligence and in Machine Learning we have Deep Learning and on the top of Machine Learning Generative AI appears.\n\n"
        "Paragraph 2: Vector embeddings turn text into numerical arrays representing meaning.\n\n"
        "Paragraph 3: Retrieval-Augmented Generation bridges private knowledge with LLMs."
    )

    print("--- Testing Fixed Token Chunking (chunk_size=15, overlap=5) ---")
    token_chunks = chunk_by_tokens(sample_text, chunk_size=15, overlap=5)
    for i, c in enumerate(token_chunks):
        print(f"Chunk {i+1} [{len(get_encoder().encode(c))} tokens]: '{c}'")

    print("\n--- Testing Paragraph Chunking (max_chunk_size=25) ---")
    para_chunks = chunk_by_paragraph(sample_text, max_chunk_size=25, overlap=5)
    for i, c in enumerate(para_chunks):
        print(f"Chunk {i+1} [{len(get_encoder().encode(c))} tokens]: '{c}'")
