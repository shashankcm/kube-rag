from typing import List
import logfire


def chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    """
    Chunk text into smaller pieces of a specified size.

    Args:
        text (str): The text to chunk.
        chunk_size (int): The size of each chunk. Defaults to 1500.

    Returns:
        List[str]: A list of text chunks.
    """

    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                chunks.append(current_chunk)
                current_chunk = paragraph + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        valid_chunks = [chunk for chunk in chunks if chunk.strip()]
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks.")
        return valid_chunks
