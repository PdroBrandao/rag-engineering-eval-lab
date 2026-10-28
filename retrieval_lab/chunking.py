from dataclasses import dataclass

import tiktoken

ENCODING_NAME = "cl100k_base"


@dataclass(frozen=True)
class TextChunk:
    page: int
    chunk_index: int
    content: str
    token_count: int


def validate_chunk_params(chunk_size: int, overlap: int) -> None:
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    if overlap < 0:
        raise ValueError("chunk_overlap must be zero or a positive integer.")
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")


def split_tokens(tokens: list[int], chunk_size: int, overlap: int) -> list[list[int]]:
    validate_chunk_params(chunk_size, overlap)
    if not tokens:
        return []
    chunks: list[list[int]] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokens[start:end])
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = 500,
    overlap: int = 50,
    encoding_name: str = ENCODING_NAME,
) -> list[TextChunk]:
    """Chunk each page on its own so every chunk keeps a single page number."""
    validate_chunk_params(chunk_size, overlap)
    encoding = tiktoken.get_encoding(encoding_name)
    chunks: list[TextChunk] = []
    chunk_index = 0
    for page_number, text in pages:
        if page_number < 1:
            raise ValueError("page numbers start at 1.")
        if not text or not text.strip():
            continue
        for piece in split_tokens(encoding.encode(text), chunk_size, overlap):
            content = encoding.decode(piece)
            if not content.strip():
                continue
            chunks.append(
                TextChunk(
                    page=page_number,
                    chunk_index=chunk_index,
                    content=content,
                    token_count=len(piece),
                )
            )
            chunk_index += 1
    return chunks
