from typing import Union, Literal

from .parser import DocumentParser, PdfParser, TxtParser
from .text_splitter import TextSplitter


def chunkenize_document(
    document_path: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
    is_separator_regex: bool,
    keep_separator: Union[bool, Literal["start", "end"]],
    strip_whitespace: bool,
) -> list[str]:
    """
    Splits a document into smaller parts (chunks) for processing.

    Args:
        document_path (str): Path to the document.
        chunk_size (int): Maximum size of chunks to return
        chunk_overlap (int): Overlap in characters between chunks
        separators (list[str]): List of separators to use for splitting
        is_separator_regex (bool): Whether the separators are regex patterns
        keep_separator (Union[bool, Literal["start", "end"]]): Whether to keep the
            separator and where to place it in each corresponding chunk (True='start')
        strip_whitespace (bool): If `True`, strips whitespace from the start and end of
            every document

    Returns:
        list[str]: List of document chunks.
    """
    # Determine the file extension
    _, extension = document_path.split(".")

    # Load the appropriate parser based on the file extension
    parsers = {
        "pdf": PdfParser,
        "txt": TxtParser,
        "md": TxtParser,
    }

    if extension not in parsers:
        raise ValueError(
            f"Unsupported file extension: {extension}. Supported: {list(parsers.keys())}"
        )

    parser: DocumentParser = parsers[extension]()
    document_text = parser.parse(document_path)

    # Create the chunker instance and split the document
    return TextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        is_separator_regex=is_separator_regex,
        keep_separator=keep_separator,
        strip_whitespace=strip_whitespace,
    ).split_text(text=document_text)
