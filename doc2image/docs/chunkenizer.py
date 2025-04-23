from langchain_text_splitters import RecursiveCharacterTextSplitter

from doc2image.docs.parser import DocumentParser, PdfParser, TxtParser


class Chunkenizer:
    """
    A class to chunk a document into smaller parts for processing.
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        parser: DocumentParser,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parser = parser

    def split(self, document_path: str) -> list[str]:
        """
        Split the document into chunks.
        Args:
            document_path (str): Path to the document.

        Returns:
            list[str]: List of document chunks.
        """
        # Parse the document using the provided parser
        document = self.parser.parse(document_path)

        # Split the document into chunks using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        return text_splitter.split_text(document)


def chunkenize_document(
    document_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Chunk a document into smaller parts for processing.

    Args:
        document_path (str): Path to the document.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.

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

    # Create the chunker instance and split the document
    return Chunkenizer(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        parser=parser,
    ).split(document_path)
