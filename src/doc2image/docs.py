from abc import ABC, abstractmethod

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentParser(ABC):
    """
    Abstract base class for document parsers.
    """

    @abstractmethod
    def parse(self, document_path: str) -> str:
        """
        Parse the document as a string.

        Args:
            document_path (str): Path to the document.

        Returns:
            str: Parsed document as a string.
        """
        pass


class PdfParser(DocumentParser):
    def __init__(self, separator: str = "\n"):
        self.separator = separator

    def parse(self, document_path: str) -> str:
        """
        Parse a PDF document using PyPDFLoader.

        Args:
            document_path (str): Path to the PDF document.

        Returns:
            str: Parsed text from the PDF document.
        """
        loader = PyPDFLoader(document_path)
        document = loader.load()
        return self.separator.join([page.page_content for page in document])


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
