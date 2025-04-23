from abc import ABC, abstractmethod

from langchain_community.document_loaders import PyPDFLoader


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


class TxtParser(DocumentParser):
    def __init__(self, separator: str = "\n"):
        self.separator = separator

    def parse(self, document_path: str) -> str:
        """
        Parse a text document (.txt and .md).

        Args:
            document_path (str): Path to the text document.

        Returns:
            str: Parsed text from the text document.
        """
        with open(document_path, "r", encoding="utf-8") as file:
            return self.separator.join(file.readlines())
