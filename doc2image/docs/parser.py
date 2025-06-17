from abc import ABC, abstractmethod

from pypdf import PdfReader
from docx import Document as DocxDocument


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
    """
    Parser for PDF documents using PyPDF.

    This class is responsible for extracting text from PDF documents.
    """

    def __init__(self, separator: str = "\n"):
        """
        Initialize the PDF parser.

        Args:
            separator (str): Separator to use when joining text from pages.
        """
        self.separator = separator

    def parse(self, document_path: str) -> str:
        """
        Parse text from a PDF document.

        Args:
            document_path (str): Path to the PDF document.

        Returns:
            str: Parsed text from the PDF document.
        """
        reader = PdfReader(document_path)
        pdf_text = ""
        for page in reader.pages:
            pdf_text += page.extract_text()
            pdf_text += "\n"

        return pdf_text


class DocxParser:
    """
    Parser for Word documents (.docx).

    This class is responsible for extracting text from Word documents using python-docx.
    """

    def __init__(self, separator: str = "\n"):
        """
        Initialize the Word document parser.
        Args:
            separator (str): Separator to use when joining paragraphs.
        """
        self.separator = separator

    def parse(self, document_path: str) -> str:
        """
        Parse a Word document (.docx).

        Args:
            document_path (str): Path to the Word document.

        Returns:
            str: Parsed text from the Word document.
        """
        doc = DocxDocument(document_path)
        return self.separator.join(paragraph.text for paragraph in doc.paragraphs)


class TxtParser(DocumentParser):
    """
    Parser for text documents (.txt, .md, etc.).
    """

    def __init__(self, separator: str = "\n"):
        """
        Initialize the text parser.

        Args:
            separator (str): Separator to use when joining lines.
        """
        self.separator = separator

    def parse(self, document_path: str) -> str:
        """
        Parse a text document (.txt, .md, etc.).

        Args:
            document_path (str): Path to the text document.

        Returns:
            str: Parsed text from the text document.
        """
        with open(document_path, "r", encoding="utf-8") as file:
            return self.separator.join(file.readlines())
