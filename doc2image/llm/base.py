from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel


class BaseLLM(ABC):
    """
    Abstract base class for all LLMs.
    Each LLM must implement the `generate` method.
    """

    @abstractmethod
    def generate(self, messages: list[dict[str, str]], output_format: Optional[type[BaseModel]] = None) -> str | BaseModel:
        """
        Generate a response from the LLM based on the given messages.

        Args:
            messages (list[dict[str, str]]): The messages to send to the LLM.
            output_format (Optional[type[BaseModel]]): The expected output format.

        Returns:
            str | BaseModel: The generated response in the expected format or as a string.
        """
        pass
