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

    @staticmethod
    def pull_model(model_name: str) -> None:
        """
        Pull the model from the LLM provider.

        This method is a placeholder and should be implemented by subclasses.

        Args:
            model_name (str): The name of the model to pull.
        """
        raise NotImplementedError("pull_model method is not implemented.")
