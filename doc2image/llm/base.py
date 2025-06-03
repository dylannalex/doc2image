from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel


class BaseLLM(ABC):
    """
    Abstract base class for all LLMs.
    Each LLM must implement the `generate` method.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        top_p: float,
        top_k: int,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the LLM model.

        Args:
            model_name (str): The name of the model to load.
            temperature (float): The temperature setting for the model.
            top_p (float): The top-p setting for the model.
            top_k (int): The top-k setting for the model.
            api_key (Optional[str]): The API key is not used in this implementation.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.api_key = api_key

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        output_format: Optional[type[BaseModel]] = None,
    ) -> str | BaseModel:
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
    def pull_model(model_name: str, api_key: str) -> None:
        """
        Pull the model from the LLM provider.

        This method is a placeholder and should be implemented by subclasses.

        Args:
            model_name (str): The name of the model to pull.
            api_key (str): The API key for the model.
        """
        raise NotImplementedError("pull_model method is not implemented.")
