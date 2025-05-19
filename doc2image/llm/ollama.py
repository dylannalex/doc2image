from typing import Optional

from ollama import chat
from pydantic import BaseModel

from .base import BaseLLM


class OllamaLLM(BaseLLM):
    """
    Ollama LLM model class.

    This class is responsible for interacting with the Ollama LLM API.
    """
    def __init__(
        self,
        model_name: str,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> None:
        """
        Initialize the Ollama LLM model.
        
        Args:
            model_name (str): The name of the model to load.
            temperature (float): The temperature setting for the model.
            top_p (float): The top-p setting for the model.
            top_k (int): The top-k setting for the model.
        """
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

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
        response = chat(
            model=self.model_name,
            messages=messages,
            format=output_format.model_json_schema() if output_format else None,
            options={
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
            },
        )

        self.last_response = response
        output = response.message.content

        if output_format:
            output = output_format.model_validate_json(response.message.content)

        return output
