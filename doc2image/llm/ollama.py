from typing import Optional

from ollama import chat, pull
from pydantic import BaseModel

from .base import BaseLLM


class OllamaLLM(BaseLLM):
    """
    Ollama LLM model class.

    This class is responsible for interacting with the Ollama LLM API.
    """

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

    @staticmethod
    def pull_model(model_name: str, api_key: str) -> None:
        """
        Pull the model from the LLM provider.

        Args:
            model_name (str): The name of the model to pull.
            api_key (str): The API key for the model.

        Raises:
            ValueError: If the model could not be pulled from Ollama.
        """
        try:
            pull(model_name)
        except Exception:
            raise ValueError(f"Model '{model_name}' could not be pulled from Ollama.")
