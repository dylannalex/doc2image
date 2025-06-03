from typing import Optional

from openai import OpenAI
from pydantic import BaseModel

from .base import BaseLLM


class OpenAILLM(BaseLLM):
    """
    OpenAI LLM model class.

    This class is responsible for interacting with the OpenAI LLM API.
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
        client = OpenAI(api_key=self.api_key)
        args = {
            "model": self.model_name,
            "input": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            # "top_k": self.top_k,  # top_k is not supported in parse
        }
        if output_format is not None:
            args["text_format"] = output_format

        completion = client.responses.parse(**args)

        self.last_response = completion

        if output_format is None:
            output = completion.output_text
        else:
            output = completion.output_parsed
        return output

    @staticmethod
    def pull_model(model_name: str, api_key: str) -> None:
        """
        Does nothing for OpenAI, as models are not pulled like in Ollama.

        Args:
            model_name (str): The name of the model to pull.
            api_key (str): The API key for the model.
        """
        client = OpenAI(api_key=api_key)
        models = [m.id for m in client.models.list().data]
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found in OpenAI models.")
