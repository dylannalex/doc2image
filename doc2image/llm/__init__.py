from typing import Optional as _Optional

from .base import BaseLLM
from .ollama import OllamaLLM as _OllamaLLM


def create_llm(
    model_name: str,
    temperature: float,
    top_p: float,
    top_k: int,
    api_name: str,
    api_key: _Optional[str] = None,
) -> BaseLLM:
    """
    Create an instance of the LLM model based on the provided configuration.

    Args:
        model_name (str): The name of the model to load.
        temperature (float): The temperature setting for the model.
        top_p (float): The top-p setting for the model.
        top_k (int): The top-k setting for the model.
        api_name (str): The name of the API to use (e.g., "ollama", "openai").
        api_key (Optional[str]): The API key for the model (if required).

    Returns:
        BaseLLM: The loaded LLM chat model.
    """
    if "ollama" == api_name.lower():
        return _OllamaLLM(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    else:
        raise ValueError(
            f"'{api_name}' API is unsupported. Supported APIs are: ollama."
        )
