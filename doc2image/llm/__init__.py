from typing import Optional as _Optional

from .base import BaseLLM
from .openai import OpenAILLM
from .ollama import OllamaLLM, OLLAMA_AVAILABLE


PROVIDERS = ["OpenAI"]
PROVIDER_TO_LLM: dict[str, BaseLLM] = {"OpenAI": OpenAILLM}

if OLLAMA_AVAILABLE:
    PROVIDERS.append("Ollama")
    PROVIDER_TO_LLM["Ollama"] = OllamaLLM


def create_llm(
    model_name: str,
    temperature: float,
    top_p: float,
    top_k: int,
    provider: str,
    api_key: _Optional[str] = None,
) -> BaseLLM:
    """
    Create an instance of the LLM model based on the provided configuration.

    Args:
        model_name (str): The name of the model to load.
        temperature (float): The temperature setting for the model.
        top_p (float): The top-p setting for the model.
        top_k (int): The top-k setting for the model.
        provider (str): The name of the API to use (e.g., "ollama", "openai").
        api_key (Optional[str]): The API key for the model (if required).

    Returns:
        BaseLLM: The loaded LLM chat model.
    """
    if provider not in PROVIDERS:
        raise ValueError(
            f"'{provider}' API is unsupported. Supported APIs are: {', '.join(PROVIDERS)}."
        )

    return PROVIDER_TO_LLM[provider](
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        api_key=api_key,
    )
