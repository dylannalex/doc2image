from langchain.chat_models.base import BaseChatModel

try:
    from langchain_ollama.chat_models import ChatOllama
except ImportError:
    print("[WARNING] Install the required package: `pip install langchain-ollama` to use the Ollama models.")

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    print("[WARNING] Install the required package: `pip install langchain-openai` to use the OpenAI models.")


def load_llm_model(
    api_name: str,
    model_name: str,
    temperature: float,
    top_p: float,
    top_k: int,
) -> BaseChatModel:
    """
    Load the specified LLM model.float

    Args:
        model_name (str): The name of the model to load.
        temperature (float): The temperature setting for the model.
        top_p (float): The top-p setting for the model.
        top_k (int): The top-k setting for the model.

    Returns:
        BaseChatModel: The loaded LLM chat model.
    """
    if "ollama" == api_name.lower():
        return ChatOllama(
            model=model_name, temperature=temperature, top_p=top_p, top_k=top_k
        )
    elif "openai" == api_name.lower():
        return ChatOpenAI(
            model=model_name, temperature=temperature, top_p=top_p, top_k=top_k
        )
    else:
        raise ValueError(
            f"'{api_name}' API is unsupported. Supported APIs are: ollama, openai."
        )
