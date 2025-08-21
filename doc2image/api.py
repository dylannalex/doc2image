import os
from time import time
from datetime import datetime
from typing import List

from . import database
from . import docs
from . import prompt
from . import pipeline
from . import llm


@database.database_session_decorator
def setup_llm_providers(session: database.Session) -> None:
    """
    Set up the LLM providers by pulling their models.
    """
    for provider_name in llm.PROVIDERS:
        exists: List[database.LlmProvider] = (
            session.query(database.LlmProvider).filter_by(name=provider_name).all()
        )
        if exists:
            continue
        provider = database.LlmProvider(name=provider_name, available=True)
        session.add(provider)
        session.flush()


# Always run this function to set up the LLM providers
# in case the database is empty.
setup_llm_providers()


def get_llm_providers() -> list[str]:
    """
    Get a list of available LLM providers.

    Returns:
        list[str]: A list of available LLM provider names.
    """
    return llm.PROVIDERS


def get_available_doc_formats() -> list[str]:
    """
    Get a list of available document formats for upload.

    Returns:
        list[str]: A list of supported document formats.
    """
    return docs.AVAILABLE_FORMATS


def get_provider_api_key(session: database.Session, provider_name: str) -> str | None:
    """
    Get the API key for a specific LLM provider.

    Args:
        session (database.Session): The database session.
        provider_name (str): The name of the LLM provider.

    Returns:
        str | None: The API key for the provider, or None if not found.
    """
    provider: database.LlmProvider = (
        session.query(database.LlmProvider).filter_by(name=provider_name).first()
    )
    assert (
        provider is not None
    ), f"LLM provider '{provider_name}' not found in the database."

    return provider.api_key


def update_provider_api_key(
    session: database.Session, provider_name: str, api_key: str
) -> None:
    """
    Update the API key for a specific LLM provider.

    Args:
        session (database.Session): The database session.
        provider_name (str): The name of the LLM provider.
        api_key (str): The new API key for the provider.
    """
    provider: database.LlmProvider = (
        session.query(database.LlmProvider).filter_by(name=provider_name).first()
    )
    assert (
        provider is not None
    ), f"LLM provider '{provider_name}' not found in the database."

    provider.api_key = api_key
    session.flush()


def get_available_providers(session: database.Session) -> List[str]:
    """
    Get a list of available LLM providers.

    Returns:
        list[str]: A list of available LLM provider names.
    """
    available_providers: List[database.LlmProvider] = (
        session.query(database.LlmProvider).filter_by(available=True).all()
    )
    return [provider.name for provider in available_providers]


def get_summary_by_id(
    session: database.Session, summary_id: int
) -> database.DocumentSummarySession | None:
    result: database.DocumentSummarySession = (
        session.query(database.DocumentSummarySession).filter_by(id=summary_id).first()
    )

    return result


def add_llm_model(
    session: database.Session,
    model_name: str,
    provider_name: str,
    api_key: str,
    available: bool = True,
) -> database.LlmModel:
    """
    Add a new LLM model to the database.

    Args:
        model_name (str): The name of the LLM model.
        provider_name (str): The name of the LLM provider.
        available (bool): Availability status of the model.
        api_key (str): The API key for the model.

    Returns:
        database.LlmModel: The created LLM model entry.
    """
    # Check if the provider exists in the database
    providers: List[database.LlmProvider] = (
        session.query(database.LlmProvider).filter_by(name=provider_name).all()
    )
    assert (
        len(providers) > 0
    ), f"LLM provider '{provider_name}' not found in the database."
    assert len(providers) == 1, "Multiple LLM providers found with the same name."
    llm_provider: database.LlmProvider = providers[0]
    assert llm_provider.available, f"LLM provider '{provider_name}' is unavailable."

    # Pull the model from the provider
    # This will raise an error if the model does not exist
    llm_cls: type[llm.BaseLLM] = llm.PROVIDER_TO_LLM[llm_provider.name]
    llm_cls.pull_model(model_name=model_name, api_key=api_key)

    # Check if the model already exists in the database
    existing_models: List[database.LlmModel] = (
        session.query(database.LlmModel)
        .filter_by(name=model_name, provider_id=llm_provider.id)
        .all()
    )
    if existing_models:
        return existing_models[0]

    # Create a new LLM model entry
    llm_model = database.LlmModel(
        name=model_name,
        available=available,
        provider_id=llm_provider.id,
    )

    session.add(llm_model)

    # Flush the session to ensure the model is saved
    session.flush()

    return llm_model


def get_all_llm_models(session: database.Session) -> List[database.LlmModel]:
    """
    Get all LLM models from the database.

    Returns:
        list[database.LlmModel]: A list of all LLM models.
    """
    return session.query(database.LlmModel).all()


def summerize_document(
    session: database.Session,
    document_path: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
    is_separator_regex: bool,
    keep_separator: str,
    strip_whitespace: bool,
    llm_api_key: str,
    llm_model_name: str,
    llm_temperature: float,
    llm_top_p: float,
    llm_top_k: int,
    llm_provider: str,
    max_document_summary_size: int,
    max_chunk_summary_size: int,
    summarize_chunk_prompt_messages: list[dict[str, str]],
    summarize_chunk_prompt_parameters: list[str],
    generate_document_summary_prompt_messages: list[dict[str, str]],
    generate_document_summary_prompt_parameters: list[str],
) -> database.DocumentSummarySession:
    """
    Summarizes a document by splitting it into chunks and generating summaries.

    Args:
        document_path (str): Path to the document.
        chunk_size (int): Maximum size of chunks to return.
        chunk_overlap (int): Overlap in characters between chunks.
        separators (list[str]): List of separators to use for splitting.
        is_separator_regex (bool): Whether the separators are regex patterns.
        keep_separator (str): Whether to keep the separator and where to place it in each corresponding chunk.
        strip_whitespace (bool): If `True`, strips whitespace from the start and end of every document.
        llm_api_key (str): The API key for the model (if required).
        llm_model_name (str): The name of the model to load.
        llm_temperature (float): The temperature setting for the model.
        llm_top_p (float): The top-p setting for the model.
        llm_top_k (int): The top-k setting for the model.
        llm_provider (str): The name of the API to use (e.g., "ollama", "openai").
        max_document_summary_size (int): Maximum size of the document summary.
        max_chunk_summary_size (int): Maximum size of each chunk summary.
        summarize_chunk_prompt_messages (list[dict[str, str]]): Messages for summarizing each chunk.
        summarize_chunk_prompt_parameters (list[str]): A list of parameter names to be used in the prompt.
        generate_document_summary_prompt_messages (list[dict[str, str]]): Messages for generating the document summary.
        generate_document_summary_prompt_parameters (list[str]): A list of parameter names to be used in the prompt.

    Returns:
        database.DocumentSummarySession: The document summary session created.
    """
    chunks = docs.chunkenize_document(
        document_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        is_separator_regex=is_separator_regex,
        keep_separator=keep_separator,
        strip_whitespace=strip_whitespace,
    )

    doc_summerizer = pipeline.DocumentSummarizer(
        llm=llm.create_llm(
            model_name=llm_model_name,
            provider=llm_provider,
            temperature=llm_temperature,
            top_p=llm_top_p,
            top_k=llm_top_k,
            api_key=llm_api_key,
        ),
        document_chunks=chunks,
        max_document_summary_size=max_document_summary_size,
        max_chunk_summary_size=max_chunk_summary_size,
        summarize_chunk_prompt=prompt.Prompt(
            messages=summarize_chunk_prompt_messages,
            parameters=summarize_chunk_prompt_parameters,
        ),
        generate_document_summary_prompt=prompt.Prompt(
            messages=generate_document_summary_prompt_messages,
            parameters=generate_document_summary_prompt_parameters,
        ),
    )

    start_time = time()
    document_summary, chunk_summaries = doc_summerizer.run()
    session_time = time() - start_time
    generation_date = datetime.now()

    # Retrieve LLM provider from the database
    llm_providers: List[database.LlmProvider] = (
        session.query(database.LlmProvider).filter_by(name=llm_provider).all()
    )
    assert bool(
        llm_providers
    ), f"LLM provider {llm_provider} not found in the database."
    assert len(llm_providers) == 1, "Multiple LLM providers found with the same name."
    llm_provider_obj: database.LlmProvider = llm_providers[0]

    # Retrieve LLM model from the database
    llm_models: List[database.LlmModel] = (
        session.query(database.LlmModel)
        .filter_by(name=llm_model_name, provider_id=llm_provider_obj.id)
        .all()
    )
    assert bool(llm_models), f"LLM model {llm_model_name} not found in the database."
    assert len(llm_models) == 1, "Multiple LLM models found with the same name."
    llm_model: database.LlmModel = llm_models[0]

    # Retrieve or create the document entry in the database
    documents: List[database.Document] = (
        session.query(database.Document)
        .filter_by(name=os.path.basename(document_path))
        .all()
    )
    if not documents:
        document = database.Document(
            name=os.path.basename(document_path), upload_date=generation_date
        )
        session.add(document)
        session.flush()
    else:
        assert len(documents) == 1, "Multiple documents found with the same name."
        document: database.Document = documents[0]

    # Create a new document summary session
    summary_session = database.DocumentSummarySession(
        document_id=document.id,
        document_summary=document_summary,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_chunk_summary_size=max_chunk_summary_size,
        max_document_summary_size=max_document_summary_size,
        llm_model_id=llm_model.id,
        llm_temperature=llm_temperature,
        llm_top_p=llm_top_p,
        llm_top_k=llm_top_k,
        generation_date=generation_date,
        session_time=session_time,
    )
    session.add(summary_session)
    session.flush()

    # Add chunk summaries to the database
    for chunk_summary_str in chunk_summaries:
        chunk_summary = database.ChunkSummary(
            chunk_summary=chunk_summary_str,
            document_summary_session_id=summary_session.id,
        )
        session.add(chunk_summary)
    session.flush()

    return summary_session


def generate_image_prompts(
    session: database.Session,
    summary_session: database.DocumentSummarySession,
    document_path: str,
    document_summary: str,
    total_prompts_to_generate: int,
    generate_image_prompts_prompt_messages: list[dict[str, str]],
    generate_image_prompts_prompt_parameters: list[str],
    llm_api_key: str,
    llm_model_name: str,
    llm_temperature: float,
    llm_top_p: float,
    llm_top_k: int,
    provider_name: str,
) -> database.ImagePromptsSession:
    """
    Generates image prompts based on the document summary.

    Args:
        summary_session (database.DocumentSummarySession): The document summary session to use.
        document_path (str): Path to the document.
        document_summary (str): The document summary to use for generating image prompts.
        total_prompts_to_generate (int): The total number of prompts to generate.
        generate_image_prompts_prompt_messages (list[dict[str, str]]): Messages for generating image prompts.
        generate_image_prompts_prompt_parameters (list[str]): A list of parameter names to be used in the prompt.
        llm_api_key (str): The API key for the model (if required).
        llm_model_name (str): The name of the model to load.
        llm_temperature (float): The temperature setting for the model.
        llm_top_p (float): The top-p setting for the model.
        llm_top_k (int): The top-k setting for the model.
        provider_name (str): The name of the API to use (e.g., "ollama", "openai").

    Returns:
        database.ImagePromptsSession: The image prompts session created.
    """
    image_prompts_generator = pipeline.ImagePromptsGenerator(
        llm=llm.create_llm(
            model_name=llm_model_name,
            temperature=llm_temperature,
            top_p=llm_top_p,
            top_k=llm_top_k,
            api_key=llm_api_key,
            provider=provider_name,
        ),
        document_summary=document_summary,
        total_prompts_to_generate=total_prompts_to_generate,
        generate_image_prompts_prompt=prompt.Prompt(
            messages=generate_image_prompts_prompt_messages,
            parameters=generate_image_prompts_prompt_parameters,
        ),
    )

    start = time()
    image_prompts = image_prompts_generator.run()
    session_time = time() - start
    generation_date = datetime.now()

    # Check if the provider exists in the database
    providers: List[database.LlmProvider] = (
        session.query(database.LlmProvider).filter_by(name=provider_name).all()
    )
    assert len(providers) == 1, "Multiple LLM providers found with the same name."
    llm_provider: database.LlmProvider = providers[0]
    assert llm_provider.available, f"LLM provider '{provider_name}' is unavailable."

    # Retrieve LLM model from the database
    llm_models: List[database.LlmModel] = (
        session.query(database.LlmModel)
        .filter_by(name=llm_model_name, provider_id=llm_provider.id)
        .all()
    )
    assert bool(llm_models), f"LLM model {llm_model_name} not found in the database."
    assert len(llm_models) == 1, "Multiple LLM models found with the same name."
    llm_model: database.LlmModel = llm_models[0]

    # Retrieve or create the document entry in the database
    documents: List[database.Document] = (
        session.query(database.Document)
        .filter_by(name=os.path.basename(document_path))
        .all()
    )
    if not documents:
        document = database.Document(
            name=os.path.basename(document_path), upload_date=generation_date
        )
        session.add(document)
        session.flush()
    else:
        assert len(documents) == 1, "Multiple documents found with the same name."
        document: database.Document = documents[0]

    # Create a new image prompts session
    image_prompts_session = database.ImagePromptsSession(
        document_summary_id=summary_session.id,
        llm_model_id=llm_model.id,
        llm_temperature=llm_temperature,
        llm_top_p=llm_top_p,
        llm_top_k=llm_top_k,
        generation_date=generation_date,
        session_time=session_time,
    )

    session.add(image_prompts_session)
    session.flush()

    # Create image prompts and add them to the database
    for prompt_ in image_prompts:
        image_prompt = database.ImagePrompt(
            image_prompts_session_id=image_prompts_session.id,
            prompt=prompt_,
        )
        session.add(image_prompt)
    session.flush()

    return image_prompts_session


def get_all_document_summary_sessions(
    session: database.Session,
) -> List[database.DocumentSummarySession]:
    """
    Get all document summary sessions from the database.

    Returns:
        list[database.DocumentSummarySession]: A list of document summary sessions.
    """
    return session.query(database.DocumentSummarySession).all()
