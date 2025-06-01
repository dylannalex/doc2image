import os
from time import time
from datetime import datetime
from typing import List

from .docs import chunkenize_document
from .prompt import Prompt
from .database import (
    LlmModel,
    LlmProvider,
    Document,
    DocumentSummarySession,
    ChunkSummary,
    ImagePromptsSession,
    ImagePrompt,
    Session,
    database_session_decorator,
)
from .pipeline import DocumentSummarizer, ImagePromptsGenerator
from .llm import create_llm, PROVIDER_TO_LLM, PROVIDERS, BaseLLM


@database_session_decorator
def setup_llm_providers(session: Session) -> None:
    """
    Set up the LLM providers by pulling their models.
    """
    for provider_name in PROVIDERS:
        exists: List[LlmProvider] = (
            session.query(LlmProvider).filter_by(name=provider_name).all()
        )
        if exists:
            continue
        provider = LlmProvider(name=provider_name, available=True)
        session.add(provider)
        session.flush()


setup_llm_providers()  # Always run this function to set up the LLM providers
# in case the database is empty.


def get_available_providers(session: Session) -> List[str]:
    """
    Get a list of available LLM providers.

    Returns:
        list[str]: A list of available LLM provider names.
    """
    available_providers: List[LlmProvider] = (
        session.query(LlmProvider).filter_by(available=True).all()
    )
    return [provider.name for provider in available_providers]


def get_summary_by_id(
    session: Session, summary_id: int
) -> DocumentSummarySession | None:
    result: DocumentSummarySession = (
        session.query(DocumentSummarySession).filter_by(id=summary_id).first()
    )

    return result


def add_llm_model(
    session: Session,
    model_name: str,
    provider_name: str,
    available: bool = True,
) -> LlmModel:
    """
    Add a new LLM model to the database.

    Args:
        model_name (str): The name of the LLM model.
        provider_name (str): The name of the LLM provider.
        available (bool): Availability status of the model.

    Returns:
        LlmModel: The created LLM model entry.
    """
    # Check if the provider exists in the database
    providers: List[LlmProvider] = (
        session.query(LlmProvider).filter_by(name=provider_name).all()
    )
    assert len(providers) == 1, "Multiple LLM providers found with the same name."
    llm_provider: LlmProvider = providers[0]
    assert llm_provider.available, f"LLM provider '{provider_name}' is unavailable."

    # Check if the model already exists in the database
    existing_models: List[LlmModel] = (
        session.query(LlmModel)
        .filter_by(name=model_name, provider_id=llm_provider.id)
        .all()
    )
    if existing_models:
        return existing_models[0]

    # Create a new LLM model entry
    llm_model = LlmModel(
        name=model_name,
        available=available,
        provider_id=llm_provider.id,
    )

    session.add(llm_model)
    session.flush()

    # Pull the model from the provider
    llm_cls: type[BaseLLM] = PROVIDER_TO_LLM[llm_provider.name]
    llm_cls.pull_model(model_name=model_name)

    return llm_model


def get_all_llm_models(session: Session) -> List[LlmModel]:
    """
    Get all LLM models from the database.

    Returns:
        list[LlmModel]: A list of all LLM models.
    """
    return session.query(LlmModel).all()


def summerize_document(
    session: Session,
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
) -> DocumentSummarySession:
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
        DocumentSummarySession: The document summary session created.
    """
    chunks = chunkenize_document(
        document_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        is_separator_regex=is_separator_regex,
        keep_separator=keep_separator,
        strip_whitespace=strip_whitespace,
    )

    doc_summerizer = DocumentSummarizer(
        llm=create_llm(
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
        summarize_chunk_prompt=Prompt(
            messages=summarize_chunk_prompt_messages,
            parameters=summarize_chunk_prompt_parameters,
        ),
        generate_document_summary_prompt=Prompt(
            messages=generate_document_summary_prompt_messages,
            parameters=generate_document_summary_prompt_parameters,
        ),
    )

    start_time = time()
    document_summary, chunk_summaries = doc_summerizer.run()
    session_time = time() - start_time
    generation_date = datetime.now()

    # Retrieve LLM provider from the database
    llm_providers: List[LlmProvider] = (
        session.query(LlmProvider).filter_by(name=llm_provider).all()
    )
    assert bool(
        llm_providers
    ), f"LLM provider {llm_provider} not found in the database."
    assert len(llm_providers) == 1, "Multiple LLM providers found with the same name."
    llm_provider_obj: LlmProvider = llm_providers[0]

    # Retrieve LLM model from the database
    llm_models: List[LlmModel] = (
        session.query(LlmModel)
        .filter_by(name=llm_model_name, provider_id=llm_provider_obj.id)
        .all()
    )
    assert bool(llm_models), f"LLM model {llm_model_name} not found in the database."
    assert len(llm_models) == 1, "Multiple LLM models found with the same name."
    llm_model: LlmModel = llm_models[0]

    # Retrieve or create the document entry in the database
    documents: List[Document] = (
        session.query(Document).filter_by(name=os.path.basename(document_path)).all()
    )
    if not documents:
        document = Document(
            name=os.path.basename(document_path), upload_date=generation_date
        )
        session.add(document)
        session.flush()
    else:
        assert len(documents) == 1, "Multiple documents found with the same name."
        document: Document = documents[0]

    # Create a new document summary session
    summary_session = DocumentSummarySession(
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
        chunk_summary = ChunkSummary(
            chunk_summary=chunk_summary_str,
            document_summary_session_id=summary_session.id,
        )
        session.add(chunk_summary)
    session.flush()

    return summary_session


def generate_image_prompts(
    session: Session,
    summary_session: DocumentSummarySession,
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
) -> ImagePromptsSession:
    """
    Generates image prompts based on the document summary.

    Args:
        summary_session (DocumentSummarySession): The document summary session to use.
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
        ImagePromptsSession: The image prompts session created.
    """
    image_prompts_generator = ImagePromptsGenerator(
        llm=create_llm(
            model_name=llm_model_name,
            temperature=llm_temperature,
            top_p=llm_top_p,
            top_k=llm_top_k,
            api_key=llm_api_key,
            provider=provider_name,
        ),
        document_summary=document_summary,
        total_prompts_to_generate=total_prompts_to_generate,
        generate_image_prompts_prompt=Prompt(
            messages=generate_image_prompts_prompt_messages,
            parameters=generate_image_prompts_prompt_parameters,
        ),
    )

    start = time()
    image_prompts = image_prompts_generator.run()
    session_time = time() - start
    generation_date = datetime.now()

    # Check if the provider exists in the database
    providers: List[LlmProvider] = (
        session.query(LlmProvider).filter_by(name=provider_name).all()
    )
    assert len(providers) == 1, "Multiple LLM providers found with the same name."
    llm_provider: LlmProvider = providers[0]
    assert llm_provider.available, f"LLM provider '{provider_name}' is unavailable."

    # Retrieve LLM model from the database
    llm_models: List[LlmModel] = (
        session.query(LlmModel)
        .filter_by(name=llm_model_name, provider_id=llm_provider.id)
        .all()
    )
    assert bool(llm_models), f"LLM model {llm_model_name} not found in the database."
    assert len(llm_models) == 1, "Multiple LLM models found with the same name."
    llm_model: LlmModel = llm_models[0]

    # Retrieve or create the document entry in the database
    documents: List[Document] = (
        session.query(Document).filter_by(name=os.path.basename(document_path)).all()
    )
    if not documents:
        document = Document(
            name=os.path.basename(document_path), upload_date=generation_date
        )
        session.add(document)
        session.flush()
    else:
        assert len(documents) == 1, "Multiple documents found with the same name."
        document: Document = documents[0]

    # Create a new image prompts session
    image_prompts_session = ImagePromptsSession(
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
    for prompt in image_prompts:
        image_prompt = ImagePrompt(
            image_prompts_session_id=image_prompts_session.id,
            prompt=prompt,
        )
        session.add(image_prompt)
    session.flush()

    return image_prompts_session


def get_all_document_summary_sessions(session: Session) -> List[DocumentSummarySession]:
    """
    Get all document summary sessions from the database.

    Returns:
        list[DocumentSummarySession]: A list of document summary sessions.
    """
    return session.query(DocumentSummarySession).all()
