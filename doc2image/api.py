import os
from time import time
from datetime import datetime
from typing import List

from sqlalchemy.orm import joinedload, selectinload

from . import schemas
from . import database
from . import docs
from . import prompt
from . import pipeline
from . import llm


# --- Setup & Configuration Functions ---
@database.database_session_decorator
def setup_llm_providers(session: database.Session) -> None:
    """
    Set up the LLM providers by creating them if they don't exist.
    """
    for provider_name in llm.PROVIDERS:
        if (
            not session.query(database.LlmProvider)
            .filter_by(name=provider_name)
            .first()
        ):
            provider = database.LlmProvider(name=provider_name, available=True)
            session.add(provider)


# Always run this function to set up the LLM providers on startup.
setup_llm_providers()


def get_llm_providers() -> list[str]:
    """Get a list of available LLM providers."""
    return llm.PROVIDERS


def get_available_doc_formats() -> list[str]:
    """Get a list of available document formats for upload."""
    return docs.AVAILABLE_FORMATS


@database.database_session_decorator
def get_provider_api_key(session: database.Session, provider_name: str) -> str | None:
    """Get the API key for a specific LLM provider."""
    provider = session.query(database.LlmProvider).filter_by(name=provider_name).first()
    return provider.api_key if provider else None


@database.database_session_decorator
def update_provider_api_key(
    session: database.Session, provider_name: str, api_key: str
) -> None:
    """Update the API key for a specific LLM provider."""
    provider = session.query(database.LlmProvider).filter_by(name=provider_name).first()
    if provider:
        provider.api_key = api_key


@database.database_session_decorator
def get_all_llm_models(session: database.Session) -> list[str]:
    """Get all LLM models from the database as DTOs."""
    models = (
        session.query(database.LlmModel)
        .options(joinedload(database.LlmModel.provider))
        .all()
    )

    return [
        schemas.LlmModelProviderDTO(model_name=m.name, provider_name=m.provider.name)
        for m in models
    ]


@database.database_session_decorator
def add_llm_model(
    session: database.Session, model_name: str, provider_name: str, api_key: str
) -> None:
    """Add a new LLM model to the database."""
    provider = session.query(database.LlmProvider).filter_by(name=provider_name).first()
    if not provider:
        raise ValueError(f"LLM provider '{provider_name}' not found.")

    llm_cls: type[llm.BaseLLM] = llm.PROVIDER_TO_LLM[provider.name]
    llm_cls.pull_model(model_name=model_name, api_key=api_key)

    existing_model = (
        session.query(database.LlmModel)
        .filter_by(name=model_name, provider_id=provider.id)
        .first()
    )
    if existing_model:
        return

    llm_model = database.LlmModel(
        name=model_name, available=True, provider_id=provider.id
    )
    session.add(llm_model)

# --- Core Pipeline & Data Retrieval Functions ---
def _get_full_summary_query(session: database.Session):
    """Helper to build a fully eager-loaded query for a summary session."""
    return session.query(database.DocumentSummarySession).options(
        joinedload(database.DocumentSummarySession.document),
        joinedload(database.DocumentSummarySession.llm_model).joinedload(
            database.LlmModel.provider
        ),
        selectinload(database.DocumentSummarySession.chunk_summaries),
        selectinload(database.DocumentSummarySession.image_prompt_sessions).options(
            joinedload(database.ImagePromptsSession.llm_model).joinedload(
                database.LlmModel.provider
            ),
            selectinload(database.ImagePromptsSession.prompts),
        ),
    )


@database.database_session_decorator
def get_summary_by_id(
    session: database.Session, summary_id: int
) -> schemas.DocumentSummarySessionDTO | None:
    """Get a single summary session by its ID, with all related data."""
    result: database.DocumentSummarySession = (
        _get_full_summary_query(session)
        .filter(database.DocumentSummarySession.id == summary_id)
        .first()
    )

    return schemas.DocumentSummarySessionDTO.model_validate(result) if result else None


@database.database_session_decorator
def get_all_document_summary_sessions(
    session: database.Session,
) -> List[schemas.DocumentSummarySessionDTO]:
    """Get all document summary sessions with related data eager loaded."""
    results = (
        _get_full_summary_query(session)
        .order_by(database.DocumentSummarySession.generation_date.desc())
        .all()
    )
    return [schemas.DocumentSummarySessionDTO.model_validate(r) for r in results]


@database.database_session_decorator
def summerize_document(
    session: database.Session, **kwargs
) -> schemas.DocumentSummarySessionDTO:
    """
    Summarizes a document and returns the created session as a DTO.
    Accepts all arguments for summarization via kwargs.
    """
    document_path = kwargs["document_path"]
    chunks = docs.chunkenize_document(
        document_path,
        chunk_size=kwargs["chunk_size"],
        chunk_overlap=kwargs["chunk_overlap"],
        separators=kwargs["separators"],
        is_separator_regex=kwargs["is_separator_regex"],
        keep_separator=kwargs["keep_separator"],
        strip_whitespace=kwargs["strip_whitespace"],
    )

    doc_summarizer = pipeline.DocumentSummarizer(
        llm=llm.create_llm(
            model_name=kwargs["llm_model_name"],
            provider=kwargs["llm_provider"],
            temperature=kwargs["llm_temperature"],
            top_p=kwargs["llm_top_p"],
            top_k=kwargs["llm_top_k"],
            api_key=kwargs["llm_api_key"],
        ),
        document_chunks=chunks,
        max_document_summary_size=kwargs["max_document_summary_size"],
        max_chunk_summary_size=kwargs["max_chunk_summary_size"],
        summarize_chunk_prompt=prompt.Prompt(
            messages=kwargs["summarize_chunk_prompt_messages"],
            parameters=kwargs["summarize_chunk_prompt_parameters"],
        ),
        generate_document_summary_prompt=prompt.Prompt(
            messages=kwargs["generate_document_summary_prompt_messages"],
            parameters=kwargs["generate_document_summary_prompt_parameters"],
        ),
    )

    start_time = time()
    document_summary, chunk_summaries = doc_summarizer.run()

    provider = (
        session.query(database.LlmProvider).filter_by(name=kwargs["llm_provider"]).one()
    )
    llm_model = (
        session.query(database.LlmModel)
        .filter_by(name=kwargs["llm_model_name"], provider_id=provider.id)
        .one()
    )

    doc_name = kwargs.get("file_name")
    doc_name = doc_name if doc_name else os.path.basename(document_path)
    document = session.query(database.Document).filter_by(name=doc_name).first()
    if not document:
        document = database.Document(name=doc_name, upload_date=datetime.now())
        session.add(document)
        session.flush()

    summary_session = database.DocumentSummarySession(
        document_id=document.id,
        document_summary=document_summary,
        llm_model_id=llm_model.id,
        session_time=(time() - start_time),
        generation_date=datetime.now(),
        # Pass through all other relevant kwargs directly from the UI
        chunk_size=kwargs["chunk_size"],
        chunk_overlap=kwargs["chunk_overlap"],
        max_chunk_summary_size=kwargs["max_chunk_summary_size"],
        max_document_summary_size=kwargs["max_document_summary_size"],
        llm_temperature=kwargs["llm_temperature"],
        llm_top_p=kwargs["llm_top_p"],
        llm_top_k=kwargs["llm_top_k"],
    )
    session.add(summary_session)

    for summary_str in chunk_summaries:
        session.add(
            database.ChunkSummary(
                chunk_summary=summary_str, document_summary=summary_session
            )
        )

    session.flush()
    # Eagerly load the data for the DTO before the session closes
    loaded_session = (
        _get_full_summary_query(session)
        .filter(database.DocumentSummarySession.id == summary_session.id)
        .one()
    )
    return schemas.DocumentSummarySessionDTO.model_validate(loaded_session)


@database.database_session_decorator
def generate_image_prompts(
    session: database.Session, **kwargs
) -> schemas.ImagePromptsSessionDTO:
    """
    Generates image prompts for a given summary session and returns the new session as a DTO.
    Accepts all arguments for prompt generation via kwargs.
    """
    summary_session_id = kwargs.pop("summary_session_id")
    summary_session = (
        session.query(database.DocumentSummarySession)
        .filter_by(id=summary_session_id)
        .one()
    )

    image_prompts_generator = pipeline.ImagePromptsGenerator(
        llm=llm.create_llm(
            model_name=kwargs["llm_model_name"],
            provider=kwargs["provider_name"],
            temperature=kwargs["llm_temperature"],
            top_p=kwargs["llm_top_p"],
            top_k=kwargs["llm_top_k"],
            api_key=kwargs["llm_api_key"],
        ),
        document_summary=kwargs["document_summary"],
        total_prompts_to_generate=kwargs["total_prompts_to_generate"],
        generate_image_prompts_prompt=prompt.Prompt(
            messages=kwargs["generate_image_prompts_prompt_messages"],
            parameters=kwargs["generate_image_prompts_prompt_parameters"],
        ),
    )

    start = time()
    image_prompts = image_prompts_generator.run()

    provider = (
        session.query(database.LlmProvider)
        .filter_by(name=kwargs["provider_name"])
        .one()
    )
    llm_model = (
        session.query(database.LlmModel)
        .filter_by(name=kwargs["llm_model_name"], provider_id=provider.id)
        .one()
    )

    image_prompts_session = database.ImagePromptsSession(
        document_summary_id=summary_session.id,
        llm_model_id=llm_model.id,
        session_time=(time() - start),
        generation_date=datetime.now(),
        llm_temperature=kwargs["llm_temperature"],
        llm_top_p=kwargs["llm_top_p"],
        llm_top_k=kwargs["llm_top_k"],
    )
    session.add(image_prompts_session)

    for prompt_text in image_prompts:
        session.add(
            database.ImagePrompt(
                prompt=prompt_text, image_prompts_session=image_prompts_session
            )
        )

    session.flush()
    # Eagerly load the data for the DTO before the session closes
    loaded_session = (
        session.query(database.ImagePromptsSession)
        .options(
            joinedload(database.ImagePromptsSession.llm_model).joinedload(
                database.LlmModel.provider
            ),
            selectinload(database.ImagePromptsSession.prompts),
        )
        .filter(database.ImagePromptsSession.id == image_prompts_session.id)
        .one()
    )
    return schemas.ImagePromptsSessionDTO.model_validate(loaded_session)
