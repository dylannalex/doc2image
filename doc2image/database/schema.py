import os
import typing
from datetime import datetime

from dotenv import load_dotenv
import sqlalchemy as sa
from sqlalchemy.orm import (
    declarative_base,
    mapped_column,
    relationship,
    Mapped,
    sessionmaker,
)

load_dotenv()
db = sa.create_engine(os.getenv("DATABASE_URL"), echo=False)
Session = sessionmaker(bind=db)
Base = declarative_base()


class LlmProvider(Base):
    """
    LLM provider model.

    Attributes:
        id (int): Unique identifier for the provider.
        name (str): Name of the provider.
        available (bool): Availability status of the provider.
        llm_models (list[LlmModel]): List of LLM models associated with the provider.
    """

    __tablename__ = "llm_provider"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(sa.String(100), unique=True)
    available: Mapped[bool] = mapped_column(sa.Boolean)
    api_key: Mapped[str] = mapped_column(sa.String(100), nullable=True)

    llm_models: Mapped[typing.List["LlmModel"]] = relationship(
        back_populates="provider"
    )


class LlmModel(Base):
    """
    LLM model model.

    Attributes:
        id (int): Unique identifier for the model.
        name (str): Name of the model.
        available (bool): Availability status of the model.
        provider_id (int): Identifier for the associated provider.
    """

    __tablename__ = "llm_model"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(sa.String(100), unique=True)
    available: Mapped[bool] = mapped_column(sa.Boolean)
    provider_id: Mapped[int] = mapped_column(sa.ForeignKey("llm_provider.id"))

    provider: Mapped["LlmProvider"] = relationship(back_populates="llm_models")
    document_summary_sessions: Mapped[list["DocumentSummarySession"]] = relationship(
        back_populates="llm_model"
    )
    image_prompt_sessions: Mapped[list["ImagePromptsSession"]] = relationship(
        back_populates="llm_model"
    )


class Document(Base):
    """
    Document model.

    Attributes:
        id (int): Unique identifier for the document.
        name (str): Name of the document.
        upload_date (datetime): Date when the document was uploaded.
    """

    __tablename__ = "document"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(sa.String(100), unique=True)
    upload_date: Mapped[datetime] = mapped_column(default=datetime)

    summaries: Mapped[typing.List["DocumentSummarySession"]] = relationship(
        back_populates="document"
    )


class DocumentSummarySession(Base):
    """
    Document summary session model.

    Attributes:
        id (int): Unique identifier for the session.
        document_id (int): Identifier for the associated document.
        document_summary (str): Summary of the document.
        max_chunk_summary_size (int): Maximum size of chunk summaries.
        max_document_summary_size (int): Maximum size of the document summary.
        llm_model_id (int): Identifier for the associated LLM model.
        llm_temperature (float): Temperature setting for the LLM model.
        llm_top_p (float): Top-p setting for the LLM model.
        llm_top_k (int): Top-k setting for the LLM model.
        generation_date (datetime): Date when the summary was generated.
        session_time (int): Duration of the session in seconds.
    """

    __tablename__ = "document_summary_session"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(sa.ForeignKey("document.id"))
    document_summary: Mapped[str] = mapped_column(sa.String(10_000))
    chunk_size: Mapped[int] = mapped_column()
    chunk_overlap: Mapped[int] = mapped_column()
    max_chunk_summary_size: Mapped[int] = mapped_column()
    max_document_summary_size: Mapped[int] = mapped_column()
    llm_model_id: Mapped[int] = mapped_column(sa.ForeignKey("llm_model.id"))
    llm_temperature: Mapped[float] = mapped_column()
    llm_top_p: Mapped[float] = mapped_column()
    llm_top_k: Mapped[int] = mapped_column()
    generation_date: Mapped[datetime] = mapped_column(default=datetime)
    session_time: Mapped[int] = mapped_column()

    document: Mapped["Document"] = relationship(back_populates="summaries")
    chunk_summaries: Mapped[typing.List["ChunkSummary"]] = relationship(
        back_populates="document_summary"
    )
    image_prompt_sessions: Mapped[typing.List["ImagePromptsSession"]] = relationship(
        back_populates="document_summary"
    )
    llm_model: Mapped["LlmModel"] = relationship(
        back_populates="document_summary_sessions"
    )


class ChunkSummary(Base):
    """
    Chunk summary model.

    Attributes:
        id (int): Unique identifier for the chunk summary.
        document_summary_session_id (int): Identifier for the associated document summary session.
        chunk_summary (str): Summary of the chunk.
    """

    __tablename__ = "chunk_summary"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_summary_session_id: Mapped[int] = mapped_column(
        sa.ForeignKey("document_summary_session.id")
    )
    chunk_summary: Mapped[str] = mapped_column(sa.String(10_000))

    document_summary: Mapped["DocumentSummarySession"] = relationship(
        back_populates="chunk_summaries"
    )


class ImagePromptsSession(Base):
    """
    Image prompts session model.

    Attributes:
        id (int): Unique identifier for the session.
        document_summary_id (int): Identifier for the associated document summary session.
        llm_model_id (int): Identifier for the associated LLM model.
        llm_temperature (float): Temperature setting for the LLM model.
        llm_top_p (float): Top-p setting for the LLM model.
        llm_top_k (int): Top-k setting for the LLM model.
        generation_date (datetime): Date when the prompts were generated.
        session_time (int): Duration of the session in seconds.
    """

    __tablename__ = "image_prompts_session"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_summary_id: Mapped[int] = mapped_column(
        sa.ForeignKey("document_summary_session.id")
    )
    llm_model_id: Mapped[int] = mapped_column(sa.ForeignKey("llm_model.id"))
    llm_temperature: Mapped[float] = mapped_column()
    llm_top_p: Mapped[float] = mapped_column()
    llm_top_k: Mapped[int] = mapped_column()
    generation_date: Mapped[datetime] = mapped_column(default=datetime)
    session_time: Mapped[int] = mapped_column()

    document_summary: Mapped["DocumentSummarySession"] = relationship(
        back_populates="image_prompt_sessions"
    )
    prompts: Mapped[typing.List["ImagePrompt"]] = relationship(
        back_populates="image_prompts_session"
    )
    llm_model: Mapped["LlmModel"] = relationship(back_populates="image_prompt_sessions")


class ImagePrompt(Base):
    """
    Image prompt model.

    Attributes:
        id (int): Unique identifier for the prompt.
        image_prompts_session_id (int): Identifier for the associated image prompts session.
        prompt (str): The image prompt text.
    """

    __tablename__ = "image_prompt"

    id: Mapped[int] = mapped_column(primary_key=True)
    image_prompts_session_id: Mapped[int] = mapped_column(
        sa.ForeignKey("image_prompts_session.id")
    )
    prompt: Mapped[str] = mapped_column(sa.String(10_000))

    image_prompts_session: Mapped["ImagePromptsSession"] = relationship(
        back_populates="prompts"
    )


# Create tables in the database if they don't exist
Base.metadata.create_all(db.engine, checkfirst=True)
