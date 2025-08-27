from typing import List
from datetime import datetime

from pydantic import BaseModel


class OrmBase(BaseModel):
    class Config:
        from_attributes = True


class LlmProviderDTO(OrmBase):
    name: str
    available: bool


class LlmModelDTO(OrmBase):
    name: str
    available: bool
    provider: LlmProviderDTO


class DocumentDTO(OrmBase):
    name: str
    upload_date: datetime


class ChunkSummaryDTO(OrmBase):
    id: int
    chunk_summary: str


class ImagePromptDTO(OrmBase):
    id: int
    prompt: str


class ImagePromptsSessionDTO(OrmBase):
    id: int
    llm_temperature: float
    llm_top_p: float
    llm_top_k: int
    generation_date: datetime
    session_time: float
    llm_model: LlmModelDTO
    prompts: List[ImagePromptDTO]


class DocumentSummarySessionDTO(OrmBase):
    id: int

    # Document Info
    document: DocumentDTO
    document_summary: str

    # Chunking Info
    chunk_size: int
    chunk_overlap: int
    max_chunk_summary_size: int
    max_document_summary_size: int
    chunk_summaries: List[ChunkSummaryDTO]

    # LLM Info
    llm_model: LlmModelDTO
    llm_temperature: float
    llm_top_p: float
    llm_top_k: int

    # Session Info
    generation_date: datetime
    session_time: float
    image_prompt_sessions: List[ImagePromptsSessionDTO]


class LlmModelProviderDTO(BaseModel):
    model_name: str
    provider_name: str
