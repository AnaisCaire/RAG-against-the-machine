from pydantic import BaseModel, Field
import uuid
from typing import List


class MinimalSource(BaseModel):
    """ Minimal amount of information """
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without an answer or retrieved sources."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question paired with its ground-truth sources and answer."""
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """data set for rag questions"""
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """One question's retrieval result: the question plus its top-k sources."""
    question_id: str
    question_str: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search result augmented with a generated answer."""
    answer: str


class StudentSearchResults(BaseModel):
    """Full dataset search output: all per-question results
    and the k value used."""
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Search results with generated answers, one per question."""
    search_results: List[MinimalAnswer]  # type: ignore[assignment]
