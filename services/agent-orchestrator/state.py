from typing import TypedDict


class CardioAgentState(TypedDict, total=False):
    """Information passed from one LangGraph node to the next."""

    classification: str
    confidence: float
    question: str
    audio_context: str
    rag_query: str
    workflow_status: str
