from langgraph.graph import END, START, StateGraph

from state import CardioAgentState


def audio_agent(state: CardioAgentState) -> dict:
    """Prepare the audio-engine result for the following agents.

    This is a placeholder for the future audio-engine HTTP call. For now,
    the classification and confidence are supplied when the graph is invoked.
    """
    classification = state.get("classification", "unknown")
    confidence = state.get("confidence")
    confidence_text = "unknown" if confidence is None else f"{confidence:.0%}"

    return {
        "audio_context": (
            f"Heart-sound classification: {classification}. "
            f"Model confidence: {confidence_text}."
        )
    }


def knowledge_rag_agent(state: CardioAgentState) -> dict:
    """Create the query that will later be sent to Qdrant."""
    question = state.get("question", "").strip()

    if question:
        rag_query = question
    else:
        classification = state.get("classification", "unknown")
        rag_query = (
            "ESC clinical guidance after a "
            f"{classification} heart-sound screening result."
        )

    return {"rag_query": rag_query}


def reasoning_llm_agent(state: CardioAgentState) -> dict:
    """Mark the handoff point for the future local Ollama call."""
    return {"workflow_status": "ready_for_llm"}


def build_graph():
    """Build and compile the Audio -> RAG -> LLM workflow."""
    workflow = StateGraph(CardioAgentState)

    workflow.add_node("audio_agent", audio_agent)
    workflow.add_node("knowledge_rag_agent", knowledge_rag_agent)
    workflow.add_node("reasoning_llm_agent", reasoning_llm_agent)

    workflow.add_edge(START, "audio_agent")
    workflow.add_edge("audio_agent", "knowledge_rag_agent")
    workflow.add_edge("knowledge_rag_agent", "reasoning_llm_agent")
    workflow.add_edge("reasoning_llm_agent", END)

    return workflow.compile()
