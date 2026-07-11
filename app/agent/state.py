"""Custom state definition for the LangGraph agent."""
from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """State passed through the LangGraph."""
    messages: Annotated[List[BaseMessage], add_messages]  # Chat history + new query
    user_id: str
    file_context: str                           # Combined summaries/excerpts from uploaded files
    file_metadata: List[Dict[str, Any]]         # Structured info about processed files
    tool_results: List[Dict[str, Any]]          # Results from tool calls (for transparency)
    final_response: str                         # The assistant's final answer
    needs_web_search: bool                      # Router decision
    metadata: Dict[str, Any]                    # Turn-level metadata (files used, tools used, etc.)
