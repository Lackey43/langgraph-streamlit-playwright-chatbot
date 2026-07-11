"""Node functions for the custom LangGraph StateGraph.
Includes router, file processor (pre-step), tool executor, LLM reasoner, and memory updater.
"""
import logging
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolNode

from app.agent.state import AgentState
from app.agent.tools import ALL_TOOLS, web_search, browse_page
from app.utils.llm import get_llm
from app.memory.db import MemoryStore
from app.config import settings

logger = logging.getLogger(__name__)

memory_store = MemoryStore()

# System prompt for the agent
SYSTEM_PROMPT = """You are an advanced, helpful AI research and document assistant powered by LangGraph.
You have access to real-time web browsing tools via Playwright and can analyze uploaded documents (PDFs, images, Office files, CSVs).

Core principles:
1. Use tools (web_search, browse_page) when the question requires current information, verification, or external knowledge.
2. When files are uploaded, deeply analyze their content and answer questions about them.
3. Be concise yet thorough. Cite sources from tools when possible.
4. If you used tools, briefly mention what you searched/browsed in your final response for transparency.
5. Respect the conversation history provided in the last 6 turns.

Always think step-by-step internally before responding.
"""

def prepare_state(state: AgentState) -> AgentState:
    """Entry node: inject system prompt and file context if present."""
    messages = state.get("messages", [])
    
    # Add system message if not already present at start
    if not messages or not isinstance(messages[0], SystemMessage):
        system_msg = SystemMessage(content=SYSTEM_PROMPT)
        if state.get("file_context"):
            system_msg = SystemMessage(
                content=SYSTEM_PROMPT + "\n\n" + state["file_context"]
            )
        messages = [system_msg] + messages
    
    state["messages"] = messages
    state.setdefault("tool_results", [])
    state.setdefault("file_metadata", [])
    state.setdefault("metadata", {})
    return state

def router_node(state: AgentState) -> Dict[str, Any]:
    """
    Simple router: decide if web tools are likely needed.
    In production, this could be an LLM classifier.
    For demo, we use a lightweight heuristic + let the LLM decide via tool calling.
    """
    last_message = state["messages"][-1].content if state["messages"] else ""
    needs_web = any(kw in last_message.lower() for kw in [
        "search", "web", "current", "latest", "news", "price", "stock", 
        "who is", "what is the", "today", "2025", "2026", "recent"
    ])
    
    # Also check if file_context is present and query is about files
    has_files = bool(state.get("file_context"))
    
    return {
        "needs_web_search": needs_web or (not has_files and len(last_message) > 40)
    }

def llm_reasoner_node(state: AgentState) -> Dict[str, Any]:
    """
    Core reasoning node: bind tools and invoke LLM.
    Uses tool-calling LLM. The graph will route to ToolNode if tool_calls present.
    """
    llm = get_llm(streaming=False)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    
    # Get recent messages (already trimmed by memory)
    messages = state["messages"]
    
    try:
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        error_msg = AIMessage(content=f"I encountered an error while reasoning: {str(e)[:200]}. Please try again.")
        return {"messages": [error_msg]}

def memory_update_node(state: AgentState) -> AgentState:
    """
    After LLM response, persist the human + AI turn to DB and trim.
    Also record tool usage in metadata.
    """
    user_id = state.get("user_id", settings.default_user_id)
    
    # Find the last human message and last AI response
    human_content = ""
    ai_content = ""
    tool_calls_made = []
    
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and not human_content:
            human_content = msg.content
        if isinstance(msg, AIMessage) and not ai_content:
            ai_content = msg.content or ""
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls_made = [tc.get("name", "unknown") for tc in msg.tool_calls]
    
    metadata = {
        "files_processed": len(state.get("file_metadata", [])),
        "tools_used": tool_calls_made,
        "web_search_performed": state.get("needs_web_search", False)
    }
    
    # Save human turn
    if human_content:
        memory_store.add_turn(user_id, "human", human_content, {"files": state.get("file_metadata", [])})
    
    # Save AI turn
    if ai_content:
        memory_store.add_turn(user_id, "ai", ai_content, metadata)
    
    state["metadata"] = metadata
    state["final_response"] = ai_content
    
    logger.info(f"Memory updated for user {user_id}. Tools used: {tool_calls_made}")
    return state

def format_response_node(state: AgentState) -> AgentState:
    """Final formatting / post-processing if needed."""
    # Could add citations, cleaning, etc. Here just pass through.
    return state
