"""LangGraph StateGraph definition and compiler.
Wires nodes with conditional edges for a flexible agentic flow.
"""
import logging
from typing import Literal
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from app.agent.state import AgentState
from app.agent.nodes import (
    prepare_state,
    router_node,
    llm_reasoner_node,
    memory_update_node,
    format_response_node,
)
from app.agent.tools import ALL_TOOLS
from app.config import settings

logger = logging.getLogger(__name__)

def should_use_tools(state: AgentState) -> Literal["tools", "llm"]:
    """
    Conditional edge after router.
    If router or LLM decided tools are needed, go to ToolNode.
    """
    # Simple: if last AI message has tool_calls, go to tools
    last_msg = state["messages"][-1] if state.get("messages") else None
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    
    # Fallback to router decision
    if state.get("needs_web_search"):
        return "tools"
    return "llm"

def build_graph():
    """Build and compile the agent graph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("prepare", prepare_state)
    workflow.add_node("router", router_node)
    workflow.add_node("llm", llm_reasoner_node)
    workflow.add_node("tools", ToolNode(ALL_TOOLS))
    workflow.add_node("memory_update", memory_update_node)
    workflow.add_node("format", format_response_node)
    
    # Define flow
    workflow.add_edge(START, "prepare")
    workflow.add_edge("prepare", "router")
    workflow.add_edge("router", "llm")          # Always go to LLM first (tool-calling style)
    
    # After LLM, decide: if it produced tool_calls -> tools, else memory_update
    workflow.add_conditional_edges(
        "llm",
        should_use_tools,
        {
            "tools": "tools",
            "llm": "memory_update",  # No tools needed
        }
    )
    
    # After tools executed, go back to LLM for final reasoning
    workflow.add_edge("tools", "llm")
    
    # After final LLM response (no more tool calls), update memory and format
    workflow.add_edge("memory_update", "format")
    workflow.add_edge("format", END)
    
    # Compile (no checkpointer here - we manage memory manually via DB for the "last 6 states" requirement)
    app = workflow.compile()
    logger.info("LangGraph agent compiled successfully with Playwright tools and memory management.")
    return app

# Singleton compiled graph
agent_graph = build_graph()
