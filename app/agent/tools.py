"""LangChain tools for the agent.
Includes Playwright web tools and placeholders for future file/dynamic tools.
"""
import logging
from typing import Optional
from langchain_core.tools import tool
from app.utils.playwright_browser import PlaywrightBrowser

logger = logging.getLogger(__name__)

# Global browser instance (reused across tool calls in one session for efficiency)
_browser_instance: Optional[PlaywrightBrowser] = None

def get_browser() -> PlaywrightBrowser:
    global _browser_instance
    if _browser_instance is None:
        _browser_instance = PlaywrightBrowser()
    return _browser_instance

@tool("web_search", return_direct=False)
def web_search(query: str) -> str:
    """
    Search the web for current information, news, facts, or research.
    Use this when the question requires up-to-date or external knowledge not in your training data.
    Input should be a clear, specific search query.
    """
    browser = get_browser()
    try:
        results = browser.web_search(query)
        logger.info(f"web_search executed for: {query[:60]}...")
        return results
    except Exception as e:
        logger.error(f"web_search tool failed: {e}")
        return f"Web search failed due to technical issue: {str(e)[:150]}. Please try rephrasing your query."

@tool("browse_page", return_direct=False)
def browse_page(url: str, task: Optional[str] = None) -> str:
    """
    Browse a specific webpage and extract/summarize its content.
    Use when you have a direct URL from previous search or user input and need detailed content.
    Args:
        url: The full URL to visit (must start with http/https).
        task: Optional specific extraction goal, e.g. 'Extract key statistics and conclusions'.
    """
    browser = get_browser()
    instructions = task or "Provide a comprehensive summary of the page content, including main arguments, data, and conclusions."
    try:
        content = browser.browse_page(url, instructions)
        return content
    except Exception as e:
        return f"Failed to browse the page: {str(e)[:120]}"

# You can add more tools here, e.g.:
# @tool("calculate")
# def calculate(expression: str) -> str: ...

# List of all tools for binding to LLM / graph
ALL_TOOLS = [web_search, browse_page]
