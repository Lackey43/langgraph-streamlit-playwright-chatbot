"""Playwright-based browser automation tool for web search and content extraction.
Designed to be used as LangChain @tool or directly in nodes.
Includes robust error handling, timeouts, and structured output.
"""
import logging
import asyncio
from typing import Optional, Dict, Any, List
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from app.config import settings

logger = logging.getLogger(__name__)

class PlaywrightBrowser:
    """Reusable headless browser for research tasks."""
    
    def __init__(self):
        self.headless = settings.playwright_headless
        self.timeout = settings.playwright_timeout_ms
        self._browser = None
        self._context = None
        self._page = None
    
    def _ensure_browser(self):
        """Lazy init browser (sync API for Streamlit compatibility)."""
        if self._page is None:
            p = sync_playwright().start()
            self._browser = p.chromium.launch(headless=self.headless)
            self._context = self._browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            self._page = self._context.new_page()
            self._page.set_default_timeout(self.timeout)
            logger.info("Playwright browser launched (headless=%s)", self.headless)
    
    def close(self):
        """Clean shutdown."""
        if self._page:
            self._page.close()
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        logger.info("Playwright browser closed.")
    
    def web_search(self, query: str, num_results: int = 8) -> str:
        """
        Perform a web search and return structured results.
        Uses DuckDuckGo HTML (no JS heavy) for reliability.
        """
        self._ensure_browser()
        try:
            # DuckDuckGo lite search (reliable, no heavy JS)
            search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
            self._page.goto(search_url, wait_until="domcontentloaded")
            
            # Extract results
            results = []
            result_elements = self._page.query_selector_all(".result")[:num_results]
            
            for elem in result_elements:
                title_elem = elem.query_selector(".result__title a")
                snippet_elem = elem.query_selector(".result__snippet")
                if title_elem:
                    title = title_elem.inner_text().strip()
                    url = title_elem.get_attribute("href") or ""
                    snippet = snippet_elem.inner_text().strip() if snippet_elem else ""
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet[:300]
                    })
            
            if not results:
                # Fallback: try to get any links
                links = self._page.query_selector_all("a.result__a")[:num_results]
                for link in links:
                    results.append({
                        "title": link.inner_text().strip(),
                        "url": link.get_attribute("href") or "",
                        "snippet": ""
                    })
            
            formatted = "\n\n".join(
                [f"{i+1}. **{r['title']}**\n   {r['snippet']}\n   URL: {r['url']}" 
                 for i, r in enumerate(results)]
            )
            return formatted or f"No search results found for '{query}'. Try rephrasing."
        
        except PlaywrightTimeout:
            return f"Search timed out for query: {query}. Please try again or use a simpler query."
        except Exception as e:
            logger.error(f"web_search error: {e}")
            return f"Error performing web search: {str(e)[:200]}. The browser tool encountered an issue."
    
    def browse_page(self, url: str, instructions: str = "Provide a detailed summary of the main content, key facts, headings, and any important data or conclusions.") -> str:
        """
        Navigate to a URL and extract content based on instructions.
        Returns LLM-friendly summary.
        """
        self._ensure_browser()
        try:
            self._page.goto(url, wait_until="domcontentloaded", timeout=self.timeout)
            # Wait a bit for dynamic content
            self._page.wait_for_timeout(1500)
            
            # Get main content
            title = self._page.title()
            # Try to get main readable text
            main_content = self._page.evaluate("""
                () => {
                    const article = document.querySelector('article') || document.querySelector('main') || document.body;
                    return article ? article.innerText.slice(0, 8000) : document.body.innerText.slice(0, 6000);
                }
            """)
            
            # Simple extraction prompt for LLM later, but here we return raw + title
            summary = f"**Page Title:** {title}\n\n**URL:** {url}\n\n**Main Content Excerpt:**\n{main_content[:6000]}..."
            
            return summary
        
        except PlaywrightTimeout:
            return f"Timeout loading page: {url}. The site may be slow, blocked, or require login."
        except Exception as e:
            logger.error(f"browse_page error for {url}: {e}")
            return f"Failed to browse {url}: {str(e)[:150]}"
    
    def extract_structured_data(self, url: str, data_type: str = "article") -> Dict[str, Any]:
        """Advanced extraction - can be extended for tables, lists, etc."""
        # Placeholder for future enhancement (tables, JSON-LD, etc.)
        content = self.browse_page(url)
        return {"url": url, "type": data_type, "extracted_content": content}
