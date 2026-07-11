"""LLM factory supporting OpenAI-compatible providers (OpenAI, xAI Grok, Groq, Together, etc.)."""
import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from app.config import settings

logger = logging.getLogger(__name__)

def get_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    streaming: bool = True
) -> BaseChatModel:
    """
    Create and return a configured ChatOpenAI instance.
    Works seamlessly with xAI Grok by setting LLM_BASE_URL=https://api.x.ai/v1
    """
    model = model or settings.llm_model
    temperature = temperature if temperature is not None else settings.llm_temperature
    max_tokens = max_tokens or settings.llm_max_tokens
    
    llm_kwargs = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "streaming": streaming,
        "api_key": settings.llm_api_key,
    }
    
    if settings.llm_base_url:
        llm_kwargs["base_url"] = settings.llm_base_url
        logger.info(f"Using custom LLM base URL: {settings.llm_base_url}")
    
    try:
        llm = ChatOpenAI(**llm_kwargs)
        logger.info(f"LLM initialized: {model} (temp={temperature}, max_tokens={max_tokens})")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise
