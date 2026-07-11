"""Configuration management using Pydantic for type safety and validation."""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""
    
    # LLM Configuration (OpenAI-compatible)
    llm_api_key: str = Field(..., alias="LLM_API_KEY")
    llm_model: str = Field("gpt-4o-mini", alias="LLM_MODEL")
    llm_base_url: Optional[str] = Field(None, alias="LLM_BASE_URL")
    llm_temperature: float = Field(0.6, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(8192, alias="LLM_MAX_TOKENS")
    
    # Memory
    db_path: str = Field("data/memory.db", alias="DB_PATH")
    max_memory_states: int = Field(6, alias="MAX_MEMORY_STATES")
    
    # Playwright
    playwright_headless: bool = Field(True, alias="PLAYWRIGHT_HEADLESS")
    playwright_timeout_ms: int = Field(45000, alias="PLAYWRIGHT_TIMEOUT_MS")
    
    # App
    app_title: str = Field("🧠 LangGraph Playwright Agentic Chatbot", alias="APP_TITLE")
    default_user_id: str = Field("demo_user", alias="DEFAULT_USER_ID")
    enable_vision: bool = Field(True, alias="ENABLE_VISION")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True
        extra = "ignore"

# Global settings instance
settings = Settings()

# Ensure data directory exists
os.makedirs(os.path.dirname(settings.db_path) if os.path.dirname(settings.db_path) else "data", exist_ok=True)
