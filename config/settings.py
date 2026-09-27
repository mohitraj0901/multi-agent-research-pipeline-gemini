"""
Production-grade configuration management using Pydantic Settings.
"""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with validation."""

    # Google Gemini Configuration
    gemini_api_key: str = Field(
        ...,
        description="Google Gemini API key"
    )

    # Tavily Search API
    tavily_api_key: str = Field(
        ...,
        description="Tavily API key for web search"
    )

    # Model Configuration
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for model responses"
    )

    llm_max_tokens: int = Field(
        default=4000,
        ge=1,
        le=128000,
        description="Maximum tokens per request"
    )

    llm_top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter"
    )

    # Agent Configuration
    max_iterations: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum iterations for agent reasoning loops"
    )

    timeout_seconds: int = Field(
        default=300,
        ge=10,
        description="Timeout for agent execution in seconds"
    )

    # Rate Limiting
    requests_per_minute: int = Field(
        default=60,
        ge=1,
        description="Maximum API requests per minute"
    )

    tokens_per_minute: int = Field(
        default=150000,
        ge=1000,
        description="Maximum tokens per minute"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )

    log_file: str = Field(
        default="logs/agent_pipeline.log",
        description="Log file path"
    )

    # Optional: Redis Configuration
    redis_host: str = Field(
        default="localhost",
        description="Redis host for distributed checkpointing"
    )

    redis_port: int = Field(
        default=6379,
        description="Redis port"
    )

    redis_db: int = Field(
        default=0,
        description="Redis database number"
    )

    # Feature Flags
    enable_streaming: bool = Field(
        default=True,
        description="Enable streaming responses"
    )

    enable_human_in_loop: bool = Field(
        default=False,
        description="Enable human-in-the-loop intervention points"
    )

    enable_visualization: bool = Field(
        default=True,
        description="Enable graph visualization"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=()
    )

    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()

        if v_upper not in valid_levels:
            raise ValueError(
                f"log_level must be one of {valid_levels}"
            )

        return v_upper


@lru_cache()
def get_settings() -> Settings:
    """Get settings singleton instance."""
    return Settings()