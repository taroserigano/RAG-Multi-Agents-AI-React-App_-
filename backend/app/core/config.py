"""
Configuration settings loaded from environment variables.
All settings are typed and validated using Pydantic.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings with validation.
    Loads from environment variables or .env file.
    """
    # Application
    app_name: str = "Policy RAG API"
    debug: bool = False
    
    # Database
    database_url: str
    
    # Pinecone Vector DB
    pinecone_api_key: str
    pinecone_index_name: str = "policy-rag"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    embed_dim: int = 768  # Default for nomic-embed-text
    
    # Ollama (local LLM)
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.1"
    ollama_embed_model: str = "nomic-embed-text"
    
    # OpenAI
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"
    
    # Anthropic
    anthropic_api_key: str = ""
    anthropic_chat_model: str = "claude-3-5-sonnet-latest"
    
    # Redis Cache
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0
    cache_enabled: bool = True
    
    # API Security (optional)
    api_key: str = ""  # If set, require X-API-Key header
    
    # File Upload
    max_file_size_mb: int = 15
    allowed_extensions: list[str] = [".pdf", ".txt"]
    
    # RAG Settings
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5
    
    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:8000"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Create cached settings instance.
    This ensures we only load .env once.
    """
    return Settings()
