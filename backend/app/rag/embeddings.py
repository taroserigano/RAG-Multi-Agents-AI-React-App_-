"""
Embedding generation for documents and queries.
Supports Ollama (local) and OpenAI embeddings.
"""
from typing import List
import requests
from langchain_openai import OpenAIEmbeddings
from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class OllamaEmbeddings:
    """
    Wrapper for Ollama embeddings API.
    Uses local Ollama server for embedding generation.
    """
    
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_embed_model
        self.endpoint = f"{self.base_url}/api/embeddings"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            text: Text string to embed
        
        Returns:
            Embedding vector
        """
        try:
            response = requests.post(
                self.endpoint,
                json={"model": self.model, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["embedding"]
        except Exception as e:
            logger.error(f"Error generating Ollama embedding: {e}")
            raise


def get_embeddings(provider: str = "ollama") -> object:
    """
    Factory function to get appropriate embedding model.
    
    Args:
        provider: "ollama" or "openai"
    
    Returns:
        Embedding model instance with embed_documents() and embed_query() methods
    """
    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        return OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=settings.openai_embed_model
        )
    elif provider == "ollama":
        return OllamaEmbeddings()
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


# Default embeddings instance (OpenAI for better quality)
default_embeddings = get_embeddings("openai")
