"""
Embedding generation for documents and queries.
Supports Ollama (local) and OpenAI embeddings.
Includes Redis caching for performance.
"""
from typing import List
import requests
from langchain_openai import OpenAIEmbeddings
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.cache import get_cache

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
        Uses Redis cache when available.
        
        Args:
            text: Text string to embed
        
        Returns:
            Embedding vector
        """
        # Try cache first
        cache = get_cache()
        cached = cache.get_embedding(text, self.model)
        if cached is not None:
            logger.debug(f"Embedding cache hit (Ollama)")
            return cached
        
        try:
            response = requests.post(
                self.endpoint,
                json={"model": self.model, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            embedding = result["embedding"]
            
            # Cache the result
            cache.set_embedding(text, embedding, self.model)
            return embedding
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
        return CachedOpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=settings.openai_embed_model
        )
    elif provider == "ollama":
        return OllamaEmbeddings()
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


class CachedOpenAIEmbeddings(OpenAIEmbeddings):
    """OpenAI embeddings with Redis caching."""
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding with caching."""
        cache = get_cache()
        cached = cache.get_embedding(text, self.model)
        if cached is not None:
            logger.debug(f"Embedding cache hit (OpenAI)")
            return cached
        
        # Generate and cache
        embedding = super().embed_query(text)
        cache.set_embedding(text, embedding, self.model)
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents with caching."""
        cache = get_cache()
        results = []
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached = cache.get_embedding(text, self.model)
            if cached is not None:
                results.append((i, cached))
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # Generate missing embeddings
        if texts_to_embed:
            new_embeddings = super().embed_documents(texts_to_embed)
            for idx, text, embedding in zip(indices_to_embed, texts_to_embed, new_embeddings):
                cache.set_embedding(text, embedding, self.model)
                results.append((idx, embedding))
        
        # Sort by original index and return
        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results]


# Default embeddings instance (OpenAI for better quality)
default_embeddings = get_embeddings("openai")
