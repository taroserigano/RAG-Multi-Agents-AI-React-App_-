"""
Redis caching module for the RAG application.
Provides caching for embeddings, retrieval results, and API responses.
"""
import json
import hashlib
from typing import Any, Optional, List, Callable
from functools import wraps
import pickle

import redis
from redis.exceptions import ConnectionError, TimeoutError

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class CacheManager:
    """
    Redis cache manager with fallback to no-cache mode.
    Gracefully handles Redis connection failures.
    """
    
    # Cache key prefixes
    PREFIX_EMBEDDING = "emb:"
    PREFIX_RETRIEVAL = "ret:"
    PREFIX_LLM = "llm:"
    PREFIX_API = "api:"
    PREFIX_IMAGE = "img:"
    
    # Default TTLs (in seconds)
    TTL_EMBEDDING = 60 * 60 * 24 * 7    # 7 days - embeddings rarely change
    TTL_RETRIEVAL = 60 * 60 * 1         # 1 hour - retrieval results
    TTL_LLM = 60 * 60 * 4               # 4 hours - LLM responses
    TTL_API = 60 * 5                     # 5 minutes - API list responses
    TTL_IMAGE_DESC = 60 * 60 * 24 * 30  # 30 days - image descriptions
    
    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._connected = False
        self._connection_attempted = False
    
    @property
    def client(self) -> Optional[redis.Redis]:
        """Lazy connection to Redis."""
        if not self._connection_attempted:
            self._connection_attempted = True
            try:
                self._client = redis.Redis(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    password=settings.redis_password or None,
                    db=settings.redis_db,
                    decode_responses=False,  # We'll handle encoding ourselves
                    socket_timeout=2,
                    socket_connect_timeout=2,
                    retry_on_timeout=True
                )
                # Test connection
                self._client.ping()
                self._connected = True
                logger.info(f"Redis connected: {settings.redis_host}:{settings.redis_port}")
            except (ConnectionError, TimeoutError, Exception) as e:
                logger.warning(f"Redis not available, running without cache: {e}")
                self._client = None
                self._connected = False
        return self._client
    
    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        if self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except:
            self._connected = False
            return False
    
    def _make_key(self, prefix: str, *args) -> str:
        """Generate a cache key from prefix and arguments."""
        # Create a deterministic hash from arguments
        key_data = json.dumps(args, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:16]
        return f"{prefix}{key_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.client:
            return None
        try:
            data = self.client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.debug(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL."""
        if not self.client:
            return False
        try:
            data = pickle.dumps(value)
            self.client.setex(key, ttl, data)
            return True
        except Exception as e:
            logger.debug(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        if not self.client:
            return False
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.debug(f"Cache delete error: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern."""
        if not self.client:
            return 0
        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.debug(f"Cache delete pattern error: {e}")
            return 0
    
    def clear_all(self) -> bool:
        """Clear all cache entries (use with caution)."""
        if not self.client:
            return False
        try:
            self.client.flushdb()
            logger.info("Cache cleared")
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    # ==================== Embedding Cache ====================
    
    def get_embedding(self, text: str, model: str = "default") -> Optional[List[float]]:
        """Get cached embedding for text."""
        key = self._make_key(self.PREFIX_EMBEDDING, text, model)
        return self.get(key)
    
    def set_embedding(self, text: str, embedding: List[float], model: str = "default") -> bool:
        """Cache an embedding."""
        key = self._make_key(self.PREFIX_EMBEDDING, text, model)
        return self.set(key, embedding, self.TTL_EMBEDDING)
    
    # ==================== Retrieval Cache ====================
    
    def get_retrieval(
        self, 
        query: str, 
        top_k: int, 
        doc_ids: Optional[List[str]] = None,
        options: Optional[dict] = None
    ) -> Optional[dict]:
        """Get cached retrieval results."""
        key = self._make_key(self.PREFIX_RETRIEVAL, query, top_k, doc_ids, options)
        return self.get(key)
    
    def set_retrieval(
        self, 
        query: str, 
        top_k: int, 
        result: dict,
        doc_ids: Optional[List[str]] = None,
        options: Optional[dict] = None
    ) -> bool:
        """Cache retrieval results."""
        key = self._make_key(self.PREFIX_RETRIEVAL, query, top_k, doc_ids, options)
        return self.set(key, result, self.TTL_RETRIEVAL)
    
    def invalidate_retrieval(self) -> int:
        """Invalidate all retrieval cache (call after document upload/delete)."""
        return self.delete_pattern(f"{self.PREFIX_RETRIEVAL}*")
    
    # ==================== LLM Response Cache ====================
    
    def get_llm_response(
        self, 
        question: str, 
        provider: str,
        model: str,
        context_hash: str
    ) -> Optional[str]:
        """Get cached LLM response."""
        key = self._make_key(self.PREFIX_LLM, question, provider, model, context_hash)
        return self.get(key)
    
    def set_llm_response(
        self, 
        question: str, 
        provider: str,
        model: str,
        context_hash: str,
        response: str
    ) -> bool:
        """Cache LLM response."""
        key = self._make_key(self.PREFIX_LLM, question, provider, model, context_hash)
        return self.set(key, response, self.TTL_LLM)
    
    # ==================== API Response Cache ====================
    
    def get_api_response(self, endpoint: str, params: dict = None) -> Optional[Any]:
        """Get cached API response."""
        key = self._make_key(self.PREFIX_API, endpoint, params)
        return self.get(key)
    
    def set_api_response(self, endpoint: str, result: Any, params: dict = None) -> bool:
        """Cache API response."""
        key = self._make_key(self.PREFIX_API, endpoint, params)
        return self.set(key, result, self.TTL_API)
    
    def invalidate_api(self, endpoint: str = None) -> int:
        """Invalidate API cache."""
        if endpoint:
            pattern = f"{self.PREFIX_API}{hashlib.md5(endpoint.encode()).hexdigest()[:8]}*"
        else:
            pattern = f"{self.PREFIX_API}*"
        return self.delete_pattern(pattern)
    
    # ==================== Image Description Cache ====================
    
    def get_image_description(self, image_hash: str, provider: str) -> Optional[str]:
        """Get cached image description."""
        key = self._make_key(self.PREFIX_IMAGE, image_hash, provider)
        return self.get(key)
    
    def set_image_description(self, image_hash: str, provider: str, description: str) -> bool:
        """Cache image description."""
        key = self._make_key(self.PREFIX_IMAGE, image_hash, provider)
        return self.set(key, description, self.TTL_IMAGE_DESC)
    
    # ==================== Stats ====================
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self.client:
            return {"connected": False, "message": "Redis not connected"}
        try:
            info = self.client.info("stats")
            memory = self.client.info("memory")
            keyspace = self.client.info("keyspace")
            
            return {
                "connected": True,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": round(
                    info.get("keyspace_hits", 0) / 
                    max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1) * 100, 
                    2
                ),
                "memory_used": memory.get("used_memory_human", "N/A"),
                "total_keys": sum(
                    db_info.get("keys", 0) 
                    for db_name, db_info in keyspace.items() 
                    if isinstance(db_info, dict)
                ),
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}
    
    def close(self) -> None:
        """Close Redis connection gracefully."""
        if self._client:
            try:
                self._client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")
            finally:
                self._client = None
                self._connected = False
                self._connection_attempted = False


# Global cache instance
_cache_manager: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


# ==================== Decorators ====================

def cached_embedding(func: Callable) -> Callable:
    """Decorator to cache embedding results."""
    @wraps(func)
    def wrapper(self, text: str, *args, **kwargs) -> List[float]:
        cache = get_cache()
        model = getattr(self, 'model', 'default')
        
        # Try cache first
        cached = cache.get_embedding(text, model)
        if cached is not None:
            logger.debug(f"Embedding cache hit for text[:50]={text[:50]}...")
            return cached
        
        # Generate and cache
        result = func(self, text, *args, **kwargs)
        cache.set_embedding(text, result, model)
        return result
    
    return wrapper


def cached_api(endpoint: str, ttl: int = None):
    """Decorator to cache API endpoint responses."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Try cache first
            cached = cache.get_api_response(endpoint, kwargs)
            if cached is not None:
                logger.debug(f"API cache hit for {endpoint}")
                return cached
            
            # Call function and cache
            result = await func(*args, **kwargs)
            cache.set_api_response(endpoint, result, kwargs)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Try cache first
            cached = cache.get_api_response(endpoint, kwargs)
            if cached is not None:
                logger.debug(f"API cache hit for {endpoint}")
                return cached
            
            # Call function and cache
            result = func(*args, **kwargs)
            cache.set_api_response(endpoint, result, kwargs)
            return result
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator
