"""
Cache management API routes.
Provides endpoints for cache statistics and management.
"""
from fastapi import APIRouter, HTTPException, status

from app.core.cache import get_cache
from app.core.logging import get_logger

router = APIRouter(prefix="/api/cache", tags=["cache"])
logger = get_logger(__name__)


@router.get("/stats")
def get_cache_stats():
    """
    Get cache statistics.
    
    Returns:
        Cache stats including hit rate, memory usage, and key counts
    """
    cache = get_cache()
    stats = cache.get_stats()
    return stats


@router.get("/health")
def check_cache_health():
    """
    Check if Redis cache is available and healthy.
    
    Returns:
        Health status of the cache
    """
    cache = get_cache()
    connected = cache.is_connected
    
    return {
        "status": "healthy" if connected else "unavailable",
        "connected": connected,
        "message": "Redis cache is connected and working" if connected else "Running without cache (Redis not available)"
    }


@router.delete("/clear")
def clear_cache(
    cache_type: str = "all"
):
    """
    Clear cache entries.
    
    Args:
        cache_type: Type of cache to clear:
            - "all": Clear everything
            - "retrieval": Clear retrieval cache only
            - "embeddings": Clear embedding cache only
            - "api": Clear API response cache only
            - "llm": Clear LLM response cache only
    
    Returns:
        Number of keys deleted
    """
    cache = get_cache()
    
    if not cache.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis cache is not available"
        )
    
    deleted = 0
    
    if cache_type == "all":
        cache.clear_all()
        logger.info("All cache cleared")
        return {"message": "All cache cleared", "cache_type": "all"}
    
    elif cache_type == "retrieval":
        deleted = cache.invalidate_retrieval()
        logger.info(f"Retrieval cache cleared: {deleted} keys")
        
    elif cache_type == "embeddings":
        deleted = cache.delete_pattern(f"{cache.PREFIX_EMBEDDING}*")
        logger.info(f"Embedding cache cleared: {deleted} keys")
        
    elif cache_type == "api":
        deleted = cache.invalidate_api()
        logger.info(f"API cache cleared: {deleted} keys")
        
    elif cache_type == "llm":
        deleted = cache.delete_pattern(f"{cache.PREFIX_LLM}*")
        logger.info(f"LLM cache cleared: {deleted} keys")
        
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid cache_type. Must be one of: all, retrieval, embeddings, api, llm"
        )
    
    return {
        "message": f"{cache_type} cache cleared",
        "cache_type": cache_type,
        "keys_deleted": deleted
    }


@router.post("/warm")
def warm_cache():
    """
    Pre-warm the cache with common queries.
    This endpoint can be called after deployment to improve initial response times.
    
    Returns:
        Status of cache warming
    """
    cache = get_cache()
    
    if not cache.is_connected:
        return {
            "status": "skipped",
            "message": "Redis not available, cache warming skipped"
        }
    
    # For now, just return status. In production, you could:
    # - Pre-embed common queries
    # - Cache frequently accessed document lists
    # - Pre-compute popular retrieval results
    
    return {
        "status": "ready",
        "message": "Cache is ready. Consider implementing query pre-warming for production."
    }
