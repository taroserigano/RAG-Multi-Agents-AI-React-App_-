"""
FastAPI application entry point.
Configures middleware, routes, and startup/shutdown events.
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.db.migrations import init_db
from app.api.routes_docs import router as docs_router
from app.api.routes_chat import router as chat_router
from app.api.routes_images import router as images_router

settings = get_settings()

# Setup logging
setup_logging(level="DEBUG" if settings.debug else "INFO")
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Runs on startup and shutdown.
    """
    # Startup
    logger.info("Starting Policy RAG API...")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Initialize database tables
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize database: {e}")
        logger.warning("Continuing without database (audit logging will be disabled)")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Policy RAG API...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="RAG-based Policy, Compliance, and Legal Document Q&A System",
    version="1.0.0",
    lifespan=lifespan
)

# ============================================================================
# Middleware
# ============================================================================

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing and logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all requests with timing information.
    """
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Log response
    logger.info(
        f"Response: {request.method} {request.url.path} "
        f"Status: {response.status_code} Duration: {duration:.3f}s"
    )
    
    return response


# Optional API key authentication middleware
@app.middleware("http")
async def check_api_key(request: Request, call_next):
    """
    Check API key if configured in settings.
    Only applied if settings.api_key is set.
    """
    if settings.api_key:
        # Skip health check
        if request.url.path == "/health":
            return await call_next(request)
        
        # Check for API key header
        api_key = request.headers.get("X-API-Key")
        if api_key != settings.api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Invalid or missing API key"}
            )
    
    return await call_next(request)


# ============================================================================
# Routes
# ============================================================================

# Include routers
app.include_router(docs_router)
app.include_router(chat_router)
app.include_router(images_router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/api/docs",
            "chat": "/api/chat",
            "openapi": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": settings.app_name
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An error occurred"
        }
    )


# ============================================================================
# Main (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
