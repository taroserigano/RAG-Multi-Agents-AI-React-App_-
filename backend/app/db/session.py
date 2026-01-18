"""
Database session management.
Provides SQLAlchemy engine and session factory with dependency injection support.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator, Optional
import logging

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Create SQLAlchemy engine
# pool_pre_ping ensures connections are valid before use
try:
    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,
        echo=settings.debug  # Log SQL queries in debug mode
    )
    # Session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    DB_AVAILABLE = True
except Exception as e:
    logger.warning(f"Database not available: {e}")
    engine = None
    SessionLocal = None
    DB_AVAILABLE = False

# Base class for ORM models
Base = declarative_base()


def get_db() -> Generator[Optional[Session], None, None]:
    """
    Dependency injection function for FastAPI routes.
    Creates a new database session for each request.
    
    Usage in FastAPI:
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    
    Yields:
        Database session that is automatically closed after use, or None if DB not available
    """
    if not DB_AVAILABLE or SessionLocal is None:
        yield None
        return
        
    db = SessionLocal()
    try:
        yield db
    finally:
        if db:
            db.close()
