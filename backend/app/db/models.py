"""
SQLAlchemy ORM models for PostgreSQL database.
Stores document metadata and chat audit logs.
"""
from sqlalchemy import Column, String, DateTime, Text, JSON, Integer, Boolean
from sqlalchemy.sql import func
import uuid

from app.db.session import Base


class Document(Base):
    """
    Document metadata table.
    Stores information about uploaded documents.
    The actual vectors are stored in Pinecone.
    """
    __tablename__ = "documents"
    
    # UUID as string primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Original filename
    filename = Column(String, nullable=False, index=True)
    
    # MIME type (application/pdf, text/plain, etc.)
    content_type = Column(String, nullable=False)
    
    # Preview text (first 500 chars) for UI display
    preview_text = Column(Text, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename})>"


class ImageDocument(Base):
    """
    Image document metadata table.
    Stores information about uploaded images for multimodal RAG.
    CLIP embeddings are stored locally as JSON (512 dimensions).
    Note: Pinecone can't be used due to dimension mismatch (CLIP=512, text=3072).
    """
    __tablename__ = "image_documents"
    
    # UUID as string primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Original filename
    filename = Column(String, nullable=False, index=True)
    
    # MIME type (image/png, image/jpeg, etc.)
    content_type = Column(String, nullable=False)
    
    # Image dimensions
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    
    # File size in bytes
    file_size = Column(Integer, nullable=True)
    
    # MD5 hash for deduplication
    content_hash = Column(String, nullable=True, index=True)
    
    # AI-generated description (for search)
    description = Column(Text, nullable=True)
    
    # Extracted text from image (OCR)
    extracted_text = Column(Text, nullable=True)
    
    # Thumbnail as base64 (for quick preview)
    thumbnail_base64 = Column(Text, nullable=True)
    
    # Associated document ID (if image is from a PDF)
    source_document_id = Column(String, nullable=True, index=True)
    
    # Page number (if from PDF)
    source_page_number = Column(Integer, nullable=True)
    
    # CLIP embedding stored as JSON (512 dimensions)
    # Used for local similarity search since Pinecone has different dimensions
    clip_embedding = Column(JSON, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<ImageDocument(id={self.id}, filename={self.filename})>"


class ChatAudit(Base):
    """
    Chat audit log table.
    Records every question/answer pair for compliance and analysis.
    """
    __tablename__ = "chat_audits"
    
    # UUID as string primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # User identifier (can be session ID or user ID)
    user_id = Column(String, nullable=False, index=True)
    
    # LLM provider used (ollama, openai, anthropic)
    provider = Column(String, nullable=False)
    
    # Specific model name
    model = Column(String, nullable=False)
    
    # User question
    question = Column(Text, nullable=False)
    
    # AI-generated answer
    answer = Column(Text, nullable=False)
    
    # Array of cited document IDs (stored as JSON for SQLite compatibility)
    cited_doc_ids = Column(JSON, nullable=True)
    
    # Array of cited image IDs (for multimodal)
    cited_image_ids = Column(JSON, nullable=True)
    
    # Whether the query included an image
    has_image_query = Column(Boolean, default=False)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<ChatAudit(id={self.id}, user_id={self.user_id}, provider={self.provider})>"


class ComplianceReport(Base):
    """
    Compliance check report table.
    Stores compliance analysis results combining documents and images.
    """
    __tablename__ = "compliance_reports"
    
    # Report ID (e.g., CR-abc123)
    id = Column(String, primary_key=True)
    
    # User who requested the check
    user_id = Column(String, nullable=False, index=True)
    
    # Report title
    title = Column(String, nullable=False)
    
    # Original compliance query
    query = Column(Text, nullable=False)
    
    # Overall compliance status
    overall_status = Column(String, nullable=False)  # compliant, non_compliant, partial, needs_review
    
    # Executive summary
    summary = Column(Text, nullable=True)
    
    # Detailed findings as JSON array
    findings_json = Column(JSON, nullable=True)
    
    # Document IDs that were analyzed
    document_ids = Column(JSON, nullable=True)
    
    # Image IDs that were analyzed
    image_ids = Column(JSON, nullable=True)
    
    # LLM provider used
    provider = Column(String, nullable=True)
    
    # Specific model name
    model = Column(String, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<ComplianceReport(id={self.id}, status={self.overall_status})>"

