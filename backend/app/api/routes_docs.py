"""
Document management API routes.
Handles file upload and document listing.
"""
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import os
import tempfile
from pathlib import Path
import uuid
import re

from app.db.session import get_db
from app.db.models import Document
from app.schemas import DocumentResponse, UploadResponse, ErrorResponse
from app.rag.indexing import index_document, delete_document_from_index
from app.core.config import get_settings
from app.core.logging import get_logger

router = APIRouter(prefix="/api/docs", tags=["documents"])
settings = get_settings()
logger = get_logger(__name__)

# Maximum file size (15 MB by default)
MAX_FILE_SIZE = settings.max_file_size_mb * 1024 * 1024


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks.
    Removes directory separators and special characters.
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    """
    # Remove path separators
    filename = os.path.basename(filename)
    # Remove special characters except dots, hyphens, underscores
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    # Remove leading/trailing whitespace
    filename = filename.strip()
    return filename


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload and index a document (PDF or TXT).
    
    Process:
    1. Validate file type and size
    2. Save temporarily
    3. Extract text and chunk
    4. Generate embeddings
    5. Store vectors in Pinecone
    6. Save metadata in PostgreSQL
    
    Returns:
        Document ID and filename
    """
    try:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Allowed: {', '.join(settings.allowed_extensions)}"
            )
        
        # Sanitize filename
        safe_filename = sanitize_filename(file.filename)
        if not safe_filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid filename"
            )
        
        # Read file content
        content = await file.read()
        
        # Check file size
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
            )
        
        # Determine content type
        content_type = "application/pdf" if file_ext == ".pdf" else "text/plain"
        
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Index document (extract, chunk, embed, upsert to Pinecone)
            logger.info(f"Indexing document: {safe_filename}")
            result = index_document(
                doc_id=doc_id,
                filename=safe_filename,
                file_path=temp_path,
                content_type=content_type
            )
            
            # Save metadata to PostgreSQL (if available)
            if db is not None:
                db_document = Document(
                    id=doc_id,
                    filename=safe_filename,
                    content_type=content_type,
                    preview_text=result["preview_text"]
                )
                db.add(db_document)
                db.commit()
                db.refresh(db_document)
            else:
                logger.warning("Database not available, document metadata not saved")
            
            logger.info(f"Document {safe_filename} uploaded successfully with ID {doc_id}")
            
            return UploadResponse(
                doc_id=doc_id,
                filename=safe_filename
            )
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )


@router.get("", response_model=List[DocumentResponse])
def list_documents(db: Session = Depends(get_db)):
    """
    List all uploaded documents.
    
    Returns:
        List of document metadata (ID, filename, type, created date)
    """
    try:
        if db is None:
            # Database not available, return empty list
            logger.warning("Database not available, returning empty document list")
            return []
        documents = db.query(Document).order_by(Document.created_at.desc()).all()
        return documents
    
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        # Return empty list instead of error when DB is not available
        return []


@router.get("/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: str, db: Session = Depends(get_db)):
    """
    Get metadata for a specific document.
    
    Args:
        doc_id: Document UUID
    
    Returns:
        Document metadata
    """
    try:
        document = db.query(Document).filter(Document.id == doc_id).first()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        return document
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving document"
        )


@router.delete("/{doc_id}", status_code=status.HTTP_200_OK)
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    """
    Delete a document and its vectors from the system.
    
    Args:
        doc_id: Document UUID
    
    Returns:
        Success message
    """
    try:
        # Check if document exists
        document = db.query(Document).filter(Document.id == doc_id).first()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        filename = document.filename
        
        # Delete vectors from Pinecone
        try:
            delete_document_from_index(doc_id)
            logger.info(f"Deleted vectors for document {doc_id}")
        except Exception as e:
            logger.warning(f"Error deleting vectors from Pinecone: {e}")
            # Continue with DB deletion even if Pinecone fails
        
        # Delete from database
        db.delete(document)
        db.commit()
        
        logger.info(f"Document {filename} (ID: {doc_id}) deleted successfully")
        
        return {
            "message": f"Document '{filename}' deleted successfully",
            "doc_id": doc_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )


@router.post("/bulk-delete", status_code=status.HTTP_200_OK)
def bulk_delete_documents(
    doc_ids: List[str],
    db: Session = Depends(get_db)
):
    """
    Delete multiple documents and their vectors from the system.
    
    Args:
        doc_ids: List of document UUIDs to delete
    
    Returns:
        Summary of deleted and failed documents
    """
    if not doc_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No document IDs provided"
        )
    
    deleted = []
    failed = []
    
    for doc_id in doc_ids:
        try:
            # Check if document exists
            document = db.query(Document).filter(Document.id == doc_id).first()
            if not document:
                failed.append({"doc_id": doc_id, "reason": "Document not found"})
                continue
            
            filename = document.filename
            
            # Delete vectors from Pinecone
            try:
                delete_document_from_index(doc_id)
            except Exception as e:
                logger.warning(f"Error deleting vectors for {doc_id}: {e}")
                # Continue with DB deletion
            
            # Delete from database
            db.delete(document)
            deleted.append({"doc_id": doc_id, "filename": filename})
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            failed.append({"doc_id": doc_id, "reason": str(e)})
    
    # Commit all deletions at once
    db.commit()
    
    logger.info(f"Bulk delete: {len(deleted)} deleted, {len(failed)} failed")
    
    return {
        "message": f"Deleted {len(deleted)} documents, {len(failed)} failed",
        "deleted": deleted,
        "failed": failed
    }
