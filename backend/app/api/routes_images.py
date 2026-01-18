"""
Image management API routes for multimodal RAG.
Handles image upload, indexing, and retrieval.
"""
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Tuple
import os
import uuid
import io
import base64
import numpy as np
from PIL import Image

from app.db.session import get_db
from app.db.models import ImageDocument
from app.schemas import (
    ImageDocumentResponse, 
    ImageUploadResponse, 
    ImageSearchRequest,
    ImageSearchResponse,
    ErrorResponse
)
from app.rag.image_processing import (
    validate_image_file,
    get_image_info,
    image_to_base64,
    create_thumbnail,
    compute_image_hash,
    prepare_image_for_embedding,
    SUPPORTED_FORMATS
)
from app.rag.image_embeddings import get_clip_embeddings, embed_image, embed_image_query
from app.rag.vision_models import get_vision_model, generate_image_description_for_indexing
from app.core.config import get_settings
from app.core.logging import get_logger

router = APIRouter(prefix="/api/images", tags=["images"])
settings = get_settings()
logger = get_logger(__name__)

# Maximum file size for images (10 MB)
MAX_IMAGE_SIZE = 10 * 1024 * 1024

# Pinecone namespace for images (separate from text)
IMAGE_NAMESPACE = "images"


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search_images_local(
    query_embedding: List[float],
    images: List[ImageDocument],
    top_k: int = 5
) -> List[Tuple[ImageDocument, float]]:
    """
    Perform local cosine similarity search on image embeddings.
    
    Args:
        query_embedding: The query vector (512 dimensions)
        images: List of ImageDocument objects with clip_embedding
        top_k: Number of results to return
        
    Returns:
        List of (ImageDocument, score) tuples sorted by score descending
    """
    results = []
    for img in images:
        if img.clip_embedding:
            score = cosine_similarity(query_embedding, img.clip_embedding)
            results.append((img, score))
    
    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


@router.post("/upload", response_model=ImageUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_image(
    file: UploadFile = File(...),
    generate_description: bool = Form(True),
    vision_provider: str = Form("openai"),
    db: Session = Depends(get_db)
):
    """
    Upload and index an image for multimodal RAG.
    
    Process:
    1. Validate image file
    2. Generate CLIP embedding
    3. Optionally generate description with vision model
    4. Store vector in Pinecone
    5. Save metadata in PostgreSQL
    
    Args:
        file: Image file (JPG, PNG, GIF, WEBP, BMP)
        generate_description: Use vision model to generate searchable description
        vision_provider: Provider for description generation (openai/anthropic/ollama)
    
    Returns:
        Image ID and metadata
    """
    try:
        # Read file content
        content = await file.read()
        
        # Validate image
        is_valid, error_msg = validate_image_file(file.filename, content)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        
        # Check file size
        if len(content) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image too large. Maximum size: {MAX_IMAGE_SIZE // (1024*1024)}MB"
            )
        
        # Open image and get info
        image = Image.open(io.BytesIO(content))
        image_info = get_image_info(image)
        
        # Generate unique ID
        image_id = str(uuid.uuid4())
        
        # Compute hash for deduplication
        content_hash = compute_image_hash(content)
        
        # Check for duplicate
        existing = db.query(ImageDocument).filter(
            ImageDocument.content_hash == content_hash
        ).first()
        
        if existing:
            logger.info(f"Duplicate image detected: {existing.id}")
            return ImageUploadResponse(
                id=existing.id,  # Include 'id' for frontend consistency
                image_id=existing.id,
                filename=existing.filename,
                content_type=existing.content_type,
                width=existing.width,
                height=existing.height,
                description=existing.description,
                thumbnail_base64=existing.thumbnail_base64,
                created_at=existing.created_at,
                message="Duplicate image found, returning existing entry"
            )
        
        # Generate CLIP embedding
        logger.info(f"Generating CLIP embedding for image: {file.filename}")
        try:
            clip_embedding = embed_image(content)
        except Exception as e:
            logger.error(f"CLIP embedding failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate image embedding"
            )
        
        # Generate description with vision model (optional)
        description = None
        if generate_description:
            try:
                logger.info(f"Generating description with {vision_provider} vision model")
                description = generate_image_description_for_indexing(content, vision_provider)
            except Exception as e:
                logger.warning(f"Vision description failed: {e}. Continuing without description.")
        
        # Create thumbnail
        thumbnail = create_thumbnail(image)
        thumb_buffer = io.BytesIO()
        thumbnail.save(thumb_buffer, format='PNG')
        thumbnail_b64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')
        
        # Note: Skipping Pinecone storage due to dimension mismatch
        # CLIP embeddings are 512 dimensions, but Pinecone index is 3072 dimensions (text-embedding-3-large)
        # Embeddings are stored in PostgreSQL instead and searched locally
        logger.info(f"Storing image embedding in PostgreSQL (dimension mismatch with Pinecone)")
        
        # Save metadata and embedding to PostgreSQL
        db_image = ImageDocument(
            id=image_id,
            filename=file.filename,
            content_type=file.content_type or "image/unknown",
            width=image_info["width"],
            height=image_info["height"],
            file_size=len(content),
            content_hash=content_hash,
            description=description,
            thumbnail_base64=thumbnail_b64,
            clip_embedding=clip_embedding  # Store embedding locally
        )
        
        db.add(db_image)
        db.commit()
        
        logger.info(f"Image uploaded successfully: {image_id}")
        
        return ImageUploadResponse(
            id=image_id,  # Include 'id' for frontend consistency
            image_id=image_id,
            filename=file.filename,
            content_type=file.content_type or "image/unknown",
            width=image_info["width"],
            height=image_info["height"],
            description=description,
            thumbnail_base64=thumbnail_b64,
            created_at=db_image.created_at,
            message="Image uploaded and indexed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image upload failed: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image upload failed: {str(e)}"
        )


@router.get("/", response_model=List[ImageDocumentResponse])
async def list_images(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    List all uploaded images.
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
    
    Returns:
        List of image metadata
    """
    images = db.query(ImageDocument).order_by(
        ImageDocument.created_at.desc()
    ).offset(skip).limit(limit).all()
    
    return images


@router.get("/{image_id}", response_model=ImageDocumentResponse)
async def get_image(
    image_id: str,
    db: Session = Depends(get_db)
):
    """
    Get image metadata by ID.
    
    Args:
        image_id: Image UUID
    
    Returns:
        Image metadata
    """
    image = db.query(ImageDocument).filter(ImageDocument.id == image_id).first()
    
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image not found: {image_id}"
        )
    
    return image


@router.delete("/{image_id}")
async def delete_image(
    image_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete an image and its vector.
    
    Args:
        image_id: Image UUID
    
    Returns:
        Success message
    """
    # Check if exists
    image = db.query(ImageDocument).filter(ImageDocument.id == image_id).first()
    
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image not found: {image_id}"
        )
    
    try:
        # Delete from PostgreSQL (embedding is stored locally, not in Pinecone)
        db.delete(image)
        db.commit()
        
        return {"message": f"Image {image_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Image deletion failed: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete image: {str(e)}"
        )


@router.post("/search", response_model=ImageSearchResponse)
async def search_images(
    request: ImageSearchRequest,
    db: Session = Depends(get_db)
):
    """
    Search images by text query or image similarity.
    
    Uses CLIP embeddings for cross-modal search:
    - Text query → Find matching images
    - Image input → Find similar images
    
    Note: Uses local cosine similarity search since Pinecone has
    different dimensions (3072 for text vs 512 for CLIP).
    
    Args:
        request: Search parameters (query text or image)
    
    Returns:
        Matching images with scores
    """
    try:
        # Generate query embedding
        if request.image_base64:
            # Image-to-image search
            logger.info("Performing image similarity search")
            query_embedding = embed_image(request.image_base64)
        elif request.query:
            # Text-to-image search
            logger.info(f"Performing text-to-image search: {request.query[:50]}...")
            query_embedding = embed_image_query(request.query)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either query or image_base64 must be provided"
            )
        
        # Get all images with embeddings from PostgreSQL
        all_images = db.query(ImageDocument).filter(
            ImageDocument.clip_embedding.isnot(None)
        ).all()
        
        # Perform local similarity search
        results = search_images_local(query_embedding, all_images, top_k=request.top_k)
        
        # Build response
        response_images = []
        for img, score in results:
            logger.debug(f"Match: {img.id} with score {score:.4f}")
            response_images.append(ImageDocumentResponse(
                id=img.id,
                filename=img.filename,
                content_type=img.content_type,
                width=img.width,
                height=img.height,
                description=img.description,
                thumbnail_base64=img.thumbnail_base64,
                source_document_id=img.source_document_id,
                source_page_number=img.source_page_number,
                created_at=img.created_at
            ))
        
        logger.info(f"Found {len(response_images)} matching images")
        
        return ImageSearchResponse(
            images=response_images,
            query=request.query
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image search failed: {str(e)}"
        )


@router.get("/formats/supported")
async def get_supported_formats():
    """
    Get list of supported image formats.
    
    Returns:
        List of supported file extensions
    """
    return {
        "formats": list(SUPPORTED_FORMATS),
        "max_size_mb": MAX_IMAGE_SIZE // (1024 * 1024)
    }
