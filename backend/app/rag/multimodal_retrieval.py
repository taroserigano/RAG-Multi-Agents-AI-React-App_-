"""
Multimodal retrieval module for combined text and image search.
Supports cross-modal retrieval using CLIP embeddings.
"""
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.retrieval import Citation, retrieve_relevant_chunks
from app.rag.image_embeddings import get_clip_embeddings, embed_image, embed_image_query
from app.rag.indexing import get_pinecone_index

settings = get_settings()
logger = get_logger(__name__)

# Namespace for image vectors in Pinecone
IMAGE_NAMESPACE = "images"


@dataclass
class ImageCitation:
    """Citation data class for retrieved images."""
    image_id: str
    filename: str
    score: float
    description: Optional[str] = None
    thumbnail_base64: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "image_id": self.image_id,
            "filename": self.filename,
            "score": round(self.score, 4),
            "description": self.description[:200] + "..." if self.description and len(self.description) > 200 else self.description,
            "thumbnail_base64": self.thumbnail_base64,
            "width": self.width,
            "height": self.height
        }


def retrieve_relevant_images(
    query: str,
    top_k: int = 5,
    image_ids: Optional[List[str]] = None
) -> List[ImageCitation]:
    """
    Retrieve relevant images using CLIP text-to-image search.
    
    Args:
        query: Text query
        top_k: Number of images to retrieve
        image_ids: Optional list of image IDs to restrict search
    
    Returns:
        List of ImageCitation objects
    """
    logger.info(f"Retrieving top {top_k} images for query")
    
    try:
        # Generate CLIP text embedding for cross-modal search
        query_embedding = embed_image_query(query)
        
        # Build filter if image IDs provided
        filter_dict = None
        if image_ids:
            filter_dict = {"image_id": {"$in": image_ids}}
        
        # Query Pinecone image namespace
        index = get_pinecone_index()
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=IMAGE_NAMESPACE,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Convert to ImageCitation objects
        citations = []
        for match in results.matches:
            metadata = match.metadata or {}
            citation = ImageCitation(
                image_id=match.id,
                filename=metadata.get("filename", "unknown"),
                score=match.score,
                description=metadata.get("description"),
                width=metadata.get("width"),
                height=metadata.get("height")
            )
            citations.append(citation)
        
        logger.info(f"Retrieved {len(citations)} images")
        return citations
        
    except Exception as e:
        logger.error(f"Image retrieval failed: {e}")
        return []


def retrieve_similar_images(
    image: Union[Image.Image, bytes, str],
    top_k: int = 5
) -> List[ImageCitation]:
    """
    Find similar images using image-to-image CLIP search.
    
    Args:
        image: Query image (PIL Image, bytes, or base64)
        top_k: Number of similar images to retrieve
    
    Returns:
        List of ImageCitation objects
    """
    logger.info(f"Finding {top_k} similar images")
    
    try:
        # Generate CLIP image embedding
        image_embedding = embed_image(image)
        
        # Query Pinecone
        index = get_pinecone_index()
        results = index.query(
            vector=image_embedding,
            top_k=top_k,
            namespace=IMAGE_NAMESPACE,
            include_metadata=True
        )
        
        # Convert to citations
        citations = []
        for match in results.matches:
            metadata = match.metadata or {}
            citation = ImageCitation(
                image_id=match.id,
                filename=metadata.get("filename", "unknown"),
                score=match.score,
                description=metadata.get("description"),
                width=metadata.get("width"),
                height=metadata.get("height")
            )
            citations.append(citation)
        
        return citations
        
    except Exception as e:
        logger.error(f"Similar image search failed: {e}")
        return []


def retrieve_multimodal(
    query: str,
    top_k_text: int = 5,
    top_k_images: int = 3,
    doc_ids: Optional[List[str]] = None,
    image_ids: Optional[List[str]] = None,
    use_hybrid: bool = False,
    image_weight: float = 0.3
) -> Tuple[List[Citation], List[ImageCitation], str]:
    """
    Combined multimodal retrieval for text and images.
    
    Retrieves both text chunks and images relevant to the query,
    combining them based on configured weights.
    
    Args:
        query: User question
        top_k_text: Number of text chunks to retrieve
        top_k_images: Number of images to retrieve
        doc_ids: Optional document ID filter
        image_ids: Optional image ID filter
        use_hybrid: Enable hybrid text search
        image_weight: Weight for image results (0-1)
    
    Returns:
        Tuple of (text_citations, image_citations, combined_context)
    """
    logger.info(f"Multimodal retrieval: text={top_k_text}, images={top_k_images}")
    
    # Retrieve text chunks
    text_citations, text_context = retrieve_relevant_chunks(
        query=query,
        top_k=top_k_text,
        doc_ids=doc_ids,
        use_hybrid=use_hybrid
    )
    
    # Retrieve images
    image_citations = []
    if top_k_images > 0:
        image_citations = retrieve_relevant_images(
            query=query,
            top_k=top_k_images,
            image_ids=image_ids
        )
    
    # Build combined context
    context_parts = []
    
    # Add text context
    if text_context:
        context_parts.append("=== Document Context ===")
        context_parts.append(text_context)
    
    # Add image descriptions as additional context
    if image_citations:
        context_parts.append("\n=== Related Images ===")
        for img in image_citations:
            if img.description:
                context_parts.append(f"- {img.filename}: {img.description}")
    
    combined_context = "\n".join(context_parts)
    
    logger.info(f"Retrieved {len(text_citations)} text chunks and {len(image_citations)} images")
    
    return text_citations, image_citations, combined_context


def retrieve_with_image_query(
    image: Union[Image.Image, bytes, str],
    text_query: Optional[str] = None,
    top_k_text: int = 5,
    top_k_images: int = 3,
    use_hybrid: bool = False
) -> Tuple[List[Citation], List[ImageCitation], str]:
    """
    Retrieve using an image as the query (with optional text).
    
    Useful for "find documents/images related to this diagram" queries.
    
    Args:
        image: Query image
        text_query: Optional accompanying text query
        top_k_text: Number of text chunks
        top_k_images: Number of images
        use_hybrid: Enable hybrid text search
    
    Returns:
        Tuple of (text_citations, image_citations, combined_context)
    """
    logger.info("Image-based multimodal retrieval")
    
    # Find similar images
    image_citations = retrieve_similar_images(image, top_k=top_k_images)
    
    # If we have a text query, use it for text retrieval
    # Otherwise, use image descriptions from similar images
    if text_query:
        search_query = text_query
    elif image_citations and image_citations[0].description:
        # Use top image's description as proxy query
        search_query = image_citations[0].description
    else:
        # Fall back to generic search
        search_query = "document related to image"
    
    # Retrieve text chunks
    text_citations, text_context = retrieve_relevant_chunks(
        query=search_query,
        top_k=top_k_text,
        use_hybrid=use_hybrid
    )
    
    # Build context
    context_parts = []
    
    if text_context:
        context_parts.append("=== Document Context ===")
        context_parts.append(text_context)
    
    if image_citations:
        context_parts.append("\n=== Similar Images Found ===")
        for img in image_citations:
            desc = img.description or "No description available"
            context_parts.append(f"- {img.filename}: {desc}")
    
    combined_context = "\n".join(context_parts)
    
    return text_citations, image_citations, combined_context


def rerank_multimodal_results(
    text_citations: List[Citation],
    image_citations: List[ImageCitation],
    query: str,
    text_weight: float = 0.7,
    image_weight: float = 0.3
) -> Tuple[List[Citation], List[ImageCitation]]:
    """
    Rerank combined multimodal results.
    
    Adjusts scores based on relevance and configurable weights.
    
    Args:
        text_citations: Text retrieval results
        image_citations: Image retrieval results
        query: Original query
        text_weight: Weight for text results
        image_weight: Weight for image results
    
    Returns:
        Reranked (text_citations, image_citations)
    """
    # Normalize and weight text scores
    if text_citations:
        max_text_score = max(c.score for c in text_citations)
        for citation in text_citations:
            citation.score = (citation.score / max_text_score) * text_weight
    
    # Normalize and weight image scores
    if image_citations:
        max_img_score = max(c.score for c in image_citations)
        for citation in image_citations:
            citation.score = (citation.score / max_img_score) * image_weight
    
    # Sort by weighted score
    text_citations.sort(key=lambda x: x.score, reverse=True)
    image_citations.sort(key=lambda x: x.score, reverse=True)
    
    return text_citations, image_citations


class MultimodalRetriever:
    """
    High-level multimodal retriever combining text and image search.
    """
    
    def __init__(
        self,
        default_top_k_text: int = 5,
        default_top_k_images: int = 3,
        default_image_weight: float = 0.3
    ):
        self.default_top_k_text = default_top_k_text
        self.default_top_k_images = default_top_k_images
        self.default_image_weight = default_image_weight
    
    def retrieve(
        self,
        query: str,
        include_images: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main retrieval method.
        
        Args:
            query: User question
            include_images: Whether to include image results
            **kwargs: Additional options
        
        Returns:
            Dictionary with text_citations, image_citations, and context
        """
        top_k_text = kwargs.get('top_k_text', self.default_top_k_text)
        top_k_images = kwargs.get('top_k_images', self.default_top_k_images) if include_images else 0
        use_hybrid = kwargs.get('use_hybrid', False)
        doc_ids = kwargs.get('doc_ids')
        image_ids = kwargs.get('image_ids')
        
        text_citations, image_citations, context = retrieve_multimodal(
            query=query,
            top_k_text=top_k_text,
            top_k_images=top_k_images,
            doc_ids=doc_ids,
            image_ids=image_ids,
            use_hybrid=use_hybrid,
            image_weight=self.default_image_weight
        )
        
        return {
            "text_citations": text_citations,
            "image_citations": image_citations,
            "context": context,
            "query": query
        }
    
    def retrieve_by_image(
        self,
        image: Union[Image.Image, bytes, str],
        text_query: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Retrieve using image as query.
        
        Args:
            image: Query image
            text_query: Optional text query
            **kwargs: Additional options
        
        Returns:
            Retrieval results
        """
        top_k_text = kwargs.get('top_k_text', self.default_top_k_text)
        top_k_images = kwargs.get('top_k_images', self.default_top_k_images)
        use_hybrid = kwargs.get('use_hybrid', False)
        
        text_citations, image_citations, context = retrieve_with_image_query(
            image=image,
            text_query=text_query,
            top_k_text=top_k_text,
            top_k_images=top_k_images,
            use_hybrid=use_hybrid
        )
        
        return {
            "text_citations": text_citations,
            "image_citations": image_citations,
            "context": context,
            "query": text_query or "image-based search"
        }


# Default retriever instance
multimodal_retriever = MultimodalRetriever()
