"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Union
from datetime import datetime


# ============================================================================
# Document Schemas
# ============================================================================

class DocumentResponse(BaseModel):
    """Response schema for document metadata."""
    id: str
    filename: str
    content_type: str
    preview_text: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    """Response schema for document upload."""
    doc_id: str
    filename: str
    message: str = "Document uploaded and indexed successfully"


# ============================================================================
# Image Schemas (Multimodal)
# ============================================================================

class ImageInfo(BaseModel):
    """Image metadata information."""
    width: int
    height: int
    format: str
    mode: str
    has_transparency: bool = False


class ImageDocumentResponse(BaseModel):
    """Response schema for image document metadata."""
    id: str
    filename: str
    content_type: str
    width: Optional[int] = None
    height: Optional[int] = None
    description: Optional[str] = None
    thumbnail_base64: Optional[str] = None
    source_document_id: Optional[str] = None
    source_page_number: Optional[int] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class ImageUploadResponse(BaseModel):
    """Response schema for image upload."""
    id: str  # Use 'id' to match ImageDocumentResponse for frontend consistency
    image_id: str  # Keep for backward compatibility
    filename: str
    content_type: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    description: Optional[str] = None
    thumbnail_base64: Optional[str] = None
    created_at: Optional[datetime] = None
    message: str = "Image uploaded and indexed successfully"


class ImageCitationResponse(BaseModel):
    """Image citation metadata in chat response."""
    image_id: str
    filename: str
    score: float
    description: Optional[str] = None
    thumbnail_base64: Optional[str] = None


# ============================================================================
# Chat Schemas
# ============================================================================

class CitationResponse(BaseModel):
    """Citation metadata in chat response."""
    doc_id: str
    filename: str
    page_number: Optional[int] = None
    chunk_index: int
    score: float
    text: Optional[str] = Field(None, description="Snippet of cited text")


class ModelInfo(BaseModel):
    """LLM model information."""
    provider: str
    name: str


class RAGOptions(BaseModel):
    """Optional advanced RAG processing options."""
    query_expansion: bool = Field(False, description="Enable query expansion/rewriting for better coverage")
    hybrid_search: bool = Field(False, description="Enable hybrid search (semantic + keyword)")
    reranking: bool = Field(False, description="Enable cross-encoder reranking for better relevance")


class MultimodalOptions(BaseModel):
    """Options for multimodal RAG queries."""
    include_images: bool = Field(True, description="Include images in retrieval")
    image_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for image results (0-1)")
    use_vision_model: bool = Field(False, description="Use vision model to analyze retrieved images")
    vision_provider: Optional[str] = Field(None, description="Vision model provider (openai/anthropic/ollama)")


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    user_id: str = Field(..., description="User or session identifier")
    provider: str = Field(..., description="LLM provider: ollama, openai, or anthropic")
    model: Optional[str] = Field(None, description="Specific model name (uses default if not provided)")
    question: str = Field(..., min_length=1, description="User's question")
    doc_ids: Optional[List[str]] = Field(None, description="Optional list of document IDs to restrict search")
    image_ids: Optional[List[str]] = Field(None, description="Optional list of image IDs to include")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of chunks to retrieve (1-20)")
    rag_options: Optional[RAGOptions] = Field(default_factory=RAGOptions, description="Advanced RAG options")
    multimodal_options: Optional[MultimodalOptions] = Field(default_factory=MultimodalOptions, description="Multimodal options")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user-123",
                "provider": "ollama",
                "question": "What is the company's data retention policy?",
                "top_k": 5,
                "multimodal_options": {
                    "include_images": True,
                    "use_vision_model": False
                }
            }
        }


class MultimodalChatRequest(BaseModel):
    """Request schema for multimodal chat with image input."""
    user_id: str = Field(..., description="User or session identifier")
    provider: str = Field(..., description="LLM provider: ollama, openai, or anthropic")
    model: Optional[str] = Field(None, description="Specific model name")
    question: str = Field(..., min_length=1, description="User's question about the image")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image to analyze")
    image_url: Optional[str] = Field(None, description="URL of image to analyze")
    doc_ids: Optional[List[str]] = Field(None, description="Document IDs for context")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    rag_options: Optional[RAGOptions] = Field(default_factory=RAGOptions, description="Advanced RAG options")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user-123",
                "provider": "openai",
                "question": "What security violations do you see in this diagram?",
                "image_base64": "base64-encoded-image-data...",
                "top_k": 5
            }
        }


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    answer: str
    citations: List[CitationResponse]
    image_citations: Optional[List[ImageCitationResponse]] = None
    model: ModelInfo
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "According to the policy document...",
                "citations": [
                    {
                        "doc_id": "abc-123",
                        "filename": "data_policy.pdf",
                        "page_number": 5,
                        "chunk_index": 2,
                        "score": 0.89
                    }
                ],
                "image_citations": [],
                "model": {
                    "provider": "ollama",
                    "name": "llama3.1"
                }
            }
        }


class ChatHistoryResponse(BaseModel):
    """Response schema for chat history entries."""
    id: str
    user_id: str
    provider: str
    model: str
    question: str
    answer: str
    cited_doc_ids: Optional[List[str]] = None
    cited_image_ids: Optional[List[str]] = None
    has_image_query: bool = False
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============================================================================
# Image Search Schemas
# ============================================================================

class ImageSearchRequest(BaseModel):
    """Request for image-based search."""
    query: Optional[str] = Field(None, description="Text query for image search")
    image_base64: Optional[str] = Field(None, description="Base64 image for similarity search")
    top_k: int = Field(5, ge=1, le=20, description="Number of images to retrieve")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "security architecture diagram",
                "top_k": 5
            }
        }


class ImageSearchResponse(BaseModel):
    """Response for image search."""
    images: List[ImageDocumentResponse]
    query: Optional[str] = None


# ============================================================================
# Error Schemas
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
