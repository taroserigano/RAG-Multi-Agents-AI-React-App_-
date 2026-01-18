"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
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


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    user_id: str = Field(..., description="User or session identifier")
    provider: str = Field(..., description="LLM provider: ollama, openai, or anthropic")
    model: Optional[str] = Field(None, description="Specific model name (uses default if not provided)")
    question: str = Field(..., min_length=1, description="User's question")
    doc_ids: Optional[List[str]] = Field(None, description="Optional list of document IDs to restrict search")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of chunks to retrieve (1-20)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user-123",
                "provider": "ollama",
                "question": "What is the company's data retention policy?",
                "top_k": 5
            }
        }


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    answer: str
    citations: List[CitationResponse]
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
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============================================================================
# Error Schemas
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
