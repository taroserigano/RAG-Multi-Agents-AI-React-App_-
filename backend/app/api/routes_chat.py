"""
Chat API routes.
Handles RAG-based question answering with LLM provider selection.
Supports both regular and streaming responses.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List
import json

from app.db.session import get_db
from app.db.models import ChatAudit, ImageDocument
from app.schemas import ChatRequest, ChatResponse, ChatHistoryResponse, ErrorResponse
from app.rag.graph import run_rag_pipeline, run_rag_pipeline_streaming
from app.rag.multimodal_retrieval import retrieve_multimodal
from app.rag.llms import get_streaming_llm
from app.core.logging import get_logger

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = get_logger(__name__)


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Process a chat question using RAG pipeline.
    
    Flow:
    1. Validate provider and model
    2. Retrieve relevant chunks from Pinecone (filtered by doc_ids if provided)
    3. Build prompt with context
    4. Call selected LLM (Ollama/OpenAI/Anthropic)
    5. Return answer with citations
    6. Log to audit table
    
    Args:
        request: ChatRequest with question, provider, optional filters
        db: Database session
    
    Returns:
        ChatResponse with answer, citations, and model info
    """
    try:
        # Validate provider
        valid_providers = ["ollama", "openai", "anthropic"]
        if request.provider not in valid_providers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid provider. Must be one of: {', '.join(valid_providers)}"
            )
        
        logger.info(f"Processing chat request from user {request.user_id} with provider {request.provider}")
        
        # Run RAG pipeline
        try:
            result = run_rag_pipeline(
                question=request.question,
                provider=request.provider,
                model=request.model,
                doc_ids=request.doc_ids,
                top_k=request.top_k or 5
            )
        except ValueError as e:
            # Handle configuration errors (missing API keys, etc.)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing question: {str(e)}"
            )
        
        # Extract cited document IDs
        cited_doc_ids = list(set([
            citation["doc_id"] for citation in result["citations"]
        ]))
        
        # Save to audit log
        try:
            audit = ChatAudit(
                user_id=request.user_id,
                provider=request.provider,
                model=request.model or result["model"]["name"],
                question=request.question,
                answer=result["answer"],
                cited_doc_ids=cited_doc_ids if cited_doc_ids else None
            )
            db.add(audit)
            db.commit()
        except Exception as e:
            # Log error but don't fail the request
            logger.error(f"Error saving chat audit: {e}")
        
        logger.info(f"Chat request completed successfully with {len(result['citations'])} citations")
        
        return ChatResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Stream chat response using Server-Sent Events (SSE).
    
    Returns tokens as they are generated, followed by citations at the end.
    
    Event types:
    - token: A text chunk from the LLM
    - citations: Array of citation objects (sent once at end)
    - done: Signals completion
    - error: Error message if something went wrong
    """
    # Validate provider upfront
    valid_providers = ["ollama", "openai", "anthropic"]
    if request.provider not in valid_providers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid provider. Must be one of: {', '.join(valid_providers)}"
        )
    
    logger.info(f"Processing streaming chat request from user {request.user_id}")
    
    # Check if multimodal mode (images selected)
    is_multimodal = request.image_ids and len(request.image_ids) > 0
    image_details = []
    
    if is_multimodal:
        logger.info(f"Multimodal chat with {len(request.image_ids)} images")
        # Fetch image details from database for context
        try:
            images = db.query(ImageDocument).filter(ImageDocument.id.in_(request.image_ids)).all()
            logger.info(f"Found {len(images)} images in database")
            for img in images:
                desc = img.description or img.extracted_text or "No description available"
                image_details.append({
                    "id": img.id,
                    "filename": img.filename,
                    "description": desc,
                    "thumbnail_base64": img.thumbnail_base64,
                    "content_type": img.content_type
                })
                logger.info(f"Image {img.filename}: description={desc[:100] if desc else 'None'}...")
        except Exception as e:
            logger.error(f"Error fetching image details: {e}")
    
    # Extract RAG options
    rag_options = None
    if request.rag_options:
        rag_options = {
            'query_expansion': request.rag_options.query_expansion,
            'hybrid_search': request.rag_options.hybrid_search,
            'reranking': request.rag_options.reranking
        }
        logger.info(f"RAG options: {rag_options}")
    
    def generate():
        full_answer = ""
        citations = []
        model_info = {"provider": request.provider, "name": request.model or "default"}
        
        try:
            # If images are selected, use image-focused chat
            if is_multimodal and image_details:
                logger.info(f"Running image-focused chat with {len(image_details)} images")
                
                # Build system prompt for image analysis
                system_prompt = """You are an AI assistant that helps users understand images. 
You have been provided with descriptions of images that the user wants to ask about.
Answer the user's question based on the image descriptions provided.
Be specific and reference the image details when answering."""

                # Build image context
                image_context_parts = ["=== IMAGE DESCRIPTIONS ==="]
                for img in image_details:
                    image_context_parts.append(f"\nImage: {img['filename']}")
                    image_context_parts.append(f"Description: {img['description']}")
                image_context = "\n".join(image_context_parts)
                
                # Create messages for LLM
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{image_context}\n\n=== USER QUESTION ===\n{request.question}"}
                ]
                
                # Get streaming LLM and generate response
                llm = get_streaming_llm(request.provider, request.model)
                model_info = {"provider": request.provider, "name": request.model or llm.model if hasattr(llm, 'model') else "default"}
                
                for token in llm.stream(messages):
                    if hasattr(token, 'content'):
                        token_text = token.content
                    else:
                        token_text = str(token)
                    full_answer += token_text
                    yield f"data: {json.dumps({'type': 'token', 'data': token_text})}\n\n"
                
                # Send empty citations (no document sources for image-only queries)
                yield f"data: {json.dumps({'type': 'citations', 'data': []})}\n\n"
                
            else:
                # Normal document RAG flow
                for event in run_rag_pipeline_streaming(
                    question=request.question,
                    provider=request.provider,
                    model=request.model,
                    doc_ids=request.doc_ids,
                    top_k=request.top_k or 5,
                    rag_options=rag_options
                ):
                    if event["type"] == "token":
                        full_answer += event["data"]
                        yield f"data: {json.dumps({'type': 'token', 'data': event['data']})}\n\n"
                    elif event["type"] == "citations":
                        citations = event["data"]
                        yield f"data: {json.dumps({'type': 'citations', 'data': citations})}\n\n"
                    elif event["type"] == "model":
                        model_info = event["data"]
                    elif event["type"] == "error":
                        yield f"data: {json.dumps({'type': 'error', 'data': event['data']})}\n\n"
                        return
            
            # Send images info if multimodal
            if is_multimodal and image_details:
                yield f"data: {json.dumps({'type': 'images', 'data': image_details})}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'done', 'data': {'model': model_info}})}\n\n"
            
            # Save to audit log
            try:
                cited_doc_ids = list(set([c["doc_id"] for c in citations])) if citations else []
                cited_image_ids = request.image_ids if is_multimodal else None
                audit = ChatAudit(
                    user_id=request.user_id,
                    provider=request.provider,
                    model=request.model or model_info["name"],
                    question=request.question,
                    answer=full_answer,
                    cited_doc_ids=cited_doc_ids if cited_doc_ids else None,
                    cited_image_ids=cited_image_ids
                )
                db.add(audit)
                db.commit()
            except Exception as e:
                logger.error(f"Error saving chat audit: {e}")
                
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/history/{user_id}", response_model=List[ChatHistoryResponse])
def get_chat_history(
    user_id: str,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get chat history for a specific user.
    
    Args:
        user_id: User identifier
        limit: Maximum number of messages to return (default 50)
    
    Returns:
        List of chat history entries ordered by most recent first
    """
    try:
        history = db.query(ChatAudit)\
            .filter(ChatAudit.user_id == user_id)\
            .order_by(ChatAudit.created_at.desc())\
            .limit(limit)\
            .all()
        
        return history
    
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching chat history"
        )


@router.delete("/history/{user_id}")
def clear_chat_history(
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Clear all chat history for a specific user.
    
    Args:
        user_id: User identifier
    
    Returns:
        Number of entries deleted
    """
    try:
        deleted_count = db.query(ChatAudit)\
            .filter(ChatAudit.user_id == user_id)\
            .delete()
        db.commit()
        
        logger.info(f"Deleted {deleted_count} chat history entries for user {user_id}")
        
        return {
            "message": f"Deleted {deleted_count} chat history entries",
            "deleted_count": deleted_count
        }
    
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error clearing chat history"
        )
