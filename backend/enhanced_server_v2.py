"""
Enhanced RAG Server - Phase 1 Complete
- PostgreSQL database persistence
- Pinecone vector store integration
- OpenAI embeddings
- Document chunking with overlap
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List, Optional, Dict
import os
import requests
import json
import base64
import hashlib
import io
from datetime import datetime
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pinecone import Pinecone
from openai import OpenAI

# Load environment
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)
print(f"[OK] Loaded environment from {env_path}")

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Models
class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    filename = Column(String, unique=True, index=True)
    content = Column(Text)
    content_hash = Column(String, unique=True, index=True)
    content_type = Column(String)
    size = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatAudit(Base):
    __tablename__ = "chat_audit"
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True)
    question = Column(Text)
    answer = Column(Text)
    model = Column(String)
    doc_ids = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class ImageRecord(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True)
    filename = Column(String, index=True)
    description = Column(Text)
    content_type = Column(String)
    size = Column(Integer)
    thumbnail = Column(Text)  # Base64 encoded thumbnail
    image_data = Column(Text)  # Base64 encoded full image
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables if they don't exist (don't drop existing data)
Base.metadata.create_all(bind=engine)
print("[DB] Tables ready (preserving existing data)")

# Initialize Pinecone and OpenAI
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "policy-rag"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Policy RAG API - Enhanced")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    question: str
    provider: str = "ollama"
    model: Optional[str] = None
    user_id: str
    doc_ids: List[str] = []

class ChatResponse(BaseModel):
    answer: str
    citations: List[dict] = []
    model: Optional[dict] = None

# Helper functions
def get_embedding(text: str) -> List[float]:
    """Get embedding from OpenAI."""
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def extract_text_from_file(content: bytes, content_type: str, filename: str) -> str:
    """Extract text from uploaded file."""
    if content_type == "text/plain" or filename.endswith(".txt"):
        return content.decode("utf-8")
    elif content_type == "application/pdf" or filename.endswith(".pdf"):
        from pypdf import PdfReader
        pdf = PdfReader(io.BytesIO(content))
        return "\n\n".join(page.extract_text() for page in pdf.pages)
    else:
        raise ValueError(f"Unsupported file type: {content_type}")

def call_ollama(model: str, messages: List[dict]) -> Optional[str]:
    """Call Ollama API."""
    try:
        response = requests.post(
            f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()["message"]["content"]
    except Exception as e:
        print(f"[ERROR] Ollama: {e}")
    return None

# Routes
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "features": ["database", "vector_store", "embeddings"]
    }

@app.get("/")
async def root():
    return {
        "name": "Policy RAG API - Enhanced",
        "version": "2.0.0"
    }

@app.post("/api/docs/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index document."""
    db = SessionLocal()
    try:
        content_bytes = await file.read()
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        
        # Check if exists
        existing = db.query(Document).filter(Document.content_hash == content_hash).first()
        if existing:
            return {"message": "Document already exists", "id": existing.id}
        
        # Extract text
        text = extract_text_from_file(content_bytes, file.content_type or "", file.filename or "")
        
        # Save to database
        doc = Document(
            filename=file.filename,
            content=text,
            content_hash=content_hash,
            content_type=file.content_type,
            size=len(content_bytes)
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        
        # Chunk and embed
        chunks = chunk_text(text)
        
        # Upload to Pinecone
        vectors = []
        for i, chunk in enumerate(chunks):
            emb = get_embedding(chunk)
            vectors.append({
                "id": f"{doc.id}-chunk-{i}",
                "values": emb,
                "metadata": {
                    "doc_id": doc.id,
                    "filename": file.filename,
                    "chunk_index": i,
                    "text": chunk[:500]
                }
            })
        
        pinecone_index.upsert(vectors=vectors)
        
        return {
            "message": "Document uploaded and indexed",
            "id": doc.id,
            "filename": file.filename,
            "chunks": len(chunks)
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/api/docs")
async def list_documents():
    """List all documents."""
    db = SessionLocal()
    try:
        docs = db.query(Document).all()
        return {
            "documents": [
                {
                    "id": str(doc.id),
                    "filename": doc.filename,
                    "content_type": doc.content_type,
                    "size": doc.size,
                    "preview_text": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                }
                for doc in docs
            ]
        }
    finally:
        db.close()

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat with RAG."""
    db = SessionLocal()
    try:
        # Embed question
        question_emb = get_embedding(request.question)
        
        # Search Pinecone
        filter_dict = None
        if request.doc_ids:
            filter_dict = {"doc_id": {"$in": [int(id) for id in request.doc_ids]}}
        
        results = pinecone_index.query(
            vector=question_emb,
            top_k=5,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Build context
        context_chunks = [
            match.metadata.get("text", "")
            for match in results.matches
            if match.score > 0.3
        ]
        context = "\n\n".join(context_chunks)
        
        # Prepare messages
        system_prompt = f"""You are a helpful assistant answering questions about company policies.
Use the following context to answer the question. If the answer is not in the context, say so.

Context:
{context}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.question}
        ]
        
        # Call LLM
        model_name = request.model or "llama3.1:8b"
        answer = call_ollama(model_name, messages)
        
        if not answer:
            answer = "Sorry, I couldn't generate a response. Please try again."
        
        # Save audit
        audit = ChatAudit(
            user_id=request.user_id,
            question=request.question,
            answer=answer,
            model=f"{request.provider}:{model_name}",
            doc_ids=",".join(request.doc_ids)
        )
        db.add(audit)
        db.commit()
        
        # Build citations
        citations = [
            {
                "document": match.metadata.get("filename"),
                "chunk": match.metadata.get("chunk_index"),
                "score": round(match.score, 3),
                "text": match.metadata.get("text", "")[:200]
            }
            for match in results.matches
            if match.score > 0.3
        ]
        
        return ChatResponse(
            answer=answer,
            citations=citations,
            model={"provider": request.provider, "name": model_name}
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# ============================================================================
# Image API endpoints (database-backed)
# ============================================================================

class ImageUploadRequest(BaseModel):
    description: Optional[str] = None
    generate_description: bool = True

@app.get("/api/images")
async def list_images():
    """List all uploaded images."""
    db = SessionLocal()
    try:
        images = db.query(ImageRecord).all()
        return [
            {
                "id": str(img.id),
                "image_id": str(img.id),
                "filename": img.filename,
                "description": img.description,
                "content_type": img.content_type,
                "size": img.size,
                "thumbnail_base64": img.thumbnail,
                "created_at": img.created_at.isoformat() if img.created_at else None
            }
            for img in images
        ]
    finally:
        db.close()

@app.get("/api/images/")
async def list_images_slash():
    """List all uploaded images (with trailing slash)."""
    return await list_images()


def generate_ai_description(image_content: bytes, filename: str) -> str:
    """Generate image description using OpenAI Vision API."""
    try:
        # Convert image to base64
        image_b64 = base64.b64encode(image_content).decode('utf-8')
        
        # Determine mime type
        img = Image.open(io.BytesIO(image_content))
        fmt = img.format or 'PNG'
        mime_type = f"image/{fmt.lower()}"
        
        # Call OpenAI Vision API
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in 1-2 sentences. Focus on the main subject and key details that would help someone search for this image later."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],
            max_tokens=150
        )
        
        description = response.choices[0].message.content.strip()
        print(f"[AI] Generated description: {description[:100]}...")
        return description
        
    except Exception as e:
        print(f"[AI] Description generation failed: {e}")
        return f"Uploaded image: {filename}"


@app.post("/api/images/upload")
async def upload_image(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    generate_description: bool = Form(True)
):
    """Upload and index an image."""
    db = SessionLocal()
    try:
        content = await file.read()
        
        # Open and validate image
        try:
            img = Image.open(io.BytesIO(content))
            img.verify()  # Verify it's a valid image
            img = Image.open(io.BytesIO(content))  # Reopen after verify
            width, height = img.size
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Create proper thumbnail (max 200x200)
        img_for_thumb = Image.open(io.BytesIO(content))
        img_for_thumb.thumbnail((200, 200), Image.Resampling.LANCZOS)
        
        # Convert to RGB if necessary (for PNG with transparency)
        if img_for_thumb.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img_for_thumb.size, (255, 255, 255))
            if img_for_thumb.mode == 'P':
                img_for_thumb = img_for_thumb.convert('RGBA')
            background.paste(img_for_thumb, mask=img_for_thumb.split()[-1] if img_for_thumb.mode == 'RGBA' else None)
            img_for_thumb = background
        elif img_for_thumb.mode != 'RGB':
            img_for_thumb = img_for_thumb.convert('RGB')
        
        # Save thumbnail to base64
        thumb_buffer = io.BytesIO()
        img_for_thumb.save(thumb_buffer, format='JPEG', quality=85)
        thumbnail_b64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')
        
        # Save full image to base64
        image_b64 = base64.b64encode(content).decode('utf-8')
        
        # Generate description - use provided description, AI generation, or fallback
        final_description = description
        if not final_description and generate_description:
            # Use AI to generate description
            final_description = generate_ai_description(content, file.filename)
        elif not final_description:
            final_description = f"Uploaded image: {file.filename}"
        
        # Save to database
        image_record = ImageRecord(
            filename=file.filename,
            description=final_description,
            content_type=file.content_type or "image/unknown",
            size=len(content),
            thumbnail=thumbnail_b64,
            image_data=image_b64
        )
        db.add(image_record)
        db.commit()
        db.refresh(image_record)
        
        return {
            "id": str(image_record.id),
            "image_id": str(image_record.id),
            "filename": image_record.filename,
            "content_type": image_record.content_type,
            "width": width,
            "height": height,
            "description": final_description,
            "thumbnail_base64": thumbnail_b64,
            "created_at": image_record.created_at.isoformat() if image_record.created_at else None,
            "message": "Image uploaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")
    finally:
        db.close()

@app.get("/api/images/{image_id}")
async def get_image(image_id: str):
    """Get image by ID."""
    db = SessionLocal()
    try:
        img = db.query(ImageRecord).filter(ImageRecord.id == int(image_id)).first()
        if not img:
            raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")
        return {
            "id": str(img.id),
            "image_id": str(img.id),
            "filename": img.filename,
            "description": img.description,
            "content_type": img.content_type,
            "size": img.size,
            "thumbnail_base64": img.thumbnail,
            "created_at": img.created_at.isoformat() if img.created_at else None
        }
    finally:
        db.close()

@app.delete("/api/images/{image_id}")
async def delete_image(image_id: str):
    """Delete an image."""
    db = SessionLocal()
    try:
        img = db.query(ImageRecord).filter(ImageRecord.id == int(image_id)).first()
        if not img:
            raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")
        db.delete(img)
        db.commit()
        return {"message": f"Image {image_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/api/images/formats/supported")
async def get_supported_formats():
    """Get list of supported image formats."""
    return {
        "formats": ["jpg", "jpeg", "png", "gif", "webp", "bmp"],
        "max_size_mb": 10
    }


# ============================================================================
# Document Management API (Phase 2)
# ============================================================================

@app.get("/api/docs/{doc_id}")
async def get_document(doc_id: int):
    """Get document metadata by ID."""
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
        return {
            "id": str(doc.id),
            "filename": doc.filename,
            "content_type": doc.content_type,
            "size": doc.size,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
            "preview_text": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
        }
    finally:
        db.close()

@app.get("/api/docs/{doc_id}/content")
async def get_document_content(doc_id: int):
    """Get full document content by ID."""
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
        return {
            "id": str(doc.id),
            "filename": doc.filename,
            "content": doc.content,
            "content_type": doc.content_type,
            "size": doc.size
        }
    finally:
        db.close()

@app.delete("/api/docs/{doc_id}")
async def delete_document(doc_id: int):
    """Delete a document and its vectors."""
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
        
        # Delete vectors from Pinecone
        try:
            # Delete all chunks for this document
            pinecone_index.delete(filter={"doc_id": doc_id})
        except Exception as e:
            print(f"[WARN] Failed to delete vectors: {e}")
        
        # Delete from database
        db.delete(doc)
        db.commit()
        
        return {"message": f"Document {doc_id} deleted successfully", "filename": doc.filename}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# ============================================================================
# Chat History API (Phase 2)
# ============================================================================

@app.get("/api/chat/history/{user_id}")
async def get_chat_history(user_id: str, limit: int = 50):
    """Get chat history for a user."""
    db = SessionLocal()
    try:
        history = db.query(ChatAudit).filter(
            ChatAudit.user_id == user_id
        ).order_by(ChatAudit.created_at.desc()).limit(limit).all()
        
        return {
            "user_id": user_id,
            "messages": [
                {
                    "id": str(h.id),
                    "question": h.question,
                    "answer": h.answer,
                    "model": h.model,
                    "created_at": h.created_at.isoformat() if h.created_at else None
                }
                for h in reversed(history)  # Oldest first
            ]
        }
    finally:
        db.close()

@app.delete("/api/chat/history/{user_id}")
async def clear_chat_history(user_id: str):
    """Clear chat history for a user."""
    db = SessionLocal()
    try:
        deleted = db.query(ChatAudit).filter(ChatAudit.user_id == user_id).delete()
        db.commit()
        return {"message": f"Cleared {deleted} messages for user {user_id}"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# ============================================================================
# Compliance Check API
# ============================================================================

class ComplianceRequest(BaseModel):
    user_id: str
    query: str
    provider: str = "openai"
    model: Optional[str] = None
    doc_ids: Optional[List[str]] = None
    image_ids: Optional[List[str]] = None


@app.post("/api/compliance/check")
async def compliance_check(request: ComplianceRequest):
    """Perform a compliance check combining documents and images."""
    db = SessionLocal()
    try:
        # Get relevant document context
        doc_context = ""
        if request.doc_ids:
            for doc_id in request.doc_ids:
                doc = db.query(Document).filter(Document.id == int(doc_id)).first()
                if doc:
                    doc_context += f"\n\n--- Document: {doc.filename} ---\n{doc.content}"
        else:
            # Use all documents if none specified
            docs = db.query(Document).limit(5).all()
            for doc in docs:
                doc_context += f"\n\n--- Document: {doc.filename} ---\n{doc.content[:2000]}"
        
        # Get image descriptions
        image_context = ""
        if request.image_ids:
            for img_id in request.image_ids:
                img = db.query(ImageRecord).filter(ImageRecord.id == int(img_id)).first()
                if img:
                    image_context += f"\n\n--- Image: {img.filename} ---\n{img.description or 'No description'}"
        
        # Build compliance prompt
        system_prompt = """You are a compliance expert. Be CONCISE and BRIEF.

Provide a short structured response:
1. Status: COMPLIANT, NON_COMPLIANT, PARTIAL, or NEEDS_REVIEW
2. Key findings (2-3 bullet points max)
3. One recommendation if needed

Keep total response under 200 words. Use short sentences."""

        user_prompt = f"""Compliance Question: {request.query}

Documents:
{doc_context if doc_context else "No documents provided."}

Images:
{image_context if image_context else "No images provided."}

Please analyze and provide your compliance assessment."""

        # Call LLM
        response = openai_client.chat.completions.create(
            model=request.model or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=4000
        )
        
        answer = response.choices[0].message.content
        
        # Determine status from response
        status = "needs_review"
        answer_lower = answer.lower()
        if "non-compliant" in answer_lower or "non_compliant" in answer_lower:
            status = "non_compliant"
        elif "partially compliant" in answer_lower or "partial" in answer_lower:
            status = "partial"
        elif "compliant" in answer_lower and "non" not in answer_lower:
            status = "compliant"
        
        return {
            "status": status,
            "answer": answer,
            "query": request.query,
            "doc_ids": request.doc_ids,
            "image_ids": request.image_ids
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.post("/api/compliance/check/stream")
async def compliance_check_stream(request: ComplianceRequest):
    """Stream compliance check with SSE events."""
    from fastapi.responses import StreamingResponse
    
    async def generate():
        db = SessionLocal()
        try:
            # Send status update
            yield f"data: {json.dumps({'type': 'status', 'data': 'Gathering documents...'})}\n\n"
            
            # Get document context
            doc_context = ""
            doc_citations = []
            if request.doc_ids:
                for doc_id in request.doc_ids:
                    doc = db.query(Document).filter(Document.id == int(doc_id)).first()
                    if doc:
                        doc_context += f"\n\n--- Document: {doc.filename} ---\n{doc.content}"
                        doc_citations.append({"id": str(doc.id), "filename": doc.filename, "type": "document", "score": 1.0})
            else:
                docs = db.query(Document).limit(5).all()
                for doc in docs:
                    doc_context += f"\n\n--- Document: {doc.filename} ---\n{doc.content[:2000]}"
                    doc_citations.append({"id": str(doc.id), "filename": doc.filename, "type": "document", "score": 1.0})
            
            yield f"data: {json.dumps({'type': 'status', 'data': 'Analyzing images...'})}\n\n"
            
            # Get image context
            image_context = ""
            image_citations = []
            if request.image_ids:
                for img_id in request.image_ids:
                    img = db.query(ImageRecord).filter(ImageRecord.id == int(img_id)).first()
                    if img:
                        image_context += f"\n\n--- Image: {img.filename} ---\n{img.description or 'No description'}"
                        image_citations.append({
                            "id": str(img.id),
                            "filename": img.filename,
                            "type": "image",
                            "thumbnail_base64": img.thumbnail
                        })
            
            # Send citations
            yield f"data: {json.dumps({'type': 'citations', 'data': {'document_citations': doc_citations, 'image_citations': image_citations}})}\n\n"
            
            yield f"data: {json.dumps({'type': 'status', 'data': 'Running compliance analysis...'})}\n\n"
            
            # Build prompt
            system_prompt = """You are a compliance expert. Be CONCISE and BRIEF.

Provide a short structured response:
1. Status: COMPLIANT, NON_COMPLIANT, PARTIAL, or NEEDS_REVIEW
2. Key findings (2-3 bullet points max)
3. One recommendation if needed

Keep total response under 200 words. Use short sentences."""

            user_prompt = f"""Compliance Question: {request.query}

Documents:
{doc_context if doc_context else "No documents provided."}

Images:
{image_context if image_context else "No images provided."}

Please analyze and provide your compliance assessment."""

            # Stream from OpenAI
            stream = openai_client.chat.completions.create(
                model=request.model or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000,
                stream=True
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"
            
            # Determine status
            status = "needs_review"
            answer_lower = full_response.lower()
            if "non-compliant" in answer_lower or "non_compliant" in answer_lower:
                status = "non_compliant"
            elif "partially compliant" in answer_lower or "partial" in answer_lower:
                status = "partial"
            elif "compliant" in answer_lower and "non" not in answer_lower:
                status = "compliant"
            
            # Count findings from response
            compliant_count = 1 if status == "compliant" else 0
            non_compliant_count = 1 if status == "non_compliant" else 0
            partial_count = 1 if status == "partial" else 0
            
            # Send final report
            report = {
                "id": f"report-{int(datetime.utcnow().timestamp())}",
                "title": "Compliance Analysis Report",
                "overall_status": status,
                "status": status,
                "created_at": datetime.utcnow().isoformat(),
                "answer": full_response,
                "query": request.query,
                "findings": [],
                "summary": full_response,
                "statistics": {
                    "total_findings": 1,
                    "compliant_count": compliant_count,
                    "non_compliant_count": non_compliant_count,
                    "partial_count": partial_count
                },
                "document_citations": doc_citations,
                "image_citations": image_citations
            }
            yield f"data: {json.dumps({'type': 'report', 'data': report})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
        finally:
            db.close()
    
    return StreamingResponse(generate(), media_type="text/event-stream")


# ============================================================================
# Image Search API (Phase 2 - Multimodal)
# ============================================================================

@app.post("/api/images/search")
async def search_images(query: str = Form(...), top_k: int = Form(5)):
    """Search images by text query using CLIP embeddings (stub - returns all images)."""
    db = SessionLocal()
    try:
        # In production, this would use CLIP embeddings for semantic search
        # For now, filter by description text match
        results = []
        query_lower = query.lower()
        
        images = db.query(ImageRecord).all()
        for img in images:
            desc = (img.description or "").lower()
            filename = (img.filename or "").lower()
            
            # Simple text matching score
            score = 0
            if query_lower in desc:
                score = 0.9
            elif query_lower in filename:
                score = 0.7
            elif any(word in desc for word in query_lower.split()):
                score = 0.5
            
            if score > 0:
                results.append({
                    "id": str(img.id),
                    "filename": img.filename,
                    "description": img.description,
                    "thumbnail_base64": img.thumbnail,
                    "score": score
                })
        
        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return {"query": query, "results": results[:top_k]}
    finally:
        db.close()


if __name__ == "__main__":
    print("=" * 70)
    print("  POLICY RAG API - Enhanced (Phase 1)")
    print("=" * 70)
    print("\n[INFO] Features:")
    print("       - PostgreSQL database persistence")
    print("       - Pinecone vector store")
    print("       - OpenAI embeddings")
    print("       - Document chunking with overlap")
    print("\nAccess points:")
    print("  - API:  http://localhost:8001")
    print("  - Docs: http://localhost:8001/docs")
    print("  - Frontend: http://localhost:5173")
    print("\n" + "=" * 70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
