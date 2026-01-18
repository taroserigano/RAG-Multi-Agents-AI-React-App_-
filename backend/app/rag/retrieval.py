"""
Retrieval module for querying Pinecone and retrieving relevant document chunks.
"""
from typing import List, Dict, Any, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.embeddings import default_embeddings
from app.rag.indexing import get_pinecone_index

settings = get_settings()
logger = get_logger(__name__)


class Citation:
    """Citation data class for storing retrieved chunk information."""
    
    def __init__(
        self,
        doc_id: str,
        filename: str,
        text: str,
        score: float,
        page_number: Optional[int] = None,
        chunk_index: int = 0
    ):
        self.doc_id = doc_id
        self.filename = filename
        self.text = text
        self.score = score
        self.page_number = page_number
        self.chunk_index = chunk_index
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 4),
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text  # Truncate for response
        }


def retrieve_relevant_chunks(
    query: str,
    top_k: int = None,
    doc_ids: Optional[List[str]] = None
) -> tuple[List[Citation], str]:
    """
    Retrieve relevant document chunks from Pinecone.
    
    Args:
        query: User question
        top_k: Number of chunks to retrieve (default from settings)
        doc_ids: Optional list of document IDs to restrict search
    
    Returns:
        Tuple of (citations_list, context_text)
        - citations_list: List of Citation objects
        - context_text: Concatenated text from all retrieved chunks
    """
    if top_k is None:
        top_k = settings.top_k
    
    logger.info(f"Retrieving top {top_k} chunks for query")
    
    # Step 1: Generate query embedding
    query_embedding = default_embeddings.embed_query(query)
    
    # Step 2: Build filter for specific documents if provided
    filter_dict = None
    if doc_ids:
        filter_dict = {"doc_id": {"$in": doc_ids}}
    
    # Step 3: Query Pinecone
    index = get_pinecone_index()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict,
        namespace=""
    )
    
    # Step 4: Parse results into Citation objects
    citations = []
    context_parts = []
    
    for match in results.matches:
        metadata = match.metadata
        
        citation = Citation(
            doc_id=metadata.get("doc_id", "unknown"),
            filename=metadata.get("filename", "unknown"),
            text=metadata.get("text", ""),
            score=match.score,
            page_number=metadata.get("page_number"),
            chunk_index=metadata.get("chunk_index", 0)
        )
        citations.append(citation)
        
        # Build context string
        source_info = f"[Source: {citation.filename}"
        if citation.page_number:
            source_info += f", Page {citation.page_number}"
        source_info += f", Chunk {citation.chunk_index}]"
        
        context_parts.append(f"{source_info}\n{citation.text}\n")
    
    context_text = "\n---\n".join(context_parts)
    
    logger.info(f"Retrieved {len(citations)} relevant chunks")
    
    return citations, context_text
