"""
Retrieval module for querying Pinecone and retrieving relevant document chunks.
Supports semantic search, hybrid search (keyword + semantic), and advanced options.
Includes Redis caching for improved performance.
"""
from typing import List, Dict, Any, Optional
import re
from collections import defaultdict

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.cache import get_cache
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
    doc_ids: Optional[List[str]] = None,
    use_hybrid: bool = False
) -> tuple[List[Citation], str]:
    """
    Retrieve relevant document chunks from Pinecone.
    Uses Redis caching for improved performance.
    
    Args:
        query: User question
        top_k: Number of chunks to retrieve (default from settings)
        doc_ids: Optional list of document IDs to restrict search
        use_hybrid: Enable hybrid search (semantic + keyword matching)
    
    Returns:
        Tuple of (citations_list, context_text)
        - citations_list: List of Citation objects
        - context_text: Concatenated text from all retrieved chunks
    """
    if top_k is None:
        top_k = settings.top_k
    
    # Try cache first
    cache = get_cache()
    cache_options = {"hybrid": use_hybrid}
    cached = cache.get_retrieval(query, top_k, doc_ids, cache_options)
    if cached is not None:
        logger.info(f"Retrieval cache hit for query")
        # Reconstruct Citation objects from cached data
        citations = [
            Citation(
                doc_id=c["doc_id"],
                filename=c["filename"],
                text=c["text"],
                score=c["score"],
                page_number=c.get("page_number"),
                chunk_index=c.get("chunk_index", 0)
            )
            for c in cached["citations"]
        ]
        return citations, cached["context"]
    
    logger.info(f"Retrieving top {top_k} chunks for query (hybrid={use_hybrid})")
    
    # Step 1: Generate query embedding
    query_embedding = default_embeddings.embed_query(query)
    
    # Step 2: Build filter for specific documents if provided
    filter_dict = None
    if doc_ids:
        filter_dict = {"doc_id": {"$in": doc_ids}}
    
    # Step 3: Query Pinecone (get more results for hybrid search)
    index = get_pinecone_index()
    fetch_k = top_k * 2 if use_hybrid else top_k
    
    results = index.query(
        vector=query_embedding,
        top_k=fetch_k,
        include_metadata=True,
        filter=filter_dict,
        namespace=""
    )
    
    # Step 4: Apply hybrid search scoring if enabled
    if use_hybrid and results.matches:
        results.matches = _apply_hybrid_scoring(query, results.matches)
        # Re-sort by combined score and take top_k
        results.matches = results.matches[:top_k]
    
    # Step 5: Parse results into Citation objects
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
    
    # Cache the results
    cache.set_retrieval(
        query, top_k,
        {
            "citations": [c.to_dict() for c in citations],
            "context": context_text
        },
        doc_ids, cache_options
    )
    
    return citations, context_text


def _apply_hybrid_scoring(query: str, matches: List) -> List:
    """
    Apply hybrid scoring by combining semantic similarity with keyword matching.
    
    Args:
        query: User question
        matches: List of Pinecone matches
    
    Returns:
        Matches with updated scores, sorted by combined score
    """
    # Extract query keywords (lowercase, remove common stop words)
    stop_words = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'what', 'which', 'who', 'when', 'where', 'why', 'how',
        'and', 'or', 'but', 'if', 'for', 'of', 'to', 'from', 'in', 'on', 'at', 'by'
    }
    
    query_words = set(word.lower() for word in re.findall(r'\b\w+\b', query) 
                      if word.lower() not in stop_words and len(word) > 2)
    
    if not query_words:
        return matches
    
    scored_matches = []
    for match in matches:
        text = match.metadata.get('text', '').lower()
        text_words = set(re.findall(r'\b\w+\b', text))
        
        # Calculate keyword overlap score
        keyword_overlap = len(query_words & text_words)
        keyword_score = keyword_overlap / len(query_words) if query_words else 0
        
        # Boost for exact phrase matches
        phrase_boost = 0.1 if query.lower() in text else 0
        
        # Combine scores: 70% semantic, 25% keyword, 5% phrase boost
        semantic_score = match.score
        combined_score = (semantic_score * 0.70) + (keyword_score * 0.25) + phrase_boost
        
        # Store combined score (update the match object)
        match.score = combined_score
        match._semantic_score = semantic_score  # Keep original for debugging
        match._keyword_score = keyword_score
        scored_matches.append(match)
    
    # Sort by combined score descending
    scored_matches.sort(key=lambda x: x.score, reverse=True)
    
    return scored_matches


def retrieve_with_multi_query(
    queries: List[str],
    top_k: int = 5,
    doc_ids: Optional[List[str]] = None,
    use_hybrid: bool = False
) -> tuple[List[Citation], str]:
    """
    Retrieve chunks using multiple query variations.
    
    Combines results from multiple queries for better coverage.
    
    Args:
        queries: List of query variations
        top_k: Number of chunks to return (final)
        doc_ids: Optional document filter
        use_hybrid: Enable hybrid scoring
    
    Returns:
        Tuple of (citations_list, context_text)
    """
    logger.info(f"Multi-query retrieval with {len(queries)} queries")
    
    # Collect results from all queries
    all_results = defaultdict(lambda: {'citation': None, 'score': 0, 'count': 0})
    
    per_query_k = max(3, top_k // len(queries) + 1)
    
    for query in queries:
        citations, _ = retrieve_relevant_chunks(
            query=query,
            top_k=per_query_k,
            doc_ids=doc_ids,
            use_hybrid=use_hybrid
        )
        
        for citation in citations:
            key = f"{citation.doc_id}_{citation.chunk_index}"
            if all_results[key]['citation'] is None:
                all_results[key]['citation'] = citation
            # Accumulate scores and count occurrences
            all_results[key]['score'] += citation.score
            all_results[key]['count'] += 1
    
    # Calculate final scores (average + frequency bonus)
    final_results = []
    for key, data in all_results.items():
        citation = data['citation']
        avg_score = data['score'] / data['count']
        freq_bonus = min(0.1 * (data['count'] - 1), 0.2)  # Max 0.2 bonus
        citation.score = avg_score + freq_bonus
        final_results.append(citation)
    
    # Sort by final score and take top_k
    final_results.sort(key=lambda x: x.score, reverse=True)
    final_citations = final_results[:top_k]
    
    # Build context
    context_parts = []
    for citation in final_citations:
        source_info = f"[Source: {citation.filename}"
        if citation.page_number:
            source_info += f", Page {citation.page_number}"
        source_info += f", Chunk {citation.chunk_index}]"
        context_parts.append(f"{source_info}\n{citation.text}\n")
    
    context_text = "\n---\n".join(context_parts)
    
    logger.info(f"Multi-query returned {len(final_citations)} unique chunks")
    
    return final_citations, context_text


def get_document_chunks(doc_id: str) -> List[Dict[str, Any]]:
    """
    Get all chunks for a specific document from Pinecone.
    
    Args:
        doc_id: Document UUID
    
    Returns:
        List of chunks with text and metadata, ordered by chunk_index
    """
    logger.info(f"Fetching all chunks for document {doc_id}")
    
    try:
        index = get_pinecone_index()
        
        # Query Pinecone with filter for this document
        # We use a dummy vector since we want all chunks, not similar ones
        # Create a zero vector of the right dimension
        dummy_vector = [0.0] * settings.embed_dim
        
        # Query with high top_k to get all chunks
        results = index.query(
            vector=dummy_vector,
            top_k=1000,  # High number to get all chunks
            include_metadata=True,
            filter={"doc_id": {"$eq": doc_id}},
            namespace=""
        )
        
        if not results.matches:
            logger.warning(f"No chunks found for document {doc_id}")
            return []
        
        # Extract chunks and sort by chunk_index
        chunks = []
        for match in results.matches:
            metadata = match.metadata
            chunks.append({
                "text": metadata.get("text", ""),
                "chunk_index": metadata.get("chunk_index", 0),
                "page_number": metadata.get("page_number"),
                "filename": metadata.get("filename", "")
            })
        
        # Sort by chunk_index to get original document order
        chunks.sort(key=lambda x: x["chunk_index"])
        
        logger.info(f"Retrieved {len(chunks)} chunks for document {doc_id}")
        return chunks
        
    except Exception as e:
        logger.error(f"Error fetching document chunks: {e}")
        raise