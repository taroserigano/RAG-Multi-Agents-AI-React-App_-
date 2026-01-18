"""
Vector Store Module - ChromaDB Integration (DEPRECATED)
Handles document embedding storage and semantic search

NOTE: This module is DEPRECATED and kept for backwards compatibility.
The production system now uses Pinecone for vector storage.
See: indexing.py, retrieval.py, embeddings.py for current implementation.
"""
import os
from typing import List, Optional, Dict, Any
from pathlib import Path

# ChromaDB for vector storage
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    # Suppress warning in production since we use Pinecone
    # print("[WARNING] chromadb not installed. Run: pip install chromadb")

# Sentence Transformers for embeddings
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    # Suppress warning in production since we use OpenAI embeddings
    # print("[WARNING] sentence-transformers not installed. Run: pip install sentence-transformers")
    pass
except Exception as e:
    pass
    # print(f"[WARNING] sentence-transformers load error: {e}")


class EmbeddingService:
    """Service for generating text embeddings using sentence-transformers or ChromaDB default."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_chromadb_embeddings: bool = False):
        """
        Initialize embedding service.
        
        Args:
            model_name: Name of the sentence-transformer model to use.
                        Options: "all-MiniLM-L6-v2" (fast), "all-mpnet-base-v2" (better quality)
            use_chromadb_embeddings: Use ChromaDB's built-in embeddings instead
        """
        self.model_name = model_name
        self._model = None
        self.use_chromadb_embeddings = use_chromadb_embeddings or not SENTENCE_TRANSFORMERS_AVAILABLE
        
        if self.use_chromadb_embeddings:
            print("[INFO] Using ChromaDB's built-in embeddings")
        
    @property
    def model(self):
        """Lazy load the embedding model."""
        if self.use_chromadb_embeddings:
            return None
            
        if self._model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers not installed")
            print(f"[INFO] Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            print(f"[OK] Embedding model loaded")
        return self._model
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if self.use_chromadb_embeddings:
            return None  # ChromaDB will handle it
        return self.model.encode(text).tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if self.use_chromadb_embeddings:
            return None  # ChromaDB will handle it
        return self.model.encode(texts).tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        if self.use_chromadb_embeddings:
            return 384  # Default dimension for ChromaDB's default embeddings
        return self.model.get_sentence_embedding_dimension()


class VectorStore:
    """ChromaDB-based vector store for document chunks."""
    
    def __init__(
        self,
        collection_name: str = "policy_documents",
        persist_directory: str = "./data/chroma_db",
        embedding_service: Optional[EmbeddingService] = None
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_service: Optional custom embedding service
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb not installed. Run: pip install chromadb")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_service = embedding_service or EmbeddingService()
        
        # Ensure persist directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Policy document chunks for RAG"}
        )
        
        print(f"[OK] Vector store initialized: {collection_name}")
        print(f"    Persist directory: {persist_directory}")
        print(f"    Documents in collection: {self.collection.count()}")
    
    def add_document(
        self,
        doc_id: str,
        chunks: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            doc_id: Unique document identifier
            chunks: List of text chunks from the document
            metadata: Optional metadata (filename, page numbers, etc.)
        
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        # Prepare IDs and metadata for each chunk
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "doc_id": doc_id,
                "chunk_index": i,
                "chunk_total": len(chunks),
                "text_length": len(chunk)
            }
            if metadata:
                chunk_metadata.update(metadata)
            metadatas.append(chunk_metadata)
        
        # Generate embeddings if using custom service
        embeddings = self.embedding_service.embed_texts(chunks)
        
        # Add to collection
        if embeddings is not None:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )
        else:
            # Let ChromaDB handle embeddings
            self.collection.add(
                ids=ids,
                documents=chunks,
                metadatas=metadatas
            )
        
        print(f"[OK] Added {len(chunks)} chunks for document: {doc_id}")
        return len(chunks)
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_doc_ids: Optional[List[str]] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks.
        
        Args:
            query: Search query text
            n_results: Maximum number of results to return
            filter_doc_ids: Optional list of document IDs to filter by
            min_score: Minimum similarity score (0-1)
        
        Returns:
            List of results with text, metadata, and score
        """
        # Build where filter if doc_ids provided
        where_filter = None
        if filter_doc_ids:
            where_filter = {"doc_id": {"$in": filter_doc_ids}}
        
        # Search collection - let ChromaDB handle query if using built-in embeddings
        query_embedding = self.embedding_service.embed_text(query)
        
        if query_embedding is not None:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        else:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        
        # Process results
        processed_results = []
        
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                # ChromaDB returns distances, convert to similarity score
                # Distance is L2, so lower is better. Convert to 0-1 score.
                distance = results["distances"][0][i] if results["distances"] else 0
                # Approximate conversion: score = 1 / (1 + distance)
                score = 1 / (1 + distance)
                
                if score >= min_score:
                    processed_results.append({
                        "text": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "score": round(score, 4),
                        "distance": round(distance, 4)
                    })
        
        return processed_results
    
    def delete_document(self, doc_id: str) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            doc_id: Document ID to delete
        
        Returns:
            Number of chunks deleted
        """
        # Get all chunk IDs for this document
        results = self.collection.get(
            where={"doc_id": doc_id},
            include=[]
        )
        
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            print(f"[OK] Deleted {len(results['ids'])} chunks for document: {doc_id}")
            return len(results["ids"])
        
        return 0
    
    def get_document_count(self) -> int:
        """Get total number of chunks in the collection."""
        return self.collection.count()
    
    def get_unique_documents(self) -> List[str]:
        """Get list of unique document IDs in the collection."""
        results = self.collection.get(include=["metadatas"])
        
        if results["metadatas"]:
            doc_ids = set()
            for metadata in results["metadatas"]:
                if "doc_id" in metadata:
                    doc_ids.add(metadata["doc_id"])
            return list(doc_ids)
        
        return []
    
    def reset(self):
        """Reset the collection (delete all data)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Policy document chunks for RAG"}
        )
        print(f"[OK] Collection reset: {self.collection_name}")


# Singleton instance
_vector_store_instance: Optional[VectorStore] = None


def get_vector_store(
    collection_name: str = "policy_documents",
    persist_directory: str = "./data/chroma_db"
) -> VectorStore:
    """Get or create the global vector store instance."""
    global _vector_store_instance
    
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
    
    return _vector_store_instance


# Testing
if __name__ == "__main__":
    print("="*60)
    print("Vector Store Test")
    print("="*60)
    
    # Initialize
    store = get_vector_store(persist_directory="./data/test_chroma")
    
    # Add test document
    test_chunks = [
        "Annual leave policy allows 20 days of paid vacation per year.",
        "Sick leave provides 10 days of paid leave for illness.",
        "Remote work is allowed 2 days per week with manager approval.",
        "Data privacy policy requires encryption of all sensitive data."
    ]
    
    store.add_document(
        doc_id="test-doc-1",
        chunks=test_chunks,
        metadata={"filename": "test_policy.txt", "category": "HR"}
    )
    
    # Search
    print("\nSearching for: 'vacation days'")
    results = store.search("vacation days", n_results=3)
    
    for r in results:
        print(f"  Score: {r['score']:.4f}")
        print(f"  Text: {r['text'][:80]}...")
        print()
    
    print(f"Total chunks in store: {store.get_document_count()}")
    print(f"Unique documents: {store.get_unique_documents()}")
