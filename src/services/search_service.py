"""
Search Service
Handles semantic search functionality for video frames
"""

from datetime import datetime
from typing import List, Dict, Optional
from src.models.clip_model import CLIPEmbedder
from src.database.vector_store import VectorStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SearchService:
    """Service for searching video frames using semantic search"""

    def __init__(self):
        """Initialize the search service"""
        self.clip = CLIPEmbedder()
        self.vector_store = VectorStore()
        logger.info("Search service initialized")

    def is_available(self) -> bool:
        """
        Check if search service is available
        
        Returns:
            True if vector store is enabled, False otherwise
        """
        return self.vector_store.enabled

    def search_frames(
        self, 
        query: str, 
        top_k: int = 10
    ) -> Dict[str, any]:
        """
        Search for frames matching a text description
        
        Args:
            query: Text description to search for
            top_k: Number of results to return (default: 10)
            
        Returns:
            Dictionary containing search results and metadata
            
        Raises:
            ValueError: If vector store is not enabled
            Exception: If search operation fails
        """
        start_time = datetime.now()
        
        logger.info(f"Search request: '{query}' (top_k={top_k})")
        
        if not self.vector_store.enabled:
            raise ValueError(
                "Vector store is not enabled. "
                "Please configure PINECONE_API_KEY."
            )
        
        try:
            # Convert text to embedding
            query_embedding = self.clip.encode_text(query)
            
            # Search for similar frames
            results = self.vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            if not results:
                return {
                    "query": query,
                    "total_results": 0,
                    "results": [],
                    "processing_time_ms": 0.0
                }
            
            # Format results
            formatted_results = []
            for result in results:
                telemetry = result.get('telemetry', {})
                location = telemetry.get('location', 'N/A') if telemetry else 'N/A'
                
                formatted_results.append({
                    "score": result['score'],
                    "timestamp": result['timestamp'],
                    "description": result['description'],
                    "frame_path": result['frame_path'],
                    "location": location
                })
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"Search completed: {len(formatted_results)} results in {processing_time:.2f}ms")
            
            return {
                "query": query,
                "total_results": len(formatted_results),
                "results": formatted_results,
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            raise
