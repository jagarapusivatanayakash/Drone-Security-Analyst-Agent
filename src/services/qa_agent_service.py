"""
Q&A Agent Service
Handles follow-up questions about processed videos using frame descriptions and vector search
"""

from datetime import datetime
from typing import List, Dict, Optional, Any
from src.models.llm_model import LLMWrapper
from src.models.clip_model import CLIPEmbedder
from src.database.vector_store import VectorStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QAAgentService:
    """Service for answering follow-up questions about processed videos"""

    def __init__(self):
        """Initialize the Q&A agent service"""
        self.llm = LLMWrapper()
        self.clip = CLIPEmbedder()
        self.vector_store = VectorStore()
        logger.info("Q&A Agent service initialized")

    def answer_question(
        self,
        question: str,
        video_context: Dict[str, Any],
        use_vector_search: bool = True,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Answer a follow-up question about a processed video
        
        Args:
            question: The user's question
            video_context: Context from the processed video including:
                - descriptions: List of all frame descriptions
                - frames: List of frame metadata (timestamps, paths, etc.)
                - summary: Overall video summary
                - alerts: List of alerts generated
            use_vector_search: Whether to use vector search for relevant frames
            top_k: Number of relevant frames to retrieve if using vector search
            
        Returns:
            Dictionary containing the answer and supporting context
        """
        start_time = datetime.now()
        
        logger.info(f"Q&A request: '{question}'")
        
        try:
            # Extract context
            all_descriptions = video_context.get("descriptions", [])
            all_frames = video_context.get("frames", [])
            video_summary = video_context.get("summary", "")
            alerts = video_context.get("alerts", [])
            
            # Decide retrieval strategy
            relevant_frames = []
            retrieval_method = "all_frames"
            
            if use_vector_search and self.vector_store.enabled and len(all_descriptions) > 20:
                # Use vector search for large videos
                logger.info("Using vector search to find relevant frames")
                retrieval_method = "vector_search"
                
                try:
                    # Convert question to embedding
                    query_embedding = self.clip.encode_text(question)
                    
                    # Search for relevant frames
                    search_results = self.vector_store.search_similar(
                        query_embedding=query_embedding,
                        top_k=min(top_k, len(all_descriptions))
                    )
                    
                    # Format relevant frames
                    for result in search_results:
                        relevant_frames.append({
                            "timestamp": result.get("timestamp", "N/A"),
                            "description": result.get("description", ""),
                            "score": result.get("score", 0.0)
                        })
                    
                    logger.info(f"Found {len(relevant_frames)} relevant frames via vector search")
                    
                except Exception as e:
                    logger.warning(f"Vector search failed, falling back to all frames: {e}")
                    retrieval_method = "all_frames_fallback"
            
            # If vector search not used or failed, use all frame descriptions
            if not relevant_frames:
                logger.info("Using all frame descriptions")
                retrieval_method = "all_frames"
                for i, desc in enumerate(all_descriptions):
                    if i < len(all_frames):
                        relevant_frames.append({
                            "timestamp": all_frames[i].get("timestamp", "N/A"),
                            "description": desc,
                            "score": 1.0
                        })
            
            # Generate answer using LLM
            answer = self._generate_answer(
                question=question,
                relevant_frames=relevant_frames,
                video_summary=video_summary,
                alerts=alerts
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"Q&A completed in {processing_time:.2f}ms")
            
            return {
                "question": question,
                "answer": answer,
                "retrieval_method": retrieval_method,
                "num_frames_used": len(relevant_frames),
                "processing_time_ms": processing_time,
                "supporting_frames": relevant_frames[:5]  # Return top 5 for reference
            }
            
        except Exception as e:
            logger.error(f"Q&A error: {e}", exc_info=True)
            raise

    def _generate_answer(
        self,
        question: str,
        relevant_frames: List[Dict[str, Any]],
        video_summary: str,
        alerts: List[Dict[str, Any]]
    ) -> str:
        """
        Generate answer using LLM with context
        
        Args:
            question: User's question
            relevant_frames: List of relevant frame descriptions
            video_summary: Overall video summary
            alerts: List of security alerts
            
        Returns:
            Generated answer
        """
        # Build context for LLM
        context_parts = []
        
        # Add video summary if available
        if video_summary and video_summary.strip():
            context_parts.append(f"VIDEO SUMMARY:\n{video_summary}\n")
        
        # Add alerts if any
        if alerts:
            alert_text = "\n".join([
                f"- [{alert.get('severity', 'UNKNOWN')}] {alert.get('timestamp', 'N/A')}: {alert.get('message', 'N/A')}"
                for alert in alerts[:10]  # Limit to top 10 alerts
            ])
            context_parts.append(f"SECURITY ALERTS:\n{alert_text}\n")
        
        # Add relevant frame descriptions
        if relevant_frames:
            frames_text = "\n".join([
                f"- Time {frame['timestamp']}: {frame['description']}"
                for frame in relevant_frames[:30]  # Limit to top 30 frames
            ])
            context_parts.append(f"RELEVANT FRAMES:\n{frames_text}\n")
        
        context = "\n".join(context_parts)
        
        # Create prompt for LLM
        system_prompt = """You are an AI assistant helping analyze surveillance video footage. 
You have access to frame-by-frame descriptions, security alerts, and video summaries.

Your task is to answer questions about the video content based on the provided context.
Be specific, cite timestamps when relevant, and provide comprehensive answers.

If the information needed to answer the question is not in the context, say so clearly."""

        user_prompt = f"""Based on the following context about a surveillance video, please answer this question:

QUESTION: {question}

CONTEXT:
{context}

Please provide a clear, comprehensive answer based on the available information."""

        try:
            # Use LLM to generate answer
            response = self.llm.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            answer = response.content.strip()
            logger.debug(f"Generated answer: {answer[:200]}...")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}", exc_info=True)
            return f"Error generating answer: {str(e)}"

    def is_available(self) -> bool:
        """
        Check if Q&A service is available
        
        Returns:
            True if LLM is configured, False otherwise
        """
        try:
            return self.llm is not None
        except Exception:
            return False
