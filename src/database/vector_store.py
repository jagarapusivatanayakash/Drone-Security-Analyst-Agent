"""
Vector Database Implementation using Pinecone
"""

from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import uuid

from pinecone import Pinecone, ServerlessSpec

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Pinecone vector store interface"""

    def __init__(self):
        """Initialize Pinecone vector store"""
        logger.info("Initializing Pinecone vector store")

        # Check if API key is available
        if not settings.PINECONE_API_KEY:
            logger.warning(
                "PINECONE_API_KEY not found. Vector store will run in "
                "disabled mode. Set PINECONE_API_KEY in .env to enable."
            )
            self.enabled = False
            self.pc = None
            self.index = None
            return

        try:
            # Initialize Pinecone
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)

            # Index name
            self.index_name = "drone-frames"

            # Create index if it doesn't exist
            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=settings.EMBEDDING_DIM,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                logger.info(f"Created Pinecone index: {self.index_name}")

            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")

            self.enabled = True
            logger.info("Pinecone initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            logger.warning("Vector store will run in disabled mode")
            self.enabled = False
            self.pc = None
            self.index = None

    def delete_frames_by_video(self, video_name: str) -> int:
        """
        Delete all frames for a specific video from Pinecone

        Args:
            video_name: Name of the video (e.g., 'stealing002')

        Returns:
            Number of frames deleted
        """
        if not self.enabled:
            logger.debug("Vector store disabled, skipping delete")
            return 0

        try:
            logger.info(f"Deleting old frames for video: {video_name}")
            
            # Query all vectors with video_name metadata
            dummy_vector = [0.0] * settings.EMBEDDING_DIM
            results = self.index.query(
                vector=dummy_vector,
                top_k=10000,
                include_metadata=True,
                filter={"video_name": {"$eq": video_name}}
            )
            
            # Extract IDs to delete
            ids_to_delete = [match.id for match in results.matches]
            
            if ids_to_delete:
                # Delete in batches (Pinecone limit is 1000 per batch)
                batch_size = 1000
                for i in range(0, len(ids_to_delete), batch_size):
                    batch = ids_to_delete[i:i + batch_size]
                    self.index.delete(ids=batch)
                
                logger.info(
                    f"Deleted {len(ids_to_delete)} old frames "
                    f"for video: {video_name}"
                )
                return len(ids_to_delete)
            else:
                logger.info(f"No existing frames for video: {video_name}")
                return 0

        except Exception as e:
            logger.error(f"Error deleting frames for {video_name}: {e}")
            # Don't raise - allow processing to continue
            return 0

    def insert_frame(
        self,
        embedding: List[float],
        timestamp: str,
        description: str,
        telemetry: Dict[str, Any],
        frame_path: str,
        video_name: Optional[str] = None,
    ) -> str:
        """
        Insert frame into Pinecone with deterministic ID

        Args:
            embedding: Frame embedding vector
            timestamp: Frame timestamp
            description: Frame description
            telemetry: Telemetry data
            frame_path: Path to frame image
            video_name: Video name (extracted from path if not provided)

        Returns:
            Frame ID
        """
        if not self.enabled:
            logger.debug("Vector store disabled, skipping frame insert")
            return f"frame_{uuid.uuid4().hex[:16]}"

        try:
            # Extract video name from frame_path if not provided
            if video_name is None:
                # frame_path: data/frames/stealing002/frame_00015_xx.jpg
                path_parts = frame_path.replace("\\", "/").split("/")
                if len(path_parts) > 2:
                    video_name = path_parts[2]
                else:
                    video_name = "unknown"
            
            # Generate deterministic ID: video_timestamp
            # This ensures same frame gets same ID (upsert updates)
            clean_timestamp = timestamp.replace(":", "-")
            frame_id = f"{video_name}_{clean_timestamp}"

            # Prepare metadata with video_name for filtering
            metadata = {
                "timestamp": timestamp,
                "description": description,
                "telemetry": json.dumps(telemetry),
                "frame_path": frame_path,
                "video_name": video_name,
                "created_at": datetime.now().isoformat(),
            }

            # Upsert to Pinecone (updates if ID exists)
            self.index.upsert(
                vectors=[
                    {"id": frame_id, "values": embedding, "metadata": metadata}
                ]
            )

            logger.debug(f"Inserted frame: {frame_id}")
            return frame_id

        except Exception as e:
            logger.error(f"Error inserting frame: {e}")
            raise

    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        time_filter: Optional[str] = None,
        exclude_recent_n: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar frames

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            time_filter: Optional time filter expression (not used in basic Pinecone)
            exclude_recent_n: Exclude N most recent frames (not implemented)

        Returns:
            List of similar frames with metadata
        """
        if not self.enabled:
            logger.debug("Vector store disabled, returning empty results")
            return []

        try:
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding, top_k=top_k, include_metadata=True
            )

            # Format results
            similar_frames = []
            for match in results.matches:
                metadata = match.metadata
                similar_frames.append(
                    {
                        "id": match.id,
                        "description": metadata.get("description", ""),
                        "timestamp": metadata.get("timestamp", ""),
                        "score": match.score,
                        "telemetry": json.loads(metadata.get("telemetry", "{}")),
                        "frame_path": metadata.get("frame_path", ""),
                    }
                )

            return similar_frames

        except Exception as e:
            logger.error(f"Error searching frames: {e}")
            return []

    def search_by_text(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search frames by text query (requires CLIP text encoding)

        Args:
            query_text: Text query
            top_k: Number of results

        Returns:
            List of matching frames
        """
        if not self.enabled:
            logger.debug("Vector store disabled, returning empty results")
            return []

        from src.models.clip_model import CLIPEmbedder

        clip = CLIPEmbedder()
        query_embedding = clip.encode_text(query_text)

        return self.search_similar(query_embedding, top_k)

    def get_all_frames(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all frames (for summary/statistics)
        Note: Pinecone doesn't have a direct "get all" method,
        so this uses a dummy query with high top_k
        """
        if not self.enabled:
            logger.debug("Vector store disabled, returning empty results")
            return []

        try:
            # Use a zero vector to get any matches
            dummy_vector = [0.0] * settings.EMBEDDING_DIM
            results = self.index.query(
                vector=dummy_vector,
                top_k=min(limit, 10000),  # Pinecone has limits
                include_metadata=True,
            )

            frames = []
            for match in results.matches:
                metadata = match.metadata
                frames.append(
                    {
                        "id": match.id,
                        "description": metadata.get("description", ""),
                        "timestamp": metadata.get("timestamp", ""),
                    }
                )

            return frames

        except Exception as e:
            logger.error(f"Error getting all frames: {e}")
            return []

    def clear_all(self):
        """Clear all data (for testing)"""
        try:
            # Delete all vectors by deleting and recreating the index
            self.pc.delete_index(self.index_name)

            # Recreate index
            self.pc.create_index(
                name=self.index_name,
                dimension=settings.EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

            # Reconnect to index
            self.index = self.pc.Index(self.index_name)

            logger.warning("Pinecone vector store cleared")

        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise
