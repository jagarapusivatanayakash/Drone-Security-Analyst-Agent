"""
Video Processing Service
Handles video processing business logic
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict
from src.config.settings import settings
from src.video.frame_extractor import FrameExtractor
from src.workflows.workflow import DroneSecurityWorkflow
from src.state import ProcessingState
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VideoProcessingService:
    """Service for processing surveillance videos"""

    def __init__(self):
        """Initialize the video processing service"""
        logger.info("=" * 80)
        logger.info("VIDEO PROCESSING SERVICE - INITIALIZING")
        logger.info("=" * 80)

        # Ensure directories exist
        settings.ensure_directories()

        # Initialize components
        self.frame_extractor = FrameExtractor(
            method=settings.FRAME_EXTRACTION_METHOD,
            interval=settings.FRAME_INTERVAL,
            max_frames=settings.MAX_FRAMES_PER_VIDEO,
        )

        self.workflow = DroneSecurityWorkflow()

        logger.info("Video processing service initialized successfully")

    def process_video(self, video_path: str) -> Dict:
        """
        Process a surveillance video

        Args:
            video_path: Path to video file

        Returns:
            Processing results dictionary
        """
        logger.info("=" * 80)
        logger.info(f"PROCESSING VIDEO: {video_path}")
        logger.info("=" * 80)

        start_time = datetime.now()

        try:
            # Step 1: Extract frames
            logger.info("Step 1: Extracting frames...")
            frames = self.frame_extractor.extract_frames(video_path)

            if not frames:
                raise ValueError("No frames extracted from video")

            logger.info(f"Extracted {len(frames)} frames")

            # Step 2: Prepare initial state
            initial_state: ProcessingState = {
                "video_path": video_path,
                "frames": frames,
                "current_frame_idx": 0,
                "descriptions": [],
                "embeddings": [],
                "stored_frame_ids": [],
                "similar_frames": [],
                "context": "",
                "past_n_frames": [],
                "analysis_results": [],
                "alerts": [],
                "alerted_frames": [],
                "video_summary": "",
                "processed_frames_count": 0,
                "errors": [],
            }

            # Step 3: Run workflow
            logger.info("Step 2: Running LangGraph workflow...")
            final_state = self.workflow.run(initial_state)

            # Step 4: Generate results
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # Build captions with timestamps
            captions = []
            for i, frame in enumerate(final_state["frames"]):
                if i < len(final_state.get("descriptions", [])):
                    captions.append({
                        "frame_number": i,
                        "timestamp": frame["timestamp"],
                        "description": final_state["descriptions"][i]
                    })

            results = {
                "video_path": video_path,
                "processing_time_seconds": processing_time,
                "total_frames": len(frames),
                "processed_frames": final_state["processed_frames_count"],
                "total_alerts": len(final_state["alerts"]),
                "alerts": final_state["alerts"],
                "summary": final_state["video_summary"],
                "captions": captions,  # Frame descriptions with timestamps
                "errors": final_state["errors"],
                "timestamp": datetime.now().isoformat(),
            }

            # Step 5: Save results
            self._save_results(results, video_path)

            logger.info("=" * 80)
            logger.info("PROCESSING COMPLETE")
            logger.info("=" * 80)

            return results

        except Exception as e:
            logger.error(f"Error processing video: {e}", exc_info=True)
            raise

    def _save_results(self, results: Dict, video_path: str):
        """
        Save processing results to JSON file
        
        Args:
            results: Processing results dictionary
            video_path: Path to the processed video
        """
        video_stem = Path(video_path).stem
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = (
            settings.OUTPUTS_DIR /
            f"results_{video_stem}_{timestamp_str}.json"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {output_file}")
