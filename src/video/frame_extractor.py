"""
Video Frame Extraction using PySceneDetect and OpenCV
"""
import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional
from scenedetect import detect, ContentDetector
import numpy as np

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FrameExtractor:
    """
    Extract key frames from video using scene detection or interval sampling
    """
    
    def __init__(
        self,
        method: str = "scene",
        interval: int = 1,
        max_frames: int = 1000
    ):
        """
        Initialize frame extractor
        
        Args:
            method: Extraction method ('scene' or 'interval')
            interval: Interval in seconds for interval method
            max_frames: Maximum number of frames to extract
        """
        self.method = method
        self.interval = interval
        self.max_frames = max_frames
        
        logger.info(
            f"FrameExtractor initialized: method={method}, "
            f"interval={interval}s"
        )
    
    def extract_frames(
        self,
        video_path: str,
        output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract frames from video
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save extracted frames
            
        Returns:
            List of frame data dictionaries
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Set output directory
        if output_dir is None:
            output_dir = settings.FRAMES_DIR / video_path.stem
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Extracting frames from: {video_path}")
        logger.info(f"Output directory: {output_dir}")
        
        # Choose extraction method
        if self.method == "scene":
            frames = self._extract_by_scene(video_path, output_dir)
        elif self.method == "interval":
            frames = self._extract_by_interval(video_path, output_dir)
        else:
            raise ValueError(f"Unknown extraction method: {self.method}")
        
        logger.info(f"Extracted {len(frames)} frames")
        return frames
    
    def _extract_by_scene(
        self,
        video_path: Path,
        output_dir: Path
    ) -> List[Dict[str, Any]]:
        """Extract frames using scene detection"""
        logger.info("Using scene detection method...")
        
        # Detect scenes
        try:
            scenes = detect(
                str(video_path),
                ContentDetector(threshold=settings.SCENE_THRESHOLD)
            )
            logger.info(f"Detected {len(scenes)} scenes")
            
            # If no scenes detected, fall back to interval method
            if not scenes:
                logger.warning(
                    "No scenes detected. Falling back to interval method."
                )
                return self._extract_by_interval(video_path, output_dir)
        except Exception as e:
            logger.warning(
                f"Scene detection failed: {e}. "
                f"Falling back to interval method."
            )
            return self._extract_by_interval(video_path, output_dir)
        
        # Extract middle frame from each scene
        frames_data = []
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        for idx, scene in enumerate(scenes):
            if idx >= self.max_frames:
                logger.warning(f"Reached max frames limit: {self.max_frames}")
                break
            
            # Get middle frame of scene
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()
            middle_frame = (start_frame + end_frame) // 2
            
            # Extract frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            ret, frame = cap.read()
            
            if ret:
                # Save frame
                timestamp = self._frame_to_timestamp(middle_frame, fps)
                frame_filename = f"frame_{idx:05d}_{timestamp.replace(':', '-')}.jpg"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                
                frames_data.append({
                    "frame_id": idx,
                    "timestamp": timestamp,
                    "frame_path": str(frame_path),
                    "frame_number": middle_frame,
                    "telemetry": self._generate_telemetry(timestamp, idx)
                })
        
        cap.release()
        return frames_data
    
    def _extract_by_interval(
        self,
        video_path: Path,
        output_dir: Path
    ) -> List[Dict[str, Any]]:
        """Extract frames at regular intervals"""
        logger.info(f"Using interval method: {self.interval}s")
        
        frames_data = []
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval
        frame_interval = int(fps * self.interval)
        
        idx = 0
        current_frame = 0
        
        while current_frame < total_frames and idx < self.max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            
            if ret:
                # Save frame
                timestamp = self._frame_to_timestamp(current_frame, fps)
                frame_filename = f"frame_{idx:05d}_{timestamp.replace(':', '-')}.jpg"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                
                frames_data.append({
                    "frame_id": idx,
                    "timestamp": timestamp,
                    "frame_path": str(frame_path),
                    "frame_number": current_frame,
                    "telemetry": self._generate_telemetry(timestamp, idx)
                })
                
                idx += 1
            
            current_frame += frame_interval
        
        cap.release()
        return frames_data
    
    def _frame_to_timestamp(self, frame_number: int, fps: float) -> str:
        """Convert frame number to timestamp string"""
        total_seconds = frame_number / fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _generate_telemetry(self, timestamp: str, frame_id: int) -> Dict[str, Any]:
        """
        Generate simulated telemetry data
        In production, this would come from actual drone sensors
        """
        # Simulate realistic telemetry
        base_altitude = 50.0
        
        # Parse timestamp for time-based variations
        time_parts = timestamp.split(':')
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        
        return {
            "location": self._get_location_from_time(hours, minutes),
            "altitude": base_altitude + np.random.uniform(-5, 5),
            "gps": {
                "latitude": 37.7749 + np.random.uniform(-0.01, 0.01),
                "longitude": -122.4194 + np.random.uniform(-0.01, 0.01)
            },
            "battery": max(20, 100 - (frame_id * 0.5)),
            "speed": np.random.uniform(0, 5),  # m/s
            "heading": np.random.uniform(0, 360),  # degrees
            "temperature": 20 + np.random.uniform(-5, 10),
            "time_of_day": "night" if hours < 6 or hours > 20 else "day"
        }
    
    def _get_location_from_time(self, hours: int, minutes: int) -> str:
        """Simulate location based on time (patrol route)"""
        locations = [
            "Main Gate",
            "North Perimeter",
            "East Wing",
            "Parking Lot",
            "South Entrance",
            "West Building",
            "Central Courtyard",
            "Rear Exit"
        ]
        
        # Cycle through locations
        location_index = (hours * 60 + minutes) // 15 % len(locations)
        return locations[location_index]
