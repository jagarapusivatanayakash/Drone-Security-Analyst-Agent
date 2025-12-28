"""
LangGraph State Definition for Drone Security Agent
"""
from typing import TypedDict, List, Dict, Any


class VideoFrameData(TypedDict):
    """Single frame data structure"""
    frame_id: int
    timestamp: str
    frame_path: str
    telemetry: Dict[str, Any]


class ProcessingState(TypedDict):
    """
    State for the LangGraph workflow
    This tracks the complete processing pipeline
    """
    # Input
    video_path: str
    frames: List[VideoFrameData]
    current_frame_idx: int

    # Gemini Vision outputs (parallel node 1)
    descriptions: List[str]

    # CLIP outputs (parallel node 2)
    embeddings: List[List[float]]

    # Vector DB results
    stored_frame_ids: List[str]
    similar_frames: List[Dict[str, Any]]

    # Agent analysis
    context: str
    past_n_frames: List[Dict[str, Any]]
    analysis_results: List[Dict[str, Any]]

    # Alerts (no operator.add - we manage appending manually)
    alerts: List[Dict[str, Any]]
    alerted_frames: List[str]

    # Summary
    video_summary: str

    # Metadata
    processed_frames_count: int
    errors: List[str]
