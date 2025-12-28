"""
Validation Service
Handles file validation logic for video uploads
"""

from pathlib import Path
from fastapi import HTTPException, status
from src.utils.logger import get_logger

logger = get_logger(__name__)

# File upload validation constants
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
ALLOWED_MIME_TYPES = {
    'video/mp4',
    'video/x-msvideo',
    'video/quicktime',
    'video/x-matroska',
    'video/x-flv',
    'video/x-ms-wmv'
}


def validate_video_file(filename: str, content_type: str, file_size: int):
    """
    Validate uploaded video file
    
    Args:
        filename: Name of the uploaded file
        content_type: MIME type of the file
        file_size: Size of the file in bytes
        
    Raises:
        HTTPException: If validation fails
    """
    # Check file extension
    file_ext = Path(filename).suffix.lower()
    if file_ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension. Allowed: "
                   f"{', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
        )
    
    # Check MIME type
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type: {content_type}. "
                   f"Must be a video file."
        )
    
    # Check file size
    if file_size > MAX_FILE_SIZE:
        max_size_mb = MAX_FILE_SIZE / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size "
                   f"of {max_size_mb}MB"
        )
    
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded"
        )
    
    logger.info(f"File validation passed: {filename} ({file_size} bytes)")
