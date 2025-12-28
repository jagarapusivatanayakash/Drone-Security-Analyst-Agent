"""
Prompts package for LLM interactions
"""

from .security_analysis import (
    SECURITY_ANALYSIS_SYSTEM_PROMPT,
    SECURITY_ANALYSIS_USER_PROMPT_TEMPLATE,
)
from .video_summary import VIDEO_SUMMARY_PROMPT_TEMPLATE
from .image_description import IMAGE_DESCRIPTION_PROMPT

__all__ = [
    "SECURITY_ANALYSIS_SYSTEM_PROMPT",
    "SECURITY_ANALYSIS_USER_PROMPT_TEMPLATE",
    "VIDEO_SUMMARY_PROMPT_TEMPLATE",
    "IMAGE_DESCRIPTION_PROMPT",
]
