"""
Gemini Vision Model for Frame Description Generation
Uses LLM wrapper with base64 image encoding
"""

from typing import Union, List
from PIL import Image
import numpy as np

from src.models.llm_model import LLMWrapper
from src.prompts import IMAGE_DESCRIPTION_PROMPT
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiVisionDescriber:
    """Gemini Vision model wrapper using LLM with base64 images"""

    def __init__(self):
        """Initialize Gemini Vision using LLM wrapper"""
        logger.info("Initializing Gemini Vision with LLM wrapper")

        # Use the existing LLM wrapper
        self.llm = LLMWrapper()

        logger.info("Gemini Vision model loaded successfully")

    def _prepare_image(
        self, image: Union[Image.Image, np.ndarray, str]
    ) -> Image.Image:
        """
        Prepare image for processing

        Args:
            image: PIL Image, numpy array, or path to image

        Returns:
            PIL Image in RGB format
        """
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def describe_frame(
        self,
        image: Union[Image.Image, np.ndarray, str],
        prompt: str = IMAGE_DESCRIPTION_PROMPT,
    ) -> str:
        """
        Generate description for a single frame using Gemini Vision

        Args:
            image: PIL Image, numpy array, or path to image
            prompt: Prompt to guide description

        Returns:
            Description string
        """
        try:
            # Prepare image
            pil_image = self._prepare_image(image)

            # Use LLM wrapper to describe image (with base64 encoding)
            description = self.llm.describe_image(pil_image, prompt)

            return description

        except Exception as e:
            logger.error(f"Error in describe_frame: {e}")
            return f"Error describing frame: {str(e)}"

    def describe_frames_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, str]],
        batch_size: int = 1
    ) -> List[str]:
        """
        Generate descriptions for multiple frames

        Note: Gemini API processes images individually
        batch_size is kept for API compatibility but processed sequentially

        Args:
            images: List of images
            batch_size: Kept for compatibility (processed sequentially)

        Returns:
            List of descriptions
        """
        descriptions = []

        for i, img in enumerate(images):
            description = self.describe_frame(img)
            descriptions.append(description)

            if (i + 1) % 5 == 0:
                logger.info(
                    f"Processed {i + 1}/{len(images)} frames"
                )

        return descriptions
