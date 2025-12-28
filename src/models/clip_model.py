"""
CLIP Model for Embedding Generation
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import Union, List
import numpy as np

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CLIPEmbedder:
    """CLIP model wrapper for generating embeddings"""

    def __init__(self, device: str = None):
        """
        Initialize CLIP model

        Args:
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing CLIP model on {self.device}")

        self.processor = CLIPProcessor.from_pretrained(
            settings.CLIP_MODEL_NAME
        )
        self.model = CLIPModel.from_pretrained(
            settings.CLIP_MODEL_NAME
        ).to(self.device)

        logger.info("CLIP model loaded successfully")

    def encode_image(
        self, image: Union[Image.Image, np.ndarray, str]
    ) -> List[float]:
        """
        Generate embedding for a single image

        Args:
            image: PIL Image, numpy array, or path to image

        Returns:
            Embedding vector as list
        """
        try:
            # Handle different input types
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert("RGB")

            # Process with CLIP
            inputs = self.processor(
                images=image, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize embeddings
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

            embedding = image_features.cpu().numpy()[0].tolist()
            return embedding

        except (OSError, IOError) as e:
            logger.error(f"Error loading image: {e}")
            return [0.0] * settings.EMBEDDING_DIM
        except RuntimeError as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * settings.EMBEDDING_DIM

    def encode_text(self, text: str) -> List[float]:
        """
        Generate embedding for text (useful for semantic search)

        Args:
            text: Text query

        Returns:
            Embedding vector as list
        """
        try:
            inputs = self.processor(
                text=[text], return_tensors="pt", padding=True
            ).to(self.device)

            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                # Normalize embeddings
                text_features = text_features / text_features.norm(
                    dim=-1, keepdim=True
                )

            embedding = text_features.cpu().numpy()[0].tolist()
            return embedding

        except RuntimeError as e:
            logger.error(f"Error generating text embedding: {e}")
            return [0.0] * settings.EMBEDDING_DIM

    def encode_images_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, str]],
        batch_size: int = 8
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple images in batches

        Args:
            images: List of images
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_embeddings = [self.encode_image(img) for img in batch]
            embeddings.extend(batch_embeddings)

            batch_num = i // batch_size + 1
            total_batches = (len(images) - 1) // batch_size + 1
            logger.info(
                f"Encoded batch {batch_num}/{total_batches}"
            )

        return embeddings
