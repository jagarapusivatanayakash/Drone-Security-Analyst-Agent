"""
Configuration settings for the Drone Security Analyst
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings"""

    # Project paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    VIDEOS_DIR = DATA_DIR / "videos"
    FRAMES_DIR = DATA_DIR / "frames"
    OUTPUTS_DIR = DATA_DIR / "outputs"

    # Model settings
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    EMBEDDING_DIM = 512

    # LLM settings
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Vector database settings (Pinecone)
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_HOST = os.getenv("PINECONE_HOST")

    # Video processing settings
    FRAME_EXTRACTION_METHOD = "interval"  # scene or interval
    SCENE_THRESHOLD = 30.0
    FRAME_INTERVAL = 2  # seconds (extract 1 frame every 2 seconds)
    MAX_FRAMES_PER_VIDEO = 1000

    # Agent settings
    MEMORY_WINDOW_SIZE = 10  # Number of recent frames to keep in memory
    SIMILARITY_SEARCH_TOP_K = 5
    ALERT_THRESHOLD = 0.7

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = str(BASE_DIR / "logs" / "app.log")

    @classmethod
    def ensure_directories(cls):
        """Create necessary directories"""
        for directory in [cls.VIDEOS_DIR, cls.FRAMES_DIR, cls.OUTPUTS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
