"""
LLM Model Wrapper (Google Gemini only)
"""

from typing import Optional, List, Dict, Any, Union
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
import base64
from io import BytesIO

from src.config.settings import settings
from src.utils.logger import get_logger
from src.prompts import (
    SECURITY_ANALYSIS_SYSTEM_PROMPT,
    SECURITY_ANALYSIS_USER_PROMPT_TEMPLATE,
    VIDEO_SUMMARY_PROMPT_TEMPLATE,
    IMAGE_DESCRIPTION_PROMPT,
)

logger = get_logger(__name__)


class LLMWrapper:
    """Unified LLM wrapper for Google Gemini"""

    def __init__(self, provider: Optional[str] = None):
        """
        Initialize LLM with Google Gemini

        Args:
            provider: LLM provider (ignored, always uses 'google')
        """
        self.provider = "google"
        logger.info(f"Initializing LLM with provider: {self.provider}")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.7,
            max_output_tokens=1024,  # Limit output token count
        )

        logger.info("LLM initialized successfully")

    def analyze_frame_with_context(
        self,
        current_description: str,
        timestamp: str,
        telemetry: Dict[str, Any],
        past_n_frames: List[Dict[str, Any]],
        similar_past_events: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze current frame with full context

        Args:
            current_description: Description of current frame
            timestamp: Timestamp of current frame
            telemetry: Drone telemetry data
            past_n_frames: Recent N frames for context
            similar_past_events: Similar events from vector DB

        Returns:
            Analysis result with alerts
        """
        # Build context - use more frames for person tracking
        past_context = "\n".join(
            [
                f"- {frame.get('timestamp')}: {frame.get('description')}"
                for frame in past_n_frames[-15:]  # Last 15 frames
            ]
        )

        similar_context = "\n".join(
            [
                f"- {event.get('timestamp')}: {event.get('description')} "
                f"(similarity: {event.get('score', 0):.2f})"
                for event in similar_past_events[:3]  # Top 3 similar events
            ]
        )

        # Create prompt using templates from prompts package
        user_prompt = SECURITY_ANALYSIS_USER_PROMPT_TEMPLATE.format(
            timestamp=timestamp,
            current_description=current_description,
            location=telemetry.get('location', 'Unknown'),
            altitude=telemetry.get('altitude', 'N/A'),
            gps=telemetry.get('gps', 'N/A'),
            past_context=past_context if past_context else "No recent activity",
            similar_context=similar_context if similar_context else "No similar events found",
        )

        messages = [
            SystemMessage(content=SECURITY_ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self.llm.invoke(messages)

            # Parse response (assuming JSON format)
            import json
            import re

            # Handle response content - it could be a string or list
            if isinstance(response.content, list):
                # Extract text from list of content parts
                content = ""
                for part in response.content:
                    if isinstance(part, dict) and "text" in part:
                        content += part["text"]
                    elif isinstance(part, str):
                        content += part
                content = content.strip()
            else:
                content = response.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```"):
                # Extract content between ```json and ``` or ``` and ```
                pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    content = match.group(1)
                else:
                    # Try to remove just the ``` markers
                    content = re.sub(r'```(?:json)?', '', content).strip()
            
            try:
                analysis = json.loads(content)
                
                # Validate required fields
                if "threat_level" not in analysis:
                    logger.warning(
                        "Response missing 'threat_level', setting to LOW"
                    )
                    analysis["threat_level"] = "LOW"
                    
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse JSON response: {json_err}")
                logger.error(f"Response content: {content[:500]}")
                
                # Try to extract threat level from text
                threat_level = "LOW"  # Default to LOW instead of UNKNOWN
                content_lower = content.lower()
                
                critical_words = ["critical", "weapon", "gun", "armed"]
                high_words = ["high", "suspicious", "danger"]
                medium_words = ["medium", "concern", "loiter"]
                
                if any(word in content_lower for word in critical_words):
                    threat_level = "CRITICAL"
                elif any(word in content_lower for word in high_words):
                    threat_level = "HIGH"
                elif any(word in content_lower for word in medium_words):
                    threat_level = "MEDIUM"
                elif "none" in content_lower or "no threat" in content_lower:
                    threat_level = "NONE"
                
                # Fallback analysis with extracted threat level
                is_attention_needed = threat_level in ["HIGH", "CRITICAL"]
                analysis = {
                    "threat_level": threat_level,
                    "analysis": content,
                    "alerts": [],
                    "objects_detected": [],
                    "requires_attention": is_attention_needed,
                }

            return analysis

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Error connecting to LLM: {e}")
            return {
                "threat_level": "ERROR",
                "analysis": f"Connection error: {str(e)}",
                "alerts": [],
                "objects_detected": [],
                "requires_attention": False,
            }
        except ValueError as e:
            logger.error(f"Error in LLM analysis: {e}")
            return {
                "threat_level": "ERROR",
                "analysis": str(e),
                "alerts": [],
                "objects_detected": [],
                "requires_attention": False,
            }

    def generate_video_summary(
        self,
        all_frames_data: List[Dict[str, Any]],
        alerts: List[Dict[str, Any]],
    ) -> str:
        """
        Generate comprehensive video summary

        Args:
            all_frames_data: All processed frames with descriptions
            alerts: All generated alerts

        Returns:
            Summary text
        """
        # Aggregate information
        total_frames = len(all_frames_data)
        total_alerts = len(alerts)

        # Sample frames for summary
        sample_frames = all_frames_data[
            :: max(1, total_frames // 10)
        ]  # Sample ~10 frames

        frames_summary = "\n".join(
            [
                f"- {frame.get('timestamp')}: {frame.get('description')}"
                for frame in sample_frames
            ]
        )

        alerts_summary = "\n".join(
            [
                f"- [{alert.get('severity')}] {alert.get('message')} "
                f"at {alert.get('timestamp')}"
                for alert in alerts
            ]
        )

        if alerts_summary:
            alerts_text = alerts_summary
        else:
            alerts_text = "No alerts generated"
        
        prompt = VIDEO_SUMMARY_PROMPT_TEMPLATE.format(
            total_frames=total_frames,
            total_alerts=total_alerts,
            frames_summary=frames_summary,
            alerts_summary=alerts_text,
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Handle response content - it could be a string or list
            if isinstance(response.content, list):
                # Extract text from list of content parts
                summary = ""
                for part in response.content:
                    if isinstance(part, dict) and "text" in part:
                        summary += part["text"]
                    elif isinstance(part, str):
                        summary += part
                return summary.strip()
            else:
                return response.content
                
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Error connecting to LLM: {e}")
            return f"Error connecting to LLM: {str(e)}"
        except ValueError as e:
            logger.error(f"Error generating summary: {e}")
            return f"Error generating summary: {str(e)}"

    def describe_image(
        self,
        image: Union[Image.Image, str],
        prompt: str = IMAGE_DESCRIPTION_PROMPT,
    ) -> str:
        """
        Generate description for an image using Gemini Vision

        Args:
            image: PIL Image or path to image file
            prompt: Prompt to guide description

        Returns:
            Description string
        """
        try:
            # Load image if path is provided
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, Image.Image):
                image = image.convert("RGB")
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            logger.debug(f"Image converted to base64, size: {len(img_base64)}")
            logger.debug(f"Gemini input prompt: {prompt}")

            # Create message with image
            message_content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{img_base64}"
                }
            ]

            response = self.llm.invoke([HumanMessage(content=message_content)])

            # Handle response content - it could be a string or list
            if isinstance(response.content, list):
                # Extract text from list of content parts
                description = ""
                for part in response.content:
                    if isinstance(part, dict) and "text" in part:
                        description += part["text"]
                    elif isinstance(part, str):
                        description += part
                description = description.strip()
            else:
                description = response.content.strip()
                
            logger.info(f"Gemini LLM output: '{description}'")

            return description

        except (OSError, IOError) as e:
            logger.error(f"Error loading image: {e}")
            return "Error loading image"
        except Exception as e:
            logger.error(f"Error describing image with Gemini: {e}")
            return f"Error describing image: {str(e)}"

