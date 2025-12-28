"""
LangGraph Workflow with Gemini Vision and CLIP Processing
"""

from langgraph.graph import StateGraph, END
from PIL import Image

from src.state import ProcessingState
from src.models.gemini_vision import GeminiVisionDescriber
from src.models.clip_model import CLIPEmbedder
from src.models.llm_model import LLMWrapper
from src.database.vector_store import VectorStore
from src.rules.alert_engine import AlertRulesEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DroneSecurityWorkflow:
    """LangGraph workflow for drone security analysis"""

    def __init__(self):
        """Initialize workflow with all components"""
        logger.info("Initializing Drone Security Workflow")

        # Initialize models
        self.gemini_vision = GeminiVisionDescriber()
        self.clip = CLIPEmbedder()
        self.llm = LLMWrapper()

        # Initialize databases
        self.vector_store = VectorStore()

        # Initialize rules engine
        self.alert_engine = AlertRulesEngine()

        # Build graph
        self.graph = self._build_graph()

        logger.info("Workflow initialized successfully")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ProcessingState)

        # Add nodes
        workflow.add_node("parallel_processing", self.parallel_process_frame)
        workflow.add_node("store_in_vector_db", self.store_frame)
        workflow.add_node("retrieve_context", self.retrieve_context)
        workflow.add_node("analyze_with_llm", self.analyze_frame)
        workflow.add_node("check_alert_rules", self.check_alerts)
        workflow.add_node("generate_summary", self.generate_summary)

        # Define edges
        workflow.set_entry_point("parallel_processing")
        workflow.add_edge("parallel_processing", "store_in_vector_db")
        workflow.add_edge("store_in_vector_db", "retrieve_context")
        workflow.add_edge("retrieve_context", "analyze_with_llm")
        workflow.add_edge("analyze_with_llm", "check_alert_rules")

        # Conditional edge: continue or summarize
        workflow.add_conditional_edges(
            "check_alert_rules",
            self.should_continue,
            {"continue": "parallel_processing", "end": "generate_summary"},
        )

        workflow.add_edge("generate_summary", END)

        return workflow.compile()

    def parallel_process_frame(self, state: ProcessingState) -> ProcessingState:
        """
        Process current frame with Gemini Vision and CLIP
        """
        current_idx = state.get("current_frame_idx", 0)
        frames = state.get("frames", [])

        if current_idx >= len(frames):
            return state

        current_frame = frames[current_idx]
        frame_path = current_frame["frame_path"]
        timestamp = current_frame["timestamp"]

        logger.info(f"Processing frame {current_idx + 1}/{len(frames)}: {timestamp}")

        try:
            # Load image
            image = Image.open(frame_path).convert("RGB")

            # Generate description with Gemini Vision
            logger.info("Generating Gemini Vision description...")
            description = self.gemini_vision.describe_frame(image)

            # Generate embedding with CLIP
            logger.info("Generating CLIP embedding...")
            embedding = self.clip.encode_image(image)

            # Update state
            descriptions = state.get("descriptions", [])
            embeddings = state.get("embeddings", [])

            descriptions.append(description)
            embeddings.append(embedding)

            state["descriptions"] = descriptions
            state["embeddings"] = embeddings

            logger.info(f"Description: {description}")

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            errors = state.get("errors", [])
            errors.append(str(e))
            state["errors"] = errors

        return state

    def store_frame(self, state: ProcessingState) -> ProcessingState:
        """Store frame data in vector database"""
        current_idx = state.get("current_frame_idx", 0)
        frames = state.get("frames", [])
        descriptions = state.get("descriptions", [])
        embeddings = state.get("embeddings", [])

        if current_idx >= len(descriptions):
            return state

        current_frame = frames[current_idx]
        description = descriptions[current_idx]
        embedding = embeddings[current_idx]
        timestamp = current_frame["timestamp"]

        logger.info(f"Storing frame {current_idx} with timestamp: {timestamp}")

        try:
            # Store in vector database
            frame_id = self.vector_store.insert_frame(
                embedding=embedding,
                timestamp=timestamp,
                description=description,
                telemetry=current_frame["telemetry"],
                frame_path=current_frame["frame_path"],
            )

            stored_ids = state.get("stored_frame_ids", [])
            stored_ids.append(frame_id)
            state["stored_frame_ids"] = stored_ids

            logger.debug(f"Stored frame {frame_id} at {timestamp}")

        except Exception as e:
            logger.error(f"Error storing frame: {e}")
            errors = state.get("errors", [])
            errors.append(str(e))
            state["errors"] = errors

        return state

    def retrieve_context(self, state: ProcessingState) -> ProcessingState:
        """Retrieve relevant context from vector database"""
        current_idx = state.get("current_frame_idx", 0)
        embeddings = state.get("embeddings", [])
        descriptions = state.get("descriptions", [])
        frames = state.get("frames", [])

        if current_idx >= len(embeddings):
            return state

        current_embedding = embeddings[current_idx]

        logger.info("Retrieving similar past frames...")

        try:
            # Search for similar frames
            similar_frames = self.vector_store.search_similar(
                query_embedding=current_embedding,
                top_k=5,
                exclude_recent_n=3,  # Exclude very recent frames
            )

            state["similar_frames"] = similar_frames

            # Get past 30 frames for sequential context
            # This tracks person-vehicle associations over time
            start_idx = max(0, current_idx - 30)
            past_frames = []
            for i in range(start_idx, current_idx):
                if i < len(descriptions):
                    past_frames.append(
                        {
                            "timestamp": frames[i]["timestamp"],
                            "description": descriptions[i],
                            "telemetry": frames[i]["telemetry"],
                        }
                    )

            state["past_n_frames"] = past_frames

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            state["errors"] = state.get("errors", []) + [str(e)]

        return state

    def analyze_frame(self, state: ProcessingState) -> ProcessingState:
        """Analyze frame with LLM using full context"""
        current_idx = state.get("current_frame_idx", 0)
        descriptions = state.get("descriptions", [])
        frames = state.get("frames", [])
        past_n_frames = state.get("past_n_frames", [])
        similar_frames = state.get("similar_frames", [])

        if current_idx >= len(descriptions):
            return state

        current_frame = frames[current_idx]
        description = descriptions[current_idx]

        logger.info("Analyzing frame with LLM...")

        try:
            # LLM analyzes with full context
            analysis = self.llm.analyze_frame_with_context(
                current_description=description,
                timestamp=current_frame["timestamp"],
                telemetry=current_frame["telemetry"],
                past_n_frames=past_n_frames,
                similar_past_events=similar_frames,
            )

            # Store analysis
            state["analysis_results"] = state.get("analysis_results", []) + [analysis]

            logger.info(f"Threat Level: {analysis.get('threat_level')}")

        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            state["errors"] = state.get("errors", []) + [str(e)]

        return state

    def check_alerts(self, state: ProcessingState) -> ProcessingState:
        """Check alert rules and generate alerts"""
        current_idx = state.get("current_frame_idx", 0)
        analysis_results = state.get("analysis_results", [])
        frames = state.get("frames", [])

        # Track which frames have been alerted already (use list not set)
        alerted_frames = state.get("alerted_frames", [])

        if current_idx >= len(analysis_results):
            return state

        current_analysis = analysis_results[current_idx]
        current_frame = frames[current_idx]
        timestamp = current_frame["timestamp"]

        logger.info(f"Checking alert rules for frame {current_idx}...")
        logger.debug(f"Alerted frames so far: {alerted_frames}")

        try:
            # Check if this frame was already alerted
            alert_key = f"{current_idx}_{timestamp}"

            if alert_key in alerted_frames:
                logger.info(
                    f"Frame {current_idx} ({timestamp}) already alerted, skipping"
                )
            else:
                # Generate alerts from analysis ONLY if requires_attention
                if current_analysis.get("requires_attention"):
                    alert_message = current_analysis.get("alerts", "")

                    if alert_message and alert_message.strip():
                        logger.info(f"Generating alert for frame {current_idx}")

                        # Create single alert with timestamp
                        alert = {
                            "severity": current_analysis.get("threat_level"),
                            "message": alert_message,
                            "timestamp": timestamp,
                            "frame_id": current_idx,
                            "description": state["descriptions"][current_idx],
                        }
                        alerts = state.get("alerts", [])
                        alerts.append(alert)
                        state["alerts"] = alerts

                        # Log with timestamp
                        severity = current_analysis.get("threat_level")
                        logger.warning(
                            f"ALERT [{severity}] {timestamp}: {alert_message}"
                        )

                        # Mark as alerted (append to list)
                        alerted_frames.append(alert_key)
                        state["alerted_frames"] = alerted_frames
                        logger.debug(f"Added {alert_key} to alerted_frames")
                else:
                    # No alerts needed for this frame
                    threat_level = current_analysis.get("threat_level", "NONE")
                    logger.info(
                        f"Frame {current_idx} ({timestamp}): "
                        f"{threat_level} - No alerts generated"
                    )

            # Increment frame counter
            state["current_frame_idx"] = current_idx + 1
            processed_count = state.get("processed_frames_count", 0)
            state["processed_frames_count"] = processed_count + 1

        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
            errors = state.get("errors", [])
            errors.append(str(e))
            state["errors"] = errors

        return state

    def should_continue(self, state: ProcessingState) -> str:
        """Determine if processing should continue or end"""
        current_idx = state.get("current_frame_idx", 0)
        total_frames = len(state.get("frames", []))

        logger.info(f"should_continue: current_idx={current_idx}, total={total_frames}")

        if current_idx < total_frames:
            logger.info("Continuing to next frame...")
            return "continue"
        else:
            logger.info("All frames processed, ending...")
            return "end"

    def generate_summary(self, state: ProcessingState) -> ProcessingState:
        """Generate comprehensive video summary"""
        logger.info("Generating video summary...")

        try:
            # Prepare data for summary
            frames_data = []
            for i, frame in enumerate(state.get("frames", [])):
                if i < len(state.get("descriptions", [])):
                    frames_data.append(
                        {
                            "timestamp": frame["timestamp"],
                            "description": state["descriptions"][i],
                            "telemetry": frame["telemetry"],
                        }
                    )

            # Generate summary using LLM
            summary = self.llm.generate_video_summary(
                all_frames_data=frames_data, alerts=state.get("alerts", [])
            )

            state["video_summary"] = summary

            logger.info("=" * 80)
            logger.info("VIDEO SUMMARY")
            logger.info("=" * 80)
            logger.info(summary)
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            state["video_summary"] = f"Error generating summary: {str(e)}"

        return state

    def run(self, initial_state: ProcessingState) -> ProcessingState:
        """Run the complete workflow"""
        logger.info("Starting workflow execution...")

        # Delete old frames for this video before processing
        video_path = initial_state.get("video_path", "")
        if video_path:
            from pathlib import Path

            video_name = Path(video_path).stem  # e.g., 'stealing002'

            if self.vector_store.enabled:
                logger.info(f"Checking for existing frames for video: {video_name}")
                deleted_count = self.vector_store.delete_frames_by_video(video_name)
                if deleted_count > 0:
                    logger.info(
                        f"✓ Deleted {deleted_count} old frames " f"before reprocessing"
                    )
                else:
                    logger.info("✓ No old frames found, processing fresh")

        # Calculate required recursion limit based on number of frames
        # Each frame goes through ~6 nodes, so we need at least frames * 6
        num_frames = len(initial_state.get("frames", []))
        # 10x for safety margin
        required_recursion_limit = max(100, num_frames * 10)

        logger.info(
            f"Processing {num_frames} frames with "
            f"recursion limit: {required_recursion_limit}"
        )

        # Invoke graph with increased recursion limit
        final_state = self.graph.invoke(
            initial_state, config={"recursion_limit": required_recursion_limit}
        )
        logger.info("Workflow completed!")
        return final_state
