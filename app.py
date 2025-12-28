"""
FastAPI Application for Drone Security Analyst
Provides REST API endpoints with comprehensive validation
"""

import argparse
import uvicorn
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field, field_validator
from src.config.settings import settings
from src.utils.logger import get_logger
from src.services import (
    VideoProcessingService,
    SearchService,
    QAAgentService,
    validate_video_file,
)

logger = get_logger(__name__)


# Pydantic Models for Request/Response Validation
class SearchRequest(BaseModel):
    """Request model for frame search"""

    query: str = Field(
        ..., min_length=3, max_length=500, description="Search query text"
    )
    top_k: int = Field(
        default=10, ge=1, le=100, description="Number of results to return"
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()


class SearchResult(BaseModel):
    """Response model for search results"""

    score: float = Field(..., description="Relevance score")
    timestamp: str = Field(..., description="Frame timestamp")
    description: str = Field(..., description="Frame description")
    frame_path: str = Field(..., description="Path to frame image")
    location: Optional[str] = Field(None, description="Telemetry location")


class SearchResponse(BaseModel):
    """Complete search response"""

    query: str
    total_results: int
    results: List[SearchResult]
    processing_time_ms: float


class ProcessingStatus(BaseModel):
    """Video processing status"""

    status: str
    message: str
    video_id: Optional[str] = None
    progress: Optional[float] = None


class ProcessingResult(BaseModel):
    """Video processing results"""

    video_id: str
    video_filename: str
    processing_time_seconds: float
    total_frames: int
    processed_frames: int
    total_alerts: int
    alerts: List[Dict]
    summary: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    vector_store_enabled: bool
    timestamp: str


class QuestionRequest(BaseModel):
    """Request model for Q&A"""

    video_id: str = Field(..., description="Video ID from processing")
    question: str = Field(
        ..., min_length=3, max_length=500, description="Question about the video"
    )
    use_vector_search: bool = Field(
        default=True, description="Use vector search for retrieval"
    )
    top_k: int = Field(
        default=10, ge=1, le=50, description="Number of frames to retrieve"
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Question cannot be empty or whitespace only")
        return v.strip()


class QuestionResponse(BaseModel):
    """Response model for Q&A"""

    question: str
    answer: str
    retrieval_method: str
    num_frames_used: int
    processing_time_ms: float
    supporting_frames: List[Dict]


# Initialize FastAPI app
app = FastAPI(
    title="Drone Security Analyst API",
    description="AI-powered surveillance video analysis with semantic search",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global instances
video_service = None
search_service = None
qa_service = None
processing_jobs = {}


# FastAPI Event Handlers
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    global video_service, search_service, qa_service
    logger.info("=" * 80)
    logger.info("DRONE SECURITY ANALYST API - STARTING")
    logger.info("=" * 80)

    try:
        video_service = VideoProcessingService()
        search_service = SearchService()
        qa_service = QAAgentService()
        logger.info("API initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise


# FastAPI Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Drone Security Analyst API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        vector_store_enabled=search_service.is_available(),
        timestamp=datetime.now().isoformat(),
    )


@app.post("/api/v1/search", response_model=SearchResponse)
async def search_frames(request: SearchRequest):
    """
    Search for frames matching a text description

    - **query**: Text description to search for
    - **top_k**: Number of results to return (1-100)
    """
    logger.info(f"Search request: '{request.query}' (top_k={request.top_k})")

    try:
        # Check if search service is available
        if not search_service.is_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector store is not enabled. "
                "Please configure PINECONE_API_KEY.",
            )

        # Perform search using service
        result = search_service.search_frames(query=request.query, top_k=request.top_k)

        # Convert to response model
        formatted_results = [SearchResult(**item) for item in result["results"]]

        return SearchResponse(
            query=result["query"],
            total_results=result["total_results"],
            results=formatted_results,
            processing_time_ms=result["processing_time_ms"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@app.post("/api/v1/qa", response_model=QuestionResponse)
async def answer_question(request: QuestionRequest):
    """
    Answer a follow-up question about a processed video

    - **video_id**: ID of the processed video
    - **question**: Question about the video content
    - **use_vector_search**: Whether to use vector search (default: True)
    - **top_k**: Number of frames to retrieve (1-50)
    """
    logger.info(f"Q&A request for video '{request.video_id}': '{request.question}'")

    try:
        # Check if video exists
        if request.video_id not in processing_jobs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video ID not found: {request.video_id}",
            )

        job = processing_jobs[request.video_id]

        # Check if processing is complete
        if job["status"] != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Video processing not complete. Status: {job['status']}",
            )

        # Check if Q&A service is available
        if not qa_service.is_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Q&A service is not available.",
            )

        # Get video context
        video_context = job.get("video_context", {})

        if not video_context:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Video context not available. "
                "Video may need to be reprocessed.",
            )

        # Answer question using Q&A service
        result = qa_service.answer_question(
            question=request.question,
            video_context=video_context,
            use_vector_search=request.use_vector_search,
            top_k=request.top_k,
        )

        return QuestionResponse(
            question=result["question"],
            answer=result["answer"],
            retrieval_method=result["retrieval_method"],
            num_frames_used=result["num_frames_used"],
            processing_time_ms=result["processing_time_ms"],
            supporting_frames=result["supporting_frames"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Q&A error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Q&A failed: {str(e)}",
        )


@app.post("/api/v1/process", response_model=ProcessingStatus)
async def process_video_endpoint(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Video file to process"),
):
    """
    Upload and process a surveillance video

    - **video**: Video file (MP4, AVI, MOV, MKV, FLV, WMV)
    - Max file size: 500MB
    """
    logger.info(f"Video upload: {video.filename}")

    try:
        # Read file to check size
        content = await video.read()
        file_size = len(content)

        # Validate file
        validate_video_file(
            filename=video.filename,
            content_type=video.content_type,
            file_size=file_size,
        )

        # Generate unique video ID
        video_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save uploaded file temporarily
        temp_dir = settings.DATA_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)
        temp_video_path = temp_dir / f"{video_id}_{video.filename}"

        with open(temp_video_path, "wb") as f:
            f.write(content)

        logger.info(f"Video saved: {temp_video_path}")

        # Add processing to background tasks
        background_tasks.add_task(
            process_video_background, str(temp_video_path), video_id, video.filename
        )

        # Store job status
        processing_jobs[video_id] = {
            "status": "processing",
            "filename": video.filename,
            "started_at": datetime.now().isoformat(),
        }

        return ProcessingStatus(
            status="accepted",
            message="Video uploaded successfully and processing started",
            video_id=video_id,
            progress=0.0,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}",
        )


@app.get("/api/v1/process/{video_id}", response_model=Dict)
async def get_processing_status(video_id: str):
    """Get processing status for a video"""
    if video_id not in processing_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video ID not found: {video_id}",
        )

    return processing_jobs[video_id]


def process_video_background(video_path: str, video_id: str, original_filename: str):
    """Background task to process video"""
    try:
        logger.info(f"Background processing started: {video_id}")

        # Process video using service
        results = video_service.process_video(video_path)

        # Update job status
        processing_jobs[video_id] = {
            "status": "completed",
            "filename": original_filename,
            "video_id": video_id,
            "started_at": processing_jobs[video_id]["started_at"],
            "completed_at": datetime.now().isoformat(),
            "results": {
                "processing_time_seconds": results["processing_time_seconds"],
                "total_frames": results["total_frames"],
                "processed_frames": results["processed_frames"],
                "total_alerts": results["total_alerts"],
                "alerts": results["alerts"],
                "summary": results["summary"],
            },
            # Store video context for Q&A
            "video_context": results.get("video_context", {}),
        }

        # Clean up temp file
        Path(video_path).unlink(missing_ok=True)

        logger.info(f"Background processing completed: {video_id}")

    except Exception as e:
        logger.error(f"Background processing error: {e}", exc_info=True)
        processing_jobs[video_id] = {
            "status": "failed",
            "filename": original_filename,
            "error": str(e),
            "started_at": processing_jobs[video_id]["started_at"],
            "failed_at": datetime.now().isoformat(),
        }


# CLI Support (for backward compatibility)
def main_cli():
    """Command-line interface (original functionality)"""
    parser = argparse.ArgumentParser(
        description="Drone Security Analyst - CLI/API Mode"
    )
    parser.add_argument(
        "video_path", type=str, nargs="?", help="Path to surveillance video"
    )
    parser.add_argument(
        "--search", type=str, help="Search frames by text description", default=None
    )
    parser.add_argument(
        "--top-k", type=int, help="Number of search results to return", default=10
    )
    parser.add_argument("--api", action="store_true", help="Run as API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")

    args = parser.parse_args()

    # Mode 1: Run API server
    if args.api:
        logger.info(f"Starting API server at {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
        return

    # Mode 2: Search frames
    if args.search:
        search_svc = SearchService()

        print("=" * 70)
        print(f"🔍 Searching for: '{args.search}'")
        print("=" * 70)

        if not search_svc.is_available():
            print("\n❌ ERROR: Vector store is not enabled!")
            return

        try:
            result = search_svc.search_frames(query=args.search, top_k=args.top_k)

            results = result["results"]
            print(f"\n✅ Found {len(results)} matching frames:\n")
            for i, item in enumerate(results, 1):
                score = item["score"]
                print(f"{i}. Score: {score:.1%} | " f"Time: {item['timestamp']}")
                print(f"   {item['description']}\n")

        except Exception as e:
            print(f"\n❌ Error: {e}")
        return

    # Mode 3: Process video
    if args.video_path:
        video_svc = VideoProcessingService()
        results = video_svc.process_video(args.video_path)

        print("\n" + "=" * 80)
        print("PROCESSING SUMMARY")
        print("=" * 80)
        print(f"Video: {results['video_path']}")
        time_sec = results["processing_time_seconds"]
        print(f"Processing Time: {time_sec:.2f} seconds")
        print(f"Frames: {results['processed_frames']}/" f"{results['total_frames']}")
        print(f"Total Alerts: {results['total_alerts']}")
        print("=" * 80 + "\n")
    else:
        parser.print_help()
        print("\n" + "=" * 70)
        print("💡 TIP: Use --api to start the API server")
        print("   python app.py --api")
        print("=" * 70)


if __name__ == "__main__":
    main_cli()
