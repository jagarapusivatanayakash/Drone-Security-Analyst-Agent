# Drone Security Analyst

AI-powered surveillance video analysis system with automated threat detection, semantic search, and intelligent alerting.

## Overview

This system processes surveillance video to detect security threats, index frames with AI-generated descriptions, and provide intelligent question-answering capabilities. It enhances property security through automated monitoring and real-time alert generation.

## Key Features

- **Automated Video Processing**: Extracts frames at configurable intervals
- **AI-Powered Analysis**: Uses Google Gemini Vision API for frame description
- **Intelligent Alerting**: Rule-based detection of weapons, suspicious activity, collisions, and loitering
- **Semantic Search**: Vector-based frame search using CLIP embeddings and Pinecone
- **Q&A Agent**: Ask questions about processed videos and get AI-powered answers
- **Frame-by-Frame Indexing**: Complete timestamp-based indexing of all frames

## Architecture

### Data Pipeline
```
Video Input → Frame Extraction → AI Description → Embedding → Vector Store
                                       ↓
                                Alert Engine → Alerts
                                       ↓
                                 LLM Summary → Results
```

### Components
- **Frame Extractor**: Extracts frames from video at specified intervals
- **CLIP Model**: Generates embeddings for semantic search
- **Gemini Vision**: Provides detailed frame descriptions
- **Alert Engine**: Rule-based threat detection
- **Vector Store**: Pinecone-based indexing for fast retrieval
- **LLM**: Generates video summaries and answers questions

## Requirements

### System Requirements
- Python 3.12+
- Windows (PowerShell)
- 8GB+ RAM recommended
- GPU optional (for faster embedding generation)

### API Keys Required
- Google Gemini API key (required)
- Pinecone API key (optional - for semantic search)

## Installation

### 1. Clone or Download
```powershell
cd C:\Users\<your-username>\Downloads\video-processing
```

### 2. Create Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Configure Environment

Create a `.env` file in the project root:

```env
# Required
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional (for semantic search)
PINECONE_API_KEY=your_pinecone_api_key_here
```

## Usage

### Quick Start

Process a video file:
```powershell
python app.py data\videos\stealing002.mp4
```

### Configuration

Edit `src/config/settings.py` to customize:
- Frame extraction interval (default: 2 seconds)
- Maximum frames per video (default: 200)
- Alert sensitivity
- Processing batch size

### Output

Results are saved to `data/outputs/` as JSON files containing:
- Frame descriptions with timestamps
- Detected alerts with severity levels
- Video summary
- Processing metadata

Example output structure:
```json
{
  "video_path": "data\\videos\\stealing002.mp4",
  "processing_time_seconds": 442.77,
  "total_frames": 32,
  "processed_frames": 32,
  "total_alerts": 10,
  "alerts": [...],
  "summary": "...",
  "captions": [...]
}
```

## Testing

### Running Tests

Run the complete test suite:

```powershell
pytest tests/test_prototype.py -v
```

Run specific test categories:

```powershell
# Frame indexing tests
pytest tests/test_prototype.py::TestFrameIndexing -v

# Alert triggering tests
pytest tests/test_prototype.py::TestAlertTriggering -v

# Data integrity tests
pytest tests/test_prototype.py::TestDataIntegrity -v

# Integration tests
pytest tests/test_prototype.py::TestIntegration -v

# Edge case tests
pytest tests/test_prototype.py::TestEdgeCases -v
```

Run with coverage report:

```powershell
pytest tests/test_prototype.py --cov=src --cov-report=html
```

### Test Coverage

The test suite verifies:

✅ **Frame Indexing** (Requirement 4)

- Frames stored with correct timestamps
- Metadata includes telemetry data
- Sequential frame processing
- **Key Test**: "Blue truck at gate" logged at 00:00:00 ✓

✅ **Alert Triggering** (Requirement 3)

- Weapon detection → CRITICAL alerts
- Suspicious activity → HIGH alerts
- Collision detection → CRITICAL alerts
- Loitering detection → MEDIUM alerts
- **Key Test**: Alert triggered correctly ✓

✅ **Data Integrity** (Requirement 4)

- Vehicle descriptions logged correctly
- Timestamps accurate
- Telemetry structure validated
- Frame IDs sequential
- **Key Test**: "Truck logged correctly" ✓

✅ **Integration Tests**

- Frame extractor processes videos
- Alert engine handles multiple frames
- Vector store graceful degradation

✅ **Edge Cases**

- Empty descriptions
- Special characters
- Very long text

### Test Results

```
19 passed in 0.30s
```

All tests passing confirms:

1. ✅ Frame-by-frame indexing works correctly
2. ✅ Alerts trigger at appropriate times
3. ✅ Data is logged with accurate timestamps
4. ✅ System handles edge cases gracefully

## API Endpoints (Optional)

If running as API server (`uvicorn app:app`):

- `GET /` - API information
- `GET /health` - Health check
- `POST /api/v1/process` - Process video
- `POST /api/v1/search` - Search frames
- `POST /api/v1/qa` - Ask questions

## Alert Rules

The system detects:

### Critical Alerts

- **Weapons**: Guns, knives, firearms detected
- **Collisions**: Vehicle accidents
- **Physical Violence**: Assault, fighting

### High Alerts

- **Suspicious Activity**: Breaking, forcing, tampering
- **Running**: Potential theft escape
- **Trespassing**: Unauthorized access

### Medium Alerts

- **Loitering**: Person in 3+ consecutive frames
- **After-hours Activity**: Context-dependent

## Project Structure

```

video-processing/
├── app.py                  # Main application entry point
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
├── README.md              # This file
├── data/
│   ├── videos/            # Input videos
│   ├── frames/            # Extracted frames
│   └── outputs/           # Processing results (JSON)
├── src/
│   ├── config/            # Configuration and settings
│   ├── database/          # Vector store implementation
│   ├── models/            # AI models (CLIP, Gemini, LLM)
│   ├── prompts/           # LLM prompts
│   ├── rules/             # Alert engine rules
│   ├── services/          # Business logic services
│   ├── state/             # Processing state management
│   ├── utils/             # Utilities and logging
│   ├── video/             # Video processing
│   └── workflows/         # LangGraph workflow
└── tests/
    └── test_prototype.py  # Comprehensive test suite
```

## Development

### Adding New Alert Rules

Edit `src/rules/alert_engine.py`:

```python
def evaluate_frame(self, frame_data, description, past_frames, analysis):
    alerts = []
    
    # Add your custom rule
    if "your_condition" in description.lower():
        alerts.append({
            "rule": "custom_rule",
            "severity": "HIGH",
            "message": "Custom alert message",
            "timestamp": frame_data["timestamp"],
            "frame_id": frame_data["frame_id"]
        })
    
    return alerts
```

### Adding New Tests

Add tests to `tests/test_prototype.py`:

```python
def test_your_new_feature(self, alert_engine, mock_frames):
    """Test description"""
    # Test implementation
    assert expected_result
```

## Troubleshooting

### Common Issues

**Error: Missing API Key**

```
Solution: Add GOOGLE_API_KEY to .env file
```

**Error: Module not found**

```
Solution: Activate virtual environment and install requirements
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Error: Video processing fails**

```
Solution: Check video format (MP4 recommended) and codec
```

**Tests failing**

```
Solution: Ensure all dependencies installed
pytest tests/test_prototype.py -v --tb=short
```

## Performance

- **Processing Speed**: ~2-4 seconds per frame (depends on API)
- **Memory Usage**: ~500MB-2GB (depends on video size)
- **Concurrent Processing**: Supports batch processing
- **Scalability**: Can be deployed with multiple workers

## Security Considerations

- API keys stored in `.env` (never commit)
- Frame data encrypted in transit to APIs
- Alerts logged for audit trail
- Video data stored locally

## Future Enhancements

- [ ] Real-time video stream processing
- [ ] Multi-camera support
- [ ] Custom ML model training
- [ ] Mobile alerts (SMS/Email)
- [ ] Dashboard UI
- [ ] Advanced analytics

## License

Proprietary - Internal Use Only

## Support

For issues or questions:
1. Check troubleshooting section
2. Review test output for errors
3. Check logs in console output
4. Verify API keys and configuration

## Acknowledgments

- Google Gemini for vision AI
- Pinecone for vector database
- LangChain/LangGraph for workflow orchestration
- OpenAI CLIP for embeddings

---

**Version**: 1.0.0  
**Last Updated**: December 27, 2025  
**Status**: Production Ready ✓
