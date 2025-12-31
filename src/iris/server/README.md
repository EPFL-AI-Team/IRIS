# IRIS Server Architecture

The IRIS server is a video inference server that receives frames, batches them, and runs VLM (Vision Language Model) inference.

## Directory Structure

```
server/
├── app.py                     # Main FastAPI app, WebSocket endpoint, lifespan
├── main.py                    # Entry point (uvicorn runner)
├── config.py                  # Server configuration (Pydantic models)
├── dependencies.py            # ServerState singleton
├── frame_buffer.py            # FrameBuffer for batching frames
├── lifecycle.py               # Startup/shutdown lifecycle hooks
├── logging_handler.py         # WebSocket log streaming handler
├── metrics.py                 # MetricsCollector for performance tracking
├── inference/                 # Inference execution layer
│   ├── executor.py           # InferenceExecutor - async worker pool
│   └── jobs/                 # Job implementations
│       ├── base.py           # BaseJob abstract class
│       ├── single_frame.py   # SingleFrameJob (deprecated)
│       └── video.py          # VideoJob - batched video inference
├── jobs/                      # Job management layer
│   ├── manager.py            # JobManager - orchestrates jobs
│   ├── factory.py            # JobFactory - creates jobs
│   ├── types.py              # Job type definitions
│   └── config.py             # Job configuration
├── routes/                    # HTTP/WS routes (modular)
│   ├── system.py             # Health, status endpoints
│   ├── jobs.py               # Job management endpoints
│   ├── video.py              # Video-specific endpoints
│   └── websocket.py          # WebSocket handlers
└── services/                  # Business logic services
```

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                          IRIS Server                                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                    WebSocket Endpoint                          │     │
│  │                      /ws/stream                                │     │
│  │  ┌─────────────┐              ┌─────────────┐                 │     │
│  │  │ Receive Loop│              │  Send Loop  │                 │     │
│  │  │ (frames in) │              │ (results out)│                │     │
│  │  └──────┬──────┘              └──────▲──────┘                 │     │
│  └─────────┼────────────────────────────┼────────────────────────┘     │
│            │                            │                               │
│  ┌─────────▼────────────────────────────┴────────────────────┐         │
│  │                    FrameBuffer                             │         │
│  │     Accumulates frames until batch is ready                │         │
│  │     (buffer_size frames, with overlap_frames overlap)      │         │
│  └─────────────────────────┬─────────────────────────────────┘         │
│                            │                                            │
│  ┌─────────────────────────▼─────────────────────────────────┐         │
│  │                    JobManager                              │         │
│  │     Creates VideoJob instances for each batch              │         │
│  │     Tracks active jobs, handles callbacks                  │         │
│  └─────────────────────────┬─────────────────────────────────┘         │
│                            │                                            │
│  ┌─────────────────────────▼─────────────────────────────────┐         │
│  │                 InferenceExecutor                          │         │
│  │  ┌─────────────────────────────────────────────────────┐  │         │
│  │  │              Async Queue                             │  │         │
│  │  │    Jobs submitted → Queue → Workers pick up         │  │         │
│  │  └─────────────────────────────────────────────────────┘  │         │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐              │         │
│  │  │  Worker 0 │  │  Worker 1 │  │  Worker N │   ...        │         │
│  │  │  (Model)  │  │  (Model)  │  │  (Model)  │              │         │
│  │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘              │         │
│  │        │              │              │                     │         │
│  │        └──────────────┴──────────────┘                     │         │
│  │                       │                                    │         │
│  │               VLM Inference (Qwen2.5-VL)                   │         │
│  └───────────────────────┬───────────────────────────────────┘         │
│                          │                                              │
│                          ▼                                              │
│              Results → Callback → WebSocket → Client                    │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### Connection Layer

**WebSocket Endpoint** (`app.py` `/ws/stream`)
- Accepts client connections
- Creates per-connection FrameBuffer
- Runs send/receive loops concurrently
- Handles graceful shutdown on disconnect

### Buffering Layer

**FrameBuffer** (`frame_buffer.py`)
- Accumulates incoming frames
- Parameters:
  - `buffer_size`: Number of frames per batch (default: 8)
  - `overlap_frames`: Frames retained between batches (default: 4)
- `is_ready()`: Returns True when buffer_size frames accumulated
- `get_batch()`: Returns current frames as list of PIL Images
- `slide_window()`: Keeps last `overlap_frames` for next batch

### Job Layer

**VideoJob** (`inference/jobs/video.py`)
- Represents a single inference batch
- Contains frames, prompt, configuration
- Executes inference via ProcessPoolExecutor
- Calls result_callback with inference output

**JobManager** (`jobs/manager.py`)
- Orchestrates job lifecycle
- Tracks active jobs
- Routes log callbacks to clients

### Execution Layer

**InferenceExecutor** (`inference/executor.py`)
- Manages worker pool (ProcessPoolExecutor for GPU isolation)
- Each worker loads its own model instance
- Async queue for job submission
- Handles backpressure (max_queue_size)

### Model Layer

**VLM Inference** (via jobs)
- Currently uses Qwen2.5-VL-3B-Instruct
- Configurable via `config.yaml` or environment
- Supports different hardware profiles (GPU, CPU, Apple Silicon)

## Data Flow

```
1. Client connects to /ws/stream
2. Client sends frames as JSON: {frame: base64, frame_id, timestamp, fps}
3. Server decodes frame, adds to FrameBuffer
4. When FrameBuffer.is_ready():
   a. Create VideoJob with batch of frames
   b. Submit to InferenceExecutor queue
   c. Slide buffer window (keep overlap frames)
5. Worker picks up job from queue:
   a. Process frames through VLM
   b. Generate response (action, tool, target, context)
   c. Call result_callback with output
6. Result sent back to client via WebSocket
```

## Configuration

### Server Config (`config.py`)
```python
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8001
    max_queue_size: int = 50      # Max pending jobs
    num_workers: int = 1          # Inference workers
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vlm_hardware: str = "auto"    # auto, cuda, mps, cpu
    model_dtype: str = "bfloat16"
```

### Job Config (via `config.yaml`)
```yaml
jobs:
  video:
    buffer_size: 8         # Frames per batch
    overlap_frames: 4      # Frame overlap between batches
    default_fps: 5         # Default FPS if not specified by client
    prompt: "Analyze..."   # VLM prompt
    max_new_tokens: 128    # Max output tokens
```

## Graceful Shutdown

1. First SIGINT/SIGTERM: Mark shutdown, stop accepting new jobs
2. Wait for active jobs to complete
3. Second signal: Force immediate shutdown

## Running the Server

```bash
# Standard start
uv run python -m iris.server.main

# With custom config
IRIS_MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct" uv run python -m iris.server.main
```

## Performance Considerations

### Batching Trade-offs
- Larger `buffer_size` = more context, higher latency
- Smaller `buffer_size` = less context, lower latency
- `overlap_frames` provides temporal continuity

### Queue Management
- `max_queue_size` prevents memory exhaustion
- When queue full, new submissions block
- Monitor queue_depth in logs

### GPU Memory
- One model instance per worker
- Multiple workers = multiple model copies
- Single worker recommended for consumer GPUs
