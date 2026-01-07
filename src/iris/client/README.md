# IRIS Client Architecture

The IRIS client is a video streaming and analysis application with a Python backend (FastAPI) and React frontend.

## Directory Structure

```
client/
├── capture/                    # Video capture modules
│   ├── camera.py              # CameraCapture - webcam/device capture
│   └── video_file.py          # VideoFileCapture - file-based capture for analysis
├── streaming/
│   └── websocket_client.py    # StreamingClient - sends frames to inference server
├── web/                        # Web application
│   ├── app.py                 # FastAPI app setup, static file mounting
│   ├── routes.py              # API endpoints and WebSocket handlers
│   ├── dependencies.py        # AppState singleton, shared state management
│   └── frontend/              # React frontend (Vite + TypeScript)
└── config.py                   # Client configuration (Pydantic models)
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         IRIS Client                                  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     React Frontend                            │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐  │   │
│  │  │  LiveView  │  │ AnalysisView│  │     Zustand Store      │  │   │
│  │  │ (Camera)   │  │ (Video File)│  │   (useAppStore.ts)     │  │   │
│  │  └─────┬──────┘  └──────┬─────┘  └───────────┬────────────┘  │   │
│  │        │                │                     │               │   │
│  │        └───────────WebSocket──────────────────┘               │   │
│  │                   /ws/client, /ws/analysis                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   FastAPI Backend                             │   │
│  │  ┌─────────────────┐    ┌─────────────────┐                  │   │
│  │  │    routes.py    │    │ dependencies.py │                  │   │
│  │  │  - /api/*       │    │   AppState      │                  │   │
│  │  │  - /ws/*        │    │   (singleton)   │                  │   │
│  │  └────────┬────────┘    └────────┬────────┘                  │   │
│  │           │                      │                            │   │
│  │  ┌────────┴──────────────────────┴─────────┐                 │   │
│  │  │              Capture Layer               │                 │   │
│  │  │  CameraCapture │ VideoFileCapture       │                 │   │
│  │  └─────────────────┬───────────────────────┘                 │   │
│  │                    │                                          │   │
│  │  ┌─────────────────┴───────────────────────┐                 │   │
│  │  │          StreamingClient                 │                 │   │
│  │  │   WebSocket → Inference Server           │                 │   │
│  │  └─────────────────────────────────────────┘                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  IRIS Server    │
                    │  (Inference)    │
                    └─────────────────┘
```

## Key Components

### Capture Layer (`capture/`)

**CameraCapture** (`camera.py`)
- Captures frames from webcam/video devices using OpenCV
- Configurable FPS, resolution
- Returns JPEG-encoded frames

**VideoFileCapture** (`video_file.py`)
- Reads video files for offline analysis/benchmarking
- Simulates real-time streaming at configurable FPS
- Tracks position for playback synchronization

### Streaming Layer (`streaming/`)

**StreamingClient** (`websocket_client.py`)
- WebSocket client that connects to the inference server
- Sends frames with metadata (frame_id, timestamp, fps)
- Receives inference results and forwards to callback
- Auto-reconnects with exponential backoff

### Web Layer (`web/`)

**FastAPI Backend** (`app.py`, `routes.py`)
- Serves React frontend as static files
- REST API for control (start/stop, config, camera selection)
- WebSocket endpoints:
  - `/ws/client` - Unified control plane (preview frames, results, status updates)
  - `/ws/analysis` - Analysis mode progress/results

**AppState** (`dependencies.py`)
- Singleton holding runtime state
- Camera instance, streaming client, results history
- Analysis session state (video capture, annotations, results)

### Frontend (`web/frontend/`)

**React + Vite + TypeScript**
- UI components in `components/`
- Global state with Zustand (`store/useAppStore.ts`)
- WebSocket hooks in `hooks/`
- Types in `types/`

**Key Views:**
- `LiveView` - Real-time camera inference
- `AnalysisView` - Video file analysis with ground truth comparison

## Data Flow

### Live Inference Mode
```
Browser ← Preview WS ← CameraCapture
                          │
                          ├─→ StreamingClient ──→ Inference Server
                          │                              │
Browser ← Results WS ←────┴──────────────────────────────┘
```

### Analysis Mode
```
VideoFileCapture ──→ Analysis WS ──→ Inference Server
       │                                    │
       └── Position/Progress ───→ Frontend  │
                                     ▲      │
                                     └──────┘
                              (Results + GT comparison)
```

## Sampling Formula

For video analysis with uniform frame sampling:

```
Given:
  T = segment duration (seconds)
  s = number of frames to sample

Then:
  capture_fps = s / T  (frames per second)

Example:
  T = 2 seconds, s = 4 frames
  capture_fps = 4 / 2 = 2 FPS
  frames sampled at: 0s, 0.5s, 1.0s, 1.5s
```

The client sends frames at `capture_fps`. The server handles:
- `frames_per_job` (buffer_size): How many frames per inference batch
- `frame_overlap` (k): Frames shared between consecutive batches

## Running the Client

```bash
# Development mode (with hot reload)
uv run python -m iris.client.web.run_dev

# Production mode
uv run python -m iris.client.web.app
```

Frontend development:
```bash
cd src/iris/client/web/frontend
npm install
npm run dev    # Development server
npm run build  # Production build
```

## Configuration

See `config.py` for:
- `ServerConfig` - Inference server connection
- `VideoConfig` - Capture settings (FPS, resolution, quality)
- `SSHTunnelConfig` - Remote server tunneling
